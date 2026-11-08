#include "targets/qwen3_6/impl/vision/sm86/linear.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::vision_sm86 {
namespace {

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("vision linear: size overflow");
    }
    return a * b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("vision linear: size overflow");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t n, std::uint64_t alignment) {
    return checked_add(n, alignment - 1) / alignment * alignment;
}

bool aligned(const void* p, std::uintptr_t alignment) {
    return p && (reinterpret_cast<std::uintptr_t>(p) % alignment == 0);
}

std::uintptr_t end_address(const void* p, std::uint64_t bytes) {
    const auto start = reinterpret_cast<std::uintptr_t>(p);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - start) {
        throw std::overflow_error("vision linear: address overflow");
    }
    return start + static_cast<std::uintptr_t>(bytes);
}

bool overlaps(const void* a, std::uint64_t as, const void* b, std::uint64_t bs) {
    return reinterpret_cast<std::uintptr_t>(a) < end_address(b, bs) &&
           reinterpret_cast<std::uintptr_t>(b) < end_address(a, as);
}

std::uint64_t validate_tensor(const Tensor& t, int rows, int cols) {
    const auto row_bytes = checked_mul(static_cast<std::uint64_t>(rows), 2);
    const auto total = checked_mul(row_bytes, static_cast<std::uint64_t>(cols));
    if (t.dtype != DType::BF16 || !aligned(t.data, 16) ||
        t.ne[0] != rows || t.ne[1] != cols || t.ne[2] != 1 || t.ne[3] != 1 ||
        row_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        total > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        (rows != 1 && t.nb[0] != 2) ||
        (cols != 1 && t.nb[1] != static_cast<std::int64_t>(row_bytes))) {
        throw std::invalid_argument("vision linear: expected aligned contiguous BF16 matrix");
    }
    (void)end_address(t.data, total);
    return total;
}

void validate_weight(const Weight& w) {
    int group = 64;
    int high_bytes = 0;
    switch (w.qtype) {
    case QType::Q4G64_F16S: break;
    case QType::Q5G64_F16S: high_bytes = 8; break;
    case QType::Q6G64_F16S: high_bytes = 16; break;
    case QType::W8G32_F16S: group = 32; break;
    default: throw std::invalid_argument("vision linear: unsupported weight format");
    }
    if (w.n <= 0 || w.k <= 0 || w.ndim != 2 || w.layout != QuantLayout::RowSplit ||
        w.group != group || w.group_size != static_cast<std::uint32_t>(group) ||
        w.scale_dtype != DType::FP16 || w.shape[0] != w.n || w.shape[1] != w.k ||
        w.shape[2] != 1 || w.shape[3] != 1 || w.padded_shape[0] != w.n ||
        w.padded_shape[2] != 1 || w.padded_shape[3] != 1 || !aligned(w.payload, 16)) {
        throw std::invalid_argument("vision linear: invalid row-split weight metadata");
    }
    const auto padded_k = align_up(static_cast<std::uint64_t>(w.k), 128);
    if (padded_k > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        w.padded_shape[1] != static_cast<std::int32_t>(padded_k)) {
        throw std::invalid_argument("vision linear: invalid padded weight shape");
    }
    const auto groups = checked_mul(static_cast<std::uint64_t>(w.n), padded_k / group);
    const auto low_size = checked_mul(groups, 32);
    const auto high_size = checked_mul(groups, static_cast<std::uint64_t>(high_bytes));
    const auto high_offset = align_up(low_size, 256);
    const auto scale_offset = checked_add(high_offset, align_up(high_size, 256));
    const auto required = checked_add(scale_offset, checked_mul(groups, 2));
    const auto base = reinterpret_cast<std::uintptr_t>(w.payload);
    (void)end_address(w.payload, required);
    if (w.payload_bytes < required || w.high_plane_bytes != high_size ||
        w.qdata != w.payload ||
        reinterpret_cast<std::uintptr_t>(w.scales) != base + scale_offset ||
        (high_size ? reinterpret_cast<std::uintptr_t>(w.qhigh) != base + high_offset
                   : w.qhigh != nullptr)) {
        throw std::invalid_argument("vision linear: invalid row-split weight planes");
    }
}

template <int Bits>
__global__ void linear_kernel(const __nv_bfloat16* x, const std::uint8_t* codes,
                              const std::uint8_t* high, const __half* scales,
                              __nv_bfloat16* out, int n, int k, int padded_k,
                              std::uint64_t total) {
    constexpr int group_size = Bits == 8 ? 32 : 64;
    constexpr int high_per_group = Bits == 5 ? 8 : (Bits == 6 ? 16 : 0);
    const int lane = threadIdx.x & 31;
    const std::uint64_t warp = (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const std::uint64_t warp_count = static_cast<std::uint64_t>(gridDim.x) * blockDim.x / 32;
    const int groups_per_row = padded_k / group_size;
    for (std::uint64_t index = warp; index < total; index += warp_count) {
        const int row = static_cast<int>(index % n);
        const std::uint64_t token = index / n;
        float sum = 0.0f;
        for (int col = lane; col < k; col += 32) {
            const std::uint64_t g = static_cast<std::uint64_t>(row) * groups_per_row + col / group_size;
            const int within = col % group_size;
            int q;
            if constexpr (Bits == 8) {
                const int byte = codes[g * 32 + within];
                q = byte < 128 ? byte : byte - 256;
            } else {
                const int packed = codes[g * 32 + within / 2];
                int value = (packed >> ((within & 1) * 4)) & 15;
                if constexpr (Bits == 5 || Bits == 6) {
                    constexpr int high_bits = Bits - 4;
                    const int bit = within * high_bits;
                    value |= ((high[g * high_per_group + bit / 8] >> (bit % 8)) &
                              ((1 << high_bits) - 1)) << 4;
                }
                q = (value ^ (1 << (Bits - 1))) - (1 << (Bits - 1));
            }
            const float weight = static_cast<float>(q) * __half2float(scales[g]);
            sum = fmaf(weight, __bfloat162float(x[token * k + col]), sum);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) out[index] = __float2bfloat16_rn(sum);
    }
}

// Reuse each decoded FP32 weight across 16 tokens. Keeping the dequantized
// values in FP32 preserves the FP16 scale rather than rounding it to BF16.
template <int Bits>
__global__ void linear_tiled_kernel(const __nv_bfloat16* x, const std::uint8_t* codes,
                                    const std::uint8_t* high, const __half* scales,
                                    __nv_bfloat16* out, int n, int k, int padded_k,
                                    int tokens, std::uint64_t tiles) {
    constexpr int tile = 16;
    constexpr int tile_k = 32;
    constexpr int group_size = Bits == 8 ? 32 : 64;
    constexpr int high_per_group = Bits == 5 ? 8 : (Bits == 6 ? 16 : 0);
    __shared__ float weights[tile_k][tile + 1];
    __shared__ float inputs[tile_k][tile + 1];
    const int tid = threadIdx.x;
    const int local_row = tid % tile;
    const int local_token = tid / tile;
    const std::uint64_t row_tiles = (static_cast<std::uint64_t>(n) + tile - 1) / tile;
    const int groups_per_row = padded_k / group_size;

    for (std::uint64_t block = blockIdx.x; block < tiles; block += gridDim.x) {
        const std::uint64_t row0 = (block % row_tiles) * tile;
        const std::uint64_t token0 = (block / row_tiles) * tile;
        const std::uint64_t row = row0 + local_row;
        const std::uint64_t token = token0 + local_token;
        float sum = 0.0f;
        for (std::int64_t k0 = 0; k0 < k; k0 += tile_k) {
            for (int item = tid; item < tile * tile_k; item += tile * tile) {
                const int kk = item % tile_k;
                const int r = item / tile_k;
                const std::uint64_t weight_row = row0 + r;
                const std::int64_t col = k0 + kk;
                float value = 0.0f;
                if (weight_row < static_cast<std::uint64_t>(n) && col < k) {
                    const std::uint64_t g = weight_row * groups_per_row + col / group_size;
                    const int within = static_cast<int>(col % group_size);
                    int q;
                    if constexpr (Bits == 8) {
                        const int byte = codes[g * 32 + within];
                        q = byte < 128 ? byte : byte - 256;
                    } else {
                        const int packed = codes[g * 32 + within / 2];
                        int code = (packed >> ((within & 1) * 4)) & 15;
                        if constexpr (Bits == 5 || Bits == 6) {
                            constexpr int high_bits = Bits - 4;
                            const int bit = within * high_bits;
                            code |= ((high[g * high_per_group + bit / 8] >> (bit % 8)) &
                                     ((1 << high_bits) - 1)) << 4;
                        }
                        q = (code ^ (1 << (Bits - 1))) - (1 << (Bits - 1));
                    }
                    value = static_cast<float>(q) * __half2float(scales[g]);
                }
                weights[kk][r] = value;
                const std::uint64_t input_token = token0 + r;
                inputs[kk][r] = input_token < static_cast<std::uint64_t>(tokens) && col < k
                    ? __bfloat162float(x[input_token * static_cast<std::uint64_t>(k) + col])
                    : 0.0f;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < tile_k; ++kk) {
                sum = fmaf(weights[kk][local_row], inputs[kk][local_token], sum);
            }
            __syncthreads();
        }
        if (row < static_cast<std::uint64_t>(n) &&
            token < static_cast<std::uint64_t>(tokens)) {
            out[token * static_cast<std::uint64_t>(n) + row] = __float2bfloat16_rn(sum);
        }
    }
}

} // namespace

void linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    validate_weight(weight);
    if (x.ne[1] <= 0) throw std::invalid_argument("vision linear: token count must be positive");
    const auto x_bytes = validate_tensor(x, weight.k, x.ne[1]);
    const auto out_bytes = validate_tensor(out, weight.n, x.ne[1]);
    if (overlaps(x.data, x_bytes, out.data, out_bytes) ||
        overlaps(weight.payload, weight.payload_bytes, out.data, out_bytes)) {
        throw std::invalid_argument("vision linear: output overlaps an input");
    }
    const auto total = checked_mul(static_cast<std::uint64_t>(weight.n),
                                   static_cast<std::uint64_t>(x.ne[1]));
    constexpr unsigned threads = 128;
    const auto wanted_blocks = (total + 3) / 4;
    const unsigned blocks = static_cast<unsigned>(wanted_blocks < 65535 ? wanted_blocks : 65535);
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* high = static_cast<const std::uint8_t*>(weight.qhigh);
    const auto* scales = static_cast<const __half*>(weight.scales);
    auto* output = static_cast<__nv_bfloat16*>(out.data);
#define VISION_LINEAR_LAUNCH(bits) \
    do { \
        if (weight.n >= 16 && x.ne[1] >= 16) { \
            const auto row_tiles = (static_cast<std::uint64_t>(weight.n) + 15) / 16; \
            const auto token_tiles = (static_cast<std::uint64_t>(x.ne[1]) + 15) / 16; \
            const auto tiles = checked_mul(row_tiles, token_tiles); \
            const unsigned tile_blocks = static_cast<unsigned>(tiles < 65535 ? tiles : 65535); \
            linear_tiled_kernel<bits><<<tile_blocks, 256, 0, stream>>>(input, codes, high, \
                scales, output, weight.n, weight.k, weight.padded_shape[1], x.ne[1], tiles); \
        } else { \
            linear_kernel<bits><<<blocks, threads, 0, stream>>>(input, codes, high, scales, \
                output, weight.n, weight.k, weight.padded_shape[1], total); \
        } \
    } while (false)
    switch (weight.qtype) {
    case QType::Q4G64_F16S: VISION_LINEAR_LAUNCH(4); break;
    case QType::Q5G64_F16S: VISION_LINEAR_LAUNCH(5); break;
    case QType::Q6G64_F16S: VISION_LINEAR_LAUNCH(6); break;
    case QType::W8G32_F16S: VISION_LINEAR_LAUNCH(8); break;
    default: throw std::invalid_argument("vision linear: unsupported weight format");
    }
#undef VISION_LINEAR_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_6::vision_sm86
