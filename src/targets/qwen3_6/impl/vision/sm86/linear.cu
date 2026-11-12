#include "targets/qwen3_6/impl/vision/sm86/linear.h"

#include "core/device.h"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"

#include <algorithm>

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

namespace detail = ninfer::ops::detail;
struct AuxQ5RowSplitSimtSchedule : detail::Q5RowSplitSimtSchedule {};
struct AuxW8RowSplitSimtSchedule : detail::W8RowSplitSimtSchedule {};

// Keep auxiliary kernel registrations distinct from text kernels compiled for another SM.

// The shared SIMT GEMM accepts a format schedule. Its scalar tail covers
// non-aligned K without reading the padded activation region.
template <int Bits>
struct ScalarCodec {
    static constexpr int kGroupK = 64;

    __device__ static void load_pair(const std::uint8_t* codes, const std::uint8_t* high,
                                     const std::uint8_t* scales, std::int64_t group, int lane,
                                     float& w0, float& w1) {
        if constexpr (Bits == 4) {
            const auto packed = codes[group * 32 + lane];
            const auto scale = *reinterpret_cast<const std::uint16_t*>(scales + group * 2);
            detail::Q4SimtDecodeAtom::decode_pair(packed, scale, w0, w1);
        } else {
            detail::Q6ScalarDecodeAtom::load_pair(codes, high, scales, group, lane, w0, w1);
        }
    }
};

template <int Bits>
struct SimtSchedule {
    using Codec = ScalarCodec<Bits>;
    static constexpr int kNibU4 = 32;
    static constexpr int kHighU4 = Bits == 6 ? 16 : 0;
    static constexpr int kScaleU32 = 8;
    static constexpr int kHighBytesPerGroup = Bits == 6 ? 16 : 0;

    __device__ static void dequant_chunk(const uint4* codes, const uint4* high,
                                         const std::uint32_t* scales, int phase, int lane,
                                         float (&weights)[8]) {
        const auto packed = reinterpret_cast<const std::uint32_t*>(codes)[phase * 32 + lane];
        const auto scale = reinterpret_cast<const std::uint16_t*>(scales)[phase * 4 + (lane >> 3)];
        if constexpr (Bits == 4) {
            detail::Q4SimtDecodeAtom::decode_eight(packed, scale, weights);
        } else {
            const auto bits = reinterpret_cast<const std::uint16_t*>(high)[phase * 32 + lane];
            detail::Q6SimtDecodeAtom::decode_eight(packed, bits, scale, weights);
        }
    }
};

template <class Schedule>
void launch_q_simt(const __nv_bfloat16* x, const Weight& weight, __nv_bfloat16* out,
                   int tokens, cudaStream_t stream) {
    constexpr int rows_per_block = 8;
    constexpr int cols_per_block = 8;
    const dim3 grid((static_cast<unsigned>(weight.n) + rows_per_block - 1) / rows_per_block,
                    (static_cast<unsigned>(tokens) + cols_per_block - 1) / cols_per_block);
    const int full_slabs = weight.k % 8 == 0 ? weight.k / 1024 : 0;
    detail::q5_rowsplit_gemm_simt_kernel<Schedule, cols_per_block, rows_per_block, 2>
        <<<grid, rows_per_block * 32, 0, stream>>>(
            x, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales), out, nullptr,
            weight.n, weight.n, weight.k, tokens, weight.padded_shape[1], full_slabs);
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
    constexpr int cols_per_block = 8;
    constexpr int max_tokens_per_launch = 65535 * cols_per_block;
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    auto* output = static_cast<__nv_bfloat16*>(out.data);
    for (std::int64_t offset = 0; offset < x.ne[1]; offset += max_tokens_per_launch) {
        const int tokens = static_cast<int>(
            std::min<std::int64_t>(max_tokens_per_launch, x.ne[1] - offset));
        const auto* x_slice = input + offset * weight.k;
        auto* out_slice = output + offset * weight.n;
        switch (weight.qtype) {
        case QType::Q4G64_F16S:
            launch_q_simt<SimtSchedule<4>>(x_slice, weight, out_slice, tokens, stream);
            break;
        case QType::Q5G64_F16S:
            launch_q_simt<AuxQ5RowSplitSimtSchedule>(x_slice, weight, out_slice, tokens, stream);
            break;
        case QType::Q6G64_F16S:
            launch_q_simt<SimtSchedule<6>>(x_slice, weight, out_slice, tokens, stream);
            break;
        case QType::W8G32_F16S: {
            const dim3 grid((static_cast<unsigned>(weight.n) + 7) / 8,
                            (static_cast<unsigned>(tokens) + 7) / 8);
            const int full_slabs = weight.k % 8 == 0 ? weight.k / 1024 : 0;
            const detail::W8ContiguousOutput output_tile{out_slice, weight.n};
            detail::w8_rowsplit_gemm_simt_kernel<AuxW8RowSplitSimtSchedule, 8, 8, 2, false>
                <<<grid, 256, 0, stream>>>(
                    x_slice, static_cast<const std::uint8_t*>(weight.qdata),
                    static_cast<const std::uint8_t*>(weight.scales), output_tile,
                    weight.n, weight.k, tokens, weight.padded_shape[1], full_slabs);
            break;
        }
        default: throw std::invalid_argument("vision linear: unsupported weight format");
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::targets::qwen3_6::vision_sm86
