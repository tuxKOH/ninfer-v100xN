#include "ops/linear/w8/w8_prepack_sm70.h"

#include "core/arena.h"
#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/volta_mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear/w8/w8_dispatch.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

__global__ void prepack_w8_qpn_kernel(const std::uint8_t* input_codes,
                                      const std::uint16_t* input_scales,
                                      std::uint8_t* output_codes,
                                      std::uint16_t* output_scales, int n, int k) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = static_cast<std::int64_t>(n) * (k / 8);
    if (index >= count) { return; }
    const int lane = static_cast<int>(index % 32);
    const int slice = static_cast<int>((index / 32) % (k / 8));
    const int tile = static_cast<int>(index / (static_cast<std::int64_t>(k / 8) * 32));
    const int row = tile * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
    reinterpret_cast<uint2*>(output_codes)[index] =
        reinterpret_cast<const uint2*>(input_codes)[static_cast<std::int64_t>(row) * (k / 8) + slice];
    if ((slice & 3) == 0) {
        output_scales[(static_cast<std::int64_t>(tile) * (k / 32) + slice / 4) * 32 + lane] =
            input_scales[static_cast<std::int64_t>(row) * (k / 32) + slice / 4];
    }
}

// Preserve the row-split SIMT route's lane-local ascending eight-value phases and
// FP32 FMA/shuffle association for T=1/2. Only the global addresses are permuted.
template <int Columns>
__global__ __launch_bounds__(256, 4) void w8_qpn_simt_kernel(
    const std::uint8_t* codes, const std::uint16_t* scales, const __nv_bfloat16* x,
    __nv_bfloat16* out, int n, int k, int t, int out_ld) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    __shared__ __align__(16) std::uint8_t code_tile[32][264];
    __shared__ std::uint16_t scale_tile[32][8];
    float accum[4][Columns] = {};
    for (int base = 0; base < k; base += 256) {
        // Load coalesced in QPN order, transpose locally to the original SIMT lane map.
        for (int index = threadIdx.x; index < 32 * 32; index += 256) {
            const int source_lane = index & 31;
            const int slice = index >> 5;
            const int local_row = ((source_lane >> 2) & 3) * 8 + (source_lane & 3) +
                                  ((source_lane & 16) ? 4 : 0);
            *reinterpret_cast<uint2*>(&code_tile[local_row][slice * 8]) =
                __ldg(reinterpret_cast<const uint2*>(codes) +
                    (static_cast<std::int64_t>(tile) * (k / 8) + base / 8 + slice) * 32 + source_lane);
        }
        const int source_lane = threadIdx.x & 31;
        const int group = threadIdx.x >> 5;
        const int local_row = ((source_lane >> 2) & 3) * 8 + (source_lane & 3) +
                              ((source_lane & 16) ? 4 : 0);
        scale_tile[local_row][group] = scales[
            (static_cast<std::int64_t>(tile) * (k / 32) + base / 32 + group) * 32 + source_lane];
        __syncthreads();
        const int kk = base + lane * 8;
        for (int row_index = 0; row_index < 4; ++row_index) {
            const int row = warp + row_index * 8;
            const uint2 packed = *reinterpret_cast<const uint2*>(&code_tile[row][lane * 8]);
            const float scale = __half2float(__ushort_as_half(scale_tile[row][lane / 4]));
            float w[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const auto word = j < 4 ? packed.x : packed.y;
                w[j] = static_cast<float>(static_cast<std::int8_t>(word >> ((j & 3) * 8))) * scale;
            }
#pragma unroll
            for (int token = 0; token < Columns; ++token) {
                if (token < t) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(
                        x + static_cast<std::int64_t>(token) * k + kk);
                    const float2 values[] = {bf16x2_bits_to_float2(raw.x), bf16x2_bits_to_float2(raw.y),
                                             bf16x2_bits_to_float2(raw.z), bf16x2_bits_to_float2(raw.w)};
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float value = (j & 1) ? values[j / 2].y : values[j / 2].x;
                        accum[row_index][token] = fmaf(w[j], value, accum[row_index][token]);
                    }
                }
            }
        }
        __syncthreads();
    }
    for (int row_index = 0; row_index < 4; ++row_index) {
        const int row = tile * 32 + warp + row_index * 8;
#pragma unroll
        for (int token = 0; token < Columns; ++token) {
            if (token < t) {
                const float value = warp_reduce_sum(accum[row_index][token]);
                if (lane == 0) {
                    out[static_cast<std::int64_t>(token) * out_ld + row] = __float2bfloat16(value);
                }
            }
        }
    }
}

// Each lane owns one represented row; warps divide the original FP32 dot product.
// This avoids the shared transpose needed by the legacy vocabulary GEMV layout.
// The MTP-only profile changes reduction association, not the decoded W8 values.
template <int Split>
__global__ __launch_bounds__(Split * 32) void w8_qpn_mtp_gemv_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint16_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) +
                    ((lane & 16) ? 4 : 0);
    const int groups = k / 32;
    const int first = warp * (groups / Split);
    const int end = warp == Split - 1 ? groups : first + groups / Split;
    float accumulator = 0.0F;
    for (int group = first; group < end; ++group) {
        const auto scale_index = (static_cast<std::int64_t>(blockIdx.x) * groups + group) * 32 + lane;
        const float scale = __half2float(__ushort_as_half(scales[scale_index]));
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const uint2 code = __ldg(reinterpret_cast<const uint2*>(codes) +
                (static_cast<std::int64_t>(blockIdx.x) * (k / 8) + group * 4 + sub) * 32 + lane);
            const uint4 raw = *reinterpret_cast<const uint4*>(x + group * 32 + sub * 8);
            const auto* activation = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const auto word = j < 4 ? code.x : code.y;
                const float weight = static_cast<float>(
                    static_cast<std::int8_t>(word >> ((j & 3) * 8))) * scale;
                accumulator = fmaf(weight, __bfloat162float(activation[j]), accumulator);
            }
        }
    }
    __shared__ float sums[Split][32];
    sums[warp][lane] = accumulator;
    __syncthreads();
    if (warp == 0 && row < n) {
        float sum = 0.0F;
#pragma unroll
        for (int part = 0; part < Split; ++part) { sum += sums[part][lane]; }
        out[row] = __float2bfloat16(sum);
    }
}

// Vocabulary heads retain four K partitions and their original association. MTP
// projections use shape-specific splits. Every eight-code load is coalesced.
template <int Split = 4, class Activation = __nv_bfloat16, int Unroll = 4>
__global__ __launch_bounds__(Split * 32, 32 / Split) void w8_qpn_prepacked_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint16_t* __restrict__ scales,
    const Activation* __restrict__ x, __nv_bfloat16* __restrict__ out,
    int n, int k, int t, int out_ld) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups = k / 32;
    const int first = warp * (groups / Split);
    const int end = warp == Split - 1 ? groups : first + groups / Split;
    const std::int64_t scale_base = static_cast<std::int64_t>(blockIdx.x) * groups * 32 + lane;
    __shared__ float sums[Split][256];
    float c[8] = {};
    const half2 bias = __half2half2(__ushort_as_half(0x6480));
#pragma unroll Unroll
    for (int group = first; group < end; ++group) {
        const half2 scale = __half2half2(__ushort_as_half(scales[scale_base + group * 32]));
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const uint2 raw = __ldg(reinterpret_cast<const uint2*>(codes) +
                (static_cast<std::int64_t>(blockIdx.x) * (k / 8) + group * 4 + sub) * 32 + lane);
            const unsigned words[2] = {raw.x ^ 0x80808080U, raw.y ^ 0x80808080U};
            half2 b[4], a[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned bits = __byte_perm(
                    words[j / 2], 0x64646464U, (j & 1) ? 0x4342U : 0x4140U);
                b[j] = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&bits), bias), scale);
            }
            if (r < t) {
                const uint4 raw_x = *reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(r) * k + group * 32 + sub * 8);
                const auto* value = reinterpret_cast<const Activation*>(&raw_x);
                half sequential[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if constexpr (std::is_same_v<Activation, half>) {
                        sequential[j] = value[j];
                    } else {
                        sequential[j] = __float2half(__bfloat162float(value[j]));
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) { a[j] = *reinterpret_cast<const half2*>(sequential + 2 * j); }
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) { a[j] = __float2half2_rn(0.0f); }
            }
            const auto* A = reinterpret_cast<const unsigned*>(a);
            const auto* B = reinterpret_cast<const unsigned*>(b);
            volta_mma_qp_n(c, A[0], A[1], B[0], B[1]);
            volta_mma_qp_n(c, A[2], A[3], B[2], B[3]);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int col = (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
        sums[warp][row * 32 + qp * 8 + col] = c[i];
    }
    __syncthreads();
    for (int index = threadIdx.x; index < 256; index += Split * 32) {
        const int row = index / 32;
        const int col = blockIdx.x * 32 + index % 32;
        if (row < t && col < n) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < Split; ++w) { sum += sums[w][index]; }
            out[static_cast<std::int64_t>(row) * out_ld + col] = __float2bfloat16(sum);
        }
    }
}

// Materialize the same BF16->FP16 operands once, instead of converting them in
// every vocabulary CTA. No quantization, reduction or output boundary changes.
__global__ void w8_input_image_kernel(const __nv_bfloat16* input, half* output, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) { output[index] = __float2half(__bfloat162float(input[index])); }
}

} // namespace

void w8_prepack_qpn_sm70(Weight& weight, cudaStream_t stream) {
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group != 32 || weight.n <= 0 ||
        !((weight.k == 5120 && (weight.n == 248320 || weight.n == 124160 ||
             weight.n == 62080 || weight.n == 5120 || weight.n == 3584 ||
             weight.n == 7168 || weight.n == 8704 || weight.n == 17408 ||
             weight.n == 4096 || weight.n == 14336 || weight.n == 34816)) ||
          (weight.n == 5120 && (weight.k == 1536 || weight.k == 2560 || weight.k == 4352 ||
                                weight.k == 3072 || weight.k == 8704 || weight.k == 4096 ||
                                weight.k == 6144 || weight.k == 10240 || weight.k == 17408 || weight.k == 25600))) ||
        weight.padded_shape[0] != weight.n || weight.padded_shape[1] != weight.k) {
        throw std::invalid_argument("W8 QPN prepack requires a registered row-split vocabulary, MTP or DFlash projection");
    }
    // A QPN tile permutes 32 complete rows. Repack independent row bands in place
    // so startup never needs a second full vocabulary head.
    const int band_rows = std::min(weight.n, 4096);
    const auto band_bytes = static_cast<std::size_t>(band_rows) * weight.k;
    DeviceBuffer scratch(band_bytes + band_bytes / 32 * sizeof(std::uint16_t));
    auto* packed_codes = static_cast<std::uint8_t*>(scratch.p);
    auto* packed_scales = reinterpret_cast<std::uint16_t*>(packed_codes + band_bytes);
    auto* codes = static_cast<std::uint8_t*>(const_cast<void*>(weight.qdata));
    auto* scales = static_cast<std::uint16_t*>(const_cast<void*>(weight.scales));
    for (int row = 0; row < weight.n; row += band_rows) {
        const int rows = std::min(band_rows, weight.n - row);
        const auto offset = static_cast<std::size_t>(row) * weight.k;
        const auto bytes = static_cast<std::size_t>(rows) * weight.k;
        prepack_w8_qpn_kernel<<<static_cast<unsigned>((bytes / 8 + 255) / 256), 256, 0, stream>>>(
            codes + offset, scales + offset / 32, packed_codes, packed_scales, rows, weight.k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(codes + offset, packed_codes, bytes, cudaMemcpyDeviceToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(scales + offset / 32, packed_scales, bytes / 32 * sizeof(*scales),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    weight.layout = QuantLayout::VoltaQpnPrepacked;
}

void launch_w8_prepacked_sm70(const Tensor& x, const Weight& weight, Tensor& out,
                               WorkspaceArena* workspace, cudaStream_t stream) {
    const int t = x.ne[1];
    const int out_ld = static_cast<int>(out.nb[1] / sizeof(__nv_bfloat16));
    const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const std::uint16_t*>(weight.scales);
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    auto* output = static_cast<__nv_bfloat16*>(out.data);
    const bool mtp = weight.n <= 34816 && !weight.w8_dflash_projection;
    if (!mtp && workspace != nullptr && w8_prepacked_workspace_bytes(weight.n, weight.k, t, t) != 0) {
        auto scope = workspace->scope();
        Tensor image = workspace->alloc(DType::FP16, {weight.k, t});
        auto* fp16 = static_cast<half*>(image.data);
        const int elements = weight.k * t;
        w8_input_image_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(input, fp16, elements);
        w8_qpn_prepacked_kernel<4, half, 1><<<weight.n / 32, 128, 0, stream>>>(
            codes, scales, fp16, output, weight.n, weight.k, t, out_ld);
    } else if (t == 1 && mtp) {
        w8_qpn_mtp_gemv_kernel<16><<<weight.n / 32, 512, 0, stream>>>(
            codes, scales, input, output, weight.n, weight.k);
    } else if (t >= 3 && t <= 8 && mtp) {
        if (weight.n == 5120) {
            w8_qpn_prepacked_kernel<16><<<weight.n / 32, 512, 0, stream>>>(
                codes, scales, input, output, weight.n, weight.k, t, out_ld);
        } else {
            w8_qpn_prepacked_kernel<8><<<weight.n / 32, 256, 0, stream>>>(
                codes, scales, input, output, weight.n, weight.k, t, out_ld);
        }
    } else if (t == 1) {
        w8_qpn_simt_kernel<1><<<weight.n / 32, 256, 0, stream>>>(
            codes, scales, input, output, weight.n, weight.k, t, out_ld);
    } else if (t == 2) {
        w8_qpn_simt_kernel<2><<<weight.n / 32, 256, 0, stream>>>(
            codes, scales, input, output, weight.n, weight.k, t, out_ld);
    } else if (t <= 8) {
        w8_qpn_prepacked_kernel<4><<<weight.n / 32, 128, 0, stream>>>(
            codes, scales, input, output, weight.n, weight.k, t, out_ld);
    } else {
        launch_w8_volta_mma(x, weight, out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
