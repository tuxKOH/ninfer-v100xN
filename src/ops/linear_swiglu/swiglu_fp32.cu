#include "ops/linear_swiglu/swiglu_fp32.h"

#include "core/device.h"
#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <bool RowScaled>
__global__ void swiglu_fp32_kernel(const float* __restrict__ projected,
                                   const __nv_bfloat16* __restrict__ scales,
                                   __nv_bfloat16* __restrict__ output,
                                   std::int64_t intermediate, std::int32_t tokens, std::int64_t output_stride) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= intermediate * tokens) { return; }
    const std::int64_t token = index / intermediate;
    const std::int64_t row = index - token * intermediate;
    const std::int64_t gate_index = token * (intermediate * 2) + row;
    float gate = projected[gate_index];
    float up = projected[gate_index + intermediate];
    if constexpr (RowScaled) {
        gate *= __bfloat162float(scales[row]);
        up *= __bfloat162float(scales[row + intermediate]);
    }
    output[token * output_stride + row] = __float2bfloat16_rn(silu(gate) * up);
}

} // namespace

void swiglu_fp32_launch(const Tensor& projected, const void* row_scales, Tensor& out,
                        cudaStream_t stream) {
    const std::int64_t intermediate = out.ne[0];
    const std::int64_t count = intermediate * out.ne[1];
    constexpr int kThreads = 256;
    const unsigned blocks = static_cast<unsigned>((count + kThreads - 1) / kThreads);
    if (row_scales != nullptr) {
        swiglu_fp32_kernel<true><<<blocks, kThreads, 0, stream>>>(
            static_cast<const float*>(projected.data),
            static_cast<const __nv_bfloat16*>(row_scales),
            static_cast<__nv_bfloat16*>(out.data), intermediate, out.ne[1], out.nb[1] / sizeof(__nv_bfloat16));
    } else {
        swiglu_fp32_kernel<false><<<blocks, kThreads, 0, stream>>>(
            static_cast<const float*>(projected.data), nullptr,
            static_cast<__nv_bfloat16*>(out.data), intermediate, out.ne[1], out.nb[1] / sizeof(__nv_bfloat16));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
