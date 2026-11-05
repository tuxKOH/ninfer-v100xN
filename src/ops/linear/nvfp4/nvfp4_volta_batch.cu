#include "ops/linear/nvfp4/nvfp4_volta_batch.cuh"

#include "core/layout.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"

#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <class Input>
__global__ void pack_batch_activation(const Input* __restrict__ input,
                                      half2* __restrict__ output, int k, int tokens,
                                      int padded) {
    const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= k * padded / 2) { return; }
    const int i = pair * 2;
    const int group = i / (padded * 16);
    const int row = (i / 16) % padded;
    const int col = i % 16;
    half2 value = __float2half2_rn(0.0F);
    if (row < tokens) {
        const auto* source = input + static_cast<std::int64_t>(row) * k + group * 16 + col;
        if constexpr (std::is_same_v<Input, half>) {
            value = *reinterpret_cast<const half2*>(source);
        } else {
            const auto represented = *reinterpret_cast<const __nv_bfloat162*>(source);
            value = __floats2half2_rn(__bfloat162float(represented.x),
                                      __bfloat162float(represented.y));
        }
    }
    output[pair] = value;
}

} // namespace

bool nvfp4_volta_batch_supported(int n, int k, int tokens) noexcept {
    // Exact TP2/TP4 dense shapes. Other TP dispatch and all T<=8 single-request
    // schedules keep their existing implementation.
    const bool shape = (k == 5120 && (n == 3584 || n == 4096 || n == 7168 ||
                                     n == 8192 || n == 8704 || n == 17408)) ||
                       (n == 5120 && (k == 1536 || k == 3072 || k == 4352 || k == 8704));
    return shape && tokens >= 9 && tokens <= 64;
}

std::size_t nvfp4_volta_batch_workspace_bytes(int n, int k, int max_tokens) {
    if (!nvfp4_volta_batch_supported(n, k, std::min(max_tokens, 64))) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP16, {16, nvfp4_volta_batch_tiles(std::min(max_tokens, 64)) * 8,
                                    k / 16});
    return layout.peak_bytes(1);
}

Tensor nvfp4_volta_batch_pack(const Tensor& input, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    const int k = input.ne[0];
    const int tokens = input.ne[1];
    const int padded = nvfp4_volta_batch_tiles(tokens) * 8;
    if (tokens < 9 || tokens > 64 || k % 16 != 0) {
        throw std::invalid_argument("NVFP4 batch activation outside the fixed decode domain");
    }
    Tensor output = workspace.alloc(DType::FP16, {16, padded, k / 16});
    const int blocks = (k * padded / 2 + 255) / 256;
    if (input.dtype == DType::FP16) {
        pack_batch_activation<<<blocks, 256, 0, stream>>>(
            static_cast<const half*>(input.data), static_cast<half2*>(output.data),
            k, tokens, padded);
    } else {
        pack_batch_activation<<<blocks, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input.data), static_cast<half2*>(output.data),
            k, tokens, padded);
    }
    CUDA_CHECK(cudaGetLastError());
    return output;
}

void launch_nvfp4_volta_batch(const Tensor& input, const Weight& weight, Tensor& output,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    launch_nvfp4_volta_batch_output(
        input, weight, Nvfp4ContiguousOutput{static_cast<__nv_bfloat16*>(output.data), weight.n},
        weight.n, workspace, stream);
}

} // namespace ninfer::ops::detail
