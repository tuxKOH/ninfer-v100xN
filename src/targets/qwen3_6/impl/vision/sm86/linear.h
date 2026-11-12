#pragma once

#include "core/tensor.h"

#include <cuda_runtime_api.h>

namespace ninfer::targets::qwen3_6::vision_sm86 {

// Enqueue out[N,T] = weight[N,K] * x[K,T]. All tensors and weight planes
// are device-resident on the current device. The caller owns them and keeps
// them alive until stream work completes. Input and output must not overlap.
// Activations are contiguous BF16; weights use canonical row-split storage.
// No workspace or synchronization is required. Invalid inputs and CUDA launch
// errors throw; asynchronous errors are reported by later stream synchronization.
void linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6::vision_sm86
