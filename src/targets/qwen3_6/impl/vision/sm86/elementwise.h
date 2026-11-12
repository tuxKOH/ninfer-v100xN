#pragma once

#include "core/tensor.h"
#include <cuda_runtime_api.h>

namespace ninfer::targets::qwen3_6::vision_sm86 {

void add_bias(const Tensor& bias, Tensor& x, cudaStream_t stream);
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps,
                Tensor& out, cudaStream_t stream);
void position_add(const Tensor& table, const Tensor& indices, const Tensor& weights,
                  Tensor& x, cudaStream_t stream);
// Positions are contiguous [P,2], with one P-element plane per axis.
// Q and K are [72,16,P] with independent byte strides between tokens.
void rope_2d(const Tensor& positions, Tensor& q, Tensor& k, cudaStream_t stream);
void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream);
void gelu_tanh(Tensor& x, cudaStream_t stream);
void gelu_exact(Tensor& x, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6::vision_sm86
