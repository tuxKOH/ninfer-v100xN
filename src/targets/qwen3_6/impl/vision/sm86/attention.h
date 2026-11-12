#pragma once

#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_6::vision_sm86 {

// Tiled attention uses no device workspace. Validates the patch and segment counts.
std::size_t attention_workspace_bytes(std::int32_t patches, std::int32_t segments);

// cu_seqlens is device I32 [segments + 1]. Its contents are checked before launch,
// which synchronizes stream. Inputs can have distinct token strides and need only
// BF16 alignment. Inputs must remain unchanged until the launch completes.
void attention_cu_seqlens(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& cu_seqlens, Tensor& out, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_6::vision_sm86
