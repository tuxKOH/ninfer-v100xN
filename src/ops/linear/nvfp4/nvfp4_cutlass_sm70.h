#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                              std::int32_t cols);
void nvfp4_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream);

// Private fused-Op projection stage; the FP32 output has no BF16 rounding boundary.
// Uses the same transient weight/input workspace query as the BF16 form.
std::size_t nvfp4_cutlass_sm70_full_swiglu_workspace_bytes(int cols);
void nvfp4_cutlass_sm70_full_swiglu_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           WorkspaceArena& ws, cudaStream_t stream);
void nvfp4_cutlass_sm70_fp32_launch(const Tensor& x, const Weight& w, Tensor& out,
                                    WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
