#pragma once

// ninfer::ops - fused residual += W @ x.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h" // ExecutionContext, PeerEvents (tp2 split form)
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

/**
 * Returns the A16-only transient capacity required by LinearAdd for every T in the inclusive
 * [min_tokens,max_tokens] interval, for every supported weight layout. The QType and dimensions
 * are the fixed implementation profile. The result can exceed the storage used by a call.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Op: linear_add
 *
 * Math / indexing:
 *   ideal[:,t] = residual[:,t] + Linear(x,w)[:,t].
 *
 * Logical shapes:
 *   Contiguous BF16 x [K,T] and residual [N,T]. Registered weights are GGML_K `ggml-k256-v1`,
 *   Q5G64_F16S RowSplit
 *   [5120,17408] or [5120,6144], W8G32_F16S RowSplit [2048,4096] or [2048,6144], NVFP4
 *   BlockScaleK16M128x4 [5120,6144] or [5120,17408], row-scaled
 *   FP8_E4M3FN_ROW_BF16S [5120,6144] or [5120,17408], or BF16_CTRL Contiguous [5120,6144]. T may
 *   be any positive value.
 *
 * Numeric:
 *   The oracle reads a registered BF16 weight directly or exact-decodes a registered packed
 *   weight, then evaluates `ideal` naively in FP64 from the represented inputs. The updated BF16
 *   residual is promoted and compared directly with that result; output storage rounding belongs
 *   to LinearAdd's selected A16, A8, or A4 criterion, not the oracle. Production routes may fuse
 *   or materialize the projection and may choose their natural accumulator, activation
 *   quantization, staging, and workspace precision; those private choices are not semantic
 *   rounding boundaries.
 *
 * Compute policy:
 *   GGML_K, Q5, W8, and BF16_CTRL admit only A16Only. NVFP4 admits A16Only and AllowA4. Row-scaled FP8
 *   admits A16Only and AllowA8. Its two semantic registrations own independent production plans:
 *   [5120,6144] resolves T<22 to A16 and T>=22 to A8, while [5120,17408] resolves T<25 to A16 and
 *   T>=25 to A8. A permissive policy allows the private resolver to select either qualified
 *   arithmetic profile; it does not itself prescribe a kernel.
 *
 * Effects:
 *   Updates the full residual tensor in place; x/weight must not alias residual.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_add_workspace_capacity_bytes(), scoped to
 *   the call. Query the selected profile even for A16: Volta routes can require projection,
 *   split-K, or dequantization storage. Other routes may have a static-zero requirement.
 *   There is no persistent state side effect.
 */
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& ws,
                cudaStream_t stream);

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                WorkspaceArena& ws, cudaStream_t stream);

// Exact GDN input permutation fused into a GGML_K projection. The represented input
// is BF16 [128,3,H,T] (H=16, or H=8 per TP2 rank); the packed weight columns are
// [128,H,3]. No weight is requantized: ideal is residual + W @ transpose_heads(x).
// FP64 decodes W's original scales/codes and applies this permutation before the dot.
// Caller-owned transient workspace for the SM70 prefill route; no persistent state. Split form
// adds the residual on rank 0 once and all-reduces the partial projections, following
// linear_add_row_parallel.
void ggml_k_gdn_output(const Tensor& x, const Weight& w, Tensor& residual,
                       WorkspaceArena& workspace, cudaStream_t stream);
void ggml_k_gdn_output(std::span<const Tensor> x, std::span<const Weight> w,
                       std::span<const Tensor> residual,
                       std::span<const Tensor> staging,
                       std::span<WorkspaceArena* const> workspace,
                       const ExecutionContext& ec,
                       const PeerEvents& events);

// --- Tensor-parallel split form (tp == 2 or 4) ------------------------------------------------
// All spans have exactly ec.device_count entries. Only registered per-format shard shapes are admitted;
// accepting a four-rank context does not register an otherwise unsupported weight geometry.
// Staging follows allreduce_sum: one output contribution per rank at TP2, four at TP4.
//
// Rank r owns BF16 x[r] [K_r,T], weight w[r] [N,K_r], and the replicated incoming residual
// [N,T]. The complete mathematical oracle is residual_in + sum_r W_r @ x_r, evaluated directly
// in FP64 from represented BF16 inputs and independently decoded packed weights. Internal
// partial storage/reduction rounding is an implementation profile, not an oracle boundary.
//
// Rank 0 folds the residual into its contribution exactly once; every other rank produces a
// residual-free contribution. One reduction leaves the full residual on every rank. Contributions
// may occupy the existing staging planes and reduce directly into the residual destinations.
// BF16_CTRL composes plain Linear and residual_add using staging's first plane; packed formats
// may fuse the rank-0 epilogue. There are no per-rank residual additions after the reduction and
// no extra hot-path allocation.
//
// Ranks agree on token count, output extent N and weight format/layout. K_r may differ. Input
// and output residency/stream requirements are the same as linear_row_parallel. GGML_K,
// NVFP4, Q5G64_F16S, row-scaled FP8 and BF16_CTRL are admitted only at their registered shapes.
// W8G32_F16S is not registered for this split form.
// SM70 TP4 prepacked NVFP4 [5120,1536/4352], A16Only, T1..8 with the published RMSNorm
// collective additionally accepts contiguous FP16 x. This consumes the BF16-rounded
// FP16 image from gated RMSNorm / column-parallel SwiGLU directly; residual and collective outputs keep
// their declared representations. Other LinearAdd profiles require BF16 x.
void linear_add_row_parallel(std::span<const Tensor> x, std::span<const Weight> w,
                             std::span<const Tensor> residual,
                             std::span<const Tensor> staging, LinearPolicy policy,
                             std::span<WorkspaceArena* const> workspace,
                             const ExecutionContext& ec, const PeerEvents& events,
                             const ReplicatedRmsNorm* norm = nullptr);

/// A16-only convenience form for profiles whose queried transient workspace is zero.
void linear_add_row_parallel(std::span<const Tensor> x, std::span<const Weight> w,
                             std::span<const Tensor> residual,
                             std::span<const Tensor> staging, const ExecutionContext& ec,
                             const PeerEvents& events);

} // namespace ninfer::ops
