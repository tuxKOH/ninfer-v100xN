#pragma once

// Volta (sm_70) tensor-core route for the plain Q5 x BF16 linear at the wide T the vision tower
// produces. Same design as the Q4 sibling (ops/linear/q4/q4_cutlass_sm70.h), which in turn
// mirrors the fused q5_linear_add_cutlass_sm70 route: dequantize the Q5 groupwise-int weight
// (4-bit low nibble + 1-bit high plane) to a scratch FP16 buffer once per call, cast the BF16
// activations to FP16, and run CUTLASS's stock Sm70 Gemm with a direct BF16 epilogue.
//
// It is a separate launcher rather than the fused one with beta = 0 because the fused dequant
// derives its row strides from K / 64. That is exact for the text model's K (6144, 17408) and
// wrong for vision mlp/fc2, whose K = 4304 is stored padded to 4352. Here the planes are
// addressed through Weight::padded_shape, and T is processed in bounded column chunks so the
// workspace stops growing at kQ5CutlassSm70ChunkCols.
//
// Not in the Q5Launch table: that signature cannot carry the workspace.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Lower bound of the routed band, shared with the Q4 sibling. Same bench and conditions,
// q5_simt_r8_c4 / q5_simt_r8_c8 -> this route, microseconds:
//
//   T:                   68            128            1024             4096
//   n=1152 k=1152     88.1 / 82.9   186.4 / 60.4   1397.8 /  96.3   5487.6 / 205.8
//   n=1152 k=4304    222.2 / 202.8  508.9 / 192.5  3823.6 / 221.2  15159.3 / 683.0
//
// T = 68 is close to parity for k = 1152 and a clear win from T = 80.
inline constexpr std::int32_t kQ5CutlassSm70MinT = 68;

// Columns per GEMM launch. Bounds the FP16 activation scratch at K * 2048 * 2 bytes (17.6 MB at
// K = 4304) while keeping >= 144 CTAs per launch at N = 1152.
inline constexpr std::int32_t kQ5CutlassSm70ChunkCols = 2048;

// The registered vision geometries only (attention/output and mlp/fc2). Every other Q5 shape
// keeps the route it had; see the Q4 sibling.
[[nodiscard]] inline bool q5_uses_cutlass_sm70(std::int32_t n, std::int32_t k,
                                               std::int32_t t) noexcept {
    return n == 1152 && (k == 1152 || k == 4304) && t >= kQ5CutlassSm70MinT;
}

[[nodiscard]] std::size_t q5_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                         std::int32_t tokens);

void q5_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
