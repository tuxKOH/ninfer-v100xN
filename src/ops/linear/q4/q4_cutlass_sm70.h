#pragma once

// Volta (sm_70) tensor-core route for the plain Q4 x BF16 linear at the wide T the vision tower
// produces (one column per image patch, so T is in the thousands). Same two-phase design as the
// fused q4_linear_swiglu_cutlass_sm70 route: dequantize the Q4 groupwise-int weight to a scratch
// FP16 buffer once per call, cast the BF16 activations to FP16, and run NVIDIA CUTLASS's stock
// Sm70 (mma.sync.m8n8k4) Gemm with a direct BF16 epilogue.
//
// Two things differ from the fused launcher, both because of the vision geometries:
//   - the weight planes are addressed through the padded group count (Weight::padded_shape), so
//     a K that is not a whole number of groups is decoded correctly;
//   - T is processed in bounded column chunks, so the FP16 activation scratch -- and with it the
//     workspace requirement -- stops growing at kQ4CutlassSm70ChunkCols instead of scaling with
//     the image size.
//
// Not in the Q4Launch table: that signature cannot carry the workspace.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Lower bound of the routed band: the first vision T (a multiple of four) above the fused
// small-T band, which ends at T = 64. Measured on a V100-PCIE-16GB with ninfer_linear_bench (cold
// cache, minimum of 20-30 samples, microseconds), q4_simt_r8_c8 -> this route:
//
//   T:              68            128            1024             4096
//   n=3456      174.1 / 90.1   313.3 / 84.0   2410.5 / 160.8   9534.5 / 549.9
//   n=4304      224.3 / 89.1   386.0 / 90.1   3060.7 / 216.1  11806.7 / 657.4
//
// The dequant pass and a single partially filled 128-row M tile make the cost nearly flat up to
// T = 256, so the route already wins at the band's lower edge and by 17-18x at T = 4096.
inline constexpr std::int32_t kQ4CutlassSm70MinT = 68;

// Columns per GEMM launch. 2048 keeps every vision geometry at >= 144 CTAs per launch (128x128
// tiles), comfortably more than the V100's 80 SMs, while bounding the FP16 activation scratch at
// K * 2048 * 2 bytes.
inline constexpr std::int32_t kQ4CutlassSm70ChunkCols = 2048;

// The registered vision geometries only (qkv and mlp/fc1). Every other Q4 shape keeps the route
// it had: the text model's plain Q4 linears are sized by callers that do not budget an FP16 copy
// of the weight (the draft head alone would need 1.3 GB).
[[nodiscard]] inline bool q4_uses_cutlass_sm70(std::int32_t n, std::int32_t k,
                                               std::int32_t t) noexcept {
    return k == 1152 && (n == 3456 || n == 4304) && t >= kQ4CutlassSm70MinT;
}

[[nodiscard]] std::size_t q4_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                         std::int32_t tokens);

void q4_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
