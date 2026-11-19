#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear_swiglu/swiglu_fp32.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The artifact stores adjacent E2M1 values in each code byte and one E4M3 scale per K16 group.
// Scales use the BlockScaleK16M128x4 swizzle. Materializing once is intentionally a wide-T
// strategy: CUTLASS can reuse the resulting FP16 matrix across every token tile instead of
// decoding the packed weights again for each tile.
__global__ void dequant_nvfp4_row_to_fp16(const std::uint8_t* __restrict__ codes,
                                           const std::uint8_t* __restrict__ scales, int n, int k,
                                           float inverse_weight_divisor,
                                           cutlass::half_t* __restrict__ out) {
    const int row      = static_cast<int>(blockIdx.y);
    const int byte_idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int bytes_per_row = k / 2;
    if (row >= n || byte_idx >= bytes_per_row) { return; }
    const int k0            = byte_idx * 2;
    const int group         = byte_idx / 8;
    const int scale_tile    = group / 4;
    const int scale_lane    = group & 3;
    const int row_inner     = row & 127;
    const int scales_per_m128 = k / 64;
    const std::int64_t scale_offset =
        static_cast<std::int64_t>((row / 128) * scales_per_m128 + scale_tile) * 512 +
        (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;
    const float coefficient = decode_nvfp4_e4m3(scales[scale_offset]) * inverse_weight_divisor;
    const float2 value = decode_nvfp4_e2m1x2(
        codes[static_cast<std::int64_t>(row) * bytes_per_row + byte_idx]);
    cutlass::half_t* out_row = out + static_cast<std::int64_t>(row) * k;
    out_row[k0]     = cutlass::half_t(value.x * coefficient);
    out_row[k0 + 1] = cutlass::half_t(value.y * coefficient);
}

__global__ void dequant_nvfp4_qpn_to_fp16(const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales, int n, int k,
                                          float inverse_weight_divisor,
                                          cutlass::half_t* __restrict__ out) {
    // Consecutive lanes read consecutive packed K16 tuples. The old row-wise traversal made
    // adjacent lanes skip 256 input bytes and decoded every tuple twice. Each lane now loads
    // its eight code bytes once and writes the complete, aligned 32-byte FP16 group.
    const std::int64_t tuple = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int groups = k / 16;
    if (tuple >= static_cast<std::int64_t>(n) * groups) { return; }
    const int lane = static_cast<int>(tuple & 31);
    const int group = static_cast<int>((tuple / 32) % groups);
    const int row = static_cast<int>(tuple / (32 * groups)) * 32 +
                    ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
    constexpr int inverse_order[16] = {0, 4, 1, 5, 2, 6, 3, 7,
                                       8, 12, 9, 13, 10, 14, 11, 15};
    const uint2 packed_words = __ldg(reinterpret_cast<const uint2*>(codes) + tuple);
    const auto* packed = reinterpret_cast<const std::uint8_t*>(&packed_words);
    const float coefficient = decode_nvfp4_e4m3(scales[tuple]) * inverse_weight_divisor;
    alignas(16) cutlass::half_t values[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int position  = inverse_order[j];
        const float2 pair   = decode_nvfp4_e2m1x2(packed[position / 2]);
        values[j] = cutlass::half_t(((position & 1) == 0 ? pair.x : pair.y) * coefficient);
    }
    auto* destination = reinterpret_cast<uint4*>(
        out + static_cast<std::int64_t>(row) * k + group * 16);
    destination[0] = reinterpret_cast<const uint4*>(values)[0];
    destination[1] = reinterpret_cast<const uint4*>(values)[1];
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = cutlass::half_t(__bfloat162float(in[i])); }
}

int div_up_i(int a, int b) { return (a + b - 1) / b; }

using ElementAccumulator     = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA          = cutlass::half_t;
using ElementInputB          = cutlass::half_t;
using LayoutInputA           = cutlass::layout::RowMajor;
using LayoutInputB           = cutlass::layout::ColumnMajor;
using LayoutOutput           = cutlass::layout::RowMajor;
using MMAOp                  = cutlass::arch::OpClassTensorOp;
using SmArch                 = cutlass::arch::Sm70;
using ShapeMMAWarp           = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp             = cutlass::gemm::GemmShape<8, 8, 4>;
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int kNumStages = 2;
template <class ElementOutput>
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
    ElementComputeEpilogue>;
template <class ElementOutput, int TileN = 128>
using GemmFor = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
                                         ElementOutput, LayoutOutput, ElementAccumulator, MMAOp,
                                         SmArch, cutlass::gemm::GemmShape<128, TileN, 32>,
                                         ShapeMMAWarp, ShapeMMAOp,
                                         EpilogueOp<ElementOutput>, SwizzleThreadBlock, kNumStages>;
using Gemm = GemmFor<cutlass::bfloat16_t>;

// Bound the FP16 weight image for the unsharded 27B matrices. A band
// contains complete 128-row scale tiles and contracts the entire K axis.
int band_rows(int n, int k) {
    const bool full = (k == 5120 && (n == 14336 || n == 16384 || n == 34816)) ||
                      (n == 5120 && (k == 6144 || k == 17408));
    if (!full) { return n; }
    constexpr int bytes = 48 * 1024 * 1024;
    return std::min(n, bytes / (k * 2 * 128) * 128);
}

// Four equal gate/up bands cover the 27B SwiGLU intermediate.
constexpr int kSwiGluBand = 4352;

template <class Allocator>
struct CutlassWorkspace {
    Tensor w_fp16;
    Tensor x_fp16;
    DeviceSpan gemm_workspace;
};

template <class Allocator>
CutlassWorkspace<Allocator> allocate_cutlass_workspace(Allocator& allocator, std::int32_t n,
                                                       std::int32_t k, std::int32_t cols,
                                                       std::size_t gemm_workspace_bytes) {
    CutlassWorkspace<Allocator> out;
    out.w_fp16 = allocator.alloc(DType::FP16, {k, band_rows(n, k)});
    out.x_fp16 = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_workspace_bytes > 0) { out.gemm_workspace = allocator.alloc_bytes(gemm_workspace_bytes); }
    return out;
}

std::size_t workspace_bytes_impl(std::int32_t n, std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    typename Gemm::Arguments arguments{
        problem_size, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    const std::size_t gemm_workspace_bytes = Gemm::get_workspace_size(arguments);
    (void)allocate_cutlass_workspace(layout, n, k, cols, gemm_workspace_bytes);
    return layout.peak_bytes(1);
}

template <class ElementOutput, int TileN, bool SwiGlu = false>
void launch_impl(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                 cudaStream_t stream) {
    using Gemm = GemmFor<ElementOutput, TileN>;
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    const std::int32_t n    = w.n;
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    typename Gemm::Arguments sizing_arguments{
        problem_size, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    const std::size_t gemm_workspace_bytes =
        Gemm::get_workspace_size(sizing_arguments);

    auto scratch_scope = ws.scope();
    CutlassWorkspace<WorkspaceArena> scratch =
        allocate_cutlass_workspace(ws, n, k, cols, gemm_workspace_bytes);
    auto* w_fp16 = static_cast<cutlass::half_t*>(scratch.w_fp16.data);
    auto* x_fp16 = static_cast<cutlass::half_t*>(scratch.x_fp16.data);

    const std::int64_t x_count = static_cast<std::int64_t>(cols) * k;
    const int x_blocks         = static_cast<int>((x_count + 255) / 256);
    bf16_to_fp16_kernel<<<x_blocks, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                      x_fp16, x_count);
    CUDA_CHECK(cudaGetLastError());

    const int band = band_rows(n, k);
    const auto project = [&](int first, int rows, ElementOutput* destination, int stride) {
        const auto offset = static_cast<std::int64_t>(first) * k;
        const auto* codes = static_cast<const std::uint8_t*>(w.qdata) + offset / 2;
        const auto* scales = static_cast<const std::uint8_t*>(w.scales) + offset / 16;
        const dim3 block(256);
        if (w.layout == QuantLayout::VoltaQpnPrepacked) {
            const dim3 grid(static_cast<unsigned>((static_cast<std::int64_t>(rows) * (k / 16) + 255) / 256));
            dequant_nvfp4_qpn_to_fp16<<<grid, block, 0, stream>>>(
                codes, scales, rows, k, 1.0F / w.weight_scale_divisor, w_fp16);
        } else {
            const dim3 grid(static_cast<unsigned>(div_up_i(k / 2, 256)), static_cast<unsigned>(rows), 1u);
            dequant_nvfp4_row_to_fp16<<<grid, block, 0, stream>>>(
                codes, scales, rows, k, 1.0F / w.weight_scale_divisor, w_fp16);
        }
        CUDA_CHECK(cudaGetLastError());

        Gemm gemm_op;
        typename Gemm::Arguments arguments{
            {cols, rows, k}, {x_fp16, k}, {w_fp16, k}, {destination, stride},
            {destination, stride},
            {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
        cutlass::Status status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS can_implement failed");
        }
        status = gemm_op.initialize(arguments, scratch.gemm_workspace.data, stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS initialize failed");
        }
        status = gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS gemm() failed");
        }
        CUDA_CHECK(cudaGetLastError());
    };
    if constexpr (SwiGlu) {
        const int intermediate = n / 2;
        Tensor pair = ws.alloc(DType::FP32, {2 * kSwiGluBand, cols});
        auto* values = static_cast<float*>(pair.data);
        for (int first = 0; first < intermediate; first += kSwiGluBand) {
            project(first, kSwiGluBand, values, 2 * kSwiGluBand);
            project(intermediate + first, kSwiGluBand, values + kSwiGluBand, 2 * kSwiGluBand);
            auto result = out.slice(0, first, kSwiGluBand);
            swiglu_fp32_launch(pair, nullptr, result, stream);
        }
    } else {
        for (int first = 0; first < n; first += band) {
            project(first, std::min(band, n - first), static_cast<ElementOutput*>(out.data) + first, n);
        }
    }
}

template <class ElementOutput>
void launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
            cudaStream_t stream) {
    // The same 64x64 warp contraction gains output-column reuse on the wide TP2 MLPs.
    // Narrow calls and other registered geometries retain their measured 128-column tile.
    if (x.ne[1] >= 2048 && ((w.n == 17408 && w.k == 5120) ||
                             (w.n == 5120 && w.k == 8704))) {
        launch_impl<ElementOutput, 256>(x, w, out, ws, stream);
    } else {
        launch_impl<ElementOutput, 128>(x, w, out, ws, stream);
    }
}

} // namespace

std::size_t nvfp4_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                               std::int32_t cols) {
    return workspace_bytes_impl(n, k, cols);
}

std::size_t nvfp4_cutlass_sm70_full_swiglu_workspace_bytes(int cols) {
    return workspace_bytes_impl(34816, 5120, cols) +
           static_cast<std::size_t>(2 * kSwiGluBand) * cols * sizeof(float);
}

void nvfp4_cutlass_sm70_full_swiglu_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           WorkspaceArena& ws, cudaStream_t stream) {
    launch_impl<float, 128, true>(x, w, out, ws, stream);
}

void nvfp4_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    launch<cutlass::bfloat16_t>(x, w, out, ws, stream);
}

void nvfp4_cutlass_sm70_fp32_launch(const Tensor& x, const Weight& w, Tensor& out,
                                    WorkspaceArena& ws, cudaStream_t stream) {
    launch<float>(x, w, out, ws, stream);
}

} // namespace ninfer::ops::detail
