#include "ops/linear/q4/q4_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Same dequant math as Q4SimtDecodeAtom::decode_pair (q4_rowsplit_storage.cuh), writing FP16
// instead of accumulating. One thread owns one code byte, i.e. the weight pair at k0 = 2 * byte.
// Output is row-major-by-n, k-contiguous, which is exactly ColumnMajor for the [K,N] matrix
// CUTLASS's B operand expects (see LayoutInputB below).
//
// Row strides come from `padded_groups`, not from k: the stored planes cover the padded K, and
// the logical K need not be a whole number of groups. Columns in [k, padded K) are never written.
__global__ void dequant_q4_rows_to_fp16(const std::uint8_t* __restrict__ codes,
                                        const std::uint8_t* __restrict__ scales, int n, int k,
                                        int padded_groups, cutlass::half_t* __restrict__ out) {
    const int row      = static_cast<int>(blockIdx.y);
    const int byte_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int k0       = byte_idx * 2;
    if (row >= n || k0 >= k) { return; }

    const int group = byte_idx / Q4RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * padded_groups *
                    Q4RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* scale_ptr =
        scales +
        (static_cast<std::int64_t>(row) * padded_groups + group) *
            Q4RowSplitStorage::kScaleBytesPerGroup;
    const std::uint16_t scale_bits = *reinterpret_cast<const std::uint16_t*>(scale_ptr);

    float w0, w1;
    Q4SimtDecodeAtom::decode_pair(code_row[byte_idx], scale_bits, w0, w1);

    cutlass::half_t* out_row = out + static_cast<std::int64_t>(row) * k;
    out_row[k0]              = cutlass::half_t(w0);
    if (k0 + 1 < k) { out_row[k0 + 1] = cutlass::half_t(w1); }
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
// Direct BF16 epilogue: the cast is from the FP32 accumulator, independent of the FP16 operands
// the tensor cores consume, so no separate narrowing pass is needed.
using ElementOutput = cutlass::bfloat16_t;

using LayoutInputA = cutlass::layout::RowMajor;    // X[T,K]: natural bf16->fp16 cast storage
using LayoutInputB = cutlass::layout::ColumnMajor; // W^T[K,N]: natural row-major-by-n dequant
using LayoutOutput = cutlass::layout::RowMajor;    // D[T,N]

using MMAOp  = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm70;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp          = cutlass::gemm::GemmShape<8, 8, 4>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
    ElementComputeEpilogue>;

constexpr int kNumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
                                         ElementOutput, LayoutOutput, ElementAccumulator, MMAOp,
                                         SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
                                         EpilogueOp, SwizzleThreadBlock, kNumStages>;

struct CutlassWorkspace {
    Tensor w_fp16;
    Tensor x_fp16;
    DeviceSpan gemm_workspace;
};

// get_workspace_size depends only on the problem shape (M=cols, N=n, K=k), not on device
// pointers, so placeholder Arguments are enough for sizing.
std::size_t gemm_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t cols) {
    const cutlass::gemm::GemmCoord problem_size(cols, n, k);
    const typename Gemm::Arguments arguments{
        problem_size, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    return Gemm::get_workspace_size(arguments);
}

// One allocation sequence for the dry-run builder and the real arena, so the reported capacity
// and the launch cannot drift apart.
template <class Allocator>
CutlassWorkspace allocate_cutlass_workspace(Allocator& allocator, std::int32_t n, std::int32_t k,
                                            std::int32_t chunk_cols) {
    CutlassWorkspace out;
    out.w_fp16 = allocator.alloc(DType::FP16, {k, n});
    out.x_fp16 = allocator.alloc(DType::FP16, {k, chunk_cols});
    if (const std::size_t bytes = gemm_workspace_bytes(n, k, chunk_cols); bytes > 0) {
        out.gemm_workspace = allocator.alloc_bytes(bytes);
    }
    return out;
}

} // namespace

std::size_t q4_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                            std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_cutlass_workspace(layout, n, k, std::min(tokens, kQ4CutlassSm70ChunkCols));
    return layout.peak_bytes(1);
}

void q4_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream) {
    const std::int32_t k      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::int32_t n      = out.ne[0];
    const std::int32_t x_ld   = static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_groups = w.padded_shape[1] / Q4RowSplitStorage::kGroupK;
    const std::int32_t chunk_cols    = std::min(tokens, kQ4CutlassSm70ChunkCols);
    // ops::linear already requires contiguous operands; the activation cast below reads x as one
    // dense [T,K] run, so restate it here rather than rely on the caller.
    if (x_ld != k) { throw std::invalid_argument("q4_cutlass_sm70: x must be contiguous"); }

    auto scratch_scope             = ws.scope();
    const CutlassWorkspace scratch = allocate_cutlass_workspace(ws, n, k, chunk_cols);
    auto* w_fp16                   = static_cast<cutlass::half_t*>(scratch.w_fp16.data);
    auto* x_fp16                   = static_cast<cutlass::half_t*>(scratch.x_fp16.data);

    {
        constexpr int kThreads = 256;
        const dim3 grid(static_cast<unsigned>(div_up_i(div_up_i(k, 2), kThreads)),
                        static_cast<unsigned>(n), 1u);
        dequant_q4_rows_to_fp16<<<grid, kThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
            n, k, padded_groups, w_fp16);
        CUDA_CHECK(cudaGetLastError());
    }

    // The weight is dequantized once; only the activation cast and the GEMM repeat per chunk.
    // Launches are stream-ordered, so reusing one activation scratch across chunks is safe.
    const auto* x_data = static_cast<const __nv_bfloat16*>(x.data);
    auto* out_data     = static_cast<ElementOutput*>(out.data);
    for (std::int32_t begin = 0; begin < tokens; begin += chunk_cols) {
        const std::int32_t cols = std::min(chunk_cols, tokens - begin);
        {
            constexpr int kThreads = 256;
            const std::int64_t count = static_cast<std::int64_t>(cols) * k;
            const auto blocks        = static_cast<unsigned>((count + kThreads - 1) / kThreads);
            bf16_to_fp16_kernel<<<blocks, kThreads, 0, stream>>>(
                x_data + static_cast<std::int64_t>(begin) * x_ld, x_fp16, count);
            CUDA_CHECK(cudaGetLastError());
        }

        Gemm gemm_op;
        ElementOutput* out_chunk = out_data + static_cast<std::int64_t>(begin) * out_ld;
        // beta = 0: the epilogue never reads C, so `out` needs no initialization.
        const typename Gemm::Arguments arguments{
            cutlass::gemm::GemmCoord(cols, n, k),
            {x_fp16, k},
            {w_fp16, k},
            {out_chunk, out_ld},
            {out_chunk, out_ld},
            {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
            1};

        cutlass::Status status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("q4_cutlass_sm70: CUTLASS can_implement failed");
        }
        status = gemm_op.initialize(arguments, scratch.gemm_workspace.data, stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("q4_cutlass_sm70: CUTLASS initialize failed");
        }
        status = gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("q4_cutlass_sm70: CUTLASS gemm() failed");
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
