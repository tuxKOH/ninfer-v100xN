// Implements: include/ninfer/ops/token_bitmask.h
// One thread owns one 32-token mask word of one column. An unconstrained column exits after one
// flag read, so the launch stays in every captured round at negligible cost.
#include "ops/launcher/token_bitmask.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

__global__ void apply_token_bitmask_kernel(__nv_bfloat16* __restrict__ logits,
                                           std::int32_t physical_rows, std::int32_t valid_rows,
                                           std::int32_t row_offset,
                                           const std::int32_t* __restrict__ constrained,
                                           const std::uint32_t* __restrict__ bitmask,
                                           std::int32_t words) {
    __shared__ std::int32_t active;
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.y);
    if (threadIdx.x == 0) { active = constrained[column]; }
    __syncthreads();
    if (active == 0) { return; }

    const std::int32_t word =
        (row_offset >> 5) + static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const std::int32_t end = row_offset + valid_rows;
    if (word * 32 >= end) { return; }
    const std::uint32_t mask =
        bitmask[static_cast<std::size_t>(column) * static_cast<std::size_t>(words) + word];
    if (mask == 0xffffffffu) { return; }

    __nv_bfloat16* rows = logits + static_cast<std::size_t>(column) * physical_rows;
    const __nv_bfloat16 negative_infinity = __ushort_as_bfloat16(0xff80u);
    const std::int32_t first = max(word * 32, row_offset);
    const std::int32_t last  = min(word * 32 + 32, end);
    for (std::int32_t token = first; token < last; ++token) {
        if (((mask >> (token & 31)) & 1u) == 0u) { rows[token - row_offset] = negative_infinity; }
    }
}

} // namespace

void apply_token_bitmask_launch(Tensor& logits, std::int32_t valid_rows, std::int32_t row_offset,
                                const std::int32_t* constrained, const std::uint32_t* bitmask,
                                std::int32_t words, cudaStream_t stream) {
    const std::int32_t columns    = logits.ne[1];
    const std::int32_t span_words = ((row_offset + valid_rows - 1) >> 5) - (row_offset >> 5) + 1;
    const dim3 grid(static_cast<unsigned int>(div_up(span_words, kBlock)),
                    static_cast<unsigned int>(columns));
    apply_token_bitmask_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(logits.data), logits.ne[0], valid_rows, row_offset,
        constrained, bitmask, words);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
