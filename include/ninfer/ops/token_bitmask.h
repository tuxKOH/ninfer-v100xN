#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Op: apply_token_bitmask
 *
 * Grammar-constrained token selection. `logits` is contiguous BF16 [physical_rows,T] holding the
 * global vocabulary rows [row_offset,row_offset+valid_rows) in its first valid_rows rows.
 * `constrained` is I32 [T] and `bitmask` is U32 [T,words]; bit (g & 31) of bitmask[t,g/32]
 * licenses global token g. Both may live in device memory or in mapped page-locked host memory
 * visible to the stream's device.
 *
 * For every column t with constrained[t] != 0 and every 0 <= v < valid_rows:
 *
 *   logits[v,t] = -inf  when bit (row_offset + v) of bitmask[t] is clear.
 *
 * Every other element, including rows [valid_rows,physical_rows), is unchanged. Requires
 * 1 <= valid_rows <= physical_rows, row_offset >= 0 and row_offset + valid_rows <= 32 * words.
 * No workspace and no other side effect; the launch shape is fixed for CUDA Graph capture.
 */
void apply_token_bitmask(Tensor& logits, std::int32_t valid_rows, std::int32_t row_offset,
                         const std::int32_t* constrained, const std::uint32_t* bitmask,
                         std::int32_t words, cudaStream_t stream);

} // namespace ninfer::ops
