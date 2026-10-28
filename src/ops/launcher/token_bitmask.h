#pragma once

// ninfer::ops::detail - private launch prototype for apply_token_bitmask.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void apply_token_bitmask_launch(Tensor& logits, std::int32_t valid_rows, std::int32_t row_offset,
                                const std::int32_t* constrained, const std::uint32_t* bitmask,
                                std::int32_t words, cudaStream_t stream);

} // namespace ninfer::ops::detail
