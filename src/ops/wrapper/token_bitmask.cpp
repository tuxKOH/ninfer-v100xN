#include "ninfer/ops/token_bitmask.h"

#include "ops/launcher/token_bitmask.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

void apply_token_bitmask(Tensor& logits, std::int32_t valid_rows, std::int32_t row_offset,
                         const std::int32_t* constrained, const std::uint32_t* bitmask,
                         std::int32_t words, cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || logits.ne[2] != 1 || logits.ne[3] != 1 ||
        !logits.is_contiguous()) {
        throw std::invalid_argument("apply_token_bitmask: logits must be contiguous BF16 [rows,T]");
    }
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0 || logits.data == nullptr) {
        throw std::invalid_argument("apply_token_bitmask: logits must be non-empty");
    }
    if (valid_rows <= 0 || valid_rows > logits.ne[0] || row_offset < 0 || words <= 0 ||
        static_cast<std::int64_t>(row_offset) + valid_rows > static_cast<std::int64_t>(words) * 32) {
        throw std::invalid_argument("apply_token_bitmask: row domain is outside the bitmask");
    }
    if (constrained == nullptr || bitmask == nullptr) {
        throw std::invalid_argument("apply_token_bitmask: constraint image must be non-null");
    }
    detail::apply_token_bitmask_launch(logits, valid_rows, row_offset, constrained, bitmask, words,
                                       stream);
}

} // namespace ninfer::ops
