#include "ninfer/ops/linear.h"

#include "ops/common/split_launch.h"
#include "ops/linear/linear_dispatch.h"
#include "ops/linear/ggml_k/ggml_k.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/ggml_k/ggml_k_cutlass_sm70.h"
#endif
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/fp8/fp8_block.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/q4/q4_cutlass_sm70.h"
#include "ops/linear/q5/q5_cutlass_sm70.h"
#endif
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/w8/w8_dispatch.h"

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

} // namespace

// Not anonymous-namespace-scoped: src/ops/linear/linear_dispatch.h exposes both of these to
// linear_add.cpp's row-parallel split form, which issues the residual-free half of a row-parallel
// rank through dispatch_linear directly (see that header for why the public linear() overloads,
// which require a bound WorkspaceArena&, cannot express a nullable workspace). Every other call
// site in this file is unqualified and unaffected -- unqualified lookup finds a plain namespace
// member exactly as it found an anonymous-namespace one.
void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T], got x=" +
                                    std::to_string(x.ne[0]) + "x" + std::to_string(x.ne[1]) +
                                    " w=" + std::to_string(w.n) + "x" + std::to_string(w.k) +
                                    " out=" + std::to_string(out.ne[0]) + "x" +
                                    std::to_string(out.ne[1]));
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    switch (w.qtype) {
    case QType::FP8_E4M3FN_BLOCK128_BF16S:
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("FP8 block128 supports A16 only");
        }
        detail::fp8_block_launch(x, w, out, workspace, stream);
        return;
    case QType::GGML_K:
#ifdef NINFER_VOLTA_BUILD
        if (workspace != nullptr && x.ne[1] >= 128) {
            detail::ggml_k_cutlass_sm70_launch(x, w, out, *workspace, stream);
            return;
        }
#endif
        detail::ggml_k_linear(x, w, out, stream);
        return;
    case QType::Q4G64_F16S:
        detail::q4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::Q5G64_F16S:
        detail::q5_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::Q6G64_F16S:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::W8G32_F16S:
        detail::w8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::BF16_CTRL:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16S:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }

    switch (qtype) {
    case QType::FP8_E4M3FN_BLOCK128_BF16S:
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("FP8 block128 supports A16 only");
        }
        return detail::fp8_block_workspace_bytes(output_rows, input_rows, min_tokens, max_tokens);
    case QType::GGML_K:
        if (output_rows <= 0 || input_rows <= 0 || input_rows % 256 != 0 ||
            policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear workspace: invalid GGML K profile");
        }
#ifdef NINFER_VOLTA_BUILD
        if (max_tokens >= 128) {
            return detail::ggml_k_cutlass_sm70_workspace_bytes(output_rows, input_rows,
                                                                 max_tokens);
        }
#endif
        return 0;
    case QType::Q4G64_F16S:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
#ifdef NINFER_VOLTA_BUILD
        {
            std::size_t capacity = 0;
            for (int t = std::max(min_tokens, 9); t <= std::min(max_tokens, 64); ++t) {
                if (detail::q4_volta_mma_supported(output_rows, input_rows, t)) {
                    capacity = std::max(capacity,
                        detail::q4_volta_mma_workspace_bytes(output_rows, input_rows, t));
                }
            }
            // Monotonic in T up to the launcher's column chunk and flat beyond it, so the
            // interval's upper end is its maximum.
            if (detail::q4_uses_cutlass_sm70(output_rows, input_rows, max_tokens)) {
                capacity = std::max(capacity, detail::q4_cutlass_sm70_workspace_bytes(
                                                  output_rows, input_rows, max_tokens));
            }
            return capacity;
        }
#endif
        return 0;
    case QType::Q5G64_F16S:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
#ifdef NINFER_VOLTA_BUILD
        if (detail::q5_uses_cutlass_sm70(output_rows, input_rows, max_tokens)) {
            return detail::q5_cutlass_sm70_workspace_bytes(output_rows, input_rows, max_tokens);
        }
#endif
        return 0;
    case QType::Q6G64_F16S:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::W8G32_F16S:
        (void)detail::select_w8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_w8_launch(output_rows, input_rows, max_tokens, policy);
        return detail::w8_prepacked_workspace_bytes(output_rows, input_rows, min_tokens, max_tokens);
    case QType::BF16_CTRL:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        if (!detail::is_nvfp4_linear_problem(output_rows, input_rows) ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("linear workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16S:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

namespace {

// Cross-rank agreement. Everything a single device can check is already checked per rank by
// validate_linear_semantics; these are the invariants only the pair makes sense of. The split
// axis's own extents are deliberately NOT required to match: an uneven split is legal and neither
// form ever needs to know the logical total.
void validate_split_pair(std::span<const Tensor> x, std::span<const Weight> w,
                         const ExecutionContext& ec, bool column_parallel) {
    detail::require_split_context(
        ec, "linear split: requires two or four distinct devices", ec.device_count);
    if (x.size() != static_cast<std::size_t>(ec.device_count) || w.size() != x.size()) {
        throw std::invalid_argument("linear split: argument width must equal TP width");
    }
    for (int rank = 1; rank < ec.device_count; ++rank) {
        if (x[0].ne[1] != x[rank].ne[1]) {
            throw std::invalid_argument("linear split: ranks must carry the same token count");
        }
        if (w[0].qtype != w[rank].qtype || w[0].layout != w[rank].layout) {
            throw std::invalid_argument("linear split: ranks must carry the same weight format");
        }
        if (column_parallel && w[0].k != w[rank].k) {
            throw std::invalid_argument(
                "linear column-parallel: ranks must consume the same input extent K");
        }
        if (!column_parallel && w[0].n != w[rank].n) {
            throw std::invalid_argument(
                "linear row-parallel: ranks must produce the same output extent N");
        }
    }
}

// Everything a rank owns must live on that rank's device. Compiled out of Release; see
// require_rank_residency.
void validate_split_residency(std::span<const Tensor> x, std::span<const Weight> w,
                              std::span<const Tensor> out, const ExecutionContext& ec) {
    for (int rank = 0; rank < ec.device_count; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, w[slot].payload, out[slot].data,
            "linear split: every per-rank argument must be resident on ec.dev[rank]");
    }
}

// Validate every active rank before issuing any work. Tensor is a non-owning view; the fixed
// four-entry host array copies no device memory and allocates nothing during graph recording.
std::array<Tensor, kMaximumExecutionDevices> validated_outputs(
    std::span<const Tensor> x, std::span<const Weight> w, std::span<const Tensor> out,
    LinearPolicy policy, std::span<WorkspaceArena* const> workspace) {
    if (out.size() != x.size() || workspace.size() != x.size()) {
        throw std::invalid_argument("linear split: output/workspace width must equal TP width");
    }
    std::array<Tensor, kMaximumExecutionDevices> destination{};
    for (std::size_t slot = 0; slot < x.size(); ++slot) {
        destination[slot] = out[slot];
        validate_linear_semantics(x[slot], w[slot], destination[slot], policy);
    }
    return destination;
}

// One rank's single-device projection, issued on that rank's own stream. This is the whole of the
// "split kernel": the shard Weight narrows N (column-parallel) or K (row-parallel), so the
// existing launcher resolves the shard geometry and its grid is already halved along that axis.
void issue_rank(int rank, std::span<const Tensor> x, std::span<const Weight> w,
                std::array<Tensor, kMaximumExecutionDevices>& out, LinearPolicy policy,
                std::span<WorkspaceArena* const> workspace, const ExecutionContext& ec) {
    const auto slot = static_cast<std::size_t>(rank);
    dispatch_linear(x[slot], w[slot], out[slot], policy, workspace[slot], ec.dev[slot]->stream);
}

} // namespace

void linear_column_parallel(std::span<const Tensor> x, std::span<const Weight> w,
                            std::span<const Tensor> out, LinearPolicy policy,
                            std::span<WorkspaceArena* const> workspace,
                            const ExecutionContext& ec) {
    validate_split_pair(x, w, ec, /*column_parallel=*/true);
    // Validate every active rank before issuing any of them.
    auto destination = validated_outputs(x, w, out, policy, workspace);
    validate_split_residency(x, w, out, ec);
    detail::for_each_rank(
        ec, [&](int rank) { issue_rank(rank, x, w, destination, policy, workspace, ec); });
}

void linear_column_parallel(std::span<const Tensor> x, std::span<const Weight> w,
                            std::span<const Tensor> out, const ExecutionContext& ec) {
    const std::array<WorkspaceArena*, kMaximumExecutionDevices> workspace{};
    linear_column_parallel(x, w, out, LinearPolicy::A16Only,
                           std::span(workspace).first(ec.device_count), ec);
}

void linear_row_parallel(std::span<const Tensor> x, std::span<const Weight> w,
                         std::span<const Tensor> out, std::span<const Tensor> staging,
                         LinearPolicy policy, std::span<WorkspaceArena* const> workspace,
                         const ExecutionContext& ec, const PeerEvents& events) {
    validate_split_pair(x, w, ec, /*column_parallel=*/false);
    auto destination = validated_outputs(x, w, out, policy, workspace);
    validate_split_residency(x, w, out, ec);
    if (!events.live() || events.ranks() != ec.device_count || staging.size() != x.size()) {
        throw std::invalid_argument("linear row-parallel: staging/events must match TP width");
    }

    // Each rank's partial lands directly in out[rank]; allreduce_sum combines in place, so no
    // separate accumulation workspace exists to get out of step with the output. The collective
    // records its inputs_ready event on the same stream the projection above was issued on, which
    // is what orders the peer's read after the partial is complete.
    detail::for_each_rank(
        ec, [&](int rank) { issue_rank(rank, x, w, destination, policy, workspace, ec); });
    // Scratch capacity follows allreduce_sum: one output contribution at TP2, four at TP4.
    allreduce_sum(out, staging, ec, events);
}

void linear_row_parallel(std::span<const Tensor> x, std::span<const Weight> w,
                         std::span<const Tensor> out, std::span<const Tensor> staging,
                         const ExecutionContext& ec, const PeerEvents& events) {
    const std::array<WorkspaceArena*, kMaximumExecutionDevices> workspace{};
    linear_row_parallel(x, w, out, staging, LinearPolicy::A16Only,
                        std::span(workspace).first(ec.device_count), ec, events);
}

} // namespace ninfer::ops
