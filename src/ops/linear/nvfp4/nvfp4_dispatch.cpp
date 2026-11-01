#include "ops/linear/nvfp4/nvfp4_dispatch.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"
#endif

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearRoute : std::uint8_t {
    A16,
    W4A4,
};

Nvfp4LinearRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                               LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || !is_nvfp4_linear_problem(output_rows, input_rows)) {
        throw std::invalid_argument("nvfp4 linear: unsupported shape");
    }
#ifdef NINFER_VOLTA_BUILD
    if (is_nvfp4_volta_tp4_problem(output_rows, input_rows)) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("nvfp4 TP4 SM70 linear admits only A16");
        }
        return Nvfp4LinearRoute::A16;
    }
#endif
    if (policy == LinearPolicy::A16Only) { return Nvfp4LinearRoute::A16; }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    }

    // A tp2 shard resolves through its tp1 parent: halving N or K shifts the measured A16/W4A4
    // crossover somewhat, but inheriting keeps the split path's route a pure function of the
    // family, so a tp2 run cannot silently take a different numerical path than its tp1 twin.
    // Re-measuring the shard crossovers is a separate, benchmarked change.
    switch (nvfp4_parent_problem(resolve_nvfp4_problem(output_rows, input_rows))) {
    case Nvfp4Problem::AttnInput:
        return tokens >= 4 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    case Nvfp4Problem::GdnInput:
        return Nvfp4LinearRoute::W4A4;
    case Nvfp4Problem::MlpGateUp:
        return tokens >= 5 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::Residual17408:
        return tokens >= 8 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    case Nvfp4Problem::AttnInputTp2Column:
    case Nvfp4Problem::GdnInputTp2Column:
    case Nvfp4Problem::MlpGateUpTp2Column:
    case Nvfp4Problem::Residual6144Tp2Row:
    case Nvfp4Problem::Residual17408Tp2Row:
        break; // nvfp4_parent_problem never returns a shard problem
    }
    throw std::logic_error("unreachable NVFP4 linear problem");
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out,
#ifdef NINFER_VOLTA_BUILD
                WorkspaceArena* workspace,
#endif
                cudaStream_t stream) {
    const std::int32_t total_t = x.ne[1];
#ifdef NINFER_VOLTA_BUILD
    if (workspace != nullptr && weight.layout == QuantLayout::VoltaQpnPrepacked &&
        nvfp4_volta_batch_supported(weight.n, weight.k, total_t)) {
        launch_nvfp4_volta_batch(x, weight, out, *workspace, stream);
        return;
    }
    if (workspace != nullptr && nvfp4_volta_qpn_precast_supported(weight.n, weight.k, total_t)) {
        launch_nvfp4_volta_qpn_precast(x, weight, out, *workspace, stream);
        return;
    }
    // Materialize each weight once for the whole prefill call, including QPN-prepacked TP2
    // shards. Repeated 32-token QPN launches reread the entire down projection per tile.
    if (workspace != nullptr && total_t >= 128) {
        nvfp4_cutlass_sm70_launch(x, weight, out, *workspace, stream);
        return;
    }
    const bool qpn            = nvfp4_volta_qpn_supported(weight.n, weight.k, kNvfp4VoltaQpnMaxTokens);
    const std::int32_t kChunk = qpn ? kNvfp4VoltaQpnMaxTokens : kNvfp4LastSmallT;
#else
    constexpr std::int32_t kChunk = kNvfp4LastSmallT;
#endif
    for (std::int32_t token_begin = 0; token_begin < total_t; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor output_chunk(output, DType::BF16, {weight.n, active});
#ifdef NINFER_VOLTA_BUILD
        if (nvfp4_volta_qpn_supported(weight.n, weight.k, active)) {
            launch_nvfp4_volta_qpn(input_chunk, weight, output_chunk, stream);
            continue;
        }
#endif
        if (active == 1) {
            launch_nvfp4_decode(input_chunk, weight, output_chunk, stream);
        } else {
            launch_nvfp4_small_t(input_chunk, weight, output_chunk, stream);
        }
    }
}

} // namespace

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    if (resolve_route(output_rows, input_rows, policy, max_tokens) == Nvfp4LinearRoute::W4A4) {
        return nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows);
    }
#ifdef NINFER_VOLTA_BUILD
    if (max_tokens >= 128) {
        return nvfp4_cutlass_sm70_workspace_bytes(output_rows, input_rows, max_tokens);
    }
    return std::max(nvfp4_volta_qpn_precast_workspace_bytes(output_rows, input_rows, max_tokens),
                    nvfp4_volta_batch_workspace_bytes(output_rows, input_rows, max_tokens));
#endif
    return 0;
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (!is_nvfp4_linear_problem(weight.n, weight.k) || x.ne[1] <= 0) {
        throw std::invalid_argument("nvfp4 linear: unsupported shape");
    }

    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Nvfp4LinearRoute::A16) {
#ifdef NINFER_VOLTA_BUILD
        launch_a16(x, weight, out, workspace, stream);
#else
        launch_a16(x, weight, out, stream);
#endif
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 linear requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    launch_nvfp4_w4a4(x, weight, out, scratch, stream);
}

} // namespace ninfer::ops::detail
