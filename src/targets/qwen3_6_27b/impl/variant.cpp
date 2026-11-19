#include "targets/qwen3_6_27b/impl/variant.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

#if defined(NINFER_SM8X_COMPAT) || defined(NINFER_VOLTA_BUILD)
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::A16Only;
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::A16Only;
#else
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::AllowA4;
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::AllowA8;
#endif

ops::LinearPolicy text_policy(const Weight& weight) {
    switch (weight.qtype) {
    case QType::NVFP4:
        return kNvfp4TextPolicy;
    case QType::FP8_E4M3FN_ROW_BF16S:
        return kFp8TextPolicy;
    default:
        return ops::LinearPolicy::A16Only;
    }
}

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

QType tp4_projection_type(WeightsProfile profile) {
    switch (profile) {
    case WeightsProfile::Qwen38QuasarNvfp4:
        return QType::NVFP4;
    case WeightsProfile::Qwen38Nvfp4:
        return QType::FP8_E4M3FN_ROW_BF16S;
    case WeightsProfile::Qwen38Fp8:
        return QType::FP8_E4M3FN_BLOCK128_BF16S;
    default:
        throw std::invalid_argument("TP4 text projection requires Qwen3.8 NVFP4, QUASAR NVFP4 or native FP8");
    }
}

std::size_t gdn_snapshot_workspace_bytes(const Tensor& hidden,
                                         const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t gdn_record_workspace_bytes(const Tensor& hidden,
                                       const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t post_mixer_workspace_bytes(QType gate_up_qtype, QType down_qtype,
                                       ops::LinearPolicy policy, std::int32_t first,
                                       std::int32_t last, int tp) {
    const std::int32_t intermediate = TextConfig::intermediate / tp;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {intermediate, last});
    {
        auto scope = layout.scope();
        const auto bytes = tp == 1
            ? ops::linear_swiglu_workspace_capacity_bytes(
                  gate_up_qtype, 2 * intermediate, TextConfig::hidden, policy, first, last)
            : ops::linear_swiglu_column_parallel_workspace_capacity_bytes(
                  gate_up_qtype, policy, first, last, tp);
        if (bytes != 0) { (void)layout.alloc_bytes(bytes); }
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
#ifdef NINFER_VOLTA_BUILD
    // Volta's producer grid keeps growing beyond 32K. Bound the idle CTAs without changing
    // the device-side active split partition; 268800 keys reaches its fixed split cap.
    for (const std::uint32_t visible_end : {65536U, 98304U, 131072U, 196608U, 268800U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
#endif
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t capacity,
                                                                  std::uint32_t draft_window,
                                                                  std::uint32_t,
                                                                  bool long_attention) {
    if (capacity == 0 || draft_window == 0) { return {}; }
    std::vector<std::uint32_t> ends;
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8192U, 16384U,
                                           16390U, 32768U, 65536U}) {
        if (visible_end > draft_window + 1U) {
            ends.push_back(visible_end - draft_window - 1U);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    auto profiles = graph_profiles_through(capacity - 1U, ends);
    if (long_attention && draft_window >= 6) {
        // The seven/eight-query long verifier replaces two attention/reduce
        // pairs with one. GraphExecUpdate cannot cross that node-count change;
        // keep both executable ownership and its memory allowance explicit.
        for (auto& profile : profiles) {
            const auto visible = std::min<std::uint64_t>(
                capacity, static_cast<std::uint64_t>(profile.max) + draft_window + 1ULL);
            profile.topology_class = visible > 16390U ? 1U : 0U;
        }
    }
    return profiles;
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        ops::attn_input_proj(hidden, split->query_key, split->gate_value, query, gate, key, value,
                             workspace, stream);
        return;
    }
    const Weight& fused = std::get<FusedAttentionProjectionPayload>(weights).query_key_gate_value;
    ops::attn_input_proj(hidden, fused, query, gate, key, value, text_policy(fused), workspace,
                         stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, weights.packed, packed, stream);
    Tensor query_heads = query.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor key_heads   = key.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor value_heads = value.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    ops::linear_pair(hidden, weights.key, weights.value, key, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.query, query, stream);
    ops::linear(hidden, weights.output_gate, gate, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj(hidden, split->query_key, split->value_z, qkv, output_gate_flat,
                            workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat, text_policy(fused), workspace,
                        stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_snapshot_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, split->query_key, split->value_z, conv_weight,
                                          conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          leaf_workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view, text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record(hidden, split->query_key, split->value_z, conv_weight,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, output_gate_view, leaf_workspace,
                                        stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value, output_gate_view,
                                    text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    if (weight.qtype == QType::GGML_K) {
        ops::ggml_k_gdn_output(hidden, weight, residual, workspace, stream);
        return;
    }
    ops::linear_add(hidden, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* split =
            std::get_if<SplitGdnControlProjectionPayload>(&weights.control_projection)) {
        ops::gdn_norm_gating_proj(residual, norm_weight, eps, split->a_projection,
                                  split->b_projection, weights.a_log, weights.dt_bias, workspace,
                                  hidden, g, beta, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnControlProjectionPayload>(weights.control_projection).a_b_projection;
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, fused, weights.a_log, weights.dt_bias,
                              workspace, hidden, g, beta, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation, text_policy(weights.gate_up), workspace,
                       stream);
    ops::linear_add(activation, weights.down, residual, text_policy(weights.down), workspace,
                    stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
    ops::linear(hidden, weights.gate_up, gate_up, stream);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                  gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(activation, weights.down, delta, stream);
    ops::residual_add(delta, residual, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(WeightsProfile profile, int tp, std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, last});
    if (profile == WeightsProfile::Qwen38Fp8) {
        const auto bytes = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_BLOCK128_BF16S, TextConfig::mtp_attention_input_rows / tp,
            TextConfig::hidden, ops::LinearPolicy::A16Only, first, last);
        if (bytes) { (void)layout.alloc_bytes(bytes); }
    }
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(WeightsProfile profile, int tp, std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    if (profile == WeightsProfile::Qwen38Fp8) {
        return ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_BLOCK128_BF16S,
            TextConfig::kv_size / tp, TextConfig::hidden, ops::LinearPolicy::A16Only, first, last);
    }
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(WeightsProfile profile, int tp, std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    if (profile == WeightsProfile::Qwen38Fp8) {
        return ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_BLOCK128_BF16S,
            TextConfig::query_size / tp, TextConfig::hidden, ops::LinearPolicy::A16Only, first, last);
    }
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile, int tp,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    if (tp == 4) {
        return ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
            tp4_projection_type(weights_profile), ops::LinearPolicy::A16Only, first, last, tp);
    }
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return 0;
    case WeightsProfile::Qwen38Fp8:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_BLOCK128_BF16S, 14336, TextConfig::hidden,
            ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen38GgmlK:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::GGML_K, 14336, TextConfig::hidden, ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
    case WeightsProfile::Qwen38QuasarNvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::NVFP4, 14336, TextConfig::hidden, kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 14336, TextConfig::hidden, kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, int tp, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    if (tp == 4) {
        return ops::linear_add_workspace_capacity_bytes(
            tp4_projection_type(weights_profile), TextConfig::hidden, TextConfig::query_size / tp,
            ops::LinearPolicy::A16Only, first, last);
    }
    switch (weights_profile) {
    case WeightsProfile::Qwen38Fp8:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_BLOCK128_BF16S,
            TextConfig::hidden, TextConfig::value_dim, ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen38GgmlK:
        return ops::linear_add_workspace_capacity_bytes(
            QType::GGML_K, TextConfig::hidden, TextConfig::value_dim,
            ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::query_size,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
    case WeightsProfile::Qwen38QuasarNvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                        TextConfig::query_size, kNvfp4TextPolicy,
                                                        first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                        TextConfig::hidden, TextConfig::query_size,
                                                        kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile, int tp,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    if (tp == 4) {
        return ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
            tp4_projection_type(weights_profile), ops::LinearPolicy::A16Only, first, last, tp);
    }
    switch (weights_profile) {
    case WeightsProfile::Qwen38Fp8:
        return ops::gdn_input_proj_workspace_capacity_bytes(QType::FP8_E4M3FN_BLOCK128_BF16S,
            16384, TextConfig::hidden, ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen38GgmlK:
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::GGML_K, 16384, TextConfig::hidden, ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return 0;
    case WeightsProfile::Qwen36Nvfp4:
    case WeightsProfile::Qwen38QuasarNvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, TextConfig::hidden,
                                                            kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile weights_profile, int tp, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    if (tp == 4) {
        return ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
            tp4_projection_type(weights_profile), ops::LinearPolicy::A16Only, batch_size, first,
            last, tp);
    }
    switch (weights_profile) {
    case WeightsProfile::Qwen38Fp8:
        return std::max(kMinimumLeafWorkspaceBytes,
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                QType::FP8_E4M3FN_BLOCK128_BF16S, 16384, TextConfig::hidden,
                ops::LinearPolicy::A16Only, batch_size, first, last));
    case WeightsProfile::Qwen38GgmlK:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::GGML_K, 16384, TextConfig::hidden,
                            ops::LinearPolicy::A16Only, batch_size, first, last));
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Qwen36Nvfp4:
    case WeightsProfile::Qwen38QuasarNvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy,
                            batch_size, first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, int tp, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    if (tp == 4) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
                            tp4_projection_type(weights_profile), ops::LinearPolicy::A16Only,
                            batch_size, first, last, tp));
    }
    switch (weights_profile) {
    case WeightsProfile::Qwen38Fp8:
        return std::max(kMinimumLeafWorkspaceBytes,
            ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                QType::FP8_E4M3FN_BLOCK128_BF16S, 16384, TextConfig::hidden,
                ops::LinearPolicy::A16Only, batch_size, first, last));
    case WeightsProfile::Qwen38GgmlK:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::GGML_K, 16384, TextConfig::hidden,
                            ops::LinearPolicy::A16Only, batch_size, first, last));
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Qwen36Nvfp4:
    case WeightsProfile::Qwen38QuasarNvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy,
                            batch_size, first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile, int tp,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    if (tp == 4) {
        return ops::linear_add_workspace_capacity_bytes(
            tp4_projection_type(weights_profile), TextConfig::hidden, TextConfig::value_dim / tp,
            ops::LinearPolicy::A16Only, first, last);
    }
    switch (weights_profile) {
    case WeightsProfile::Qwen38Fp8:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_BLOCK128_BF16S,
            TextConfig::hidden, TextConfig::value_dim, ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen38GgmlK:
        return ops::linear_add_workspace_capacity_bytes(
            QType::GGML_K, TextConfig::hidden, TextConfig::value_dim,
            ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::value_dim,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
    case WeightsProfile::Qwen38QuasarNvfp4:
        return ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::hidden, TextConfig::value_dim, kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                        TextConfig::hidden, TextConfig::value_dim,
                                                        kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, int tp, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    if (tp == 4) {
        return ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(first, last);
    }
    if (weights_profile == WeightsProfile::Qwen38GgmlK) {
        return ops::linear_workspace_capacity_bytes(
            QType::GGML_K, TextConfig::gdn_value_heads, TextConfig::hidden,
            ops::LinearPolicy::A16Only, first, last);
    }
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile, int tp,
                                                         qwen3_6::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38Fp8:
        return post_mixer_workspace_bytes(QType::FP8_E4M3FN_BLOCK128_BF16S,
            QType::FP8_E4M3FN_BLOCK128_BF16S, ops::LinearPolicy::A16Only, first, last, tp);
    case WeightsProfile::Qwen38GgmlK:
        return post_mixer_workspace_bytes(QType::GGML_K, QType::GGML_K,
                                          ops::LinearPolicy::A16Only, first, last, tp);
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return post_mixer_workspace_bytes(QType::Q4G64_F16S, QType::Q5G64_F16S,
                                          ops::LinearPolicy::A16Only, first, last, tp);
    case WeightsProfile::Qwen36Nvfp4:
    case WeightsProfile::Qwen38QuasarNvfp4:
        return post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first,
                                          last, tp);
    case WeightsProfile::Qwen38Nvfp4: {
        const std::size_t nvfp4 =
            post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first, last, tp);
        const std::size_t fp8 = post_mixer_workspace_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, QType::FP8_E4M3FN_ROW_BF16S, kFp8TextPolicy, first, last, tp);
        return std::max(nvfp4, fp8);
    }
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

// --- tp == 2 split leaves ----------------------------------------------------------------------
//
// Each of these is the tp1 leaf above with every per-rank argument taken as an array and the
// matching `*_column_parallel` / `*_row_parallel` Op called ONCE for both ranks. The weight
// variant (fused vs split storage) is resolved from rank 0 and required to agree on rank 1: both
// ranks bind the same artifact objects through the same profile, so a disagreement is a loader
// bug, not a supported configuration.

namespace {

template <class Payload, class Weights>
std::array<const Payload*, kMaximumExecutionDevices>
require_same_alternative(std::span<const Weights* const> w, const char* label) {
    std::array<const Payload*, kMaximumExecutionDevices> out{};
    for (std::size_t rank = 0; rank < w.size(); ++rank) {
        out[rank] = std::get_if<Payload>(w[rank]);
        if (out[rank] == nullptr) {
            throw std::logic_error(std::string(label) + ": ranks disagree on weight storage form");
        }
    }
    return out;
}

template <class T, std::size_t N>
auto active(const std::array<T, N>& values, const ExecutionContext& ec) {
    return std::span(values).first(ec.device_count);
}

template <class Range, class Payload>
std::array<Weight, kMaximumExecutionDevices>
weight_shards(const Range& payloads, Weight Payload::* member, const ExecutionContext& ec) {
    std::array<Weight, kMaximumExecutionDevices> out{};
    for (int rank = 0; rank < ec.device_count; ++rank) { out[rank] = payloads[rank]->*member; }
    return out;
}

template <class Weights, class Payload>
std::array<const Payload*, kMaximumExecutionDevices>
payload_shards(std::span<const Weights* const> weights, Payload Weights::* member) {
    std::array<const Payload*, kMaximumExecutionDevices> out{};
    for (std::size_t rank = 0; rank < weights.size(); ++rank) {
        out[rank] = &(weights[rank]->*member);
    }
    return out;
}

std::array<std::optional<WorkspaceArena::Scope>, kMaximumExecutionDevices>
workspace_scopes(std::span<WorkspaceArena* const> workspace) {
    std::array<std::optional<WorkspaceArena::Scope>, kMaximumExecutionDevices> out{};
    for (std::size_t rank = 0; rank < workspace.size(); ++rank) {
        out[rank].emplace(workspace[rank]->scope());
    }
    return out;
}

// Current-device save/restore around a per-rank kernel issue. Only ONE tp2 leaf below needs it --
// `mtp_attention_projection`, whose second stage (`ops::mtp_split_attn_in`) is a purely
// elementwise remap with no cross-device form and no reason for one, so it is issued once per
// rank on that rank's own stream. Every other split leaf delegates wholly to an Op that owns its
// own per-rank issue. This mirrors ops::detail::for_each_rank rather than including an Op's
// private header, exactly as the family schedule does (text_context_impl.h).
class CurrentDevice {
public:
    CurrentDevice() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDevice() { (void)cudaSetDevice(previous_); }

    CurrentDevice(const CurrentDevice&)            = delete;
    CurrentDevice& operator=(const CurrentDevice&) = delete;

private:
    int previous_ = 0;
};

template <class Body>
void for_each_rank(const ExecutionContext& ec, Body&& body) {
    const CurrentDevice restore;
    for (int rank = 0; rank < ec.device_count; ++rank) {
        CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
        body(rank);
    }
}

// Both registered MTP codecs use A16 projections without transient Linear storage.
void require_mtp_shard(const Weight& a, const Weight& b, const char* label) {
    if (a.qtype != b.qtype ||
        (a.qtype != QType::W8G32_F16S && a.qtype != QType::GGML_K &&
         a.qtype != QType::FP8_E4M3FN_BLOCK128_BF16S)) {
        throw std::logic_error(std::string(label) +
                               ": unsupported or inconsistent tp2 MTP weight format");
    }
    if (a.k != b.k || a.n != b.n) {
        throw std::logic_error(std::string(label) + ": tp2 MTP shards disagree on shape");
    }
}

} // namespace

void Variant::attention_projection(std::span<const Tensor> hidden,
                                   std::span<const FullAttentionProjectionWeights* const> w,
                                   std::span<const Tensor> query,
                                   std::span<const Tensor> gate,
                                   std::span<const Tensor> key,
                                   std::span<const Tensor> value, qwen3_6::TextPhase,
                                   std::span<WorkspaceArena* const> workspace,
                                   const ExecutionContext& ec) {
    if (std::holds_alternative<SplitAttentionProjectionPayload>(*w[0])) {
        const auto split = require_same_alternative<SplitAttentionProjectionPayload>(
            w, "attention projection");
        ops::attn_input_proj_column_parallel(
            hidden, active(weight_shards(split, &SplitAttentionProjectionPayload::query_key, ec), ec),
            active(weight_shards(split, &SplitAttentionProjectionPayload::gate_value, ec), ec), query, gate, key, value,
            workspace, ec);
        return;
    }
    const auto fused =
        require_same_alternative<FusedAttentionProjectionPayload>(w, "attention projection");
    ops::attn_input_proj_column_parallel(
        hidden, active(weight_shards(fused, &FusedAttentionProjectionPayload::query_key_gate_value, ec), ec), query,
        gate, key, value, text_policy(fused[0]->query_key_gate_value), workspace, ec);
}

void Variant::attention_output_projection(std::span<const Tensor> attention,
                                          std::span<const Weight> weight,
                                          std::span<const Tensor> residual,
                                          std::span<const Tensor> staging, qwen3_6::TextPhase,
                                          std::span<WorkspaceArena* const> workspace,
                                          const ExecutionContext& ec, const ops::PeerEvents& ev,
                                    const ops::ReplicatedRmsNorm* norm) {
    ops::linear_add_row_parallel(attention, weight, residual, staging, text_policy(weight[0]),
                                 workspace, ec, ev, norm);
}

void Variant::gdn_input_projection(std::span<const Tensor> hidden,
                                   std::span<const GdnProjectionWeights* const> w,
                                   std::span<const Tensor> qkv,
                                   std::span<const Tensor> output_gate, qwen3_6::TextPhase,
                                   std::span<WorkspaceArena* const> workspace,
                                   const ExecutionContext& ec) {
    // The caller holds `z` as [head_dim, value_heads, T]; the Op wants the flat [value_dim, T],
    // exactly as the tp1 leaf above does. The shard's value_dim is read off the tensor rather than
    // assumed, so this stays correct if the head split ever changes.
    std::array<Tensor, kMaximumExecutionDevices> output_gate_flat{};
    for (int rank = 0; rank < ec.device_count; ++rank) {
        output_gate_flat[rank] = output_gate[rank].view(
            {output_gate[rank].ne[0] * output_gate[rank].ne[1], hidden[rank].ne[1]});
    }
    const auto input = payload_shards(w, &GdnProjectionWeights::input_projection);
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(*input[0])) {
        const auto split =
            require_same_alternative<SplitGdnInputProjectionPayload>(active(input, ec), "GDN input projection");
        ops::gdn_input_proj_column_parallel(hidden,
                                            active(weight_shards(split, &SplitGdnInputProjectionPayload::query_key, ec), ec),
                                            active(weight_shards(split, &SplitGdnInputProjectionPayload::value_z, ec), ec), qkv,
                                            active(output_gate_flat, ec), ec);
        return;
    }
    const auto fused =
        require_same_alternative<FusedGdnInputProjectionPayload>(active(input, ec), "GDN input projection");
    ops::gdn_input_proj_column_parallel(
        hidden, active(weight_shards(fused, &FusedGdnInputProjectionPayload::query_key_value_z, ec), ec), qkv,
        active(output_gate_flat, ec), text_policy(fused[0]->query_key_value_z), workspace, ec);
}

void Variant::gdn_input_projection_snapshot(
    std::span<const Tensor> hidden, std::span<const GdnProjectionWeights* const> w,
    std::span<const Tensor> conv_weight, std::span<const Tensor> conv_states,
    std::span<const Tensor> valid_columns, std::span<const Tensor> initial_slot,
    std::span<const Tensor> snapshot_base_slot, std::span<const Tensor> query,
    std::span<const Tensor> key, std::span<const Tensor> value,
    std::span<const Tensor> output_gate, qwen3_6::TextPhase,
    std::span<WorkspaceArena* const> workspace, const ExecutionContext& ec) {
    const auto input = payload_shards(w, &GdnProjectionWeights::input_projection);
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(*input[0])) {
        const auto split = require_same_alternative<SplitGdnInputProjectionPayload>(active(input, ec), "GDN snapshot projection");
        ops::gdn_input_proj_conv_snapshot_column_parallel(
            hidden, active(weight_shards(split, &SplitGdnInputProjectionPayload::query_key, ec), ec),
            active(weight_shards(split, &SplitGdnInputProjectionPayload::value_z, ec), ec), conv_weight, conv_states, valid_columns,
            initial_slot, snapshot_base_slot, query, key, value, output_gate, workspace, ec);
        return;
    }
    const auto fused =
        require_same_alternative<FusedGdnInputProjectionPayload>(active(input, ec), "GDN snapshot projection");
    ops::gdn_input_proj_conv_snapshot_column_parallel(
        hidden, active(weight_shards(fused, &FusedGdnInputProjectionPayload::query_key_value_z, ec), ec), conv_weight,
        conv_states, valid_columns, initial_slot, snapshot_base_slot, query, key, value,
        output_gate, text_policy(fused[0]->query_key_value_z), workspace, ec);
}

void Variant::gdn_output_projection(std::span<const Tensor> hidden,
                                    std::span<const Weight> weight,
                                    std::span<const Tensor> residual,
                                    std::span<const Tensor> staging, qwen3_6::TextPhase,
                                    std::span<WorkspaceArena* const> workspace,
                                    const ExecutionContext& ec, const ops::PeerEvents& ev,
                                    const ops::ReplicatedRmsNorm* norm) {
    if (weight[0].qtype == QType::GGML_K) {
        ops::ggml_k_gdn_output(hidden, weight, residual, staging, workspace, ec, ev);
        if (norm != nullptr) for_each_rank(ec, [&](int rank) {
            Tensor normalized = norm->output[rank];
            ops::rmsnorm(residual[rank], norm->weight[rank], norm->eps, norm->unit_offset,
                         normalized, ec.dev[rank]->stream);
        });
        return;
    }
    ops::linear_add_row_parallel(hidden, weight, residual, staging, text_policy(weight[0]),
                                 workspace, ec, ev, norm);
}

void Variant::gdn_control_projection(std::span<const Tensor> hidden,
                                     std::span<const GdnProjectionWeights* const> w,
                                     std::span<const Tensor> g,
                                     std::span<const Tensor> beta,
                                     std::span<WorkspaceArena* const> workspace,
                                     const ExecutionContext& ec) {
    // The tp1 leaf fuses the input RMSNorm into the gating GEMM. There is no split form of the
    // fused kernel and no reason for one: the norm is replicated elementwise work over the
    // full-width residual, so the caller runs it per rank and this leaf takes the normalized
    // hidden directly.
    const auto control = payload_shards(w, &GdnProjectionWeights::control_projection);
    std::array<Tensor, kMaximumExecutionDevices> a_log{}, dt_bias{};
    for (int rank = 0; rank < ec.device_count; ++rank) {
        a_log[rank] = w[rank]->a_log;
        dt_bias[rank] = w[rank]->dt_bias;
    }
    if (std::holds_alternative<SplitGdnControlProjectionPayload>(*control[0])) {
        const auto split = require_same_alternative<SplitGdnControlProjectionPayload>(active(control, ec), "GDN control projection");
        ops::gdn_gating_proj_column_parallel(
            hidden, active(weight_shards(split, &SplitGdnControlProjectionPayload::a_projection, ec), ec),
            active(weight_shards(split, &SplitGdnControlProjectionPayload::b_projection, ec), ec), active(a_log, ec), active(dt_bias, ec), workspace, g,
            beta, ec);
        return;
    }
    const auto fused = require_same_alternative<FusedGdnControlProjectionPayload>(active(control, ec), "GDN control projection");
    ops::gdn_gating_proj_column_parallel(
        hidden, active(weight_shards(fused, &FusedGdnControlProjectionPayload::a_b_projection, ec), ec), active(a_log, ec), active(dt_bias, ec),
        workspace, g, beta, ec);
}

DType Variant::gdn_output_input_dtype(const Weight& w, int tp, int tokens) {
    return tp == 4 && tokens >= 1 && tokens <= 8 && w.qtype == QType::NVFP4 &&
           w.layout == QuantLayout::VoltaQpnPrepacked && w.n == 5120 && w.k == 1536
               ? DType::FP16 : DType::BF16;
}

DType Variant::post_mixer_input_dtype(const PostMixerWeights& w, int tp, int tokens) {
    return tp == 4 && tokens >= 1 && tokens <= 8 &&
           w.gate_up.qtype == QType::NVFP4 &&
           w.gate_up.layout == QuantLayout::VoltaQpnPrepacked &&
           w.gate_up.n == 8704 && w.gate_up.k == 5120 ? DType::FP16 : DType::BF16;
}

void Variant::post_mixer(std::span<const Tensor> hidden,
                         std::span<const PostMixerWeights* const> w,
                         std::span<const Tensor> residual,
                         std::span<const Tensor> staging, qwen3_6::TextPhase,
                         std::span<WorkspaceArena* const> workspace,
                         const ExecutionContext& ec, const ops::PeerEvents& ev,
                         const ops::ReplicatedRmsNorm* norm) {
    // The activation width is this rank's own gate/up shard, read off the weight rather than
    // assumed: `gate_up` is [2 * intermediate_shard, hidden], so half its rows is the shard.
    const std::int32_t shard_intermediate = w[0]->gate_up.n / 2;
    for (int rank = 1; rank < ec.device_count; ++rank) {
        if (w[rank]->gate_up.n != w[0]->gate_up.n) {
        throw std::logic_error("post_mixer: tp2 gate/up shards disagree on width");
        }
    }
    std::array<Tensor, kMaximumExecutionDevices> activation{};
    auto scopes = workspace_scopes(workspace);
    for (std::size_t rank = 0; rank < ec.device_count; ++rank) {
        const bool fp16_activation = norm && norm->weight.size() == 4 &&
            norm->output.size() == 4 && ev.packet_storage(0) &&
            post_mixer_input_dtype(*w[rank], ec.device_count, hidden[rank].ne[1]) == DType::FP16 &&
            w[rank]->down.qtype == QType::NVFP4 &&
            w[rank]->down.layout == QuantLayout::VoltaQpnPrepacked &&
            w[rank]->down.n == 5120 && w[rank]->down.k == 4352;
        const auto activation_dtype = fp16_activation ? DType::FP16 : DType::BF16;
        activation[rank] =
            workspace[rank]->alloc(activation_dtype, {shard_intermediate, hidden[0].ne[1]});
    }
    ops::linear_swiglu_column_parallel(hidden, active(weight_shards(w, &PostMixerWeights::gate_up, ec), ec), active(activation, ec),
                                       text_policy(w[0]->gate_up), workspace, ec);
    ops::linear_add_row_parallel(active(activation, ec), active(weight_shards(w, &PostMixerWeights::down, ec), ec), residual, staging,
                                 text_policy(w[0]->down), workspace, ec, ev, norm);
}

void Variant::gdn_input_projection_record(
    std::span<const Tensor> hidden, std::span<const GdnProjectionWeights* const> w,
    std::span<const Tensor> conv_weight, std::span<const Tensor> conv_states,
    std::span<const Tensor> valid_columns, std::span<const Tensor> initial_slots,
    std::span<const Tensor> conv_record, std::span<const Tensor> query,
    std::span<const Tensor> key, std::span<const Tensor> value,
    std::span<const Tensor> output_gate, qwen3_6::TextPhase,
    std::span<WorkspaceArena* const> workspace, const ExecutionContext& ec) {
    // The record twin of `gdn_input_projection_snapshot` above, reached only from the speculative
    // verify round: instead of snapshotting the post-round conv state it writes a per-column
    // `conv_record` that the peer's `ops::gdn_replay_fold` later folds at
    // FoldGeometry<48, 8, 24, 5120>. Everything else -- the shard extents, the weight storage
    // form, the per-rank issue -- is the snapshot leaf's.
    const auto input = payload_shards(w, &GdnProjectionWeights::input_projection);
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(*input[0])) {
        const auto split = require_same_alternative<SplitGdnInputProjectionPayload>(active(input, ec), "GDN record projection");
        ops::gdn_input_proj_conv_record_column_parallel(
            hidden, active(weight_shards(split, &SplitGdnInputProjectionPayload::query_key, ec), ec),
            active(weight_shards(split, &SplitGdnInputProjectionPayload::value_z, ec), ec), conv_weight, conv_states, valid_columns,
            initial_slots, conv_record, query, key, value, output_gate, workspace, ec);
        return;
    }
    const auto fused =
        require_same_alternative<FusedGdnInputProjectionPayload>(active(input, ec), "GDN record projection");
    ops::gdn_input_proj_conv_record_column_parallel(
        hidden, active(weight_shards(fused, &FusedGdnInputProjectionPayload::query_key_value_z, ec), ec), conv_weight,
        conv_states, valid_columns, initial_slots, conv_record, query, key, value, output_gate,
        text_policy(fused[0]->query_key_value_z), workspace, ec);
}

void Variant::mtp_attention_projection(
    std::span<const Tensor> hidden,
    std::span<const MtpAttentionProjectionWeights* const> w,
    std::span<const Tensor> query, std::span<const Tensor> gate,
    std::span<const Tensor> key, std::span<const Tensor> value,
    std::span<WorkspaceArena* const> workspace, const ExecutionContext& ec) {
    for (int rank = 1; rank < ec.device_count; ++rank) {
        require_mtp_shard(w[0]->packed, w[rank]->packed, "MTP attention projection");
    }
    const std::int32_t columns   = hidden[0].ne[1];
    const std::int32_t attn_rows = w[0]->packed.n;
    for (int rank = 1; rank < ec.device_count; ++rank) {
        if (hidden[rank].ne[1] != columns) {
        throw std::logic_error("MTP attention projection: tp2 ranks disagree on token count");
        }
    }
    std::array<Tensor, kMaximumExecutionDevices> packed{};
    auto scopes = workspace_scopes(workspace);
    for (std::size_t rank = 0; rank < ec.device_count; ++rank) {
        packed[rank] = workspace[rank]->alloc(DType::BF16, {attn_rows, columns});
    }
    ops::linear_column_parallel(hidden, active(weight_shards(w, &MtpAttentionProjectionWeights::packed, ec), ec), active(packed, ec), ops::LinearPolicy::A16Only, workspace, ec);
    // `mtp_split_attn_in` selects its section boundaries from the packed row count alone, so the
    // shard geometry (rows [0,3072) Q | [3072,3584) K | [3584,6656) Gate | [6656,7168) V) needs
    // no rank argument. The resulting head indices are DEVICE-LOCAL, which is exactly what the
    // head-local 12|2 gqa_attention downstream consumes.
    for_each_rank(ec, [&](int rank) {
        const auto r          = static_cast<std::size_t>(rank);
        const std::int32_t qh = query[r].numel() / (TextConfig::head_dim * columns);
        const std::int32_t kh = key[r].numel() / (TextConfig::head_dim * columns);
        Tensor query_heads    = query[r].view({TextConfig::head_dim, qh, columns});
        Tensor gate_heads     = gate[r].view({TextConfig::head_dim, qh, columns});
        Tensor key_heads      = key[r].view({TextConfig::head_dim, kh, columns});
        Tensor value_heads    = value[r].view({TextConfig::head_dim, kh, columns});
        ops::mtp_split_attn_in(packed[r], query_heads, key_heads, gate_heads, value_heads,
                               ec.dev[rank]->stream);
    });
}

void Variant::mtp_kv_projection(std::span<const Tensor> hidden,
                                std::span<const MtpAttentionProjectionWeights* const> w,
                                std::span<const Tensor> key,
                                std::span<const Tensor> value,
                                std::span<WorkspaceArena* const> workspace, const ExecutionContext& ec) {
    // The tp1 leaf fuses these two into one `linear_pair`; the shard's key and value row views
    // are separate blocks of the same packed shard, so at tp2 they are two column-parallel calls.
    for (int rank = 1; rank < ec.device_count; ++rank) {
        require_mtp_shard(w[0]->key, w[rank]->key, "MTP key projection");
    }
    for (int rank = 1; rank < ec.device_count; ++rank) {
        require_mtp_shard(w[0]->value, w[rank]->value, "MTP value projection");
    }
    ops::linear_column_parallel(hidden, active(weight_shards(w, &MtpAttentionProjectionWeights::key, ec), ec), key, ops::LinearPolicy::A16Only, workspace, ec);
    ops::linear_column_parallel(hidden, active(weight_shards(w, &MtpAttentionProjectionWeights::value, ec), ec), value, ops::LinearPolicy::A16Only, workspace, ec);
}

void Variant::mtp_q_gate_projection(std::span<const Tensor> hidden,
                                    std::span<const MtpAttentionProjectionWeights* const> w,
                                    std::span<const Tensor> query,
                                    std::span<const Tensor> gate,
                                    std::span<WorkspaceArena* const>,
                                    const ExecutionContext& ec) {
    for (int rank = 1; rank < ec.device_count; ++rank) {
        require_mtp_shard(w[0]->query, w[rank]->query, "MTP query projection");
    }
    for (int rank = 1; rank < ec.device_count; ++rank) {
        require_mtp_shard(w[0]->output_gate, w[rank]->output_gate, "MTP gate projection");
    }
    ops::linear_column_parallel(hidden, active(weight_shards(w, &MtpAttentionProjectionWeights::query, ec), ec), query, ec);
    ops::linear_column_parallel(hidden, active(weight_shards(w, &MtpAttentionProjectionWeights::output_gate, ec), ec), gate, ec);
}

void Variant::mtp_post_mixer(std::span<const Tensor> hidden,
                             std::span<const MtpPostMixerWeights* const> w,
                             std::span<const Tensor> residual,
                             std::span<const Tensor> staging,
                             std::span<WorkspaceArena* const> workspace,
                             const ExecutionContext& ec, const ops::PeerEvents& ev) {
    // The MTP post-mixer is composed exactly the way the tp1 leaf above composes it -- separate
    // `linear` / `silu_mul` / `linear` / `residual_add`, NOT the fused linear_swiglu + linear_add
    // pair the text post-mixer uses. That is not a stylistic choice: neither
    // `linear_swiglu_column_parallel` nor `linear_add_row_parallel` registers W8G32_F16S, which
    // is the format of every MTP object, and the tp1 MTP leaf already avoids both fused Ops for
    // the same reason. `tests/ops/test_mtp_split.cpp`'s Leg A proves this exact composition at
    // tp2 -- column-parallel gate_up, a shard-local silu_mul over the shard's own gate/up halves,
    // then row-parallel down plus the all-reduce.
    for (int rank = 1; rank < ec.device_count; ++rank) {
        require_mtp_shard(w[0]->gate_up, w[rank]->gate_up, "MTP post mixer gate/up");
    }
    for (int rank = 1; rank < ec.device_count; ++rank) {
        require_mtp_shard(w[0]->down, w[rank]->down, "MTP post mixer down");
    }
    const std::int32_t shard_intermediate = w[0]->gate_up.n / 2;
    const std::int32_t columns            = hidden[0].ne[1];
    std::array<Tensor, kMaximumExecutionDevices> gate_up{};
    std::array<Tensor, kMaximumExecutionDevices> activation{};
    std::array<Tensor, kMaximumExecutionDevices> delta{};
    auto scopes = workspace_scopes(workspace);
    for (std::size_t rank = 0; rank < ec.device_count; ++rank) {
        gate_up[rank]    = workspace[rank]->alloc(DType::BF16, {w[rank]->gate_up.n, columns});
        activation[rank] = workspace[rank]->alloc(DType::BF16, {shard_intermediate, columns});
        delta[rank]      = workspace[rank]->alloc(DType::BF16, {TextConfig::hidden, columns});
    }
    ops::linear_column_parallel(hidden, active(weight_shards(w, &PostMixerWeights::gate_up, ec), ec), active(gate_up, ec), ops::LinearPolicy::A16Only, workspace, ec);
    // Each rank's gate and up halves are its OWN shard's halves -- the ShardPlan splits gate_up
    // as two independent column blocks, so rank r holds gate rows [r*I/2 ...] and up rows in the
    // matching block, and the SiLU pairing is rank-local with nothing to communicate.
    for_each_rank(ec, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::silu_mul(gate_up[r].slice(0, 0, shard_intermediate),
                      gate_up[r].slice(0, shard_intermediate, shard_intermediate), activation[r],
                      ec.dev[rank]->stream);
    });
    ops::linear_row_parallel(active(activation, ec), active(weight_shards(w, &PostMixerWeights::down, ec), ec), active(delta, ec), staging, ops::LinearPolicy::A16Only, workspace, ec, ev);
    // `delta` is identical on both ranks after the collective, so the residual fold is replicated
    // elementwise work and keeps the residual bit-identical across devices.
    for_each_rank(ec, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::residual_add(delta[r], const_cast<Tensor&>(residual[r]), ec.dev[rank]->stream);
    });
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(WeightsProfile profile, int tp, std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    if (profile == WeightsProfile::Qwen38Fp8) {
        const auto bytes = std::max(
            ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_BLOCK128_BF16S,
                TextConfig::mtp_mlp_gate_up_rows / tp, TextConfig::hidden,
                ops::LinearPolicy::A16Only, first, last),
            ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_BLOCK128_BF16S,
                TextConfig::hidden, TextConfig::intermediate / tp,
                ops::LinearPolicy::A16Only, first, last));
        if (bytes) { (void)layout.alloc_bytes(bytes); }
    }
    return layout.peak_bytes(1);
}

} // namespace ninfer::targets::qwen3_6_27b::detail
