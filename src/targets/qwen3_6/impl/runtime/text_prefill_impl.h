#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/token_bitmask.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

DFlashFeatureSink make_dflash_prefill_sink(PrefillContext& state) {
    if (!state.execution.io.dflash_decode || state.dflash_host_ingress == nullptr) {
        throw std::logic_error("DFlash prefill controls are unavailable");
    }
    return dflash_feature_sink(
        state, [&state](const Tensor& features, const Tensor& positions, bool rewrite_checkpoint) {
            auto& frame  = *state.execution.io.dflash_decode;
            Tensor count = frame.append_counts.slice(0, 0, 1);
            Tensor lane  = frame.lanes.slice(0, 0, 1);
            Tensor row   = frame.dflash_kv_table_rows.slice(0, 0, 1);
            ops::set_i32_scalar(count, features.ne[1], state.execution.device.stream);
            const auto exact = static_cast<std::uint32_t>(features.ne[1]);
            dflash_append_context(state, features, positions, count, lane, row, {exact, exact});
            if (rewrite_checkpoint) {
                state.dflash->save_rewrite_checkpoint(state.dflash_host_ingress->lanes[0],
                                                      state.execution.device.stream);
            }
        });
}

} // namespace

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t current_state_slot,
                         std::int32_t rewrite_checkpoint_state_slot,
                         std::uint32_t mtp_proposal_extent) {
    card.set_sampling(sampling);
    card.set_volta_long_attention(execution.volta_long_attention);
    card.set_pipeline(pipeline_execution(execution), execution.pipeline_upload);
    card.set_linear_state_slots(current_state_slot, rewrite_checkpoint_state_slot);
    card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    card.set_mtp_proposal_extent(mtp_proposal_extent);
    if (execution.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_ids() == nullptr ||
        card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

PrefillChunkResult prefill_text_chunk(
    PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier, bool finalize_at_end) {
    auto tp = tp_execution(state.execution);
    for (std::size_t index = 0; index < state.execution.peers.size(); ++index) {
        tp[index].mtp_kv = state.mtp_kv_peers[index];
        if (tp[index].mtp_kv.valid() != state.mtp_kv.valid()) {
            throw std::logic_error("tensor-parallel MTP KV windows disagree between ranks");
        }
    }
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                     std::span(tp).first(state.execution.peers.size()));
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_retrieval_query_capture(state.capture_retrieval_query);
    card.set_token_constraints(state.execution.token_constraints);
    card.set_prefill_rewrite_checkpoint_frontier(
        rewrite_checkpoint_capture_frontier
            ? static_cast<std::int64_t>(*rewrite_checkpoint_capture_frontier)
            : -1);
    const std::span<const int> prompt(ids.data(), ids.size());
    card.set_rope_delta(state.rope_delta);
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                  sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end);
}

PrefillChunkResult
prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier,
                         bool finalize_at_end) {
    if (state.dflash != nullptr) {
        throw std::logic_error("DFlash staged multimodal prefill is unavailable");
    }
    auto tp = tp_execution(state.execution);
    for (std::size_t index = 0; index < state.execution.peers.size(); ++index) {
        tp[index].mtp_kv = state.mtp_kv_peers[index];
    }
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                     std::span(tp).first(state.execution.peers.size()));
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_token_constraints(state.execution.token_constraints);
    card.set_prefill_rewrite_checkpoint_frontier(
        rewrite_checkpoint_capture_frontier
            ? static_cast<std::int64_t>(*rewrite_checkpoint_capture_frontier)
            : -1);
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision, finalize_at_end);
}

void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge) {
    if (!state.mtp_kv.valid() || bridge.previous_hidden == nullptr || state.text_kv_base == 0 ||
        bridge.position < 0 ||
        static_cast<std::uint32_t>(bridge.position) + 1 != state.text_kv_base) {
        throw std::logic_error("multimodal MTP bridge does not match the reusable frontier");
    }

    Tensor bridge_token = state.execution.io.mtp->target_input_ids.slice(0, 0, 1);
    const TokenId token = prompt.token_ids[state.text_kv_base];
    CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                               state.execution.device.stream));

    Tensor visual_embedding;
    const Tensor* composed_embedding = nullptr;
    if (prompt.token_types[state.text_kv_base] != 0) {
        const VisionChunk chunk = vision.prepare_chunk(state.text_kv_base, 1);
        if (chunk.control == nullptr) {
            throw std::logic_error("visual MTP bridge has no encoded Vision item");
        }
        const auto& scatter = chunk.control->scatter_indices;
        const auto column   = std::lower_bound(scatter.begin(), scatter.end(),
                                               static_cast<std::int32_t>(state.text_kv_base));
        if (column == scatter.end() || *column != static_cast<std::int32_t>(state.text_kv_base) ||
            static_cast<std::uint8_t>(chunk.control->modality) !=
                prompt.token_types[state.text_kv_base]) {
            throw std::logic_error("visual MTP bridge does not match Vision scatter metadata");
        }
        visual_embedding =
            chunk.embeddings.slice(1, static_cast<std::int32_t>(column - scatter.begin()), 1);
        composed_embedding = &visual_embedding;
    }

    mtp_bridge_and_propose(state, bridge_token, *bridge.previous_hidden, bridge.position,
                           bridge.rope_position, false, composed_embedding);
}

std::array<Tensor, kMaximumExecutionDevices>
resume_hidden(ExecutionCore& execution, const Tensor& hidden) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden || hidden.ne[1] != 1 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("prefix resume requires BF16 [hidden,1]");
    }
    auto result = rank_views(execution, [&](int rank) {
        return rank == 0 ? hidden : execution.peers[rank - 1].prefill_hidden->slice(1, 0, 1);
    });
    if (!execution.peers.empty()) {
        const auto& peer = execution.peers[0];
        ops::broadcast_rank0(hidden, std::span(result).subspan(1, execution.peers.size()),
                             *peer.execution, *peer.events);
    }
    return result;
}

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose) {
    const auto restored = resume_hidden(state.execution, hidden);
    state.execution.work.reset();
    Tensor logits = state.execution.io.logits.slice(1, 0, 1);
    auto tp = tp_execution(state.execution);
    if (!state.execution.peers.empty()) {
        for (const auto& peer : state.execution.peers) { peer.work->reset(); }
        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, state.text_kv,
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk,
                         state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                         std::span(tp).first(state.execution.peers.size()));
        const auto rank_logits = rank_views(state.execution, [&](int rank) {
            return rank == 0 ? logits : state.execution.peers[rank - 1].io->logits.slice(1, 0, 1);
        });
        card.target_logits(restored, rank_logits);
    } else {
        ops::linear(hidden, state.execution.model.output_head, logits,
                     state.execution.device.stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(state.execution.io.pos.data, &absolute_position,
                               sizeof(absolute_position), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    if (const TokenConstraintView& constraints = state.execution.token_constraints) {
        ops::apply_token_bitmask(logits, TextConfig::token_domain, 0, constraints.constrained,
                                 constraints.bitmask, constraints.words,
                                 state.execution.device.stream);
    }
    ops::sample(logits, state.execution.io.token, TextConfig::token_domain, state.sampling,
                state.execution.io.pos, purpose, state.execution.work,
                state.execution.device.stream);
    state.execution.work.reset();
    for (const auto& peer : state.execution.peers) { peer.work->reset(); }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
