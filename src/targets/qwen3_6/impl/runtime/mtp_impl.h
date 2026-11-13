#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

// One rank's window into ITS OWN MtpDecodeState, sliced to the round's batch. Both ranks are
// sliced by the same function so a shape mistake cannot apply to one device only -- which matters
// because rank 1's buffers live on the other GPU, where an out-of-bounds write is silent: a
// peer-side write that is exactly in bounds at batch 1 runs off the end of the buffer at batch 2
// with nothing on the local device noticing.
struct MtpRoundView {
    Tensor anchors;
    Tensor frontiers;
    Tensor budgets;
    Tensor current_extents;
    Tensor target_valid;
    Tensor current_drafts;
    Tensor target_rope;
    Tensor text_rows;
    Tensor mtp_rows;
    Tensor lanes;
    Tensor rope_deltas;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_tokens;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor selected_hidden;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted;
    Tensor next_extents;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;
    Tensor next_drafts;
    Tensor proposal_logits;
    const ops::SamplingConfig* sampling = nullptr;
};

MtpRoundView slice_mtp_frame(qwen3_6::MtpDecodeState& frame, std::int32_t batch_size) {
    MtpRoundView out;
    out.anchors           = frame.anchors.slice(0, 0, batch_size);
    out.frontiers         = frame.base_frontiers.slice(0, 0, batch_size);
    out.budgets           = frame.remaining_budgets.slice(0, 0, batch_size);
    out.current_extents   = frame.current_extents.slice(0, 0, batch_size);
    out.target_valid      = frame.target_valid_columns.slice(0, 0, batch_size);
    out.current_drafts    = frame.current_drafts.slice(1, 0, batch_size);
    out.target_rope       = frame.target_rope_positions.slice(1, 0, batch_size);
    out.text_rows         = frame.text_kv_table_rows.slice(0, 0, batch_size);
    out.mtp_rows          = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
    out.lanes             = frame.lanes.slice(0, 0, batch_size);
    out.rope_deltas       = frame.rope_deltas.slice(0, 0, batch_size);
    out.verify_ids        = frame.verify_ids.slice(1, 0, batch_size);
    out.target_positions  = frame.target_positions.slice(1, 0, batch_size);
    out.target_tokens     = frame.target_argmax.slice(1, 0, batch_size);
    out.target_logits     = frame.target_logits.slice(2, 0, batch_size);
    out.target_hidden     = frame.target_hidden.slice(2, 0, batch_size);
    out.selected_hidden   = frame.target_continuation_hidden.slice(1, 0, batch_size);
    out.licensed_tokens   = frame.licensed_tokens.slice(1, 0, batch_size);
    out.licensed_counts   = frame.licensed_counts.slice(0, 0, batch_size);
    out.accepted          = frame.accepted_drafts.slice(0, 0, batch_size);
    out.next_extents      = frame.next_extents.slice(0, 0, batch_size);
    out.alignment_ids     = frame.alignment_ids.slice(1, 0, batch_size);
    out.alignment_hidden  = frame.alignment_hidden.slice(2, 0, batch_size);
    out.ar_hidden         = frame.ar_hidden.slice(1, 0, batch_size);
    out.ar_positions      = frame.ar_positions.slice(0, 0, batch_size);
    out.ar_rope_positions = frame.ar_rope_positions.slice(0, 0, batch_size);
    out.ar_valid_columns  = frame.ar_valid_columns.slice(0, 0, batch_size);
    out.next_drafts       = frame.next_drafts.slice(0, 0, batch_size);
    out.proposal_logits   = frame.proposal_logits.slice(1, 0, batch_size);
    out.sampling          = frame.sampling;
    return out;
}

TargetVerifyFrameView verify_view(const MtpRoundView& v, const GdnReplayRecords* records) {
    return TargetVerifyFrameView{
        .ids             = v.verify_ids,
        .cache_positions = v.target_positions,
        .rope_positions  = v.target_rope,
        .valid_columns   = v.target_valid,
        .kv_table_rows   = v.text_rows,
        .lanes           = v.lanes,
        .target_hidden   = v.target_hidden,
        .target_logits   = v.target_logits,
        .target_tokens   = v.target_tokens,
        .drafts          = v.current_drafts,
        .current_extents = v.current_extents,
        .frontiers       = v.frontiers,
        .anchors         = v.anchors,
        .licensed_tokens = v.licensed_tokens,
        .licensed_counts = v.licensed_counts,
        .accepted_drafts = v.accepted,
        .selected_hidden = v.selected_hidden,
        .replay_records  = records,
        .sampling        = v.sampling,
    };
}

void mtp_bridge_tp2(PrefillContext& state, const Tensor& next_token,
                    const Tensor& previous_hidden, std::int32_t position,
                    std::span<const std::int32_t> rope_position, bool build_proposal,
                    const Tensor* next_embedding) {
    auto tp = tp_execution(state.execution);
    for (std::size_t index = 0; index < state.execution.peers.size(); ++index) {
        tp[index].mtp_kv = state.mtp_kv_peers[index];
        if (!tp[index].mtp_kv.valid() || !tp[index].io->mtp) {
            throw std::logic_error("parallel MTP bridge requires every rank's KV window");
        }
        tp[index].work->reset();
    }
    state.execution.work.reset();
    const auto restored = resume_hidden(state.execution, previous_hidden);
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                     std::span(tp).first(state.execution.peers.size()));
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);
    const auto work = rank_views(state.execution, [&](int rank) {
        return rank == 0 ? &state.execution.work : tp[rank - 1].work;
    });
    const auto io = rank_views(state.execution, [&](int rank) {
        return rank == 0 ? &state.execution.io : tp[rank - 1].io;
    });
    const auto& ec = *tp[0].execution;
    std::array<Tensor, kMaximumExecutionDevices> positions{}, rope{}, ar_hidden{}, logits{}, ar_positions{};
    for_each_rank(ec, [&](int rank) {
        const auto stream = ec.dev[rank]->stream;
        positions[rank] = io[rank]->mtp->target_positions.slice(0, 0, 1);
        ops::set_i32_scalar(positions[rank], position, stream);
        rope[rank] = work[rank]->alloc(DType::I32, {1, 3});
        CUDA_CHECK(cudaMemcpyAsync(rope[rank].data, rope_position.data(), rope_position.size_bytes(),
                                   cudaMemcpyHostToDevice, stream));
        ar_hidden[rank] = io[rank]->mtp->ar_hidden;
        logits[rank] = io[rank]->logits.slice(1, 0, 1);
        ar_positions[rank] = io[rank]->mtp->position.slice(0, 0, 1);
    });
    Tensor draft0 = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    const auto visible = static_cast<std::uint32_t>(position + 1);
    card.mtp_forward_batch(next_token, restored, positions, rope, {visible, visible}, ar_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, next_embedding);
    if (build_proposal) {
        for_each_rank(ec, [&](int rank) {
            ops::set_i32_scalar(ar_positions[rank], position + 1, ec.dev[rank]->stream);
        });
        for (int index = 1; index < static_cast<int>(state.mtp_proposal_extent); ++index) {
            Tensor previous = state.execution.io.mtp->draft_tokens.slice(0, index - 1, 1);
            Tensor next = state.execution.io.mtp->draft_tokens.slice(0, index, 1);
            const auto ar_visible = static_cast<std::uint32_t>(position + index + 1);
            card.mtp_forward_ar_step(previous, ar_hidden, ar_positions, {ar_visible, ar_visible},
                                     ar_hidden, logits, next);
            for_each_rank(ec, [&](int rank) {
                ops::increment_i32_scalar(ar_positions[rank], ec.dev[rank]->stream);
            });
        }
    }
    state.execution.device.synchronize();
    state.execution.work.reset();
    for (const auto& peer : state.execution.peers) {
        peer.device->synchronize();
        peer.work->reset();
    }
}

} // namespace

void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    if (build_proposal &&
        (state.mtp_proposal_extent == 0 ||
         state.mtp_proposal_extent >
             static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0]))) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }
    if (!state.execution.peers.empty()) {
        mtp_bridge_tp2(state, next_token, previous_hidden, position, rope_position, build_proposal,
                       next_embedding);
        return;
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.execution.rope_frequency, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::GqaExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::GqaExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, state.execution.io.mtp->ar_hidden, logits, next_draft);
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpGqaEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_6::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));
        auto tp = tp_execution(state.execution);
        for (const auto& peer : state.execution.peers) {
            if (!peer.io->mtp_decode || peer.mtp_host_ingress == nullptr) {
                throw std::logic_error("parallel MTP decode requires every peer frame and ingress");
            }
            const CurrentDevice restore;
            CUDA_CHECK(cudaSetDevice(peer.device->device));
            CUDA_CHECK(cudaMemcpyAsync(peer.io->mtp_decode->ingress.data, peer.mtp_host_ingress,
                                       sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                       peer.device->stream));
        }

        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, {}, state.execution.linear_attention,
                         state.execution.io, state.execution.prefill_hidden,
                         state.execution.prefill_chunk, 0, {}, &state.text_cache,
                         &state.mtp_cache, std::span(tp).first(state.execution.peers.size()));
        card.set_pipeline(pipeline_execution(state.execution));
        card.set_volta_long_attention(state.execution.volta_long_attention);
        card.set_token_constraints(state.execution.token_constraints);

        MtpRoundView v = slice_mtp_frame(frame, batch_size);

        if (state.execution.peers.empty()) {
            ops::speculative_prepare_verify_inputs(v.anchors, v.current_drafts, v.frontiers,
                                                   v.current_extents, v.verify_ids,
                                                   v.target_positions,
                                                   state.execution.device.stream);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 verify_view(v, state.execution.replay_records),
                                 envelopes.target_verify, state.greedy_target);

            ops::mtp_prepare_next_round(v.verify_ids, v.anchors, v.accepted, v.frontiers,
                                        v.budgets, v.licensed_counts, v.rope_deltas,
                                        v.alignment_ids, v.next_extents, v.ar_positions,
                                        v.ar_rope_positions, v.ar_valid_columns,
                                        static_cast<std::int32_t>(state.text_cache.max_context()),
                                        state.execution.device.stream);
            card.mtp_forward_decode_batch(v.alignment_ids, v.target_hidden, v.target_positions,
                                          v.target_rope, v.licensed_counts, v.mtp_rows,
                                          envelopes.batch, v.alignment_hidden);
            ops::speculative_select_accepted_hidden(v.alignment_hidden, v.accepted, v.ar_hidden,
                                                    state.execution.device.stream);

            Tensor draft0 = v.next_drafts.slice(1, 0, 1).view({batch_size});
            card.mtp_propose_batch(v.ar_hidden, v.proposal_logits, draft0);
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                Tensor previous =
                    v.next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
                Tensor next = v.next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1)
                                  .view({batch_size});
                Tensor position =
                    v.ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1,
                                                                                      batch_size});
                Tensor rope = v.ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                  .view({1, batch_size});
                Tensor valid = v.ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                                   .view({batch_size});
                Tensor previous_batch    = previous.view({1, batch_size});
                Tensor hidden_batch      = v.ar_hidden.view({TextConfig::hidden, 1, batch_size});
                card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                              v.mtp_rows, envelopes.ar[step], hidden_batch);
                card.mtp_propose_batch(v.ar_hidden, v.proposal_logits, next);
            }
        } else {
            const auto& ec = *tp[0].execution;
            auto views = rank_views(state.execution, [&](int rank) {
                return rank == 0 ? v : slice_mtp_frame(*tp[rank - 1].io->mtp_decode, batch_size);
            });
            const auto field = [&](Tensor MtpRoundView::* member) {
                return rank_views(state.execution, [&](int rank) { return views[rank].*member; });
            };
            for_each_rank(ec, [&](int rank) {
                auto& value = views[rank];
                ops::speculative_prepare_verify_inputs(
                    value.anchors, value.current_drafts, value.frontiers, value.current_extents,
                    value.verify_ids, value.target_positions, ec.dev[rank]->stream);
            });
            std::array<TargetVerifyFrameView, kMaximumExecutionDevices - 1> peer_frames{};
            for (int rank = 1; rank < ec.device_count; ++rank) {
                peer_frames[rank - 1] = verify_view(views[rank], tp[rank - 1].replay_records);
            }
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 verify_view(v, state.execution.replay_records),
                                 std::span(peer_frames).first(state.execution.peers.size()),
                                 envelopes.target_verify, state.greedy_target);
            for_each_rank(ec, [&](int rank) {
                auto& value = views[rank];
                ops::mtp_prepare_next_round(
                    value.verify_ids, value.anchors, value.accepted, value.frontiers, value.budgets,
                    value.licensed_counts, value.rope_deltas, value.alignment_ids,
                    value.next_extents, value.ar_positions, value.ar_rope_positions,
                    value.ar_valid_columns, static_cast<std::int32_t>(state.text_cache.max_context()),
                    ec.dev[rank]->stream);
            });
            card.mtp_forward_decode_batch(
                v.alignment_ids, field(&MtpRoundView::target_hidden),
                field(&MtpRoundView::target_positions), field(&MtpRoundView::target_rope),
                field(&MtpRoundView::licensed_counts), field(&MtpRoundView::mtp_rows),
                envelopes.batch, field(&MtpRoundView::alignment_hidden));
            for_each_rank(ec, [&](int rank) {
                auto& value = views[rank];
                ops::speculative_select_accepted_hidden(
                    value.alignment_hidden, value.accepted, value.ar_hidden, ec.dev[rank]->stream);
            });
            const auto proposal_logits = field(&MtpRoundView::proposal_logits);
            Tensor draft0 = v.next_drafts.slice(1, 0, 1).view({batch_size});
            card.mtp_propose_batch(field(&MtpRoundView::ar_hidden), proposal_logits, draft0);
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                Tensor previous = v.next_drafts.slice(1, step, 1).view({1, batch_size});
                Tensor next = v.next_drafts.slice(1, step + 1, 1).view({batch_size});
                const auto positions = rank_views(state.execution, [&](int rank) {
                    return views[rank].ar_positions.slice(1, step, 1).view({1, batch_size});
                });
                const auto rope = rank_views(state.execution, [&](int rank) {
                    return views[rank].ar_rope_positions.slice(1, step, 1).view({1, batch_size});
                });
                const auto valid = rank_views(state.execution, [&](int rank) {
                    return views[rank].ar_valid_columns.slice(1, step, 1).view({batch_size});
                });
                const auto hidden = rank_views(state.execution, [&](int rank) {
                    return views[rank].ar_hidden.view({TextConfig::hidden, 1, batch_size});
                });
                card.mtp_forward_decode_batch(previous, hidden, positions, rope, valid,
                                              field(&MtpRoundView::mtp_rows),
                                              envelopes.ar[step], hidden);
                card.mtp_propose_batch(field(&MtpRoundView::ar_hidden), proposal_logits, next);
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpGqaEnvelopes envelopes, DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpGqaEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
