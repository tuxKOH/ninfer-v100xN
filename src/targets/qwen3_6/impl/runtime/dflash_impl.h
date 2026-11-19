#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/dflash_conv.h"
#include "ninfer/ops/dflash_selector.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/swa.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.model.dflash.has_value()) {
        throw std::logic_error("DFlash schedule requires DFlash weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

template <class V>
DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const int>(Config::target_feature_layers),
        };
    }
}

template <class V, class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash context append is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != Config::feature_rows || features.ne[3] != 1 ||
            positions.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
            commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
            lanes.dtype != DType::I32 || lanes.ne[0] != batch || table_rows.dtype != DType::I32 ||
            table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window = batch == 1 && width > Config::local_capacity;
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset = replace_local_window ? width - Config::local_capacity : 0;
        const int local_width  = replace_local_window ? Config::local_capacity : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, Config::local_capacity,
                                state.execution.device.stream);
        }

        const auto context_roots =
            workspace_recipe::dflash_context<Config>(state.execution.work, columns);
        Tensor projected = context_roots.projected;
        ops::linear(features.view({Config::feature_rows, columns}),
                    state.execution.model.dflash->feature_projection, projected,
                    ops::LinearPolicy::A16Only, state.execution.work, state.execution.device.stream);
        Tensor context = context_roots.normalized;
        ops::rmsnorm(projected, state.execution.model.dflash->context_norm, Config::rms_epsilon,
                     false, context, state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope = state.execution.work.scope();
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            const bool local_layer  = layer < Config::local_layers;
            const int layer_width   = local_layer ? local_width : width;
            const int layer_columns = layer_width * batch;
            Tensor layer_context    = local_layer && replace_local_window
                                          ? context.slice(1, local_offset, local_width)
                                          : context;
            Tensor layer_positions  = local_layer && replace_local_window
                                          ? positions.slice(0, local_offset, local_width)
                                          : positions;
            auto layer_roots =
                workspace_recipe::dflash_context_layer<Config>(state.execution.work, layer_columns);
            Tensor key_raw =
                layer_roots.key_raw.view({Config::head_dim, Config::kv_heads, layer_columns});
            Tensor value =
                layer_roots.value.view({Config::head_dim, Config::kv_heads, layer_columns});
            Tensor key_flat   = key_raw.view({Config::kv_size, layer_columns});
            Tensor value_flat = value.view({Config::kv_size, layer_columns});
            ops::linear(layer_context, weight.key, key_flat,
                        state.execution.device.stream);
            ops::linear(layer_context, weight.value, value_flat,
                        state.execution.device.stream);
            Tensor key = layer_roots.key.view({Config::head_dim, Config::kv_heads, layer_columns});
            ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                         state.execution.device.stream);
            ops::rope(layer_positions.view({layer_columns}), Config::head_dim, Config::rope_theta,
                      key, state.execution.device.stream);
            Tensor key_batch = key.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor value_batch =
                value.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor position_batch = layer_positions.view({layer_width, batch});
            if (local_layer) {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                    dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                    state.execution.device.stream);
            } else {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                    dflash_state(state).full_batch_layer(0), state.execution.device.stream);
            }
        }
    }
}

template <class V>
void propose_batch_impl(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash proposal is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        Tensor anchors             = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers           = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor lanes               = frame.lanes.slice(0, 0, batch_size);
        Tensor full_rows           = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor ids                 = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions           = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts              = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream);
        Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                       state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            {
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace_recipe::dflash_attention<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, roots.hidden,
                             state.execution.device.stream);
                ops::linear(roots.hidden, weight.attention_conv_kernel_projection, roots.conv_dynamic,
                            state.execution.device.stream);
                ops::dflash2_conv(roots.hidden, roots.conv_dynamic, weight.attention_conv_base_kernel, 0,
                                  roots.conv_input, state.execution.device.stream);
                Tensor query_raw =
                    roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
                Tensor key_raw = roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
                Tensor value   = roots.value.view({Config::head_dim, Config::kv_heads, columns});
                Tensor query_flat = query_raw.view({Config::query_size, columns});
                Tensor key_flat   = key_raw.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                ops::linear(roots.conv_input, weight.query, query_flat, ops::LinearPolicy::A16Only,
                            state.execution.work, state.execution.device.stream);
                ops::linear(roots.conv_input, weight.key, key_flat, state.execution.device.stream);
                ops::linear(roots.conv_input, weight.value, value_flat, state.execution.device.stream);
                Tensor query = roots.query.view({Config::head_dim, Config::query_heads, columns});
                Tensor key   = roots.key.view({Config::head_dim, Config::kv_heads, columns});
                ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                             state.execution.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(positions.view({columns}), Config::head_dim, Config::rope_theta, query,
                          key, state.execution.device.stream);
                Tensor query_batch =
                    query.view({Config::head_dim, Config::query_heads, width, batch_size});
                Tensor key_batch =
                    key.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor value_batch =
                    value.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {Config::head_dim, Config::query_heads, width, batch_size});
                if (layer < Config::local_layers) {
                    ops::swa(query_batch, key_batch, value_batch, positions, valid_columns, lanes,
                             Config::attention_scale,
                             dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                             envelopes.local, state.execution.work, attention_batch,
                             state.execution.device.stream);
                } else {
                    ops::bidirectional_gqa_attention(
                        query_batch, key_batch, value_batch, frontiers, valid_columns, full_rows,
                        Config::attention_scale, dflash_state(state).full_batch_layer(0),
                        envelopes.full, state.execution.work, attention_batch,
                        state.execution.device.stream);
                }
                ops::linear(roots.attention.view({Config::query_size, columns}),
                            weight.attention_output, roots.attention_projected,
                            ops::LinearPolicy::A16Only, state.execution.work, state.execution.device.stream);
                ops::dflash2_conv(roots.attention_projected, roots.conv_dynamic,
                                  weight.attention_conv_base_kernel, 1, roots.attention_convolved,
                                  state.execution.device.stream);
                ops::residual_add(roots.attention_convolved, residual, state.execution.device.stream);
            }
            {
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                             roots.hidden, state.execution.device.stream);
                ops::linear(roots.hidden, weight.mlp_conv_kernel_projection, roots.dynamic,
                            state.execution.device.stream);
                ops::dflash2_conv(roots.hidden, roots.dynamic, weight.mlp_conv_base_kernel, 0,
                                  roots.input, state.execution.device.stream);
                Tensor gate_up = state.execution.work.alloc(
                    DType::BF16, {2 * Config::intermediate, columns});
                ops::linear(roots.input, weight.gate_up, gate_up, ops::LinearPolicy::A16Only,
                            state.execution.work, state.execution.device.stream);
                ops::silu_mul(gate_up.slice(0, 0, Config::intermediate),
                              gate_up.slice(0, Config::intermediate, Config::intermediate),
                              roots.intermediate, state.execution.device.stream);
                ops::linear(roots.intermediate, weight.down, roots.projected,
                            ops::LinearPolicy::A16Only, state.execution.work, state.execution.device.stream);
                ops::dflash2_conv(roots.projected, roots.dynamic, weight.mlp_conv_base_kernel, 1,
                                  roots.convolved, state.execution.device.stream);
                ops::residual_add(roots.convolved, residual, state.execution.device.stream);
            }
        }

        Tensor packed = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes =
            static_cast<std::size_t>(Config::hidden) * static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(Config::hidden) * width * element_bytes;
        const auto* source = static_cast<const std::byte*>(residual.data) +
                             static_cast<std::size_t>(Config::hidden) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream));
        Tensor proposal_hidden = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        ops::rmsnorm(packed, state.execution.model.dflash->final_norm, Config::rms_epsilon, false,
                     proposal_hidden, state.execution.device.stream);
        // DFlash2's selector codebooks are indexed by tokenizer IDs. In TP2, retain the exact
        // full-vocabulary candidate set while avoiding a full-vocabulary peer transfer: each rank
        // produces its local top-16 ordered keys and rank 0 merges the two 16-entry lists.
        const std::int32_t selector_columns = static_cast<std::int32_t>(k) * batch_size;
        const std::int32_t shard   = TextConfig::output_rows / 2;
        if (state.execution.peers.empty()) {
            Tensor logits = state.execution.work.alloc(
                DType::BF16, {TextConfig::output_rows, selector_columns});
            ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                        state.execution.device.stream);
            Tensor partial = state.execution.work.alloc(
                DType::I64, {16, (TextConfig::output_rows + ops::kDFlashSelectorTile - 1) /
                                     ops::kDFlashSelectorTile,
                             selector_columns});
            Tensor topk = state.execution.work.alloc(DType::I32, {16, selector_columns});
            Tensor selector_gate = state.execution.work.alloc(DType::BF16, {256, selector_columns});
            ops::linear(proposal_hidden, state.execution.model.dflash->selector_hidden_projection,
                        selector_gate, state.execution.device.stream);
            Tensor selector_logits = logits.view({TextConfig::output_rows,
                                                  static_cast<std::int32_t>(k), batch_size});
            Tensor selector_topk = topk.view({16, static_cast<std::int32_t>(k), batch_size});
            Tensor gate_batch = selector_gate.view({256, static_cast<std::int32_t>(k), batch_size});
            ops::dflash2_select(
                selector_logits, gate_batch, state.execution.model.dflash->selector_predecessor_codebook,
                state.execution.model.dflash->selector_successor_codebook, anchors, partial,
                selector_topk, drafts, state.execution.device.stream);
        } else {
            Tensor peer_hidden = state.execution.peers[0].work->alloc(DType::BF16,
                                                                     {Config::hidden, selector_columns});
            ops::broadcast_rank0(proposal_hidden, std::span<const Tensor>(&peer_hidden, 1),
                                 *state.execution.peers[0].execution,
                                 *state.execution.peers[0].events);
            Tensor part0 = state.execution.work.alloc(DType::BF16, {shard, selector_columns});
            Tensor part1 = state.execution.peers[0].work->alloc(DType::BF16, {shard, selector_columns});
            ops::linear_column_parallel(
                std::array<Tensor, 2>{proposal_hidden, peer_hidden},
                std::array<Weight, 2>{state.execution.model.output_head, state.execution.peers[0].model->output_head},
                std::array<Tensor, 2>{part0, part1}, *state.execution.peers[0].execution);
            const std::int32_t local_parts =
                (shard + ops::kDFlashSelectorTile - 1) / ops::kDFlashSelectorTile;
            Tensor partial0 = state.execution.work.alloc(DType::I64,
                                                         {16, local_parts, selector_columns});
            Tensor local_keys0 = state.execution.work.alloc(DType::I64, {16, selector_columns});
            Tensor partial1 = state.execution.peers[0].work->alloc(DType::I64,
                                                                 {16, local_parts, selector_columns});
            Tensor local_keys1 = state.execution.peers[0].work->alloc(DType::I64, {16, selector_columns});
            ops::dflash2_local_topk(part0, partial0, local_keys0, 0,
                                    state.execution.device.stream);
            {
                const CurrentDevice restore;
                CUDA_CHECK(cudaSetDevice(state.execution.peers[0].device->device));
                ops::dflash2_local_topk(part1, partial1, local_keys1, shard,
                                        state.execution.peers[0].device->stream);
            }
            // gather_columns_rank0 concatenates the first tensor dimension (the local top-k
            // entries), so the two [16,C] lists become one [32,C] list.  Keeping columns in the
            // second dimension preserves the selector's column-major tensor convention.
            Tensor gathered_keys = state.execution.work.alloc(DType::I64, {32, selector_columns});
            ops::gather_columns_rank0(
                gathered_keys, std::array<Tensor, 2>{local_keys0, local_keys1}, *state.execution.peers[0].execution,
                *state.execution.peers[0].events);
            Tensor global_keys = state.execution.work.alloc(DType::I64, {16, selector_columns});
            Tensor selector_gate = state.execution.work.alloc(DType::BF16, {256, selector_columns});
            ops::linear(proposal_hidden, state.execution.model.dflash->selector_hidden_projection,
                        selector_gate, state.execution.device.stream);
            // Projections flatten K*B columns. The lattice walk must retain each request's
            // independent K-step chain and anchor rather than treating the whole batch as one.
            Tensor gate_batch = selector_gate.view({256, static_cast<std::int32_t>(k), batch_size});
            ops::dflash2_select_sharded(
                gathered_keys, gate_batch,
                state.execution.model.dflash->selector_predecessor_codebook,
                state.execution.model.dflash->selector_successor_codebook, anchors, global_keys,
                drafts, state.execution.device.stream);
        }
        state.execution.work.reset();
    }
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DFlashEnvelopes envelopes,
                              ops::GqaExecutionEnvelope target_envelope) {
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        qwen3_6::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));
        auto tp_ranks = tp_execution(state.execution);
        const PeerExecution* tp = state.execution.peers.empty() ? nullptr : &tp_ranks[0];
        qwen3_6::DFlashDecodeState* peer_frame = nullptr;
        if (tp) {
            if (!tp->io->dflash_decode || state.execution.peers[0].dflash_host_ingress == nullptr) {
                throw std::logic_error("tensor-parallel DFlash decode requires a peer frame and ingress");
            }
            peer_frame = &*tp->io->dflash_decode;
            const CurrentDevice restore;
            CUDA_CHECK(cudaSetDevice(tp->device->device));
            CUDA_CHECK(cudaMemcpyAsync(peer_frame->ingress.data,
                                       state.execution.peers[0].dflash_host_ingress,
                                       sizeof(qwen3_6::DFlashDecodeIngress),
                                       cudaMemcpyHostToDevice, tp->device->stream));
        }

        Tensor anchors          = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers        = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts   = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents          = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns    = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows      = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes            = frame.lanes.slice(0, 0, batch_size);
        Tensor append_positions = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts    = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts           = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids       = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor target_tokens    = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits    = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden    = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden  = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens  = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts  = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted         = frame.accepted_drafts.slice(0, 0, batch_size);

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {Variant::DFlashConfig::feature_rows, width, batch_size});
        ops::prepare_ragged_prefix(dflash_state(state).pending_features, lanes, context_starts,
                                   frontiers, compact_features, append_positions, append_counts,
                                   state.execution.device.stream);
        append_context_impl<Variant>(state, compact_features, append_positions, append_counts,
                                     lanes, dflash_rows, envelopes.append);

        propose_batch_impl<Variant>(state, frame, batch_size, k, envelopes);
        ops::speculative_prepare_verify_ids(anchors, drafts, extents, verify_ids,
                                            state.execution.device.stream);
        if (peer_frame != nullptr) {
            Tensor peer_anchors = peer_frame->anchors.slice(0, 0, batch_size);
            Tensor peer_frontiers = peer_frame->execution_frontiers.slice(0, 0, batch_size);
            Tensor peer_valid = peer_frame->target_valid_columns.slice(0, 0, batch_size);
            Tensor peer_ids = peer_frame->proposal_ids.slice(1, 0, batch_size);
            Tensor peer_positions = peer_frame->proposal_positions.slice(1, 0, batch_size);
            Tensor peer_drafts = peer_frame->draft_tokens.slice(1, 0, batch_size);
            const CurrentDevice restore;
            CUDA_CHECK(cudaSetDevice(tp->device->device));
            ops::prepare_masked_block(peer_anchors, peer_frontiers, peer_valid,
                                      Variant::DFlashConfig::mask_token, peer_ids,
                                      peer_positions, tp->device->stream);
            ops::broadcast_rank0(drafts, std::span<const Tensor>(&peer_drafts, 1),
                                 *tp->execution, *tp->events);
            CUDA_CHECK(cudaSetDevice(tp->device->device));
            Tensor peer_verify_ids = peer_frame->verify_ids.slice(1, 0, batch_size);
            ops::speculative_prepare_verify_ids(
                peer_anchors, peer_drafts, peer_frame->proposal_extents.slice(0, 0, batch_size),
                peer_verify_ids, tp->device->stream);
        }
        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, {}, state.execution.linear_attention,
                         state.execution.io, state.execution.prefill_hidden,
                         state.execution.prefill_chunk, 0, {}, &state.text_cache, nullptr,
                         std::span(tp_ranks).first(state.execution.peers.size()));
        card.set_volta_long_attention(state.execution.volta_long_attention);
        DFlashFeatureSink sink =
            batch_feature_sink_impl<Variant>(state, lanes, valid_columns, width, batch_size);
        auto verify_view = [batch_size](qwen3_6::DFlashDecodeState& f,
                                        const GdnReplayRecords* replay,
                                        DFlashFeatureSink* feature_sink) {
            return TargetVerifyFrameView{
                .ids             = f.verify_ids.slice(1, 0, batch_size),
                .cache_positions = f.proposal_positions.slice(1, 0, batch_size),
                .rope_positions  = f.proposal_positions.slice(1, 0, batch_size),
                .valid_columns   = f.target_valid_columns.slice(0, 0, batch_size),
                .kv_table_rows   = f.text_kv_table_rows.slice(0, 0, batch_size),
                .lanes           = f.lanes.slice(0, 0, batch_size),
                .target_hidden   = f.target_hidden.slice(2, 0, batch_size),
                .target_logits   = f.target_logits.slice(2, 0, batch_size),
                .target_tokens   = f.target_argmax.slice(1, 0, batch_size),
                .drafts          = f.draft_tokens.slice(1, 0, batch_size),
                .current_extents = f.proposal_extents.slice(0, 0, batch_size),
                .frontiers       = f.execution_frontiers.slice(0, 0, batch_size),
                .anchors         = f.anchors.slice(0, 0, batch_size),
                .licensed_tokens = f.licensed_tokens.slice(1, 0, batch_size),
                .licensed_counts = f.licensed_counts.slice(0, 0, batch_size),
                .accepted_drafts = f.accepted_drafts.slice(0, 0, batch_size),
                .selected_hidden = f.target_continuation_hidden.slice(1, 0, batch_size),
                .replay_records  = replay,
                .sampling        = f.sampling,
                .feature_sink    = feature_sink,
            };
        };
        if (peer_frame != nullptr) {
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 verify_view(frame, state.execution.replay_records, &sink),
                                 std::array<TargetVerifyFrameView, 1>{
                                     verify_view(*peer_frame, tp->replay_records, nullptr)},
                                 target_envelope, state.greedy_target);
        } else {
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 verify_view(frame, state.execution.replay_records, &sink),
                                 target_envelope, state.greedy_target);
        }
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::DFlashDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    capture_graph(state, definition, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
