#include "targets/qwen3_6/impl/runtime/instance.h"
#include "core/peer_copy.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/nvtx.h"
#include "targets/qwen3_6/impl/runtime/visual_scatter.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/kv_retrieval.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/proposal_argmax.h"
#include "ninfer/ops/token_bitmask.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(const Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::GqaExecutionEnvelope*& slot,
                   const ops::GqaExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::GqaExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(
    DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
    const std::array<ops::RopeFrequencyOverride, kMaximumExecutionDevices>& rope_frequency,
    qwen3_6::PagedKVCacheView kv, LinearAttentionStatePool& state, qwen3_6::RoundState& io,
    Tensor& prefill_hidden, std::uint32_t prefill_chunk, std::uint32_t text_kv_base,
    qwen3_6::PagedKVCacheView mtp_kv, const qwen3_6::PagedKVCache* batch_text_kv,
    const qwen3_6::PagedKVCache* batch_mtp_kv, std::span<const PeerExecution> tp)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(mtp_kv), state_(state), io_(io),
      prefill_hidden_(prefill_hidden), prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base),
      rope_frequency_(rope_frequency), batch_text_kv_(batch_text_kv), batch_mtp_kv_(batch_mtp_kv),
      tp_(tp) {
    for (const PeerExecution& peer : tp_) {
        if (!peer.complete() || peer.execution->device_count != static_cast<int>(tp_.size() + 1) ||
            !peer.events->live()) {
            throw std::invalid_argument("tensor-parallel TextContext binding is incomplete");
        }
        if (mtp_enabled() != (peer.mtp_kv.valid() || peer.batch_mtp_kv != nullptr)) {
            throw std::invalid_argument("tensor-parallel MTP storage disagrees between ranks");
        }
    }
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, state_.slot_count() > 1 ? 1 : 0);
    bind();
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t current_slot,
                                         std::int32_t rewrite_checkpoint_slot) {
    if (current_slot < 0 || current_slot >= state_.slot_count() || rewrite_checkpoint_slot < 0 ||
        rewrite_checkpoint_slot >= state_.slot_count() || current_slot == rewrite_checkpoint_slot) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_current_slot_            = current_slot;
    linear_state_rewrite_checkpoint_slot_ = rewrite_checkpoint_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::bind() {
    using TargetBindings = LoadedModelData;
    using TargetMlp      = MlpWeights;
    const auto bind_mlp  = [](const TargetMlp& source) { return MlpW{&source}; };

    embed_      = &weights_.token_embedding;
    final_norm_ = &weights_.final_norm;
    lm_head_    = &weights_.output_head;
    for (std::size_t peer_index = 0; peer_index < tp_.size(); ++peer_index) {
        // Per-rank bindings point into ITS OWN model view, whose sharded extents are already
        // halved by the loader. Norms and the embedding table are replicated, so both ranks bind
        // structurally identical -- but physically distinct, per-device -- objects.
        const LoadedModelData& peer = *tp_[peer_index].weights;
        embed_peer_[peer_index]     = &peer.token_embedding;
        final_norm_peer_[peer_index] = &peer.final_norm;
        lm_head_peer_[peer_index]    = &peer.output_head;
        for (int layer = 0; layer < kCfg.n_layers; ++layer) {
            if (ModelConfig::is_full(layer)) {
                const std::size_t fidx = static_cast<std::size_t>(ModelConfig::full_idx(layer));
                FullLayerW& out        = full_peer_[peer_index][fidx];
                const auto& source     = peer.full_layers[fidx];
                out.input_norm         = &source.input_norm;
                out.projection         = &source.projection;
                out.o_proj             = &source.output;
                out.q_norm             = &source.query_norm;
                out.k_norm             = &source.key_norm;
                out.post_attn_norm     = &source.post_attention_norm;
                out.mlp                = bind_mlp(source.post_mixer);
            } else {
                const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
                GdnLayerW& out         = gdn_peer_[peer_index][gidx];
                const auto& source     = peer.gdn_layers[gidx];
                out.input_norm         = &source.input_norm;
                out.projection         = &source.projection;
                out.conv1d             = &source.convolution;
                out.gdn_norm           = &source.norm;
                out.out_proj           = &source.output;
                out.post_attn_norm     = &source.post_attention_norm;
                out.mlp                = bind_mlp(source.post_mixer);
            }
        }
    }
    if (weights_.optimized_proposal) {
        const auto& proposal = *weights_.optimized_proposal;
        set_proposal_head(&proposal.head, static_cast<const std::int32_t*>(proposal.token_ids.data),
                          proposal.head.n);
        for (std::size_t peer_index = 0; peer_index < tp_.size(); ++peer_index) {
            if (!tp_[peer_index].weights->optimized_proposal) {
                throw std::invalid_argument("tensor-parallel peer has no proposal head shard");
            }
            const auto& peer_proposal = *tp_[peer_index].weights->optimized_proposal;
            proposal_head_peer_[peer_index] = &peer_proposal.head;
            // `draft_head_token_ids` is REPLICATED, so this is the peer's own device copy of
            // the whole [131072] map, not a 65536-entry slice. Rank 1's copy
            // is DEAD STORAGE in this build: the remap runs where the argmax runs, which is rank
            // 0. It is bound anyway, and its presence checked below, because that check is what
            // proves the loader actually replicated the map rather than sharding it -- 512 KiB
            // against a ~400 MiB draft head, and the alternative is a loader special case whose
            // only effect would be to make the placement asymmetric.
            proposal_head_ids_peer_[peer_index] =
                static_cast<const std::int32_t*>(peer_proposal.token_ids.data);
        }
    }

    const auto bind_mtp_weights = [](const auto& source) {
        return MtpW{&source,
                    &source.input_projection,
                    &source.embedding_norm,
                    &source.hidden_norm,
                    &source.input_norm,
                    &source.query_norm,
                    &source.key_norm,
                    &source.output,
                    &source.post_attention_norm,
                    &source.final_norm};
    };
    if (mtp_enabled()) {
        if (!weights_.mtp) {
            throw std::invalid_argument("MTP state was enabled without materialized MTP weights");
        }
        mtp_ = bind_mtp_weights(*weights_.mtp);
        for (std::size_t peer_index = 0; peer_index < tp_.size(); ++peer_index) {
            if (!tp_[peer_index].weights->mtp) {
                throw std::invalid_argument("tensor-parallel peer has no MTP weight shard");
            }
            mtp_peer_[peer_index] = bind_mtp_weights(*tp_[peer_index].weights->mtp);
        }
    }

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            const auto& source =
                weights_.full_layers[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.post_mixer);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            const auto& source     = weights_.gdn_layers[gidx];
            out.input_norm         = &source.input_norm;
            out.projection         = &source.projection;
            out.conv1d             = &source.convolution;
            out.gdn_norm           = &source.norm;
            out.out_proj           = &source.output;
            out.post_attn_norm     = &source.post_attention_norm;
            out.mlp                = bind_mlp(source.post_mixer);
        }
    }
}

const MtpW& TextContext::mtp_weights() const {
    if (!mtp_enabled()) { throw std::runtime_error("MTP draft weights are not enabled"); }
    return mtp_;
}

const MtpW& TextContext::mtp_weights_for(int rank) const {
    if (!mtp_enabled()) { throw std::runtime_error("MTP draft weights are not enabled"); }
    if (rank == 0) { return mtp_; }
    if (mtp_peer_[rank - 1].payload == nullptr) {
        throw std::logic_error("tensor-parallel peer MTP weights are unbound");
    }
    return mtp_peer_[rank - 1];
}

const GdnReplayRecords* TextContext::replay_records_for(int rank) const {
    if (rank == 0) { return replay_records_; }
    if (tp_.empty()) { throw std::logic_error("TextContext has no tensor-parallel context"); }
    // The two ranks must agree: a peer with no record storage while rank 0 records would fold a
    // stale half of the GDN state on device 1 and diverge silently from the next round on.
    if ((replay_records_ == nullptr) != (tp_[rank - 1].replay_records == nullptr)) {
        throw std::logic_error("tensor-parallel replay-record bindings disagree between ranks");
    }
    return tp_[rank - 1].replay_records;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({kCfg.hidden, T});

    auto roots = workspace_recipe::mtp_stem<TextConfig>(work_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 || input_embeddings->ne[0] != kCfg.hidden ||
            input_embeddings->numel() != static_cast<std::int64_t>(kCfg.hidden) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({kCfg.hidden, T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, *mtp_.pre_fc_norm_embedding, kCfg.rms_eps, true, e, s);
    ops::rmsnorm(flat_hidden, *mtp_.pre_fc_norm_hidden, kCfg.rms_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    ops::linear(fc_in, *mtp_.fc, x, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, *mtp_.input_norm, kCfg.rms_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace_recipe::mtp_attention_projection<TextConfig>(work_, T);
    Tensor q              = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k              = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor gate           = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor v              = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat         = q.view({kCfg.q_size, T});
    Tensor gate_flat      = gate.view({kCfg.q_size, T});
    Tensor k_flat         = k.view({kCfg.kv_size, T});
    Tensor v_flat         = v.view({kCfg.kv_size, T});
    Variant::mtp_attention_projection(ah, mtp_.payload->attention, q_flat, gate_flat, k_flat,
                                      v_flat, work_, s);

    const auto results = workspace_recipe::mtp_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, rope_frequency_[0], s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
                           *active_backend_kv_table_rows_, kAttnScale,
                           batch_mtp_kv_->batch_layer_view(0), attention_envelope(envelope), work_, a_batch, s);
    } else {
        ops::gqa_attention(qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row, kAttnScale,
                           batch_mtp_kv_->batch_layer_view(0), attention_envelope(envelope), work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace_recipe::mtp_post_attention<TextConfig>(work_, T);
    Tensor o        = post.output;
    ops::linear(a.view({kCfg.q_size, T}), *mtp_.o_proj, o, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x, work_, s);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({kCfg.hidden, T});
    ops::rmsnorm(x, *mtp_.norm, kCfg.rms_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::GqaExecutionEnvelope envelope, bool final_chunk,
                                    Tensor* final_hidden, Tensor* logits, Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {kCfg.hidden, 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ah_last = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Tensor v_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Variant::mtp_kv_projection(ah, mtp_.payload->attention, k_flat, v_flat, work_, s);
        Tensor k  = k_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor v  = v_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
        ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
        ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, kn, rope_frequency_[0], s);
        ops::gqa_kv_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(kCfg.hidden) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat    = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Tensor gate_flat = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Variant::mtp_q_gate_projection(ah_last, mtp_.payload->attention, q_flat, gate_flat, work_,
                                       s);
        Tensor q    = q_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor gate = gate_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor qn   = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        ops::rope(last_rope_position, kCfg.rotary_dim, kCfg.rope_theta, qn, rope_frequency_[0],
                  s);

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::gqa_attention_cached(qn, last_position, kAttnScale, mtp_kv_.layer_view(0), attention_envelope(envelope),
                                  work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::linear(a.view({kCfg.q_size, 1}), *mtp_.o_proj, o, s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::rmsnorm(x_last, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x_last, work_, s);
        }
        ops::rmsnorm(x_last, *mtp_.norm, kCfg.rms_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    const int T = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, kCfg.vocab, T, "proposal logits");
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        ops::linear(hidden, *proposal_head_, proposal_logits, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens, proposal_head_n_, ctx_.stream);
        ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                      ctx_.stream);
    } else {
        Tensor output_logits = matrix_window(logits, T);
        ops::linear(hidden, *lm_head_, output_logits, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens, kCfg.token_domain, ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, ops::GqaExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position, ops::GqaExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_slots,
                                        ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                        Tensor& logits) {
    if (tp2()) {
        ordinary_decode_batch_tp2(ids, cache_positions, rope_positions, kv_table_rows,
                                  linear_state_slots, envelope, hidden, logits);
        return;
    }
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, batch});
        ops::embedding(ids, *embed_, x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, stream);
        ops::linear(hidden, *lm_head_, logits, stream);
    }
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_slots,
                                           ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                           Tensor& logits, Tensor& target_tokens, Tap& tap) {
    if (tp2()) {
        throw std::logic_error("use tensor-parallel target_verify_batch overload");
    }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {kCfg.hidden, columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({kCfg.hidden, columns});
        Tensor flat_logits = logits.view({kCfg.vocab, columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, flat_hidden, stream);
        ops::linear(flat_hidden, *lm_head_, flat_logits, stream);
        constrain_logits(flat_logits, 0, kCfg.token_domain, stream);
        ops::argmax(flat_logits, flat_tokens, kCfg.token_domain, stream);
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                      Tensor& logits, Tensor& target_tokens, bool greedy_target) {
    (void)greedy_target;
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_slots, envelope, hidden, logits, target_tokens, tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                      Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink, bool greedy_target) {
    (void)greedy_target;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_slots, envelope, hidden, logits, target_tokens, sink);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
    fidx -= weights_.layer_begin / 4;
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_gqa_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace_recipe::text_attention_projection<TextConfig>(work_, T);
    Tensor h              = projection.hidden;
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    Tensor q         = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor gate      = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor v         = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({kCfg.q_size, T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    Variant::attention_projection(h, *w.projection, q_flat, gate_flat, k_flat, v_flat, ph, work_,
                                  s);

    const auto results = workspace_recipe::text_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    if (batch_text_kv_->retrieval_keys.data && ph == Phase::Prefill && capture_retrieval_query_) {
        ops::kv_retrieval_query_summary(qn, cache_positions.view({T}), batch_text_kv_->retrieval_span,
            batch_text_kv_->retrieval_query.slice(2, fidx, 1),
            batch_text_kv_->retrieval_fallback.slice(2, fidx, 1),
            batch_text_kv_->retrieval_counts.slice(0, fidx, 1), s);
    }
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, rope_frequency_[0], s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
                           kAttnScale, batch_text_kv_->batch_layer_view(fidx),
                           attention_envelope(*active_gqa_envelope_), work_, a_batch, s);
    } else {
        ops::gqa_attention(qn, kn, v, cache_positions, Tensor{}, kv_table_rows, kAttnScale,
                           batch_text_kv_->batch_layer_view(fidx), attention_envelope(*active_gqa_envelope_), work_, a,
                           s);
    }
    if (batch_text_kv_->retrieval_keys.data) {
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::kv_retrieval_key_summary(batch_text_kv_->batch_layer_view(fidx), cache_positions.view({T}),
            valid, batch_text_kv_->retrieval_keys.slice(3, fidx, 1), rope_frequency_[0], s);
    }
    ops::sigmoid_mul(gate, a, s);

    Variant::attention_output_projection(a.view({kCfg.q_size, T}), *w.o_proj, x, ph, work_, s);
}

void TextContext::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    gidx -= weights_.layer_begin - weights_.layer_begin / 4;
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto control = workspace_recipe::gdn_control<TextConfig>(work_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    Variant::gdn_norm_control_projection(x, *w.input_norm, kCfg.rms_eps, *w.projection, h, g, beta,
                                         work_, s);

    const auto projection = workspace_recipe::gdn_projection<TextConfig>(work_, T);
    Tensor z              = projection.output_gate.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor qc             = projection.query;
    Tensor kc             = projection.key;
    Tensor vc             = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        Tensor projection_input = h.view({kCfg.hidden, width, active_sequence_batch_});
        Tensor query_output     = qc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor key_output       = kc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor value_output     = vc.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor gate_output      = z.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor& conv_states     = state_.conv.at(static_cast<std::size_t>(gidx));
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            Variant::gdn_input_projection_record(projection_input, *w.projection, *w.conv1d,
                                                 conv_states, valid, *active_linear_state_slots_,
                                                 records.conv, query_output, key_output,
                                                 value_output, gate_output, ph, work_, s);
        } else {
            Variant::gdn_input_projection_snapshot(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_slots_, *active_linear_state_slots_, query_output, key_output,
                value_output, gate_output, ph, work_, s);
        }
    } else {
        const auto conv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, T);
        Tensor qkv      = conv.projected;
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s);
        Tensor qkv_c = conv.convolved;
        Tensor conv_state =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
        ops::causal_conv1d_silu(qkv, *w.conv1d, conv_state, conv_state, qkv_c, s);
        ops::extract_bf16_columns(qkv_c, 0, qc, s);
        ops::extract_bf16_columns(qkv_c, kCfg.key_dim, kc, s);
        ops::extract_bf16_columns(qkv_c, 2 * kCfg.key_dim, vc, s);
    }

    Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});

    Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o  = workspace_recipe::gdn_recurrent_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor& recurrent_states = state_.recurrent.at(static_cast<std::size_t>(gidx));
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch =
            q_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor k_batch =
            k_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor v_batch = vv.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor g_batch = g.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor beta_batch = beta.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor out_batch =
            o.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                               kGdnScale, recurrent_states, valid,
                                               *active_linear_state_slots_, records.key,
                                               records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_snapshot(q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale,
                                          /*normalize_qk=*/true, recurrent_states, valid,
                                          *active_linear_state_slots_, *active_linear_state_slots_,
                                          out_batch, s);
        }
    } else {
        Tensor recurrent_state =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
        ops::gated_delta_net(q_recurrent, k_recurrent, vv, g, beta, kGdnScale,
                             /*normalize_qk=*/true, work_, recurrent_state, o, s);
    }

    Tensor on = workspace_recipe::gdn_normalized_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    ops::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, on, s);

    Variant::gdn_output_projection(on.view({kCfg.value_dim, T}), *w.out_proj, x, ph, work_, s);
}

void TextContext::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    Tensor h       = workspace_recipe::post_mixer_hidden<TextConfig>(work_, T);
    ops::rmsnorm(x, *post_norm, kCfg.rms_eps, true, h, s);

    Variant::post_mixer(h, *m.payload, x, ph, work_, s);
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    if (pipeline_) { run_pipeline_stage(x, ph); }
    run_owned_layers(x, ph, tap);
}

template <class Tap>
void TextContext::run_owned_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    for (int layer = weights_.layer_begin; layer < weights_.layer_end; ++layer) {
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                attn_mix(full, x, fidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(full.post_attn_norm, full.mlp, x, ph);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            nvtx::ScopedRange layer_range(prefill ? nvtx::Name::PrefillLayerGdn
                                                  : nvtx::Name::VerifyLayerGdn,
                                          nvtx::Category::Gdn, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn,
                    static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                gdn_mix(gdn, x, gidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, ph);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

void TextContext::pipeline_prefill_layers(Tensor& x, const Tensor& positions,
                                          ops::GqaExecutionEnvelope envelope, std::int32_t begin) {
    const auto& peer = *pipeline_;
    auto frequency = rope_frequency_;
    frequency[0] = rope_frequency_[1];
    Tensor peer_positions = peer.work->alloc(DType::I32, {positions.ne[0]});
    ops::fill_i32_positions(peer_positions, begin, peer.device->stream);
    Tensor peer_rope = peer_positions;
    if (rope_delta_ != 0) {
        peer_rope = peer.work->alloc(DType::I32, {positions.ne[0]});
        ops::set_i32_scalar(peer.io->rope_delta, rope_delta_, peer.device->stream);
        ops::offset_i32_positions(peer_positions, peer.io->rope_delta, peer_rope, peer.device->stream);
    }
    TextContext stage(*peer.device, *peer.weights, *peer.work, frequency, {}, *peer.state,
                       *peer.io, *peer.prefill_hidden, prefill_chunk_, text_kv_base_, {}, peer.batch_kv);
    stage.set_linear_state_slots(linear_state_current_slot_, linear_state_rewrite_checkpoint_slot_);
    stage.set_retrieval_query_capture(capture_retrieval_query_);
    stage.active_cache_positions_ = &peer_positions;
    stage.active_rope_positions_ = &peer_rope;
    stage.active_gqa_envelope_ = &envelope;
    stage.run_layers(x, Phase::Prefill);
}

void TextContext::run_pipeline_stage(Tensor& x, Phase phase) {
    if (phase != Phase::Verify) { throw std::logic_error("PP prefill requires a peer-local stem"); }
    const auto& peer = *pipeline_;
    const CurrentDevice restore;
    CUDA_CHECK(cudaSetDevice(peer.device->device));
    peer.work->reset();
    Tensor peer_x = peer.work->alloc(x.dtype, {x.ne[0], x.ne[1]});
    peer_copy_begin(ctx_, *peer.device, peer.events->inputs_ready(0));
    peer_copy(x, peer_x, ctx_, *peer.device);
    const auto mirror = [&](const Tensor* source) {
        if (source == nullptr || source->data == nullptr) { return Tensor{}; }
        Tensor result = peer.work->alloc(source->dtype,
                                         {source->ne[0], source->ne[1], source->ne[2], source->ne[3]});
        peer_copy(*source, result, ctx_, *peer.device);
        return result;
    };
    Tensor positions = mirror(active_cache_positions_);
    Tensor rope = mirror(active_rope_positions_);
    Tensor rows = mirror(active_kv_table_rows_);
    Tensor slots = mirror(active_linear_state_slots_);
    Tensor valid = mirror(active_valid_columns_);
    peer_copy_end(ctx_, *peer.device, peer.events->pull_done(1));
    CUDA_CHECK(cudaSetDevice(peer.device->device));
    auto frequency = rope_frequency_;
    frequency[0] = rope_frequency_[1];
    TextContext stage(*peer.device, *peer.weights, *peer.work, frequency, {}, *peer.state,
                       *peer.io, *peer.prefill_hidden, prefill_chunk_, text_kv_base_, {}, peer.batch_kv);
    stage.set_linear_state_slots(linear_state_current_slot_, linear_state_rewrite_checkpoint_slot_);
    stage.set_gdn_state_action(gdn_state_action_, gdn_state_action_ == GdnStateAction::RecordForReplay
                                                   ? peer.replay_records : nullptr);
    stage.active_cache_positions_ = positions.data != nullptr ? &positions : nullptr;
    stage.active_rope_positions_ = rope.data != nullptr ? &rope : nullptr;
    stage.active_kv_table_rows_ = rows.data != nullptr ? &rows : nullptr;
    stage.active_linear_state_slots_ = slots.data != nullptr ? &slots : nullptr;
    stage.active_valid_columns_ = valid.data != nullptr ? &valid : nullptr;
    stage.active_gqa_envelope_ = active_gqa_envelope_;
    stage.active_sequence_batch_ = active_sequence_batch_;
    stage.active_sequence_width_ = active_sequence_width_;
    stage.run_layers(peer_x, phase);
    peer_copy_begin(*peer.device, ctx_, peer.events->inputs_ready(1));
    peer_copy(peer_x, x, *peer.device, ctx_);
    peer_copy_end(*peer.device, ctx_, peer.events->pull_done(0));
    CUDA_CHECK(cudaSetDevice(ctx_.device));
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;
    if (pipeline_) {
        if (pipeline_upload_ == nullptr) { throw std::logic_error("PP prefill requires pinned ingress"); }
        std::copy(ids.begin(), ids.end(), pipeline_upload_);
    }

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64         = static_cast<std::int64_t>(base);
    const std::int64_t checkpoint_abs = prefill_rewrite_checkpoint_frontier_;
    const bool has_rewrite_checkpoint =
        checkpoint_abs > base64 && checkpoint_abs <= base64 + static_cast<std::int64_t>(T);
    const int checkpoint_rel =
        has_rewrite_checkpoint ? static_cast<int>(checkpoint_abs - base64) : -1;
    const std::int32_t rewrite_checkpoint_slot = linear_state_rewrite_checkpoint_slot_;

    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    int tail_column = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (checkpoint_rel > 0 && t0 < checkpoint_rel && t0 + len > checkpoint_rel) {
            len = checkpoint_rel - t0;
        }
        work_.reset();
        if (pipeline_) { pipeline_->work->reset(); }

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace_recipe::text_prefill_roots<TextConfig>(
                work_, len, rope_axes, static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            if (!pipeline_) { copy_i32(ids.data() + t0, ids_device, s); }

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::GqaExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_gqa_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            if (!pipeline_) { ops::embedding(ids_device, *embed_, x, s); }
            if (pipeline_) {
                const auto& peer = *pipeline_;
                const CurrentDevice restore;
                CUDA_CHECK(cudaSetDevice(peer.device->device));
                Tensor peer_ids = peer.work->alloc(DType::I32, {len});
                copy_i32(pipeline_upload_ + t0, peer_ids, peer.device->stream);
                Tensor peer_x = peer.work->alloc(DType::BF16, {kCfg.hidden, len});
                ops::embedding(peer_ids, peer.weights->token_embedding, peer_x, peer.device->stream);
                // The peer stem must not wait for the previous primary chunk.
                pipeline_prefill_layers(peer_x, positions, chunk_envelope, base_i + t0);
                peer_copy_begin(*peer.device, ctx_, peer.events->inputs_ready(1));
                peer_copy(peer_x, x, *peer.device, ctx_);
                peer_copy_end(*peer.device, ctx_, peer.events->pull_done(0));
                CUDA_CHECK(cudaSetDevice(ctx_.device));
            }
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_owned_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {kCfg.hidden, len});
            ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                ops::linear(last_xf, *lm_head_, logits, s);
                constrain_logits(logits, 0, kCfg.token_domain, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_, io_.pos,
                                ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token, kCfg.token_domain, s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_6::MtpAlignmentWindow mtp_window = qwen3_6::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                std::vector<int> mtp_ids_host(static_cast<std::size_t>(len));
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                for (int j = 0; j < prompt_columns; ++j) {
                    mtp_ids_host[static_cast<std::size_t>(j)] =
                        alignment_ids[static_cast<std::size_t>(mtp_window.shifted_embedding_begin) +
                                      static_cast<std::size_t>(j)];
                }
                if (mtp_window.final_column_uses_generated_token) {
                    int next_token = 0;
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    CUDA_CHECK(cudaMemcpy(&next_token, io_.token.data, sizeof(next_token),
                                          cudaMemcpyDeviceToHost));
                    mtp_ids_host[static_cast<std::size_t>(len - 1)] = next_token;
                }

                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                if (pipeline_) {
                    std::copy(mtp_ids_host.begin(), mtp_ids_host.end(), pipeline_upload_ + T + t0);
                    copy_i32(pipeline_upload_ + T + t0, mtp_ids, s);
                } else {
                    copy_i32(mtp_ids_host.data(), mtp_ids, s);
                }
                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_6::MtpVisualOverlap overlap = qwen3_6::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace_recipe::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_6::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token     = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token     = io_.mtp->draft_tokens.slice(0, i, 1);
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::GqaExecutionEnvelope ar_envelope{ar_visible, ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, io_.mtp->ar_hidden, logits, next_token);
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (checkpoint_rel > 0 && t0 + len == checkpoint_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {kCfg.hidden, 1}, "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, checkpoint_rel > 0 && t0 + len == checkpoint_rel);
        }

        if (checkpoint_rel > 0 && t0 + len == checkpoint_rel) {
            state_.copy_slot(linear_state_current_slot_, rewrite_checkpoint_slot, s);
            if (pipeline_) {
                const CurrentDevice restore;
                CUDA_CHECK(cudaSetDevice(pipeline_->device->device));
                pipeline_->state->copy_slot(linear_state_current_slot_, rewrite_checkpoint_slot,
                                            pipeline_->device->stream);
            }
        }

        tail_column = len - 1;
        t0 += len;
        if (!pipeline_) { break; }
    }

    prefill_rewrite_checkpoint_frontier_ = -1;

    ctx_.synchronize();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .tail_column      = static_cast<std::uint32_t>(tail_column)};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    if (tp2()) {
        return prefill_impl_tp2(full_ids.subspan(begin, nominal_length), text_prefill,
                                finalize_at_end);
    }
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    if (tp2()) {
        const TextPrefill text_prefill{full_ids, begin};
        return prefill_impl_tp2(full_ids.subspan(begin, nominal_length), text_prefill,
                                finalize_at_end, &sink);
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_6::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    if (tp2()) {
        return prefill_impl_tp2(tokens.subspan(begin, nominal_length), TextPrefill{tokens, begin},
                                finalize_at_end, nullptr, &multimodal);
    }
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}


// =================================================================================================
// tp == 2 forward
// =================================================================================================
//
// SHAPE OF THE SCHEDULE. Every layer runs the same pattern:
//
//   1. a replicated elementwise stage (RMSNorm, RoPE, gating multiply) issued once per rank on
//      that rank's own stream over that rank's own copy of the data;
//   2. a COLUMN-parallel projection, one call driving both ranks, no communication;
//   3. head-local mixing -- attention over this device's 12 of 24 query heads against its own 2 of
//      4 KV heads, GDN over its own 24 of 48 value heads -- again per rank, no communication;
//   4. a ROW-parallel output projection whose single `allreduce_sum` is the ONLY collective in the
//      mixer, and which folds the residual in exactly once (rank 0, pre-reduce);
//   5. the MLP tail, whose `down` projection carries the layer's second and last collective.
//
// Two all-reduces per layer, 128 per token for the 64-layer model.
//
// STREAMS AND EVENTS. There is no host synchronization anywhere inside the layer loop. Rank r's
// work is enqueued on `ec.dev[r]->stream`; the collectives' own four-event choreography
// (`inputs_ready` / `pull_done`, include/ninfer/ops/allreduce.h) is what orders the two streams
// against each other, both within a call and across calls. The single `PeerEvents` instance lives
// in the Program, is created once, and is reused by every collective of every layer. Kernel
// launches go to the CURRENT device, so every per-rank issue runs inside `for_each_rank`, which
// sets and restores it.
//
// WHY THE TWO HALVES STAY IN LOCKSTEP. The residual is the only tensor both ranks must agree on
// bit-for-bit, and they do: `allreduce_sum` leaves rank 0 holding `p0 + p1` and rank 1 holding
// `p1 + p0`, and IEEE addition is commutative, so both store the identical BF16. Everything
// derived from the residual -- the next layer's norm, the GDN convolution and recurrent state, the
// KV pages -- is therefore identical on both devices with no further agreement protocol. Prefill
// chunk boundaries are driven by the token count alone, so both ranks also see the same chunks.

const ExecutionContext& TextContext::ec() const {
    if (tp_.empty()) { throw std::logic_error("TextContext has no tensor-parallel context"); }
    return *tp_[0].execution;
}

std::array<WorkspaceArena*, kMaximumExecutionDevices> TextContext::workspaces() const {
    return rank_map([&](int rank) { return rank == 0 ? &work_ : tp_[rank - 1].work; });
}

std::array<std::optional<WorkspaceArena::Scope>, kMaximumExecutionDevices>
TextContext::workspace_scopes() const {
    std::array<std::optional<WorkspaceArena::Scope>, kMaximumExecutionDevices> out{};
    const auto work = workspaces();
    for (int rank = 0; rank < ec().device_count; ++rank) { out[rank].emplace(work[rank]->scope()); }
    return out;
}

void TextContext::synchronize_all() const {
    ctx_.synchronize();
    for (const PeerExecution& peer : tp_) { peer.device->synchronize(); }
}

const Tensor& TextContext::rank_cache_positions(int rank) const {
    if (rank == 0) {
        return active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    }
    if (peer_cache_positions_[rank - 1] == nullptr) {
        throw std::logic_error("tensor-parallel peer cache positions are unbound");
    }
    return *peer_cache_positions_[rank - 1];
}

const Tensor& TextContext::rank_rope_positions(int rank) const {
    if (rank == 0) {
        return active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    }
    if (peer_rope_positions_[rank - 1] == nullptr) {
        throw std::logic_error("tensor-parallel peer RoPE positions are unbound");
    }
    return *peer_rope_positions_[rank - 1];
}

const Tensor& TextContext::rank_kv_table_rows(int rank) const {
    if (rank == 0) {
        return active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    }
    if (peer_kv_table_rows_[rank - 1] == nullptr) {
        throw std::logic_error("tensor-parallel peer KV table rows are unbound");
    }
    return *peer_kv_table_rows_[rank - 1];
}

Tensor TextContext::rank_valid_columns(int rank) const {
    // Unlike its siblings, an ABSENT binding is legal here: the prefill and ordinary-decode paths
    // never bind valid columns, and every consumer reads an empty Tensor as "every column counts".
    // What is not legal is the two ranks DISAGREEING -- a peer binding present while rank 0's is
    // absent (or the reverse) would have the two devices mask different columns and diverge
    // silently, which is exactly the trap the sibling accessors' throws exist to prevent. The
    // paths that do bind it (speculative verify, MTP) are guarded off at tp2 today; this keeps the
    // invariant checked rather than assumed for whoever lifts that guard.
    if (rank != 0 && (active_valid_columns_ == nullptr) != (peer_valid_columns_[rank - 1] == nullptr)) {
        throw std::logic_error("tensor-parallel valid-column bindings disagree between ranks");
    }
    if (rank == 0) {
        return active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
    }
    return peer_valid_columns_[rank - 1] != nullptr ? *peer_valid_columns_[rank - 1] : Tensor{};
}

const Tensor& TextContext::rank_backend_kv_table_rows(int rank) const {
    if (rank == 0) {
        if (active_backend_kv_table_rows_ == nullptr) {
            throw std::logic_error("MTP backend KV table rows are unbound");
        }
        return *active_backend_kv_table_rows_;
    }
    if (peer_backend_kv_table_rows_[rank - 1] == nullptr) {
        throw std::logic_error("tensor-parallel peer backend KV table rows are unbound");
    }
    return *peer_backend_kv_table_rows_[rank - 1];
}

const Tensor& TextContext::rank_linear_state_slots(int rank) const {
    if (rank == 0) {
        if (active_linear_state_slots_ == nullptr) {
            throw std::logic_error("Linear Attention state slots are unbound");
        }
        return *active_linear_state_slots_;
    }
    if (peer_linear_state_slots_[rank - 1] == nullptr) {
        throw std::logic_error("tensor-parallel peer state slots are unbound");
    }
    return *peer_linear_state_slots_[rank - 1];
}

void TextContext::attn_mix_tp2(const std::array<const FullLayerW*, kMaximumExecutionDevices>& w, std::array<Tensor, kMaximumExecutionDevices>& x,
                               int fidx, Phase ph, const std::array<Tensor, kMaximumExecutionDevices>& staging,
                              const ops::ReplicatedRmsNorm* norm,
                              const std::array<Tensor, kMaximumExecutionDevices>* normalized_input) {
    const ExecutionContext& execution = ec();
    const int T                       = x[0].ne[1];
    if (active_gqa_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }
    const std::array<WorkspaceArena*, kMaximumExecutionDevices> ws  = workspaces();

    std::array<Tensor, kMaximumExecutionDevices> h;
    std::array<Tensor, kMaximumExecutionDevices> q;
    std::array<Tensor, kMaximumExecutionDevices> gate;
    std::array<Tensor, kMaximumExecutionDevices> k;
    std::array<Tensor, kMaximumExecutionDevices> v;
    std::array<Tensor, kMaximumExecutionDevices> q_flat;
    std::array<Tensor, kMaximumExecutionDevices> gate_flat;
    std::array<Tensor, kMaximumExecutionDevices> k_flat;
    std::array<Tensor, kMaximumExecutionDevices> v_flat;
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        const auto projection =
            workspace_recipe::text_attention_projection<TextConfig>(*ws[r], T,
                                                                    ec().device_count);
        h[r]         = normalized_input ? (*normalized_input)[r] : projection.hidden;
        q[r]         = projection.query.view({kCfg.head_dim, (kCfg.n_q / ec().device_count), T});
        gate[r]      = projection.gate.view({kCfg.head_dim, (kCfg.n_q / ec().device_count), T});
        k[r]         = projection.key.view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), T});
        v[r]         = projection.value.view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), T});
        q_flat[r]    = q[r].view({(kCfg.q_size / ec().device_count), T});
        gate_flat[r] = gate[r].view({(kCfg.q_size / ec().device_count), T});
        k_flat[r]    = k[r].view({(kCfg.kv_size / ec().device_count), T});
        v_flat[r]    = v[r].view({(kCfg.kv_size / ec().device_count), T});
    }
    if (normalized_input == nullptr) for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::rmsnorm(x[r], *w[r]->input_norm, kCfg.rms_eps, true, h[r], stream_for(rank));
    });
    Variant::attention_projection(active(h), active(rank_map([&](int rank) { return w[rank]->projection; })), active(q_flat), active(gate_flat), active(k_flat),
                                  active(v_flat), ph, active(ws), execution);

    std::array<Tensor, kMaximumExecutionDevices> qn;
    std::array<Tensor, kMaximumExecutionDevices> kn;
    std::array<Tensor, kMaximumExecutionDevices> a;
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        const auto results =
            workspace_recipe::text_attention_results<TextConfig>(*ws[r], T, ec().device_count);
        qn[r] = results.normalized_query.view({kCfg.head_dim, (kCfg.n_q / ec().device_count), T});
        kn[r] = results.normalized_key.view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), T});
        a[r]  = results.attention.view({kCfg.head_dim, (kCfg.n_q / ec().device_count), T});
    }
    for_each_rank(execution, [&](int rank) {
        const auto r        = static_cast<std::size_t>(rank);
        cudaStream_t s      = stream_for(rank);
        const Tensor& cache = rank_cache_positions(rank);
        const Tensor& rope  = rank_rope_positions(rank);
        ops::rmsnorm(q[r], *w[r]->q_norm, kCfg.rms_eps, true, qn[r], s);
        ops::rmsnorm(k[r], *w[r]->k_norm, kCfg.rms_eps, true, kn[r], s);
        const qwen3_6::PagedKVCache& pages = rank == 0 ? *batch_text_kv_ : *tp_[rank - 1].batch_kv;
        if (pages.retrieval_keys.data && ph == Phase::Prefill && capture_retrieval_query_) {
            ops::kv_retrieval_query_summary(qn[r], cache.view({T}), pages.retrieval_span,
                pages.retrieval_query.slice(2, fidx, 1), pages.retrieval_fallback.slice(2, fidx, 1),
                pages.retrieval_counts.slice(0, fidx, 1), s);
        }
        Tensor rope_for_op = active_sequence_batch_ != 0 ? rope.view({T}) : rope;
        ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn[r], kn[r],
                  rope_frequency_[r], s);

        if (active_sequence_batch_ != 0) {
            const std::int32_t width = active_sequence_width_;
            if (width <= 0 || width * active_sequence_batch_ != T) {
                throw std::logic_error(
                    "Text sequence batch binding does not match aggregate columns");
            }
            Tensor q_batch =
                qn[r].view({kCfg.head_dim, (kCfg.n_q / ec().device_count), width, active_sequence_batch_});
            Tensor k_batch =
                kn[r].view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), width, active_sequence_batch_});
            Tensor v_batch =
                v[r].view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), width, active_sequence_batch_});
            Tensor a_batch =
                a[r].view({kCfg.head_dim, (kCfg.n_q / ec().device_count), width, active_sequence_batch_});
            Tensor position_batch = cache.view({width, active_sequence_batch_});
            ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, rank_valid_columns(rank),
                               rank_kv_table_rows(rank), kAttnScale, pages.batch_layer_view(fidx),
                               attention_envelope(*active_gqa_envelope_), *ws[r], a_batch, s);
        } else {
            ops::gqa_attention(qn[r], kn[r], v[r], cache, Tensor{}, rank_kv_table_rows(rank),
                               kAttnScale, pages.batch_layer_view(fidx), attention_envelope(*active_gqa_envelope_),
                               *ws[r], a[r], s);
        }
        if (pages.retrieval_keys.data) {
            ops::kv_retrieval_key_summary(pages.batch_layer_view(fidx), cache.view({T}),
                rank_valid_columns(rank), pages.retrieval_keys.slice(3, fidx, 1), rope_frequency_[r], s);
        }
        ops::sigmoid_mul(gate[r], a[r], s);
    });

    Variant::attention_output_projection(active(rank_map([&](int rank) { return a[rank].view({kCfg.q_size / ec().device_count, T}); })),
                                         active(rank_map([&](int rank) { return *w[rank]->o_proj; })), active(x), active(staging), ph, active(ws), execution,
                                         *tp_[0].events, norm);
}

void TextContext::gdn_mix_tp2(const std::array<const GdnLayerW*, kMaximumExecutionDevices>& w, std::array<Tensor, kMaximumExecutionDevices>& x,
                              int gidx, Phase ph, const std::array<Tensor, kMaximumExecutionDevices>& staging,
                              const ops::ReplicatedRmsNorm* norm,
                              const std::array<Tensor, kMaximumExecutionDevices>* normalized_input) {
    const ExecutionContext& execution       = ec();
    const int T                             = x[0].ne[1];
    const std::array<WorkspaceArena*, kMaximumExecutionDevices> ws = workspaces();

    std::array<Tensor, kMaximumExecutionDevices> h;
    std::array<Tensor, kMaximumExecutionDevices> g;
    std::array<Tensor, kMaximumExecutionDevices> beta;
    std::array<Tensor, kMaximumExecutionDevices> z;
    std::array<Tensor, kMaximumExecutionDevices> qc;
    std::array<Tensor, kMaximumExecutionDevices> kc;
    std::array<Tensor, kMaximumExecutionDevices> vc;
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        const auto control = workspace_recipe::gdn_control<TextConfig>(*ws[r], T,
                                                                      ec().device_count);
        h[r]               = normalized_input ? (*normalized_input)[r] : control.hidden;
        g[r]               = control.g;
        beta[r]            = control.beta;
        const auto projection =
            workspace_recipe::gdn_projection<TextConfig>(*ws[r], T, ec().device_count);
        z[r]  = projection.output_gate.view({kCfg.gdn_v_dim, (kCfg.gdn_v_heads / ec().device_count), T});
        qc[r] = projection.query;
        kc[r] = projection.key;
        vc[r] = projection.value;
    }
    // The tp1 leaf fuses this RMSNorm into the gating GEMM. There is no split form of the fused
    // kernel and no need for one: the norm is replicated elementwise work over the full-width
    // residual, so it runs per rank and the gating GEMM is the column-parallel leaf.
    if (normalized_input == nullptr) for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::rmsnorm(x[r], *w[r]->input_norm, kCfg.rms_eps, true, h[r], stream_for(rank));
    });
    Variant::gdn_control_projection(active(h), active(rank_map([&](int rank) { return w[rank]->projection; })), active(g), active(beta), active(ws), execution);

    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0) {
            throw std::logic_error("Verify GDN requires an explicit sequence batch");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        std::array<Tensor, kMaximumExecutionDevices> projection_input;
        std::array<Tensor, kMaximumExecutionDevices> query_output;
        std::array<Tensor, kMaximumExecutionDevices> key_output;
        std::array<Tensor, kMaximumExecutionDevices> value_output;
        std::array<Tensor, kMaximumExecutionDevices> gate_output;
        std::array<Tensor, kMaximumExecutionDevices> conv_weight;
        std::array<Tensor, kMaximumExecutionDevices> conv_states;
        std::array<Tensor, kMaximumExecutionDevices> valid;
        std::array<Tensor, kMaximumExecutionDevices> slots;
        for (std::size_t r = 0; r < ec().device_count; ++r) {
            const int rank      = static_cast<int>(r);
            projection_input[r] = h[r].view({kCfg.hidden, width, active_sequence_batch_});
            query_output[r]     = qc[r].view({(kCfg.key_dim / ec().device_count), width, active_sequence_batch_});
            key_output[r]       = kc[r].view({(kCfg.key_dim / ec().device_count), width, active_sequence_batch_});
            value_output[r]     = vc[r].view({(kCfg.value_dim / ec().device_count), width, active_sequence_batch_});
            gate_output[r]      = z[r].view({(kCfg.value_dim / ec().device_count), width, active_sequence_batch_});
            conv_weight[r]      = *w[r]->conv1d;
            conv_states[r]      = state_for(rank).conv.at(static_cast<std::size_t>(gidx));
            valid[r]            = rank_valid_columns(rank);
            slots[r]            = rank_linear_state_slots(rank);
        }
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            // The speculative verify round records this device's own head/channel shard instead
            // of committing it: the round's accepted prefix is only known after the target's
            // argmax, so the recurrent and conv updates are replayed afterwards by
            // ops::gdn_replay_fold on EACH device, at the registered
            // FoldGeometry<48, 8, 24, 5120>. Both ranks record the same
            // columns of the same rows -- only the head range differs -- so the two folds commit
            // the same accepted prefix without any agreement protocol.
            std::array<Tensor, kMaximumExecutionDevices> conv_record;
            for (std::size_t r = 0; r < ec().device_count; ++r) {
                const GdnReplayRecords* records = replay_records_for(static_cast<int>(r));
                if (records == nullptr) {
                    throw std::logic_error("Replay-record GDN has no record storage");
                }
                conv_record[r] =
                    records->layer(gidx, active_sequence_batch_).conv;
            }
            Variant::gdn_input_projection_record(active(projection_input), active(rank_map([&](int rank) { return w[rank]->projection; })),
                                                 active(conv_weight), active(conv_states), active(valid), active(slots),
                                                 active(conv_record), active(query_output), active(key_output),
                                                 active(value_output), active(gate_output), ph, active(ws), execution);
        } else {
            Variant::gdn_input_projection_snapshot(
                active(projection_input), active(rank_map([&](int rank) { return w[rank]->projection; })), active(conv_weight), active(conv_states), active(valid),
                active(slots), active(slots), active(query_output), active(key_output), active(value_output), active(gate_output), ph, active(ws),
                execution);
        }
    } else {
        std::array<Tensor, kMaximumExecutionDevices> qkv;
        std::array<Tensor, kMaximumExecutionDevices> qkv_c;
        for (std::size_t r = 0; r < ec().device_count; ++r) {
            const auto conv =
                workspace_recipe::gdn_prefill_conv<TextConfig>(*ws[r], T, ec().device_count);
            qkv[r]   = conv.projected;
            qkv_c[r] = conv.convolved;
        }
        Variant::gdn_input_projection(active(h), active(rank_map([&](int rank) { return w[rank]->projection; })), active(qkv), active(z), ph, active(ws), execution);
        for_each_rank(execution, [&](int rank) {
            const auto r      = static_cast<std::size_t>(rank);
            cudaStream_t s    = stream_for(rank);
            Tensor conv_state = state_for(rank).conv_slot(static_cast<std::uint32_t>(gidx),
                                                          linear_state_current_slot_);
            ops::causal_conv1d_silu(qkv[r], *w[r]->conv1d, conv_state, conv_state, qkv_c[r], s);
            // Shard-local section offsets: this device's convolved block is its own
            // q(1024) | k(1024) | v(3072), not the model's 2048 | 2048 | 6144.
            ops::extract_bf16_columns(qkv_c[r], 0, qc[r], s);
            ops::extract_bf16_columns(qkv_c[r], (kCfg.key_dim / ec().device_count), kc[r], s);
            ops::extract_bf16_columns(qkv_c[r], 2 * (kCfg.key_dim / ec().device_count), vc[r], s);
        });
    }

    std::array<Tensor, kMaximumExecutionDevices> o;
    std::array<Tensor, kMaximumExecutionDevices> on;
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        o[r] = workspace_recipe::gdn_recurrent_output<TextConfig>(*ws[r], T, ec().device_count)
                   .view({kCfg.gdn_v_dim, (kCfg.gdn_v_heads / ec().device_count), T});
        const auto output_dtype = norm && tp_[0].events->packet_storage(0)
            ? Variant::gdn_output_input_dtype(*w[r]->out_proj, ec().device_count, T)
            : DType::BF16;
        on[r] = workspace_recipe::gdn_normalized_output<TextConfig>(*ws[r], T, ec().device_count,
                                                                    output_dtype)
                    .view({kCfg.gdn_v_dim, (kCfg.gdn_v_heads / ec().device_count), T});
    }
    for_each_rank(execution, [&](int rank) {
        const auto r       = static_cast<std::size_t>(rank);
        cudaStream_t s     = stream_for(rank);
        bool fused_norm    = false;
        Tensor q_recurrent = qc[r].view({kCfg.gdn_k_dim, (kCfg.gdn_k_heads / ec().device_count), T});
        Tensor k_recurrent = kc[r].view({kCfg.gdn_k_dim, (kCfg.gdn_k_heads / ec().device_count), T});
        Tensor vv          = vc[r].view({kCfg.gdn_v_dim, (kCfg.gdn_v_heads / ec().device_count), T});
        if (ph == Phase::Verify) {
            Tensor& recurrent_states = state_for(rank).recurrent.at(static_cast<std::size_t>(gidx));
            const std::int32_t width = active_sequence_width_;
            Tensor q_batch =
                q_recurrent.view({kCfg.gdn_k_dim, (kCfg.gdn_k_heads / ec().device_count), width, active_sequence_batch_});
            Tensor k_batch =
                k_recurrent.view({kCfg.gdn_k_dim, (kCfg.gdn_k_heads / ec().device_count), width, active_sequence_batch_});
            Tensor v_batch =
                vv.view({kCfg.gdn_v_dim, (kCfg.gdn_v_heads / ec().device_count), width, active_sequence_batch_});
            Tensor g_batch    = g[r].view({(kCfg.gdn_v_heads / ec().device_count), width, active_sequence_batch_});
            Tensor beta_batch = beta[r].view({(kCfg.gdn_v_heads / ec().device_count), width, active_sequence_batch_});
            Tensor out_batch =
                o[r].view({kCfg.gdn_v_dim, (kCfg.gdn_v_heads / ec().device_count), width, active_sequence_batch_});
            const Tensor valid = rank_valid_columns(rank);
            const Tensor slots = rank_linear_state_slots(rank);
            if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
                GdnReplayRecordLayer records =
                    replay_records_for(rank)->layer(gidx, active_sequence_batch_);
                const Tensor counters = io_for(rank).gdn_record_norm_counts;
                fused_norm = q_batch.ne[1] == 4 && v_batch.ne[1] == 12 &&
                             active_sequence_batch_ == 1 && width <= 8 &&
                             on[r].dtype == DType::FP16 && counters.data != nullptr;
                if (fused_norm) {
                    const ops::GdnReplayGatedNorm epilogue{
                        *w[r]->gdn_norm, z[r].view({128, 12, width, 1}), kCfg.rms_eps,
                        on[r].view({128, 12, width, 1}), counters};
                    ops::gated_delta_net_replay_record(
                        q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale, recurrent_states,
                        valid, slots, records.key, records.value, records.gate, out_batch,
                        epilogue, s);
                } else {
                    ops::gated_delta_net_replay_record(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                                       kGdnScale, recurrent_states, valid, slots,
                                                       records.key, records.value, records.gate,
                                                       out_batch, s);
                }
            } else {
                ops::gated_delta_net_snapshot(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                              kGdnScale,
                                              /*normalize_qk=*/true, recurrent_states, valid,
                                              slots, slots, out_batch, s);
            }
        } else {
            Tensor recurrent_state = state_for(rank).recurrent_slot(
                static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
            ops::gated_delta_net(q_recurrent, k_recurrent, vv, g[r], beta[r], kGdnScale,
                                 /*normalize_qk=*/true, *ws[r], recurrent_state, o[r], s);
        }
        // `gdn_norm` is the per-head-DIMENSION gain {128}: replicated, so each rank applies the
        // whole weight over its own 24 value heads.
        if (!fused_norm) {
            ops::gated_rmsnorm(o[r], *w[r]->gdn_norm, z[r], kCfg.rms_eps, on[r], s);
        }
    });

    Variant::gdn_output_projection(
        active(rank_map([&](int rank) { return on[rank].view({kCfg.value_dim / ec().device_count, T}); })),
        active(rank_map([&](int rank) { return *w[rank]->out_proj; })), active(x), active(staging), ph, active(ws), execution, *tp_[0].events, norm);
}

void TextContext::mlp_tail_tp2(
    const std::array<const Tensor*, kMaximumExecutionDevices>& norm,
    const std::array<const MlpW*, kMaximumExecutionDevices>& mlp,
    std::array<Tensor, kMaximumExecutionDevices>& x, Phase phase,
    const std::array<Tensor, kMaximumExecutionDevices>& staging,
    const std::array<Tensor, kMaximumExecutionDevices>* normalized,
    const ops::ReplicatedRmsNorm* next_input_norm) {
    const auto work = workspaces();
    const auto hidden = normalized != nullptr ? *normalized : rank_map([&](int rank) {
        return workspace_recipe::post_mixer_hidden<TextConfig>(*work[rank], x[rank].ne[1]);
    });
    if (normalized == nullptr) for_each_rank(ec(), [&](int rank) {
        Tensor output = hidden[rank];
        ops::rmsnorm(x[rank], *norm[rank], kCfg.rms_eps, true, output, stream_for(rank));
    });
    Variant::post_mixer(active(hidden),
                       active(rank_map([&](int rank) { return mlp[rank]->payload; })),
                       active(x), active(staging), phase, active(work), ec(), *tp_[0].events, next_input_norm);
}

void TextContext::run_layers_tp2(std::array<Tensor, kMaximumExecutionDevices>& x, Phase phase,
                                const std::array<Tensor, kMaximumExecutionDevices>& staging,
                                DFlashFeatureSink* sink) {
    if (sink != nullptr) { sink->begin(x[0]); }
    const bool prefill = phase == Phase::Prefill;
    auto input_norm_scopes = workspace_scopes();
    const bool handoff_norm = x[0].ne[1] <= 8;
    const auto prepared_input = rank_map([&](int rank) {
        return handoff_norm ? workspace_recipe::post_mixer_hidden<TextConfig>(*workspaces()[rank], x[rank].ne[1]) : Tensor{};
    });
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        auto layer_scopes = workspace_scopes();
        const bool normalized_mixer_output = x[0].ne[1] <= 8;
        const auto normalized = rank_map([&](int rank) {
            if (!normalized_mixer_output) return Tensor{};
            const MlpW* mlp;
            if (ModelConfig::is_full(layer)) {
                const auto i = ModelConfig::full_idx(layer);
                mlp = rank == 0 ? &full_[i].mlp : &full_peer_[rank - 1][i].mlp;
            } else {
                const auto i = ModelConfig::gdn_idx(layer);
                mlp = rank == 0 ? &gdn_[i].mlp : &gdn_peer_[rank - 1][i].mlp;
            }
            // The FP16 image is consumed by the packet-published TP4 projection.
            // Staged transport composes the BF16 norm and ordinary projection,
            // just as the GDN output above does without packet storage.
            const auto dtype = tp_[0].events->packet_storage(0)
                ? Variant::post_mixer_input_dtype(*mlp->payload, ec().device_count, x[rank].ne[1])
                : DType::BF16;
            return workspaces()[rank]->alloc(dtype, {kCfg.hidden, x[rank].ne[1]});
        });
        const auto next_weights = rank_map([&](int rank) -> Tensor {
            if (layer + 1 >= kCfg.n_layers || !handoff_norm) return {};
            if (ModelConfig::is_full(layer + 1)) {
                const auto index = ModelConfig::full_idx(layer + 1);
                return *(rank == 0 ? full_[index].input_norm : full_peer_[rank - 1][index].input_norm);
            }
            const auto index = ModelConfig::gdn_idx(layer + 1);
            return *(rank == 0 ? gdn_[index].input_norm : gdn_peer_[rank - 1][index].input_norm);
        });
        const ops::ReplicatedRmsNorm next_norm{active(next_weights), active(prepared_input), kCfg.rms_eps, true};
        const auto* next_input_norm = handoff_norm && layer + 1 < kCfg.n_layers ? &next_norm : nullptr;
        const auto* input = handoff_norm && layer > 0 ? &prepared_input : nullptr;
        if (ModelConfig::is_full(layer)) {
            const int index = ModelConfig::full_idx(layer);
            const auto weights = rank_map([&](int rank) -> const FullLayerW* {
                return rank == 0 ? &full_[index] : &full_peer_[rank - 1][index];
            });
            const auto norm_weights = rank_map([&](int rank) { return *weights[rank]->post_attn_norm; });
            const ops::ReplicatedRmsNorm norm{active(norm_weights), active(normalized), kCfg.rms_eps, true};
            const auto* norm_output = normalized_mixer_output ? &norm : nullptr;
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, layer);
            {
                auto scopes = workspace_scopes();
                attn_mix_tp2(weights, x, index, phase, staging, norm_output, input);
            }
            {
                auto scopes = workspace_scopes();
                mlp_tail_tp2(
                    rank_map([&](int rank) { return weights[rank]->post_attn_norm; }),
                    rank_map([&](int rank) -> const MlpW* { return &weights[rank]->mlp; }),
                    x, phase, staging, normalized_mixer_output ? &normalized : nullptr, next_input_norm);
            }
        } else {
            const int index = ModelConfig::gdn_idx(layer);
            const auto weights = rank_map([&](int rank) -> const GdnLayerW* {
                return rank == 0 ? &gdn_[index] : &gdn_peer_[rank - 1][index];
            });
            const auto norm_weights = rank_map([&](int rank) { return *weights[rank]->post_attn_norm; });
            const ops::ReplicatedRmsNorm norm{active(norm_weights), active(normalized), kCfg.rms_eps, true};
            const auto* norm_output = normalized_mixer_output ? &norm : nullptr;
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerGdn : nvtx::Name::VerifyLayerGdn,
                nvtx::Category::Gdn, layer);
            {
                auto scopes = workspace_scopes();
                gdn_mix_tp2(weights, x, index, phase, staging, norm_output, input);
            }
            {
                auto scopes = workspace_scopes();
                mlp_tail_tp2(
                    rank_map([&](int rank) { return weights[rank]->post_attn_norm; }),
                    rank_map([&](int rank) -> const MlpW* { return &weights[rank]->mlp; }),
                    x, phase, staging, normalized_mixer_output ? &normalized : nullptr, next_input_norm);
            }
        }
        if (sink != nullptr) { sink->capture_layer(layer, x[0], stream_for(0)); }
    }
    if (sink != nullptr) { sink->capture_positions(rank_cache_positions(0), stream_for(0)); }
}

void TextContext::target_logits(
    const std::array<Tensor, kMaximumExecutionDevices>& hidden,
    const std::array<Tensor, kMaximumExecutionDevices>& logits) {
    if (!tp2()) { throw std::logic_error("tensor-parallel logits require peer bindings"); }
    logits_tp2(hidden, logits);
}

void TextContext::logits_tp2(
    const std::array<Tensor, kMaximumExecutionDevices>& hidden,
    const std::array<Tensor, kMaximumExecutionDevices>& logits) {
    const int columns = hidden[0].ne[1];
    const int rows = kCfg.vocab / ec().device_count;
    const auto work = workspaces();
    auto scopes = workspace_scopes();
    const auto shards = rank_map([&](int rank) {
        require_tensor_shape(hidden[rank], DType::BF16, {kCfg.hidden, columns}, "head hidden");
        require_tensor_shape(logits[rank], DType::BF16, {kCfg.vocab, columns}, "head logits");
        return work[rank]->alloc(DType::BF16, {rows, columns});
    });
    ops::linear_column_parallel(active(hidden),
        active(rank_map([&](int rank) { return rank == 0 ? *lm_head_ : *lm_head_peer_[rank - 1]; })),
        active(shards), ops::LinearPolicy::A16Only, active(work), ec());
    for (int column = 0; column < columns; ++column) {
        const auto source = rank_map([&](int rank) {
            return shards[rank].slice(1, column, 1).view({1, rows});
        });
        const auto output = rank_map([&](int rank) {
            return logits[rank].slice(1, column, 1).view({1, kCfg.vocab});
        });
        ops::allgather_rows(active(output), active(source), ec(), *tp_[0].events);
    }
}

void TextContext::constrain_logits(const Tensor& logits, std::int32_t row_offset,
                                   std::int32_t valid_rows, cudaStream_t stream) const {
    if (!token_constraints_) { return; }
    Tensor view = logits;
    ops::apply_token_bitmask(view, valid_rows, row_offset, token_constraints_.constrained,
                             token_constraints_.bitmask, token_constraints_.words, stream);
}

void TextContext::target_argmax_tp2(
    const std::array<Tensor, kMaximumExecutionDevices>& hidden,
    const std::array<Tensor, kMaximumExecutionDevices>& target_tokens) {
    head_argmax_tp(hidden,
        rank_map([&](int rank) { return rank == 0 ? *lm_head_ : *lm_head_peer_[rank - 1]; }),
        kCfg.token_domain, target_tokens);
}

void TextContext::head_argmax_tp(
    const std::array<Tensor, kMaximumExecutionDevices>& hidden,
    const std::array<Weight, kMaximumExecutionDevices>& heads,
    std::int32_t valid_rows,
    const std::array<Tensor, kMaximumExecutionDevices>& target_tokens, bool proposal) {
    const int columns = hidden[0].ne[1];
    const int rows = heads[0].n;
    const auto work = workspaces();
    auto scopes = workspace_scopes();
    const auto shards = rank_map([&](int rank) {
        require_tensor_shape(hidden[rank], DType::BF16, {kCfg.hidden, columns}, "head hidden");
        require_tensor_shape(target_tokens[rank], DType::I32, {columns}, "head tokens");
        return work[rank]->alloc(DType::BF16, {rows, columns});
    });
    const auto values = rank_map([&](int rank) {
        return work[rank]->alloc(DType::FP32, {1, columns});
    });
    const auto indices = rank_map([&](int rank) {
        return work[rank]->alloc(DType::I32, {1, columns});
    });
    if (proposal) {
        for_each_rank(ec(), [&](int rank) {
            Tensor scratch = shards[rank];
            Tensor value = values[rank].view({columns});
            Tensor index = indices[rank].view({columns});
            ops::proposal_argmax(hidden[rank], heads[rank], scratch, value, index, *work[rank],
                                 std::min(rows, valid_rows - rank * rows), ec().dev[rank]->sm(),
                                 stream_for(rank));
        });
    } else {
        ops::linear_column_parallel(active(hidden), active(heads), active(shards),
                                    ops::LinearPolicy::A16Only, active(work), ec());
        for_each_rank(ec(), [&](int rank) {
            const int valid = std::min(rows, valid_rows - rank * rows);
            if (valid <= 0) { throw std::logic_error("vocabulary shard does not cover token domain"); }
            Tensor value = values[rank].view({columns});
            Tensor index = indices[rank].view({columns});
            constrain_logits(shards[rank], rank * rows, valid, stream_for(rank));
            ops::argmax_with_value(shards[rank], value, index, valid, stream_for(rank));
        });
    }
    Tensor gathered_values = work_.alloc(DType::FP32, {ec().device_count, columns});
    Tensor gathered_indices = work_.alloc(DType::I32, {ec().device_count, columns});
    ops::allreduce_argmax(active(values), active(indices), active(target_tokens),
                          gathered_values, gathered_indices, rows, ec(), *tp_[0].events,
                          proposal && proposal_head_ != nullptr ? proposal_head_ids_ : nullptr);
}

PrefillChunkResult TextContext::prefill_impl_tp2(std::span<const int> ids,
                                                 const TextPrefill& text_prefill,
                                                 bool finalize_at_end,
                                                 DFlashFeatureSink* dflash_sink,
                                                 const MultimodalPrefill* multimodal) {
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    // The same chunk-belongs-to-this-prompt check the tp1 path makes: this context was built for
    // one cache base, and a chunk that starts somewhere else would append at the wrong positions.
    if (text_kv_base_ != text_prefill.begin ||
        text_prefill.token_ids.size() < static_cast<std::size_t>(text_kv_base_) + ids.size()) {
        throw std::invalid_argument("text prefill chunk does not match its full prompt");
    }
    const ExecutionContext& execution       = ec();
    const std::array<WorkspaceArena*, kMaximumExecutionDevices> ws = workspaces();
    const int T                             = static_cast<int>(ids.size());
    const int chunk                         = static_cast<int>(prefill_chunk_);
    // Prefix-append prefill continues an existing cache, so positions are absolute and can leave
    // int32 even when the chunk itself is small.
    if (static_cast<std::uint64_t>(text_kv_base_) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(text_kv_base_);
    if (multimodal != nullptr) {
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size() ||
            multimodal->vision == nullptr || multimodal->begin != text_kv_base_) {
            throw std::invalid_argument("invalid tensor-parallel multimodal prefill");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    for_each_rank(execution, [&](int rank) {
        ops::set_i32_scalar(io_for(rank).rope_delta, rope_delta_, stream_for(rank));
    });

    const std::int64_t base64         = static_cast<std::int64_t>(text_kv_base_);
    const std::int64_t checkpoint_abs = prefill_rewrite_checkpoint_frontier_;
    const bool has_rewrite_checkpoint =
        checkpoint_abs > base64 && checkpoint_abs <= base64 + static_cast<std::int64_t>(T);
    const int checkpoint_rel =
        has_rewrite_checkpoint ? static_cast<int>(checkpoint_abs - base64) : -1;

    int len = std::min(chunk, T);
    if (checkpoint_rel > 0 && len > checkpoint_rel) { len = checkpoint_rel; }
    work_.reset();
    for (const PeerExecution& peer : tp_) { peer.work->reset(); }
    VisionChunk vision_chunk;
    if (multimodal != nullptr) {
        vision_chunk = multimodal->vision->prepare_chunk(text_kv_base_, len);
        len = vision_chunk.length;
    }
    std::vector<std::int32_t> scatter_indices;
    std::int32_t visual_begin = 0;
    if (vision_chunk.control != nullptr) {
        const auto& scatter = vision_chunk.control->scatter_indices;
        const auto begin = std::lower_bound(scatter.begin(), scatter.end(), base_i);
        const auto end = std::lower_bound(begin, scatter.end(), base_i + len);
        visual_begin = static_cast<std::int32_t>(begin - scatter.begin());
        for (auto position = begin; position != end; ++position) {
            scatter_indices.push_back(*position - base_i);
        }
    }
    std::vector<std::int32_t> rope_host;
    if (multimodal != nullptr) {
        rope_host.resize(static_cast<std::size_t>(3) * len);
        for (int axis = 0; axis < 3; ++axis) {
            std::copy_n(multimodal->positions.data() +
                            static_cast<std::size_t>(axis) * multimodal->token_ids.size() + base_i,
                        len, rope_host.data() + static_cast<std::size_t>(axis) * len);
        }
    }
    const bool is_last                = finalize_at_end && len == T;
    const bool prepare_mtp_prompt     = mtp_enabled() && io_.mtp.has_value();
    nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                  static_cast<std::uint64_t>(len));
    work_.reset();
    for (const PeerExecution& peer : tp_) { peer.work->reset(); }
    {
        auto scopes = workspace_scopes();
        std::array<Tensor, kMaximumExecutionDevices> ids_device;
        std::array<Tensor, kMaximumExecutionDevices> positions;
        std::array<Tensor, kMaximumExecutionDevices> rope_positions;
        std::array<Tensor, kMaximumExecutionDevices> scatter_device;
        std::array<Tensor, kMaximumExecutionDevices> x;
        std::array<Tensor, kMaximumExecutionDevices> staging;
        for (std::size_t r = 0; r < ec().device_count; ++r) {
            const auto roots = workspace_recipe::text_prefill_roots<TextConfig>(
                *ws[r], len, multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0),
                static_cast<std::int32_t>(scatter_indices.size()));
            ids_device[r]    = roots.ids;
            positions[r]     = roots.positions;
            rope_positions[r] = roots.rope_positions.data != nullptr ? roots.rope_positions
                                                                     : positions[r];
            scatter_device[r] = roots.scatter_indices;
            x[r]             = roots.residual;
            staging[r]       = ws[r]->alloc(DType::BF16, {kCfg.hidden, len * (ec().device_count == 4 ? 4 : 1)});
        }
        for_each_rank(execution, [&](int rank) {
            const auto r   = static_cast<std::size_t>(rank);
            cudaStream_t s = stream_for(rank);
            copy_i32(ids.data(), ids_device[r], s);
            ops::fill_i32_positions(positions[r], base_i, s);
            if (multimodal != nullptr) {
                copy_i32(rope_host.data(), rope_positions[r], s);
            } else if (rope_delta_ != 0) {
                ops::offset_i32_positions(positions[r], io_for(rank).rope_delta,
                                          rope_positions[r], s);
            }
            ops::embedding(ids_device[r], embedding_for(rank), x[r], s);
        });
        if (!scatter_indices.empty()) {
            const CurrentDevice restore;
            CUDA_CHECK(cudaSetDevice(ctx_.device));
            copy_i32(scatter_indices.data(), scatter_device[0], ctx_.stream);
            Tensor embeddings = vision_chunk.embeddings.slice(
                1, visual_begin, static_cast<std::int32_t>(scatter_indices.size()));
            ops::scatter(embeddings, scatter_device[0], x[0], ctx_.stream);
            CUDA_CHECK(cudaEventRecord(tp_[0].events->inputs_ready(0), ctx_.stream));
            for_each_rank(execution, [&](int rank) {
                if (rank == 0) { return; }
                const auto r = static_cast<std::size_t>(rank);
                CUDA_CHECK(cudaSetDevice(execution.dev[r]->device));
                CUDA_CHECK(
                    cudaStreamWaitEvent(stream_for(rank), tp_[0].events->inputs_ready(0), 0));
                CUDA_CHECK(cudaMemcpyAsync(x[r].data, x[0].data, x[0].bytes(),
                                           cudaMemcpyDeviceToDevice, stream_for(rank)));
                CUDA_CHECK(cudaEventRecord(tp_[0].events->pull_done(rank), stream_for(rank)));
                CUDA_CHECK(cudaSetDevice(ctx_.device));
                CUDA_CHECK(
                    cudaStreamWaitEvent(ctx_.stream, tp_[0].events->pull_done(rank), 0));
            });
        }

        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_cache(peer_cache_positions_, peer_positions(positions));
        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_rope(peer_rope_positions_, peer_positions(rope_positions));
        const auto table_rows = rank_map([&](int rank) { return io_for(rank).text_kv_table_row; });
        ScopedValue peer_rows(peer_kv_table_rows_, peer_positions(table_rows));
        ScopedPositions scoped_cache(active_cache_positions_, positions[0]);
        ScopedPositions scoped_rope(active_rope_positions_, rope_positions[0]);
        const auto visible = static_cast<std::uint32_t>(base_i + len);
        const ops::GqaExecutionEnvelope chunk_envelope{visible, visible};
        ScopedEnvelope scoped_envelope(active_gqa_envelope_, chunk_envelope);

        run_layers_tp2(x, Phase::Prefill, staging, dflash_sink);

        std::array<Tensor, kMaximumExecutionDevices> xf;
        xf[0] = prefill_hidden_.data != nullptr ? matrix_window(prefill_hidden_, len)
                                                : ws[0]->alloc(DType::BF16, {kCfg.hidden, len});
        for (int rank = 1; rank < ec().device_count; ++rank) {
            Tensor* hidden = tp_[rank - 1].prefill_hidden;
            xf[rank] = hidden != nullptr && hidden->data != nullptr
                           ? matrix_window(*hidden, len)
                           : ws[rank]->alloc(DType::BF16, {kCfg.hidden, len});
        }
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::rmsnorm(x[r], final_norm_for(rank), kCfg.rms_eps, true,
                         xf[r], stream_for(rank));
        });

        if (is_last) {
            const auto last = rank_map([&](int rank) { return xf[rank].slice(1, len - 1, 1); });
            const auto rank_logits = rank_map([&](int rank) { return matrix_window(io_for(rank).logits, 1); });
            Tensor logits = rank_logits[0];
            logits_tp2(last, rank_logits);
            // Sampling belongs to rank 0 alone: it consumes the reconstructed FULL logits and
            // writes the single committed token.
            const CurrentDevice restore;
            CUDA_CHECK(cudaSetDevice(ctx_.device));
            ops::set_i32_scalar(io_.pos, base_i + T, ctx_.stream);
            ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, ctx_.stream);
            constrain_logits(logits, 0, kCfg.token_domain, ctx_.stream);
            if (sampling_config_ != nullptr) {
                ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_, io_.pos,
                            ops::kSamplePurposePrefill, work_, ctx_.stream);
            } else {
                ops::argmax(logits, io_.token, kCfg.token_domain, ctx_.stream);
            }
        }


        // MTP prompt alignment, chunk by chunk, exactly as the tp1 path drives it: the MTP head
        // consumes the SHIFTED token stream against the text model's own final hidden, so its KV
        // is built one column behind the target's. Only rank 0 needs the shifted ids (its fc
        // shard is the embedding half); rank 1 works from its own copy of the final hidden.
        if (prepare_mtp_prompt) {
            for (const auto& peer : tp_) {
                if (!peer.io->mtp) {
                    throw std::logic_error("tensor-parallel MTP prefill requires every peer MTP frame");
                }
            }
            const auto alignment_tokens = static_cast<std::uint32_t>(text_prefill.token_ids.size());
            const qwen3_6::MtpAlignmentWindow mtp_window = qwen3_6::plan_mtp_alignment_window(
                alignment_tokens, text_kv_base_, static_cast<std::uint32_t>(len));
            std::vector<int> mtp_ids_host(static_cast<std::size_t>(len));
            const int prompt_columns =
                len - static_cast<int>(mtp_window.final_column_uses_generated_token);
            for (int j = 0; j < prompt_columns; ++j) {
                mtp_ids_host[static_cast<std::size_t>(j)] =
                    text_prefill
                        .token_ids[static_cast<std::size_t>(mtp_window.shifted_embedding_begin) +
                                   static_cast<std::size_t>(j)];
            }
            Tensor mtp_ids = ws[0]->alloc(DType::I32, {len});
            {
                const CurrentDevice restore;
                CUDA_CHECK(cudaSetDevice(ctx_.device));
                if (mtp_window.final_column_uses_generated_token) {
                    int next_token = 0;
                    CUDA_CHECK(cudaStreamSynchronize(ctx_.stream));
                    CUDA_CHECK(cudaMemcpy(&next_token, io_.token.data, sizeof(next_token),
                                          cudaMemcpyDeviceToHost));
                    mtp_ids_host[static_cast<std::size_t>(len - 1)] = next_token;
                }
                copy_i32(mtp_ids_host.data(), mtp_ids, ctx_.stream);
            }

            Tensor mtp_embeddings;
            const Tensor* mtp_embeddings_ptr = nullptr;
            if (multimodal != nullptr) {
                const CurrentDevice restore;
                CUDA_CHECK(cudaSetDevice(ctx_.device));
                mtp_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, len});
                ops::embedding(mtp_ids, *embed_, mtp_embeddings, ctx_.stream);
                if (vision_chunk.control != nullptr) {
                    const auto overlap = qwen3_6::shifted_visual_overlap(
                        vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                    if (!overlap.empty()) {
                        Tensor indices = workspace_recipe::visual_scatter_indices(
                            work_, static_cast<std::int32_t>(overlap.size()));
                        qwen3_6::detail::scatter_shifted_visual_embeddings(
                            mtp_embeddings, vision_chunk.embeddings, overlap, indices, ctx_.stream);
                    }
                }
                mtp_embeddings_ptr = &mtp_embeddings;
            }
            const auto ar_hidden = rank_map([&](int rank) { return io_for(rank).mtp->ar_hidden; });
            const auto mtp_logits = rank_map([&](int rank) { return matrix_window(io_for(rank).logits, 1); });
            if (is_last && mtp_proposal_extent_ != 0) {
                if (mtp_proposal_extent_ >
                    static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
                    throw std::logic_error("MTP proposal extent exceeds the configured window");
                }
                Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                mtp_prefill_chunk_tp2(mtp_ids, xf, positions, rope_positions, chunk_envelope,
                                      /*final_chunk=*/true, &ar_hidden, &mtp_logits, &draft0,
                                      mtp_embeddings_ptr);

                const auto ar_position = rank_map([&](int rank) {
                    return io_for(rank).mtp->position.slice(0, 0, 1);
                });
                for_each_rank(execution, [&](int rank) {
                    const auto r = static_cast<std::size_t>(rank);
                    ops::set_i32_scalar(const_cast<Tensor&>(ar_position[r]), base_i + T,
                                        stream_for(rank));
                });
                for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                    auto ar_scopes = workspace_scopes();
                    Tensor prev_token = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                    Tensor next_token = io_.mtp->draft_tokens.slice(0, i, 1);
                    const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                    const ops::GqaExecutionEnvelope ar_envelope{ar_visible, ar_visible};
                    mtp_forward_ar_step(prev_token, ar_hidden, ar_position, ar_envelope,
                                        ar_hidden, mtp_logits, next_token);
                    for_each_rank(execution, [&](int rank) {
                        const auto r   = static_cast<std::size_t>(rank);
                        cudaStream_t s = stream_for(rank);
                        ops::increment_i32_scalar(const_cast<Tensor&>(ar_position[r]), s);
                    });
                }
            } else {
                mtp_prefill_chunk_tp2(mtp_ids, xf, positions, rope_positions, chunk_envelope,
                                      /*final_chunk=*/false, nullptr, nullptr, nullptr,
                                      mtp_embeddings_ptr);
            }
        }

        if (checkpoint_rel > 0 && len == checkpoint_rel &&
            rewrite_checkpoint_hidden_output_ != nullptr) {
            require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16, {kCfg.hidden, 1},
                                 "rewrite checkpoint hidden output");
            const Tensor checkpoint_hidden = xf[0].slice(1, len - 1, 1);
            const CurrentDevice restore;
            CUDA_CHECK(cudaSetDevice(ctx_.device));
            CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                       checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, ctx_.stream));
        }
    }

    if (checkpoint_rel > 0 && len == checkpoint_rel) {
        for_each_rank(execution, [&](int rank) {
            state_for(rank).copy_slot(linear_state_current_slot_,
                                      linear_state_rewrite_checkpoint_slot_, stream_for(rank));
        });
    }

    prefill_rewrite_checkpoint_frontier_ = -1;
    synchronize_all();
    if (dflash_sink != nullptr) {
        dflash_sink->consume_prefill_chunk(len, checkpoint_rel > 0 && len == checkpoint_rel);
        synchronize_all();
    }
    work_.reset();
    for (const PeerExecution& peer : tp_) { peer.work->reset(); }
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(len),
                              .finalized        = finalize_at_end && len == T,
                              .tail_column      = static_cast<std::uint32_t>(len - 1)};
}

void TextContext::ordinary_decode_batch_tp2(
    const Tensor& ids, const Tensor& cache_positions, const Tensor& rope_positions,
    const Tensor& kv_table_rows, const Tensor& linear_state_slots,
    ops::GqaExecutionEnvelope envelope, Tensor& hidden, Tensor& logits) {
    const int batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<int>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    const auto work = workspaces();
    const auto rank_ids = rank_map([&](int rank) {
        return rank == 0 ? ids : io_for(rank).ordinary->tokens.slice(0, 0, batch);
    });
    const auto positions = rank_map([&](int rank) {
        return rank == 0 ? cache_positions : io_for(rank).ordinary->cache_positions.slice(0, 0, batch);
    });
    const auto rope = rank_map([&](int rank) {
        return rank == 0 ? rope_positions : io_for(rank).ordinary->rope_positions.slice(0, 0, batch);
    });
    const auto rows = rank_map([&](int rank) {
        return rank == 0 ? kv_table_rows : io_for(rank).ordinary->text_kv_table_rows.slice(0, 0, batch);
    });
    const auto lanes = rank_map([&](int rank) {
        return rank == 0 ? linear_state_slots : io_for(rank).ordinary->lanes.slice(0, 0, batch);
    });
    const auto rank_hidden = rank_map([&](int rank) {
        return rank == 0 ? hidden : io_for(rank).ordinary->hidden.slice(1, 0, batch);
    });
    const auto rank_logits = rank_map([&](int rank) {
        return rank == 0 ? logits : io_for(rank).ordinary->logits.slice(1, 0, batch);
    });
    work_.reset();
    for (const auto& peer : tp_) { peer.work->reset(); }
    {
        ScopedPositions cache_binding(active_cache_positions_, positions[0]);
        ScopedPositions rope_binding(active_rope_positions_, rope[0]);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> rows_binding(active_kv_table_rows_, &rows[0]);
        ScopedValue<const Tensor*> lanes_binding(active_linear_state_slots_, &lanes[0]);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);
        ScopedValue peer_cache(peer_cache_positions_, peer_positions(positions));
        ScopedValue peer_rope(peer_rope_positions_, peer_positions(rope));
        ScopedValue peer_rows(peer_kv_table_rows_, peer_positions(rows));
        ScopedValue peer_lanes(peer_linear_state_slots_, peer_positions(lanes));
        auto scopes = workspace_scopes();
        auto x = rank_map([&](int rank) { return work[rank]->alloc(DType::BF16, {kCfg.hidden, batch}); });
        const auto staging = rank_map([&](int rank) {
            return work[rank]->alloc(DType::BF16, {kCfg.hidden, batch * (ec().device_count == 4 ? 4 : 1)});
        });
        for_each_rank(ec(), [&](int rank) {
            ops::embedding(rank_ids[rank], embedding_for(rank), x[rank], stream_for(rank));
        });
        run_layers_tp2(x, Phase::Verify, staging);
        for_each_rank(ec(), [&](int rank) {
            Tensor out = rank_hidden[rank];
            ops::rmsnorm(x[rank], final_norm_for(rank), kCfg.rms_eps, true, out, stream_for(rank));
        });
        logits_tp2(rank_hidden, rank_logits);
    }
    work_.reset();
    for (const auto& peer : tp_) { peer.work->reset(); }
}

// --- tp == 2 MTP -------------------------------------------------------------------------------
//
// The MTP head is a single decoder layer plus a vocabulary head, so its split schedule is the
// text layer's split schedule with one instance of each stage. Three collectives per MTP call:
// the stem's fc all-reduce, the attention output projection's all-reduce, and the post-mixer's
// all-reduce. The residual `x` is replicated and bit-identical on both ranks after every one of
// them, exactly as in the text layers, which is what keeps the MTP KV pages and the proposal
// argmax in lockstep without any further agreement protocol.

void TextContext::mtp_forward_stem_tp2(
    const Tensor& ids, const std::array<Tensor, kMaximumExecutionDevices>& hidden,
    std::array<Tensor, kMaximumExecutionDevices>& x,
    std::array<Tensor, kMaximumExecutionDevices>& ah,
    const std::array<Tensor, kMaximumExecutionDevices>& staging,
    const Tensor* input_embeddings) {
    const int columns = ids.ne[0] * ids.ne[1];
    const int input_rows = kCfg.mtp_fc_in / ec().device_count;
    const auto work = workspaces();
    const int embedding_ranks = ec().device_count / 2;
    const auto roots = rank_map([&](int rank) {
        require_tensor_shape(hidden[rank].view({kCfg.hidden, columns}), DType::BF16,
                             {kCfg.hidden, columns}, "MTP hidden");
        return workspace_recipe::mtp_stem<TextConfig>(*work[rank], columns,
                                                      rank < embedding_ranks, ec().device_count);
    });
    Tensor flat_ids = ids.view({columns});
    const auto rank_ids = rank_map([&](int rank) {
        return rank == 0 ? flat_ids : embedding_ranks > 1
            ? work[rank]->alloc(DType::I32, {columns}) : Tensor{};
    });
    // Ranks 0..TP/2-1 contract normalized embedding slices. Others contract hidden slices.
    // TP2 has no remote embedding reader; TP4 still needs the rank-0 token broadcast.
    if (embedding_ranks > 1) {
        ops::broadcast_rank0(flat_ids, std::span(rank_ids).subspan(1, ec().device_count - 1),
                             ec(), *tp_[0].events);
    }
    std::array<Tensor, kMaximumExecutionDevices> inputs{};
    const auto weights = rank_map([&](int rank) { return *mtp_weights_for(rank).fc; });
    for_each_rank(ec(), [&](int rank) {
        const auto stream = stream_for(rank);
        Tensor normalized = rank < embedding_ranks ? roots[rank].normalized_embedding
                                                   : roots[rank].normalized_hidden;
        if (rank < embedding_ranks) {
            Tensor embedding = roots[rank].embedding;
            if (rank == 0 && input_embeddings != nullptr) {
                require_tensor_shape(*input_embeddings, DType::BF16, {kCfg.hidden, columns},
                                     "MTP input embeddings");
                embedding = *input_embeddings;
            } else {
                ops::embedding(rank_ids[rank], embedding_for(rank), embedding, stream);
            }
            ops::rmsnorm(embedding, *mtp_weights_for(rank).pre_fc_norm_embedding,
                         kCfg.rms_eps, true, normalized, stream);
        } else {
            ops::rmsnorm(hidden[rank].view({kCfg.hidden, columns}),
                         *mtp_weights_for(rank).pre_fc_norm_hidden,
                         kCfg.rms_eps, true, normalized, stream);
        }
        if (input_rows == kCfg.hidden) {
            // The complete normalized plane is already contiguous and remains live
            // through FC. TP4 still materializes each strided half-width slice.
            inputs[rank] = normalized;
        } else {
            inputs[rank] = work[rank]->alloc(DType::BF16, {input_rows, columns});
            const int offset = (rank % embedding_ranks) * input_rows;
            const auto* source = static_cast<const std::uint16_t*>(normalized.data) + offset;
            CUDA_CHECK(cudaMemcpy2DAsync(inputs[rank].data, input_rows * sizeof(std::uint16_t),
                                         source, kCfg.hidden * sizeof(std::uint16_t),
                                         input_rows * sizeof(std::uint16_t), columns,
                                         cudaMemcpyDeviceToDevice, stream));
        }
        x[rank] = roots[rank].residual;
        ah[rank] = roots[rank].attention_hidden;
    });
    if (ec().device_count == 2) {
        // Each shard keeps its represented BF16 partial in the existing staging plane.
        // The collective retains the BF16 sum before evaluating the input norm.
        const auto partials = rank_map([&](int rank) {
            return staging[rank].view({kCfg.hidden, columns});
        });
        for_each_rank(ec(), [&](int rank) {
            Tensor partial = partials[rank];
            ops::linear(inputs[rank], weights[rank], partial, stream_for(rank));
        });
        const auto norms = rank_map([&](int rank) { return *mtp_weights_for(rank).input_norm; });
        const ops::ReplicatedRmsNorm norm{active(norms), active(ah), kCfg.rms_eps, true};
        ops::allreduce_sum_rmsnorm_to(active(partials), active(x), norm, ec(), *tp_[0].events);
    } else {
        ops::linear_row_parallel(active(inputs), active(weights), active(x), active(staging),
                                 ec(), *tp_[0].events);
        for_each_rank(ec(), [&](int rank) {
            ops::rmsnorm(x[rank], *mtp_weights_for(rank).input_norm, kCfg.rms_eps, true,
                         ah[rank], stream_for(rank));
        });
    }
}

void TextContext::mtp_forward_tail_tp2(std::array<Tensor, kMaximumExecutionDevices>& x, const std::array<Tensor, kMaximumExecutionDevices>& ah,
                                       const std::array<Tensor, kMaximumExecutionDevices>& positions,
                                       const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                                       ops::GqaExecutionEnvelope envelope,
                                       const std::array<Tensor, kMaximumExecutionDevices>& mtp_hidden,
                                       const std::array<Tensor, kMaximumExecutionDevices>& staging) {
    const ExecutionContext& execution       = ec();
    const std::array<WorkspaceArena*, kMaximumExecutionDevices> ws = workspaces();
    const int T                             = x[0].ne[1];

    std::array<Tensor, kMaximumExecutionDevices> q;
    std::array<Tensor, kMaximumExecutionDevices> k;
    std::array<Tensor, kMaximumExecutionDevices> gate;
    std::array<Tensor, kMaximumExecutionDevices> v;
    std::array<Tensor, kMaximumExecutionDevices> q_flat;
    std::array<Tensor, kMaximumExecutionDevices> gate_flat;
    std::array<Tensor, kMaximumExecutionDevices> k_flat;
    std::array<Tensor, kMaximumExecutionDevices> v_flat;
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        const auto projection =
            workspace_recipe::mtp_attention_projection<TextConfig>(*ws[r], T,
                                                                   ec().device_count);
        q[r]         = projection.query.view({kCfg.head_dim, (kCfg.n_q / ec().device_count), T});
        k[r]         = projection.key.view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), T});
        gate[r]      = projection.gate.view({kCfg.head_dim, (kCfg.n_q / ec().device_count), T});
        v[r]         = projection.value.view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), T});
        q_flat[r]    = q[r].view({(kCfg.q_size / ec().device_count), T});
        gate_flat[r] = gate[r].view({(kCfg.q_size / ec().device_count), T});
        k_flat[r]    = k[r].view({(kCfg.kv_size / ec().device_count), T});
        v_flat[r]    = v[r].view({(kCfg.kv_size / ec().device_count), T});
    }
    Variant::mtp_attention_projection(
        active(ah), active(rank_map([&](int rank) { return &mtp_weights_for(rank).payload->attention; })),
        active(q_flat), active(gate_flat), active(k_flat), active(v_flat), active(ws), execution);

    std::array<Tensor, kMaximumExecutionDevices> qn;
    std::array<Tensor, kMaximumExecutionDevices> kn;
    std::array<Tensor, kMaximumExecutionDevices> a;
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        const auto results =
            workspace_recipe::mtp_attention_results<TextConfig>(*ws[r], T, ec().device_count);
        qn[r] = results.normalized_query.view({kCfg.head_dim, (kCfg.n_q / ec().device_count), T});
        kn[r] = results.normalized_key.view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), T});
        a[r]  = results.attention.view({kCfg.head_dim, (kCfg.n_q / ec().device_count), T});
    }
    for_each_rank(execution, [&](int rank) {
        const auto r    = static_cast<std::size_t>(rank);
        cudaStream_t s  = stream_for(rank);
        const MtpW& mtp = mtp_weights_for(rank);
        ops::rmsnorm(q[r], *mtp.q_norm, kCfg.rms_eps, true, qn[r], s);
        ops::rmsnorm(k[r], *mtp.k_norm, kCfg.rms_eps, true, kn[r], s);
        Tensor rope_for_op =
            active_sequence_batch_ != 0 ? rope_positions[r].view({T}) : rope_positions[r];
        ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn[r], kn[r],
                  rope_frequency_[r], s);

        const qwen3_6::PagedKVCache& pages = rank == 0 ? *batch_mtp_kv_ : *tp_[rank - 1].batch_mtp_kv;
        if (active_sequence_batch_ != 0) {
            const std::int32_t width = active_sequence_width_;
            if (width <= 0 || width * active_sequence_batch_ != T) {
                throw std::logic_error("MTP sequence batch binding is incomplete");
            }
            Tensor q_batch =
                qn[r].view({kCfg.head_dim, (kCfg.n_q / ec().device_count), width, active_sequence_batch_});
            Tensor k_batch =
                kn[r].view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), width, active_sequence_batch_});
            Tensor v_batch =
                v[r].view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), width, active_sequence_batch_});
            Tensor a_batch =
                a[r].view({kCfg.head_dim, (kCfg.n_q / ec().device_count), width, active_sequence_batch_});
            Tensor position_batch = positions[r].view({width, active_sequence_batch_});
            ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, rank_valid_columns(rank),
                               rank_backend_kv_table_rows(rank), kAttnScale,
                               pages.batch_layer_view(0), attention_envelope(envelope), *ws[r], a_batch, s);
        } else {
            ops::gqa_attention(qn[r], kn[r], v[r], positions[r], Tensor{},
                               io_for(rank).backend_kv_table_row, kAttnScale,
                               pages.batch_layer_view(0), attention_envelope(envelope), *ws[r], a[r], s);
        }
        ops::sigmoid_mul(gate[r], a[r], s);
    });

    // The MTP output projection is `linear` + `residual_add`, not the fused `linear_add` the text
    // layers use -- the tp1 MTP leaf composes it the same way, because W8G32_F16S has no
    // linear_add profile. So the row-parallel split is `linear_row_parallel` + a replicated
    // per-rank `residual_add` over the all-reduced result.
    std::array<Tensor, kMaximumExecutionDevices> o;
    std::array<Tensor, kMaximumExecutionDevices> mh;
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        const auto post   = workspace_recipe::mtp_post_attention<TextConfig>(*ws[r], T);
        o[r]              = post.output;
        mh[r]             = post.post_mixer_hidden;
    }
    ops::linear_row_parallel(active(rank_map([&](int rank) { return a[rank].view({kCfg.q_size / ec().device_count, T}); })),
                             active(rank_map([&](int rank) { return *mtp_weights_for(rank).o_proj; })), active(o), active(staging),
                             execution, *tp_[0].events);
    for_each_rank(execution, [&](int rank) {
        const auto r   = static_cast<std::size_t>(rank);
        cudaStream_t s = stream_for(rank);
        ops::residual_add(o[r], x[r], s);
        ops::rmsnorm(x[r], *mtp_weights_for(rank).post_attn_norm, kCfg.rms_eps, true, mh[r], s);
    });

    {
        auto scopes = workspace_scopes();
        Variant::mtp_post_mixer(
            active(mh), active(rank_map([&](int rank) { return &mtp_weights_for(rank).payload->post_mixer; })),
            active(x), active(staging), active(ws), execution, *tp_[0].events);
    }

    for_each_rank(execution, [&](int rank) {
        const auto r           = static_cast<std::size_t>(rank);
        Tensor flat_mtp_hidden = mtp_hidden[r].view({kCfg.hidden, T});
        ops::rmsnorm(x[r], *mtp_weights_for(rank).norm, kCfg.rms_eps, true, flat_mtp_hidden,
                     stream_for(rank));
    });
}

void TextContext::mtp_forward_core_tp2(const Tensor& ids, const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                                       const std::array<Tensor, kMaximumExecutionDevices>& positions,
                                       const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                                       ops::GqaExecutionEnvelope envelope,
                                       const std::array<Tensor, kMaximumExecutionDevices>& mtp_hidden,
                                       const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr || std::any_of(tp_.begin(), tp_.end(), [](const auto& peer) { return peer.batch_mtp_kv == nullptr; })) {
        throw std::runtime_error("MTP forward is not enabled");
    }
    const std::array<WorkspaceArena*, kMaximumExecutionDevices> ws = workspaces();
    auto scope_all = workspace_scopes();
    const int T                             = ids.ne[0] * ids.ne[1];
    std::array<Tensor, kMaximumExecutionDevices> staging;
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        staging[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, T * (ec().device_count == 4 ? 4 : 1)});
    }
    std::array<Tensor, kMaximumExecutionDevices> x;
    std::array<Tensor, kMaximumExecutionDevices> ah;
    mtp_forward_stem_tp2(ids, hidden, x, ah, staging, input_embeddings);
    mtp_forward_tail_tp2(x, ah, positions, rope_positions, envelope, mtp_hidden, staging);
}

void TextContext::proposal_argmax_tp2(
    const std::array<Tensor, kMaximumExecutionDevices>& hidden,
    const std::array<Tensor, kMaximumExecutionDevices>&, Tensor& tokens) {
    const int columns = hidden[0].ne[1];
    const auto heads = rank_map([&](int rank) {
        if (proposal_head_ == nullptr) {
            return rank == 0 ? *lm_head_ : *lm_head_peer_[rank - 1];
        }
        const Weight* head = rank == 0 ? proposal_head_ : proposal_head_peer_[rank - 1];
        if (head == nullptr || head->n != proposal_head_n_) {
            throw std::logic_error("proposal head shards disagree on width");
        }
        return *head;
    });
    const int total = proposal_head_ == nullptr ? kCfg.token_domain : proposal_head_n_ * ec().device_count;
    const auto work = workspaces();
    auto scopes = workspace_scopes();
    const auto destinations = rank_map([&](int rank) {
        return rank == 0 ? tokens : work[rank]->alloc(DType::I32, {columns});
    });
    head_argmax_tp(hidden, heads, total, destinations, true);
}

void TextContext::target_verify_batch(const std::array<Tensor, kMaximumExecutionDevices>& ids,
                                      const std::array<Tensor, kMaximumExecutionDevices>& cache_positions,
                                      const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                                      const std::array<Tensor, kMaximumExecutionDevices>& valid_columns,
                                      const std::array<Tensor, kMaximumExecutionDevices>& kv_table_rows,
                                      const std::array<Tensor, kMaximumExecutionDevices>& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope,
                                      const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                                      const std::array<Tensor, kMaximumExecutionDevices>& logits,
                                      const std::array<Tensor, kMaximumExecutionDevices>& target_tokens,
                                      bool greedy_target) {
    if (!tp2()) { throw std::logic_error("tensor-parallel target verify requires a peer"); }
    const ExecutionContext& execution       = ec();
    const std::array<WorkspaceArena*, kMaximumExecutionDevices> ws = workspaces();
    const std::int32_t width                = ids[0].ne[0];
    const std::int32_t batch                = ids[0].ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    // Both ranks' extents are validated, not just rank 0's: rank 1's destinations live on the
    // other device, where an undersized buffer is a silent out-of-bounds write -- a defect of
    // exactly this class was once live here, invisible at batch 1 and out of bounds at batch > 1.
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        require_tensor_shape(ids[r], DType::I32, {width, batch}, "target verify batch ids");
        require_tensor_shape(cache_positions[r], DType::I32, {width, batch},
                             "target verify batch cache positions");
        require_tensor_shape(rope_positions[r], DType::I32, {width, batch},
                             "target verify batch RoPE positions");
        require_tensor_shape(valid_columns[r], DType::I32, {batch},
                             "target verify batch valid columns");
        require_tensor_shape(kv_table_rows[r], DType::I32, {batch}, "target verify batch KV rows");
        require_tensor_shape(linear_state_slots[r], DType::I32, {batch},
                             "target verify batch Linear Attention slots");
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, width, batch},
                             "target verify batch hidden");
        require_tensor_shape(logits[r], DType::BF16, {kCfg.vocab, width, batch},
                             "target verify batch logits");
        require_tensor_shape(target_tokens[r], DType::I32, {width, batch},
                             "target verify batch tokens");
    }

    work_.reset();
    for (const PeerExecution& peer : tp_) { peer.work->reset(); }
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions[0]);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions[0]);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows[0]);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_,
                                                 &linear_state_slots[0]);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns[0]);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
        // SYMMETRIC valid-column binding. `rank_valid_columns` throws if only one rank is bound,
        // precisely because two devices masking different columns would diverge silently; that
        // invariant is only exercised from here.
        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_cache_binding(peer_cache_positions_, peer_positions(cache_positions));
        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_rope_binding(peer_rope_positions_, peer_positions(rope_positions));
        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_rows_binding(peer_kv_table_rows_, peer_positions(kv_table_rows));
        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_slots_binding(peer_linear_state_slots_, peer_positions(linear_state_slots));
        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_valid_binding(peer_valid_columns_, peer_positions(valid_columns));

        auto scopes = workspace_scopes();
        std::array<Tensor, kMaximumExecutionDevices> x;
        std::array<Tensor, kMaximumExecutionDevices> staging;
        for (std::size_t r = 0; r < ec().device_count; ++r) {
            x[r]       = ws[r]->alloc(DType::BF16, {kCfg.hidden, columns});
            staging[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, columns * (ec().device_count == 4 ? 4 : 1)});
        }
        for_each_rank(execution, [&](int rank) {
            const auto r    = static_cast<std::size_t>(rank);
            Tensor flat_ids = ids[r].view({columns});
            ops::embedding(flat_ids, embedding_for(rank), x[r], stream_for(rank));
        });
        run_layers_tp2(x, Phase::Verify, staging);

        std::array<Tensor, kMaximumExecutionDevices> flat_hidden;
        std::array<Tensor, kMaximumExecutionDevices> flat_logits;
        for (std::size_t r = 0; r < ec().device_count; ++r) {
            flat_hidden[r] = hidden[r].view({kCfg.hidden, columns});
            flat_logits[r] = logits[r].view({kCfg.vocab, columns});
        }
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::rmsnorm(x[r], final_norm_for(rank), kCfg.rms_eps, true,
                         flat_hidden[r], stream_for(rank));
        });
        if (greedy_target) {
            target_argmax_tp2(flat_hidden,
                              rank_map([&](int rank) { return target_tokens[rank].view({columns}); }));
        } else {
            logits_tp2(flat_hidden, flat_logits);
            // The argmax is replicated, not rank 0's alone: both ranks hold the identical gathered
            // logits and rank 1 needs its own target tokens for acceptance without a control transfer.
            for_each_rank(execution, [&](int rank) {
                const auto r       = static_cast<std::size_t>(rank);
                Tensor flat_tokens = target_tokens[r].view({columns});
                constrain_logits(flat_logits[r], 0, kCfg.token_domain, stream_for(rank));
                ops::argmax(flat_logits[r], flat_tokens, kCfg.token_domain, stream_for(rank));
            });
        }
    }
    work_.reset();
    for (const PeerExecution& peer : tp_) { peer.work->reset(); }
}

void TextContext::target_verify_batch(const std::array<Tensor, kMaximumExecutionDevices>& ids,
                                      const std::array<Tensor, kMaximumExecutionDevices>& cache_positions,
                                      const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                                      const std::array<Tensor, kMaximumExecutionDevices>& valid_columns,
                                      const std::array<Tensor, kMaximumExecutionDevices>& kv_table_rows,
                                      const std::array<Tensor, kMaximumExecutionDevices>& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope,
                                      const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                                      const std::array<Tensor, kMaximumExecutionDevices>& logits,
                                      const std::array<Tensor, kMaximumExecutionDevices>& target_tokens,
                                      DFlashFeatureSink& sink, bool greedy_target) {
    if (!tp2()) { throw std::logic_error("tensor-parallel target verify requires a peer"); }
    const ExecutionContext& execution = ec();
    const std::array<WorkspaceArena*, kMaximumExecutionDevices> ws = workspaces();
    const std::int32_t width = ids[0].ne[0];
    const std::int32_t batch = ids[0].ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("tensor-parallel DFlash target verify shape is invalid");
    }
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        require_tensor_shape(ids[r], DType::I32, {width, batch}, "DFlash target verify ids");
        require_tensor_shape(cache_positions[r], DType::I32, {width, batch},
                             "DFlash target verify cache positions");
        require_tensor_shape(rope_positions[r], DType::I32, {width, batch},
                             "DFlash target verify rope positions");
        require_tensor_shape(valid_columns[r], DType::I32, {batch},
                             "DFlash target verify valid columns");
        require_tensor_shape(kv_table_rows[r], DType::I32, {batch},
                             "DFlash target verify KV rows");
        require_tensor_shape(linear_state_slots[r], DType::I32, {batch},
                             "DFlash target verify state slots");
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, width, batch},
                             "DFlash target verify hidden");
        require_tensor_shape(logits[r], DType::BF16, {kCfg.vocab, width, batch},
                             "DFlash target verify logits");
        require_tensor_shape(target_tokens[r], DType::I32, {width, batch},
                             "DFlash target verify tokens");
    }
    work_.reset();
    for (const PeerExecution& peer : tp_) { peer.work->reset(); }
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions[0]);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions[0]);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows[0]);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slots[0]);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns[0]);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_cache_binding(peer_cache_positions_, peer_positions(cache_positions));
        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_rope_binding(peer_rope_positions_, peer_positions(rope_positions));
        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_rows_binding(peer_kv_table_rows_, peer_positions(kv_table_rows));
        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_slots_binding(peer_linear_state_slots_, peer_positions(linear_state_slots));
        ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_valid_binding(peer_valid_columns_, peer_positions(valid_columns));
        auto scopes = workspace_scopes();
        std::array<Tensor, kMaximumExecutionDevices> x;
        std::array<Tensor, kMaximumExecutionDevices> staging;
        for (std::size_t r = 0; r < ec().device_count; ++r) {
            x[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, width * batch});
            staging[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, width * batch * (ec().device_count == 4 ? 4 : 1)});
        }
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::embedding(ids[r].view({width * batch}), embedding_for(rank),
                           x[r], stream_for(rank));
        });
        run_layers_tp2(x, Phase::Verify, staging, &sink);
        std::array<Tensor, kMaximumExecutionDevices> flat_hidden;
        std::array<Tensor, kMaximumExecutionDevices> flat_logits;
        for (std::size_t r = 0; r < ec().device_count; ++r) {
            flat_hidden[r] = hidden[r].view({kCfg.hidden, width * batch});
            flat_logits[r] = logits[r].view({kCfg.vocab, width * batch});
            for_each_rank(execution, [&](int rank) {
                if (static_cast<std::size_t>(rank) == r) {
                    ops::rmsnorm(x[r], final_norm_for(rank), kCfg.rms_eps,
                                 true, flat_hidden[r], stream_for(rank));
                }
            });
        }
        if (greedy_target) {
            target_argmax_tp2(flat_hidden,
                              rank_map([&](int rank) { return target_tokens[rank].view({width * batch}); }));
        } else {
            logits_tp2(flat_hidden, flat_logits);
            for_each_rank(execution, [&](int rank) {
                const auto r = static_cast<std::size_t>(rank);
                Tensor flat_tokens = target_tokens[r].view({width * batch});
                ops::argmax(flat_logits[r], flat_tokens, kCfg.token_domain, stream_for(rank));
            });
        }
    }
    work_.reset();
    for (const PeerExecution& peer : tp_) { peer.work->reset(); }
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids,
                                           const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                                           const std::array<Tensor, kMaximumExecutionDevices>& cache_positions,
                                           const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                                           const std::array<Tensor, kMaximumExecutionDevices>& valid_columns,
                                           const std::array<Tensor, kMaximumExecutionDevices>& kv_table_rows,
                                           ops::GqaExecutionEnvelope envelope,
                                           const std::array<Tensor, kMaximumExecutionDevices>& mtp_hidden) {
    if (!tp2()) { throw std::logic_error("tensor-parallel MTP decode requires a peer"); }
    if (batch_mtp_kv_ == nullptr || std::any_of(tp_.begin(), tp_.end(), [](const auto& peer) { return peer.batch_mtp_kv == nullptr; })) {
        throw std::runtime_error("MTP forward is not enabled");
    }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, width, batch},
                             "MTP decode batch target hidden");
        require_tensor_shape(cache_positions[r], DType::I32, {width, batch},
                             "MTP decode batch cache positions");
        require_tensor_shape(rope_positions[r], DType::I32, {width, batch},
                             "MTP decode batch RoPE positions");
        require_tensor_shape(valid_columns[r], DType::I32, {batch},
                             "MTP decode batch valid columns");
        require_tensor_shape(kv_table_rows[r], DType::I32, {batch}, "MTP decode batch KV rows");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {kCfg.hidden, width, batch},
                             "MTP decode batch hidden");
    }

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows[0]);
    ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_backend_binding(peer_backend_kv_table_rows_, peer_positions(kv_table_rows));
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns[0]);
    ScopedValue<std::array<const Tensor*, kMaximumExecutionDevices - 1>> peer_valid_binding(peer_valid_columns_, peer_positions(valid_columns));
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core_tp2(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_propose_batch(const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                                    const std::array<Tensor, kMaximumExecutionDevices>& logits, Tensor& draft_tokens) {
    if (!tp2()) { throw std::logic_error("tensor-parallel MTP proposal requires a peer"); }
    const std::int32_t batch = hidden[0].ne[1];
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, batch},
                             "MTP proposal batch hidden");
        require_tensor_shape(logits[r], DType::BF16, {kCfg.vocab, batch},
                             "MTP proposal batch logits");
    }
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    auto scopes = workspace_scopes();
    proposal_argmax_tp2(hidden, logits, draft_tokens);
}

void TextContext::mtp_forward_batch(const Tensor& ids, const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                                    const std::array<Tensor, kMaximumExecutionDevices>& positions,
                                    const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                                    ops::GqaExecutionEnvelope envelope,
                                    const std::array<Tensor, kMaximumExecutionDevices>& mtp_hidden, int logits_column,
                                    const std::array<Tensor, kMaximumExecutionDevices>* logits, Tensor* draft_token,
                                    const Tensor* input_embeddings) {
    if (!tp2()) { throw std::logic_error("tensor-parallel MTP batch requires a peer"); }
    if (batch_mtp_kv_ == nullptr || std::any_of(tp_.begin(), tp_.end(), [](const auto& peer) { return peer.batch_mtp_kv == nullptr; })) {
        throw std::runtime_error("MTP forward is not enabled");
    }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        require_tensor_shape(positions[r], DType::I32, {T}, "MTP positions");
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, T}, "MTP hidden");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
        const Tensor& rope = rope_positions[r];
        if (rope.dtype != DType::I32 || rope.ne[0] != T ||
            (rope.ne[1] != 1 && rope.ne[1] != 3) || rope.ne[2] != 1 || rope.ne[3] != 1 ||
            !rope.is_contiguous() || rope.data == nullptr) {
            throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
        }
    }
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        for (const Tensor& destination : std::span(*logits).first(ec().device_count)) {
            require_tensor_shape(destination, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        }
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    mtp_forward_core_tp2(ids, hidden, positions, rope_positions, envelope, mtp_hidden,
                         input_embeddings);
    if (logits_column >= 0) {
        const auto columns = rank_map([&](int rank) { return mtp_hidden[rank].slice(1, logits_column, 1); });
        proposal_argmax_tp2(columns, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token,
                                      const std::array<Tensor, kMaximumExecutionDevices>& previous_hidden,
                                      const std::array<Tensor, kMaximumExecutionDevices>& position,
                                      ops::GqaExecutionEnvelope envelope,
                                      const std::array<Tensor, kMaximumExecutionDevices>& mtp_hidden,
                                      const std::array<Tensor, kMaximumExecutionDevices>& logits, Tensor& draft_token) {
    if (!tp2()) { throw std::logic_error("tensor-parallel MTP AR step requires a peer"); }
    if (batch_mtp_kv_ == nullptr || std::any_of(tp_.begin(), tp_.end(), [](const auto& peer) { return peer.batch_mtp_kv == nullptr; })) {
        throw std::runtime_error("MTP forward is not enabled");
    }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        require_tensor_shape(position[r], DType::I32, {1}, "MTP AR position");
        require_tensor_shape(previous_hidden[r], DType::BF16, {kCfg.hidden, 1},
                             "MTP AR previous hidden");
        require_tensor_shape(mtp_hidden[r], DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
        require_tensor_shape(logits[r], DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    }
    const ExecutionContext& execution       = ec();
    const std::array<WorkspaceArena*, kMaximumExecutionDevices> ws = workspaces();
    auto position_scope_all = workspace_scopes();
    std::array<Tensor, kMaximumExecutionDevices> rope_position;
    for (std::size_t r = 0; r < ec().device_count; ++r) { rope_position[r] = ws[r]->alloc(DType::I32, {1}); }
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::offset_i32_positions(position[r], io_for(rank).rope_delta, rope_position[r],
                                  stream_for(rank));
    });
    mtp_forward_core_tp2(token, previous_hidden, position, rope_position, envelope, mtp_hidden);
    auto logits_scope_all = workspace_scopes();
    proposal_argmax_tp2(mtp_hidden, logits, draft_token);
}

void TextContext::mtp_prefill_chunk_tp2(const Tensor& ids, const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                                        const std::array<Tensor, kMaximumExecutionDevices>& positions,
                                        const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                                        ops::GqaExecutionEnvelope envelope, bool final_chunk,
                                        const std::array<Tensor, kMaximumExecutionDevices>* final_hidden,
                                        const std::array<Tensor, kMaximumExecutionDevices>* logits,
                                        Tensor* draft_token,
                                        const Tensor* input_embeddings) {
    if (!mtp_kv_.valid() || std::any_of(tp_.begin(), tp_.end(),
                                     [](const auto& peer) { return !peer.mtp_kv.valid(); })) {
        throw std::runtime_error("MTP prefill is not enabled");
    }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        require_tensor_shape(hidden[r], DType::BF16, {kCfg.hidden, T}, "MTP prefill hidden");
        require_tensor_shape(positions[r], DType::I32, {T}, "MTP prefill positions");
        const Tensor& rope = rope_positions[r];
        if (rope.dtype != DType::I32 || rope.ne[0] != T ||
            (rope.ne[1] != 1 && rope.ne[1] != 3) || rope.ne[2] != 1 || rope.ne[3] != 1 ||
            !rope.is_contiguous() || rope.data == nullptr) {
            throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
        }
    }
    if (final_chunk && (final_hidden == nullptr || logits == nullptr || draft_token == nullptr)) {
        throw std::invalid_argument("MTP final prefill outputs are required");
    }

    const ExecutionContext& execution       = ec();
    const std::array<WorkspaceArena*, kMaximumExecutionDevices> ws = workspaces();
    auto scratch_scope_all = workspace_scopes();
    std::array<Tensor, kMaximumExecutionDevices> staging;
    std::array<Tensor, kMaximumExecutionDevices> last_staging;
    std::array<Tensor, kMaximumExecutionDevices> x_last;
    std::array<Tensor, kMaximumExecutionDevices> ah_last;
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        staging[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, T * (ec().device_count == 4 ? 4 : 1)});
        if (final_chunk) {
            // The final-chunk stage reduces ONE column, and `linear_row_parallel` requires the
            // staging buffer to match its output's shape exactly on both devices -- the chunk's
            // [hidden, T] staging would be the wrong shape for it.
            last_staging[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, 1 * (ec().device_count == 4 ? 4 : 1)});
            x_last[r]       = ws[r]->alloc(DType::BF16, {kCfg.hidden, 1});
            ah_last[r]      = ws[r]->alloc(DType::BF16, {kCfg.hidden, 1});
        }
    }

    {
        auto bulk_scope_all = workspace_scopes();
        std::array<Tensor, kMaximumExecutionDevices> x;
        std::array<Tensor, kMaximumExecutionDevices> ah;
        mtp_forward_stem_tp2(ids, hidden, x, ah, staging, input_embeddings);

        std::array<Tensor, kMaximumExecutionDevices> k_flat;
        std::array<Tensor, kMaximumExecutionDevices> v_flat;
        for (std::size_t r = 0; r < ec().device_count; ++r) {
            k_flat[r] = ws[r]->alloc(DType::BF16, {(kCfg.kv_size / ec().device_count), T});
            v_flat[r] = ws[r]->alloc(DType::BF16, {(kCfg.kv_size / ec().device_count), T});
        }
        Variant::mtp_kv_projection(
            active(ah), active(rank_map([&](int rank) { return &mtp_weights_for(rank).payload->attention; })),
            active(k_flat), active(v_flat), active(ws), execution);
        for_each_rank(execution, [&](int rank) {
            const auto r    = static_cast<std::size_t>(rank);
            cudaStream_t s  = stream_for(rank);
            const MtpW& mtp = mtp_weights_for(rank);
            Tensor k        = k_flat[r].view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), T});
            Tensor v        = v_flat[r].view({kCfg.head_dim, (kCfg.n_kv / ec().device_count), T});
            Tensor kn       = ws[r]->alloc(DType::BF16, {kCfg.head_dim, (kCfg.n_kv / ec().device_count), T});
            ops::rmsnorm(k, *mtp.k_norm, kCfg.rms_eps, true, kn, s);
            ops::rope(rope_positions[r], kCfg.rotary_dim, kCfg.rope_theta, kn,
                      rope_frequency_[r], s);
            qwen3_6::PagedKVCacheView pages = rank == 0 ? mtp_kv_ : tp_[rank - 1].mtp_kv;
            ops::gqa_kv_append(kn, v, positions[r], pages.layer_view(0), s);
            if (final_chunk) {
                const std::size_t column_bytes =
                    static_cast<std::size_t>(kCfg.hidden) * dtype_size(DType::BF16);
                const auto* x_src = static_cast<const unsigned char*>(x[r].data) +
                                    static_cast<std::size_t>(T - 1) * column_bytes;
                const auto* ah_src = static_cast<const unsigned char*>(ah[r].data) +
                                     static_cast<std::size_t>(T - 1) * column_bytes;
                CUDA_CHECK(cudaMemcpyAsync(x_last[r].data, x_src, column_bytes,
                                           cudaMemcpyDeviceToDevice, s));
                CUDA_CHECK(cudaMemcpyAsync(ah_last[r].data, ah_src, column_bytes,
                                           cudaMemcpyDeviceToDevice, s));
            }
        });
    }

    if (!final_chunk) { return; }

    std::array<Tensor, kMaximumExecutionDevices> q_flat;
    std::array<Tensor, kMaximumExecutionDevices> gate_flat;
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        q_flat[r]    = ws[r]->alloc(DType::BF16, {(kCfg.q_size / ec().device_count), 1});
        gate_flat[r] = ws[r]->alloc(DType::BF16, {(kCfg.q_size / ec().device_count), 1});
    }
    Variant::mtp_q_gate_projection(
        active(ah_last), active(rank_map([&](int rank) { return &mtp_weights_for(rank).payload->attention; })),
        active(q_flat), active(gate_flat), active(ws), execution);
    std::array<Tensor, kMaximumExecutionDevices> a;
    std::array<Tensor, kMaximumExecutionDevices> o;
    std::array<Tensor, kMaximumExecutionDevices> mh;
    for (std::size_t r = 0; r < ec().device_count; ++r) {
        a[r]  = ws[r]->alloc(DType::BF16, {kCfg.head_dim, (kCfg.n_q / ec().device_count), 1});
        o[r]  = ws[r]->alloc(DType::BF16, {kCfg.hidden, 1});
        mh[r] = ws[r]->alloc(DType::BF16, {kCfg.hidden, 1});
    }
    for_each_rank(execution, [&](int rank) {
        const auto r    = static_cast<std::size_t>(rank);
        cudaStream_t s  = stream_for(rank);
        const MtpW& mtp = mtp_weights_for(rank);
        Tensor q        = q_flat[r].view({kCfg.head_dim, (kCfg.n_q / ec().device_count), 1});
        Tensor gate     = gate_flat[r].view({kCfg.head_dim, (kCfg.n_q / ec().device_count), 1});
        Tensor qn       = ws[r]->alloc(DType::BF16, {kCfg.head_dim, (kCfg.n_q / ec().device_count), 1});
        ops::rmsnorm(q, *mtp.q_norm, kCfg.rms_eps, true, qn, s);
        Tensor last_position = positions[r].slice(0, T - 1, 1);
        Tensor last_rope;
        if (rope_positions[r].ne[1] == 1) {
            last_rope = rope_positions[r].slice(0, T - 1, 1);
        } else {
            last_rope = ws[r]->alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(last_rope.data) + axis,
                                           static_cast<const std::int32_t*>(rope_positions[r].data) +
                                               axis * T + T - 1,
                                           sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        ops::rope(last_rope, kCfg.rotary_dim, kCfg.rope_theta, qn, rope_frequency_[r], s);
        qwen3_6::PagedKVCacheView pages = rank == 0 ? mtp_kv_ : tp_[rank - 1].mtp_kv;
        ops::gqa_attention_cached(qn, last_position, kAttnScale, pages.layer_view(0), attention_envelope(envelope),
                                  *ws[r], a[r], s);
        ops::sigmoid_mul(gate, a[r], s);
    });
    ops::linear_row_parallel(active(rank_map([&](int rank) { return a[rank].view({kCfg.q_size / ec().device_count, 1}); })),
                             active(rank_map([&](int rank) { return *mtp_weights_for(rank).o_proj; })), active(o),
                             active(last_staging), execution, *tp_[0].events);
    for_each_rank(execution, [&](int rank) {
        const auto r   = static_cast<std::size_t>(rank);
        cudaStream_t s = stream_for(rank);
        ops::residual_add(o[r], x_last[r], s);
        ops::rmsnorm(x_last[r], *mtp_weights_for(rank).post_attn_norm, kCfg.rms_eps, true, mh[r],
                     s);
    });
    {
        auto post_scope_all = workspace_scopes();
        Variant::mtp_post_mixer(
            active(mh), active(rank_map([&](int rank) { return &mtp_weights_for(rank).payload->post_mixer; })),
            x_last, active(last_staging), active(ws), execution, *tp_[0].events);
    }
    for_each_rank(execution, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        require_tensor_shape((*final_hidden)[r], DType::BF16, {kCfg.hidden, 1},
                             "MTP final prefill hidden");
        ops::rmsnorm(x_last[r], *mtp_weights_for(rank).norm, kCfg.rms_eps, true,
                     const_cast<Tensor&>((*final_hidden)[r]), stream_for(rank));
    });
    proposal_argmax_tp2(*final_hidden, *logits, *draft_token);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
