#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/allreduce.h"
#include "core/gdn_replay_records.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/rope.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/round_state.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

// Target-private compatibility vocabulary for the mechanically preserved fixed schedule. It is
// data-only: TextContext is constructed on the stack for one schedule recording/execution and owns
// neither weights nor device state.
struct ModelConfig {
    static constexpr int hidden              = TextConfig::hidden;
    static constexpr int n_layers            = TextConfig::layers;
    static constexpr int intermediate        = TextConfig::intermediate;
    static constexpr int vocab               = TextConfig::output_rows;
    static constexpr int token_domain        = TextConfig::token_domain;
    static constexpr int gdn_k_heads         = TextConfig::gdn_key_heads;
    static constexpr int gdn_k_dim           = TextConfig::gdn_key_head_dim;
    static constexpr int gdn_v_heads         = TextConfig::gdn_value_heads;
    static constexpr int gdn_v_dim           = TextConfig::gdn_value_head_dim;
    static constexpr int n_q                 = TextConfig::query_heads;
    static constexpr int n_kv                = TextConfig::kv_heads;
    static constexpr int head_dim            = TextConfig::head_dim;
    static constexpr int rotary_dim          = TextConfig::rotary_dim;
    static constexpr int key_dim             = TextConfig::key_dim;
    static constexpr int value_dim           = TextConfig::value_dim;
    static constexpr int conv_dim            = TextConfig::convolution_dim;
    static constexpr int q_size              = TextConfig::query_size;
    static constexpr int kv_size             = TextConfig::kv_size;
    static constexpr int mtp_fc_in           = TextConfig::mtp_input_rows;
    static constexpr int mtp_attn_in         = TextConfig::mtp_attention_input_rows;
    static constexpr int mtp_mlp_gateup_rows = TextConfig::mtp_mlp_gate_up_rows;
    static constexpr float rms_eps           = TextConfig::rms_epsilon;
    static constexpr float rope_theta        = TextConfig::rope_theta;
    static constexpr int mtp_layers          = TextConfig::mtp_layers;

    [[nodiscard]] static constexpr bool is_full(int layer) {
        return TextConfig::is_full_attention(layer);
    }

    [[nodiscard]] static constexpr int n_full() { return TextConfig::full_attention_layers(); }

    [[nodiscard]] static constexpr int n_gdn() { return TextConfig::gdn_layers(); }

    [[nodiscard]] static constexpr int full_idx(int layer) {
        return TextConfig::full_attention_index(layer);
    }

    [[nodiscard]] static constexpr int gdn_idx(int layer) { return TextConfig::gdn_index(layer); }
};

inline constexpr ModelConfig kCfg{};

inline constexpr float kAttnScale                     = kAttentionScale;
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;

struct MlpW {
    const MlpWeights* payload = nullptr;
};

struct FullLayerW {
    const Tensor* input_norm                         = nullptr;
    const FullAttentionProjectionWeights* projection = nullptr;
    const Weight* o_proj                             = nullptr;
    const Tensor* q_norm                             = nullptr;
    const Tensor* k_norm                             = nullptr;
    const Tensor* post_attn_norm                     = nullptr;
    MlpW mlp;
};

struct GdnLayerW {
    const Tensor* input_norm               = nullptr;
    const GdnProjectionWeights* projection = nullptr;
    const Tensor* conv1d                   = nullptr;
    const Tensor* gdn_norm                 = nullptr;
    const Weight* out_proj                 = nullptr;
    const Tensor* post_attn_norm           = nullptr;
    MlpW mlp;
};

struct MtpW {
    const MtpWeights* payload           = nullptr;
    const Weight* fc                    = nullptr;
    const Tensor* pre_fc_norm_embedding = nullptr;
    const Tensor* pre_fc_norm_hidden    = nullptr;
    const Tensor* input_norm            = nullptr;
    const Tensor* q_norm                = nullptr;
    const Tensor* k_norm                = nullptr;
    const Weight* o_proj                = nullptr;
    const Tensor* post_attn_norm        = nullptr;
    const Tensor* norm                  = nullptr;
};

using Phase = qwen3_6::TextPhase;

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    std::uint32_t tail_column       = 0;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const int> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;

// Everything a TextContext needs to drive a SECOND device in lockstep with the first. Absent (a
// null pointer) at tp == 1, in which case not one line of the single-device schedule changes.
//
// The rank-1 members mirror the rank-0 constructor arguments one for one: rank 1 has its own
// device/stream, its own SHARD of the weights, its own workspace arena, its own halved GDN state
// pool, its own RoundState mirror, and its own half of the paged KV cache. What it does NOT have
// is its own bookkeeping: page allocation, lane state, sampling and the host round buffers all
// live once, on rank 0, and rank 1's pools are driven through the identical call sequence so its
// block tables match rank 0's by construction.
struct PeerExecution {
    const ExecutionContext* execution = nullptr;
    const ops::PeerEvents* events     = nullptr;
    DeviceContext* device             = nullptr;
    const LoadedModelData* weights    = nullptr;
    WorkspaceArena* work              = nullptr;
    LinearAttentionStatePool* state   = nullptr;
    qwen3_6::RoundState* io           = nullptr;
    Tensor* prefill_hidden            = nullptr;
    qwen3_6::PagedKVCacheView kv;
    const qwen3_6::PagedKVCache* batch_kv = nullptr;
    // Rank 1's own MTP KV pool and replay-record storage. Both are absent unless the sequence
    // plan enables MTP; when present they mirror rank 0's exactly, with the head/channel extents
    // halved, so the two devices' MTP pages and GDN records stay in lockstep by construction.
    qwen3_6::PagedKVCacheView mtp_kv;
    const qwen3_6::PagedKVCache* batch_mtp_kv = nullptr;
    const GdnReplayRecords* replay_records    = nullptr;

    [[nodiscard]] bool complete() const noexcept {
        return execution != nullptr && events != nullptr && device != nullptr &&
               weights != nullptr && work != nullptr && state != nullptr && io != nullptr;
    }
};

// Structured-output constraint rows in mapped page-locked host memory, readable by every rank.
// Row r applies to logit column r of the selection that binds the view.
struct TokenConstraintView {
    const std::int32_t* constrained = nullptr; // [rows]
    const std::uint32_t* bitmask    = nullptr; // [rows, words]
    std::int32_t words              = 0;

    [[nodiscard]] TokenConstraintView from_row(std::int32_t row) const noexcept {
        return {constrained + row, bitmask + static_cast<std::size_t>(row) * words, words};
    }

    [[nodiscard]] explicit operator bool() const noexcept { return constrained != nullptr; }
};

class TextContext {
public:
    // `rope_frequency` is the per-RANK YaRN rotary override (see `rope_frequency_` below). It has
    // no default: a new construction site must state which tables it ropes with, because a
    // silently-defaulted null descriptor is a correct-looking native run at extended positions.
    // `ExecutionCore::rope_frequency` is what every production caller passes.
    TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                const std::array<ops::RopeFrequencyOverride, kMaximumExecutionDevices>& rope_frequency,
                qwen3_6::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_6::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_6::PagedKVCacheView mtp_kv           = qwen3_6::PagedKVCacheView(),
                const qwen3_6::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_6::PagedKVCache* batch_mtp_kv  = nullptr,
                std::span<const PeerExecution> tp = {});
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    // Rank 1's half is bound in bind() and cleared here together with rank 0's: the only caller
    // that passes a null weight is configure_text_card's "this request uses the full LM head"
    // path, and leaving the peer bound while rank 0 is not would have the two ranks take
    // different proposal branches.
    void set_proposal_head(const Weight* weight, const std::int32_t* ids, int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
        if (weight == nullptr) {
            proposal_head_peer_.fill(nullptr);
            proposal_head_ids_peer_.fill(nullptr);
        }
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }
    void set_volta_long_attention(bool enabled) noexcept { volta_long_attention_ = enabled; }
    // Applies to target-token selection only: prefill's final token and MTP verification.
    void set_token_constraints(TokenConstraintView view) noexcept { token_constraints_ = view; }
    void set_rope_delta(std::int32_t delta) noexcept { rope_delta_ = delta; }
    void set_retrieval_query_capture(bool enabled) noexcept { capture_retrieval_query_ = enabled; }
    void set_pipeline(std::optional<PeerExecution> stage, TokenId* upload = nullptr) {
        pipeline_ = std::move(stage);
        pipeline_upload_ = upload;
    }

    void set_prefill_rewrite_checkpoint_frontier(std::int64_t position) noexcept {
        prefill_rewrite_checkpoint_frontier_ = position;
    }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    void set_linear_state_slots(std::int32_t current_slot, std::int32_t rewrite_checkpoint_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    [[nodiscard]] const Weight* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_6::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_slots, ops::GqaExecutionEnvelope envelope,
                               Tensor& hidden, Tensor& logits);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                             ops::GqaExecutionEnvelope envelope, Tensor& hidden, Tensor& logits,
                             Tensor& target_tokens, bool greedy_target = false);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                             ops::GqaExecutionEnvelope envelope, Tensor& hidden, Tensor& logits,
                             Tensor& target_tokens, DFlashFeatureSink& sink,
                             bool greedy_target = false);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::GqaExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);

    // --- tp == 2 entry points ------------------------------------------------------------------
    //
    // Every per-rank tensor is passed explicitly as an array rather than derived here, for the
    // same reason `logits_tp2` takes both destinations: there is no single "the frame", and an
    // undersized peer destination is an out-of-bounds write on the OTHER device that nothing local
    // would notice. The schedule owns both frames and is the only place that can name them
    // correctly.
    //
    // `ids` and the draft-token outputs stay single tensors: they belong to rank 0 alone, because
    // rank 1's MTP stem contracts the normalized-hidden half of the fc input and never embeds a
    // token, and the proposal leaves through rank 0's egress.
    // Computes the full target vocabulary on both ranks from replicated final-normalized hidden.
    // The caller owns the hidden replicas, output storage, and subsequent sampling.
    void target_logits(const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                        const std::array<Tensor, kMaximumExecutionDevices>& logits);
    void target_verify_batch(const std::array<Tensor, kMaximumExecutionDevices>& ids,
                             const std::array<Tensor, kMaximumExecutionDevices>& cache_positions,
                             const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                             const std::array<Tensor, kMaximumExecutionDevices>& valid_columns,
                             const std::array<Tensor, kMaximumExecutionDevices>& kv_table_rows,
                             const std::array<Tensor, kMaximumExecutionDevices>& linear_state_slots,
                             ops::GqaExecutionEnvelope envelope,
                             const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                             const std::array<Tensor, kMaximumExecutionDevices>& logits,
                             const std::array<Tensor, kMaximumExecutionDevices>& target_tokens,
                             bool greedy_target = false);
    void target_verify_batch(const std::array<Tensor, kMaximumExecutionDevices>& ids,
                             const std::array<Tensor, kMaximumExecutionDevices>& cache_positions,
                             const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                             const std::array<Tensor, kMaximumExecutionDevices>& valid_columns,
                             const std::array<Tensor, kMaximumExecutionDevices>& kv_table_rows,
                             const std::array<Tensor, kMaximumExecutionDevices>& linear_state_slots,
                             ops::GqaExecutionEnvelope envelope,
                             const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                             const std::array<Tensor, kMaximumExecutionDevices>& logits,
                             const std::array<Tensor, kMaximumExecutionDevices>& target_tokens,
                             DFlashFeatureSink& sink, bool greedy_target = false);
    // Greedy TP2 verification can reduce each vocabulary shard to one exact (value, id) winner,
    // exchange two tiny [1,T] tensors, and merge them. This avoids the full [V,T] all-gather; the
    // full-logit overload above remains the route for temperature/penalty sampling.
    void target_argmax_tp2(const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                           const std::array<Tensor, kMaximumExecutionDevices>& target_tokens);
    void mtp_forward_decode_batch(const Tensor& ids, const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                                  const std::array<Tensor, kMaximumExecutionDevices>& cache_positions,
                                  const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                                  const std::array<Tensor, kMaximumExecutionDevices>& valid_columns,
                                  const std::array<Tensor, kMaximumExecutionDevices>& kv_table_rows,
                                  ops::GqaExecutionEnvelope envelope,
                                  const std::array<Tensor, kMaximumExecutionDevices>& mtp_hidden);
    void mtp_propose_batch(const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                           const std::array<Tensor, kMaximumExecutionDevices>& logits, Tensor& draft_tokens);
    void mtp_forward_batch(const Tensor& ids, const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                           const std::array<Tensor, kMaximumExecutionDevices>& positions,
                           const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                           ops::GqaExecutionEnvelope envelope,
                           const std::array<Tensor, kMaximumExecutionDevices>& mtp_hidden, int logits_column,
                           const std::array<Tensor, kMaximumExecutionDevices>* logits, Tensor* draft_token,
                           const Tensor* input_embeddings = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const std::array<Tensor, kMaximumExecutionDevices>& previous_hidden,
                             const std::array<Tensor, kMaximumExecutionDevices>& position,
                             ops::GqaExecutionEnvelope envelope,
                             const std::array<Tensor, kMaximumExecutionDevices>& mtp_hidden,
                             const std::array<Tensor, kMaximumExecutionDevices>& logits, Tensor& draft_token);
private:
    void bind();

    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    [[nodiscard]] const MtpW& mtp_weights() const;
    void attn_mix(const FullLayerW& weights, Tensor& x, int index, Phase phase);
    void gdn_mix(const GdnLayerW& weights, Tensor& x, int index, Phase phase);
    void mlp_tail(const Tensor* post_norm, const MlpW& weights, Tensor& x, Phase phase);
    void run_layers(Tensor& x, Phase phase);
    void run_pipeline_stage(Tensor& x, Phase phase);
    void pipeline_prefill_layers(Tensor& x, const Tensor& positions,
                                  ops::GqaExecutionEnvelope envelope, std::int32_t begin);

    // --- tp == 2 forward -----------------------------------------------------------------------
    // Deliberately separate functions rather than branches inside the tp1 ones: the tp1 schedule
    // stays byte-identical, and the split schedule reads as the linear sequence it is. The
    // residual `x` is REPLICATED (bitwise identical on both ranks -- the reduce sums the same two
    // BF16 partials on both sides and IEEE addition is commutative), which is what keeps every
    // per-device GDN state and KV page in lockstep without any extra synchronization.
    [[nodiscard]] bool tp2() const noexcept { return !tp_.empty(); }
    [[nodiscard]] const ExecutionContext& ec() const;
    [[nodiscard]] std::array<WorkspaceArena*, kMaximumExecutionDevices> workspaces() const;
    [[nodiscard]] std::array<std::optional<WorkspaceArena::Scope>, kMaximumExecutionDevices>
    workspace_scopes() const;
    template <class Body>
    auto rank_map(Body&& body) const {
        using Value = std::decay_t<decltype(body(0))>;
        std::array<Value, kMaximumExecutionDevices> out{};
        for (int rank = 0; rank < ec().device_count; ++rank) { out[rank] = body(rank); }
        return out;
    }
    template <class T, std::size_t N>
    auto active(const std::array<T, N>& values) const {
        return std::span(values).first(ec().device_count);
    }
    [[nodiscard]] std::array<const Tensor*, kMaximumExecutionDevices - 1>
    peer_positions(const std::array<Tensor, kMaximumExecutionDevices>& tensors) const {
        std::array<const Tensor*, kMaximumExecutionDevices - 1> out{};
        for (std::size_t index = 0; index < tp_.size(); ++index) { out[index] = &tensors[index + 1]; }
        return out;
    }
    [[nodiscard]] const Weight& embedding_for(int rank) const {
        return rank == 0 ? *embed_ : *embed_peer_[rank - 1];
    }
    [[nodiscard]] const Tensor& final_norm_for(int rank) const {
        return rank == 0 ? *final_norm_ : *final_norm_peer_[rank - 1];
    }
    [[nodiscard]] cudaStream_t stream_for(int rank) const noexcept {
        return rank == 0 ? ctx_.stream : tp_[rank - 1].device->stream;
    }
    [[nodiscard]] qwen3_6::RoundState& io_for(int rank) const noexcept {
        return rank == 0 ? io_ : *tp_[rank - 1].io;
    }
    [[nodiscard]] LinearAttentionStatePool& state_for(int rank) const noexcept {
        return rank == 0 ? state_ : *tp_[rank - 1].state;
    }
    // Rank 1's own device copies of the per-call I32 control tensors. Rank 0 keeps using the
    // existing `active_*` bindings unchanged; these are their mirrors, set by the same call that
    // sets those. A null peer binding means "the same thing rank 0 has", which is only ever valid
    // for the empty Tensor{} cases.
    [[nodiscard]] const Tensor& rank_cache_positions(int rank) const;
    [[nodiscard]] const Tensor& rank_rope_positions(int rank) const;
    [[nodiscard]] const Tensor& rank_kv_table_rows(int rank) const;
    [[nodiscard]] const Tensor& rank_backend_kv_table_rows(int rank) const;
    [[nodiscard]] Tensor rank_valid_columns(int rank) const;
    [[nodiscard]] const Tensor& rank_linear_state_slots(int rank) const;
    void synchronize_all() const;
    void attn_mix_tp2(const std::array<const FullLayerW*, kMaximumExecutionDevices>& w, std::array<Tensor, kMaximumExecutionDevices>& x,
                      int index, Phase phase, const std::array<Tensor, kMaximumExecutionDevices>& staging,
                      const ops::ReplicatedRmsNorm* norm = nullptr,
                       const std::array<Tensor, kMaximumExecutionDevices>* normalized_input = nullptr);
    void gdn_mix_tp2(const std::array<const GdnLayerW*, kMaximumExecutionDevices>& w, std::array<Tensor, kMaximumExecutionDevices>& x, int index,
                     Phase phase, const std::array<Tensor, kMaximumExecutionDevices>& staging,
                      const ops::ReplicatedRmsNorm* norm = nullptr,
                       const std::array<Tensor, kMaximumExecutionDevices>* normalized_input = nullptr);
    void mlp_tail_tp2(const std::array<const Tensor*, kMaximumExecutionDevices>& norm,
                      const std::array<const MlpW*, kMaximumExecutionDevices>& mlp, std::array<Tensor, kMaximumExecutionDevices>& x, Phase phase,
                      const std::array<Tensor, kMaximumExecutionDevices>& staging,
                      const std::array<Tensor, kMaximumExecutionDevices>* normalized = nullptr,
                       const ops::ReplicatedRmsNorm* next_input_norm = nullptr);
    void run_layers_tp2(std::array<Tensor, kMaximumExecutionDevices>& x, Phase phase,
                        const std::array<Tensor, kMaximumExecutionDevices>& staging,
                        DFlashFeatureSink* dflash_sink = nullptr);
    // Vocabulary-split head: each rank computes its own half of the logits, then one allgather
    // per column leaves the FULL logits on both ranks. Sampling then runs on rank 0 alone.
    void logits_tp2(const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                    const std::array<Tensor, kMaximumExecutionDevices>& logits);
    void head_argmax_tp(const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                        const std::array<Weight, kMaximumExecutionDevices>& heads,
                        std::int32_t valid_rows,
                        const std::array<Tensor, kMaximumExecutionDevices>& tokens, bool proposal = false);
    void ordinary_decode_batch_tp2(const Tensor& ids, const Tensor& cache_positions,
                                   const Tensor& rope_positions, const Tensor& kv_table_rows,
                                   const Tensor& linear_state_slots,
                                   ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                   Tensor& logits);
    // --- tp == 2 MTP -----------------------------------------------------------------------
    // The MTP head is one layer, so its split schedule is the text layer's split schedule with a
    // single instance of each stage: a ROW-parallel stem (no pack -- each rank's K-slice already
    // IS one of the two normalized halves), a COLUMN-parallel packed attention projection with
    // head-local 12|2 attention over this device's own MTP KV pages, a ROW-parallel output
    // projection, and the MTP post-mixer's column/row-parallel pair. `staging` is one
    // [hidden, T] scratch per rank, allocated once per MTP call and reused by all three of the
    // call's reduces exactly as the text layer loop reuses its own.
    // `ids` is rank 0's alone: rank 0's fc shard contracts the NORMALIZED EMBEDDING half and
    // rank 1's the NORMALIZED HIDDEN half, so device 1 never embeds a token in the MTP stem.
    void mtp_forward_stem_tp2(const Tensor& ids, const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                              std::array<Tensor, kMaximumExecutionDevices>& x, std::array<Tensor, kMaximumExecutionDevices>& ah,
                              const std::array<Tensor, kMaximumExecutionDevices>& staging,
                              const Tensor* input_embeddings = nullptr);
    void mtp_forward_tail_tp2(std::array<Tensor, kMaximumExecutionDevices>& x, const std::array<Tensor, kMaximumExecutionDevices>& ah,
                              const std::array<Tensor, kMaximumExecutionDevices>& positions,
                              const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                              ops::GqaExecutionEnvelope envelope,
                              const std::array<Tensor, kMaximumExecutionDevices>& mtp_hidden,
                              const std::array<Tensor, kMaximumExecutionDevices>& staging);
    void mtp_forward_core_tp2(const Tensor& ids, const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                              const std::array<Tensor, kMaximumExecutionDevices>& positions,
                              const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                              ops::GqaExecutionEnvelope envelope,
                              const std::array<Tensor, kMaximumExecutionDevices>& mtp_hidden,
                              const Tensor* input_embeddings = nullptr);
    void mtp_prefill_chunk_tp2(const Tensor& ids, const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                               const std::array<Tensor, kMaximumExecutionDevices>& positions,
                               const std::array<Tensor, kMaximumExecutionDevices>& rope_positions,
                               ops::GqaExecutionEnvelope envelope, bool final_chunk,
                               const std::array<Tensor, kMaximumExecutionDevices>* final_hidden,
                               const std::array<Tensor, kMaximumExecutionDevices>* logits, Tensor* draft_token,
                               const Tensor* input_embeddings = nullptr);
    // Vocabulary-split proposal head: each rank computes its own half of the proposal logits and
    // one allgather leaves the FULL vector on both, because the winning row is a GLOBAL argmax
    // that can land in either half and `draft_head_token_ids` is replicated for exactly that
    // reason. Both ranks then run the identical argmax + remap over bit-identical gathered
    // logits, which is what puts the next round's draft ids on device 1 without a control-tensor
    // transfer.
    // The winning token id is needed on rank 0 only -- it leaves through the egress and, for the
    // AR proposal steps, feeds only rank 0's MTP stem (rank 1's stem contracts the hidden half
    // and never embeds a token). The gather still lands on both ranks because `allgather_rows`
    // writes both destinations; rank 1's copy is simply not read.
    void proposal_argmax_tp2(const std::array<Tensor, kMaximumExecutionDevices>& hidden,
                             const std::array<Tensor, kMaximumExecutionDevices>& logits, Tensor& proposal_tokens);
    [[nodiscard]] const MtpW& mtp_weights_for(int rank) const;
    [[nodiscard]] const GdnReplayRecords* replay_records_for(int rank) const;
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    template <class Tap>
    void run_owned_layers(Tensor& x, Phase phase, Tap& tap);
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                  ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                          Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                          Tensor& mtp_hidden, const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::GqaExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);
    // The tp2 text prefill. Declared here rather than beside its siblings above because it names
    // TextPrefill, which is declared just above this line.
    [[nodiscard]] PrefillChunkResult prefill_impl_tp2(std::span<const int> ids,
                                                 const TextPrefill& text_prefill,
                                                 bool finalize_at_end,
                                                 DFlashFeatureSink* dflash_sink = nullptr,
                                                 const MultimodalPrefill* multimodal = nullptr);
    DeviceContext& ctx_;
    const LoadedModelData& weights_;
    WorkspaceArena& work_;
    qwen3_6::PagedKVCacheView kv_;
    qwen3_6::PagedKVCacheView mtp_kv_;
    const qwen3_6::PagedKVCache* batch_text_kv_ = nullptr;
    const qwen3_6::PagedKVCache* batch_mtp_kv_  = nullptr;
    LinearAttentionStatePool& state_;
    qwen3_6::RoundState& io_;
    Tensor& prefill_hidden_;
    std::uint32_t prefill_chunk_;
    std::uint32_t text_kv_base_;
    const Tensor* active_cache_positions_                 = nullptr;
    const Tensor* active_rope_positions_                  = nullptr;
    const Tensor* active_kv_table_rows_                   = nullptr;
    const Tensor* active_linear_state_slots_              = nullptr;
    const Tensor* active_valid_columns_                   = nullptr;
    const Tensor* active_backend_kv_table_rows_           = nullptr;
    const ops::GqaExecutionEnvelope* active_gqa_envelope_ = nullptr;
    bool volta_long_attention_ = false;
    TokenConstraintView token_constraints_{};
    // Masks rows [row_offset,row_offset+valid_rows) of every bound logit column. No-op unbound.
    void constrain_logits(const Tensor& logits, std::int32_t row_offset, std::int32_t valid_rows,
                          cudaStream_t stream) const;
    [[nodiscard]] ops::GqaExecutionEnvelope attention_envelope(ops::GqaExecutionEnvelope value) const {
        value.volta_long_attention = volta_long_attention_;
        return value;
    }
    std::int32_t active_sequence_batch_                   = 0;
    std::int32_t active_sequence_width_                   = 0;
    std::int32_t rope_delta_                              = 0;
    std::int32_t linear_state_current_slot_               = 0;
    std::int32_t linear_state_rewrite_checkpoint_slot_    = 0;
    GdnStateAction gdn_state_action_                      = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_               = nullptr;
    std::int64_t prefill_rewrite_checkpoint_frontier_     = -1;
    Tensor* rewrite_checkpoint_hidden_output_             = nullptr;
    std::uint32_t mtp_proposal_extent_                    = 0;
    std::optional<PeerExecution> pipeline_;
    TokenId* pipeline_upload_ = nullptr;
    bool capture_retrieval_query_ = true;

    // YaRN rotary frequency override, indexed by rank: each device ropes its own head-local q/k
    // (12Q:2KV at tp == 2), so each device needs its own resident copy of the corrected table.
    // Default-constructed here, which is the native constant-table rope path, bit-for-bit — the
    // Op is what interprets a null `inv_frequency`. Under `RopeMode::Yarn` they are filled from
    // EngineOptions with per-device buffers allocated once at load, never per call: CUDA Graph
    // capture bakes the pointer into the replayed launch node, so the address must outlive every
    // replay.
    //
    // Every full-attention call site that ropes q/k with `rope_frequency_[r]` (this class's
    // `ops::rope(...)` calls, including the MTP tail at `mtp_forward_tail`) passes `kAttnScale`
    // (== `Variant::attention_scale`, a compile-time rsqrt(head_dim) constant) unmodified to the
    // following `ops::gqa_attention`/`gqa_attention_cached` call, in both native and yarn mode:
    // yarn's mscale is entirely a rope-path effect (`ops::RopeFrequencyOverride::mscale`) and
    // there is no attention-side factor, so `kAttnScale` never depends on `rope_frequency_`. See
    // `src/targets/qwen3_6/impl/runtime/yarn_rope.h` for the full account.
    std::array<ops::RopeFrequencyOverride, kMaximumExecutionDevices> rope_frequency_{};

    std::span<const PeerExecution> tp_;
    std::array<const Tensor*, kMaximumExecutionDevices - 1> peer_cache_positions_{};
    std::array<const Tensor*, kMaximumExecutionDevices - 1> peer_rope_positions_{};
    std::array<const Tensor*, kMaximumExecutionDevices - 1> peer_kv_table_rows_{};
    std::array<const Tensor*, kMaximumExecutionDevices - 1> peer_linear_state_slots_{};
    std::array<const Tensor*, kMaximumExecutionDevices - 1> peer_valid_columns_{};

    std::array<const Tensor*, kMaximumExecutionDevices - 1> peer_backend_kv_table_rows_{};

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const Weight* lm_head_                      = nullptr;
    const Weight* proposal_head_                = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    MtpW mtp_;
    std::array<FullLayerW, TextConfig::full_attention_layers()> full_{};
    std::array<GdnLayerW, TextConfig::gdn_layers()> gdn_{};
    // Rank 1's own shard bindings; populated only at tp == 2.
    std::array<const Weight*, kMaximumExecutionDevices - 1> embed_peer_{};
    std::array<const Tensor*, kMaximumExecutionDevices - 1> final_norm_peer_{};
    std::array<const Weight*, kMaximumExecutionDevices - 1> lm_head_peer_{};
    std::array<std::array<FullLayerW, TextConfig::full_attention_layers()>, kMaximumExecutionDevices - 1> full_peer_{};
    std::array<std::array<GdnLayerW, TextConfig::gdn_layers()>, kMaximumExecutionDevices - 1> gdn_peer_{};
    std::array<MtpW, kMaximumExecutionDevices - 1> mtp_peer_{};
    // Rank 1's own vocabulary half of the draft head plus its own device copy of the REPLICATED
    // [131072] id map. Both are cleared together with rank 0's when the request runs on the full
    // LM head instead (`set_proposal_head(nullptr, ...)`).
    std::array<const Weight*, kMaximumExecutionDevices - 1> proposal_head_peer_{};
    std::array<const std::int32_t*, kMaximumExecutionDevices - 1> proposal_head_ids_peer_{};
    std::array<Weight, TextConfig::gdn_layers()> gdn_in_a_{};
    std::array<Weight, TextConfig::gdn_layers()> gdn_in_b_{};
    std::array<Tensor, TextConfig::gdn_layers()> gdn_conv1d_views_{};
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
