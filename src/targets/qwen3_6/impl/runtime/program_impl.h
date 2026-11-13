#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/yarn_rope.h"
#include "targets/qwen3_6/impl/runtime/ram_kv_selection.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/kv_retrieval.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

using Clock = std::chrono::steady_clock;

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token) {
    const std::size_t tokens = prompt.token_ids.size();
    if (token >= tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("MTP bridge position is outside prepared prompt metadata");
    }
    return {prompt.positions[token], prompt.positions[tokens + token],
            prompt.positions[2 * tokens + token]};
}

// Makes one device current for the duration of a statement and restores the caller's. Peer-side
// pool operations issue memsets and copies on the peer's stream, which must be done with the
// peer's device current.
// Sums this device's GDN recurrent + convolution state: TP splits heads/channels,
// whereas PP splits layers. This is already the per-device number.
// The single construction of this Program's YaRN parameters. Both readers -- the `yarn_mscale`
// member initializer and the per-device table upload -- go through here, so the factor/origin/theta
// /rotary-pair tuple the descriptor's mscale describes is by construction the tuple the uploaded
// frequencies were computed from. Only meaningful when `plan.rope_mode == RopeMode::Yarn`.
qwen3_6::detail::YarnParams plan_yarn_params(const SequencePlanImpl& plan) {
    return qwen3_6::detail::YarnParams{
        .factor       = static_cast<float>(plan.yarn_factor),
        .original_max = static_cast<int>(plan.yarn_origin),
        .theta        = TextConfig::rope_theta,
        .rotary_pairs = TextConfig::rotary_dim / 2,
    };
}

std::size_t linear_attention_state_bytes(const LinearAttentionStatePoolLayout& layout) {
    std::size_t total = 0;
    for (const LayoutRegion& region : layout.conv) { total += region.bytes; }
    for (const LayoutRegion& region : layout.recurrent) { total += region.bytes; }
    return total;
}

class ScopedDevice {
public:
    explicit ScopedDevice(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        CUDA_CHECK(cudaSetDevice(device));
    }

    ~ScopedDevice() { (void)cudaSetDevice(previous_); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    int previous_ = 0;
};

schedule::MtpGqaEnvelopes mtp_gqa_envelopes(std::uint32_t max_frontier, std::uint32_t k,
                                            std::uint32_t capacity) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    schedule::MtpGqaEnvelopes out;
    out.target_verify = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + 1ULL)};
    out.batch         = out.target_verify;
    for (std::uint32_t step = 0; step + 1 < k; ++step) {
        out.ar[step] = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + step + 2ULL)};
    }
    return out;
}

schedule::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                           std::uint32_t k) {
    (void)min_frontier;
    return schedule::DFlashEnvelopes{
        .local  = {0, max_frontier},
        .full   = {0, max_frontier},
        .append = {0, k + 1},
    };
}

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label,
                                         bool greedy_target = false, bool constrained = false) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier &&
                   profile.greedy_target == greedy_target && profile.constrained == constrained;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label) {
    if (profiles.empty() || profiles.front().min != 0 || profiles.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (profiles[i].min > profiles[i].max ||
            (i != 0 && profiles[i].min != profiles[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label) {
    const auto it = std::find_if(family.topologies.begin(), family.topologies.end(),
                                 [topology_class](const DecodeGraphTopology& topology) {
                                     return topology.topology_class == topology_class;
                                 });
    if (it == family.topologies.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
    }
    return *it;
}

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        topology.executable.update(profile.definition);
        topology.installed_profile = profile_index;
    }
    return topology.executable;
}

template <class Prepare, class Synchronize>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device,
                              Prepare&& prepare, Synchronize&& synchronize_all) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " CUDA Graph family has no profiles");
    }

    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        DecodeGraphProfile& profile = family.profiles[i];
        if (!profile.definition.ready()) {
            throw std::logic_error(std::string(label) + " CUDA Graph definition is empty");
        }
        const auto existing =
            std::find_if(family.topologies.begin(), family.topologies.end(),
                         [&](const DecodeGraphTopology& topology) {
                             return topology.topology_class == profile.topology_class;
                         });
        if (existing != family.topologies.end()) { continue; }

        family.topologies.emplace_back();
        DecodeGraphTopology& topology = family.topologies.back();
        topology.topology_class       = profile.topology_class;
        topology.executable.instantiate(profile.definition);
        topology.installed_profile = i;
    }

    const auto install_and_upload = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        DecodeGraphProfile& profile = family.profiles[profile_index];
        if (topology.installed_profile != profile_index) {
            topology.executable.update(profile.definition);
            topology.installed_profile = profile_index;
        }
        // A dual-device executable uploads and launches from rank 0's stream as one unit; the
        // synchronize covers both devices because the graph's rank-1 nodes retire on rank 1.
        topology.executable.upload(device.stream);
        synchronize_all();
    };

    for (DecodeGraphTopology& topology : family.topologies) {
        std::optional<std::size_t> first_profile;
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                if (!first_profile) {
                    first_profile = i;
                    install_and_upload(topology, i);

                    DecodeGraphProfile& profile = family.profiles[i];
                    prepare(profile.min_execution_frontier, profile.batch_size);
                    synchronize_all();
                    topology.executable.launch(device.stream);
                    synchronize_all();
                    continue;
                }
                install_and_upload(topology, i);
            }
        }
        if (!first_profile) {
            throw std::logic_error(std::string(label) + " CUDA Graph topology has no definitions");
        }
        if (topology.installed_profile != *first_profile) {
            install_and_upload(topology, *first_profile);
        }
    }
}

} // namespace

PeerRuntime::PeerRuntime(DeviceContext& peer_device, const LoadedModelData& peer_model,
                         const SequencePlanImpl& plan, int rank_in)
    : rank(rank_in), device(peer_device), model(peer_model), persistent(plan.persistent.bytes),
      workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), workspace_storage.capacity()}) {
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder = std::make_unique<qwen3_6::DecoderState>(backing, plan.persistent.decoder);
    if (plan.persistent.dflash) { dflash.emplace(backing, *plan.persistent.dflash); }
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
    }
    io             = qwen3_6::RoundState(backing, plan.persistent.round);
    if (io.gdn_record_norm_counts.data != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(io.gdn_record_norm_counts.data, 0,
                                   io.gdn_record_norm_counts.bytes(), device.stream));
    }
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    token_counts   = plan.persistent.token_counts.bind(backing);
}

ProgramImplCore::ProgramImplCore(const LoadedModelData& model_in,
                                 std::span<const LoadedModelData* const> peer_models, const SequencePlanImpl& plan,
                                 ExecutionContext& execution_in)
    : model(model_in), execution(execution_in), device(execution_in.primary()), device_count(execution_in.device_count),
      pp(plan.pp), pipeline_chunks(plan.pipeline_chunks),
      pipeline_upload(plan.pp == 2
                          ? std::make_optional<PinnedHostBuffer>(2ULL * plan.prefill_chunk * plan.pipeline_chunks * sizeof(TokenId))
                          : std::nullopt),
      capacity(plan.capacity), kv_capacity(plan.kv_capacity),
      max_concurrency(plan.max_concurrency), prefix_cache_tokens(plan.prefix_cache_tokens), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), ram_kv(plan.ram_kv), volta_long_attention(plan.volta_long_attention),
      speculative_backend(plan.speculative_backend),
      kv_dtype(plan.kv_dtype), kv_quant_group(plan.kv_quant_group),
      proposal_head(plan.proposal_head), vision_enabled(plan.features.vision),
      use_cuda_graph(plan.use_cuda_graph), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      gdn_state_bytes(linear_attention_state_bytes(plan.persistent.decoder.linear_attention)),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), workspace_storage.capacity()}),
      rope_mode(plan.rope_mode), effective_max_context(plan.effective_max_context),
      yarn_mscale(plan.rope_mode == RopeMode::Yarn
                      ? static_cast<double>(
                            qwen3_6::detail::yarn_rope_mscale(plan_yarn_params(plan)))
                      : 1.0),
      round_host(sizeof(TokenId)),
      token_constraint_host(
          static_cast<std::size_t>(kTokenConstraintRows) *
          (sizeof(std::int32_t) +
           sizeof(std::uint32_t) * static_cast<std::size_t>(
                                       runtime::token_bitmask_words(TextConfig::token_domain)))),
      ordinary_host(
          plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_6::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::MtpDecodeIngress) +
                                                          sizeof(qwen3_6::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(plan.speculative_backend == SpeculativeBackend::DFlash
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::DFlashDecodeIngress) +
                                                             sizeof(qwen3_6::DFlashDecodeEgress))
                      : std::nullopt) {
    if (model.weights_arena == nullptr) {
        throw std::invalid_argument("Qwen3.6 model view has no owning weight arena");
    }
    if (model.features != plan.features || model.mtp.has_value() != plan.features.mtp() ||
        model.dflash.has_value() != plan.features.dflash() ||
        model.optimized_proposal.has_value() != plan.features.optimized_proposal() ||
        model.vision.has_value() != plan.features.vision) {
        throw std::invalid_argument(
            "Qwen3.6 loaded weights do not match the frozen startup features");
    }
    if (model.mtp.has_value() && model.dflash.has_value()) {
        throw std::invalid_argument("MTP and DFlash model views are mutually exclusive");
    }
    if (model.dflash.has_value() && model.vision.has_value()) {
        throw std::invalid_argument("DFlash and Vision model views are mutually exclusive");
    }
    if (device_count != plan.tp * plan.pp || peer_models.size() != static_cast<std::size_t>(device_count - 1)) {
        throw std::invalid_argument("Qwen3.6 program tensor-parallel width is inconsistent");
    }
    peers.reserve(device_count - 1);
    for (int rank = 1; rank < device_count; ++rank) {
        if (!execution.dev[rank] || peer_models[rank - 1] == nullptr) {
            throw std::invalid_argument("tensor-parallel program requires every rank's model and device");
        }
        const ScopedDevice scope(execution.dev[rank]->device);
        peers.push_back(std::make_unique<PeerRuntime>(
            *execution.dev[rank], *peer_models[rank - 1], plan, rank));
        if (plan.use_cuda_graph) {
            peers.back()->graph_bridge.emplace(device.device, execution.dev[rank]->device);
        }
    }
    if (device_count > 1) {
        const bool direct_peer_access = ops::enable_peer_access(execution);
        peer_events.emplace(execution, direct_peer_access);
    }
    if (rope_mode == RopeMode::Yarn) {
        // ONE resident 32-float corrected inverse-frequency table PER DEVICE, uploaded once, here.
        // cudaMalloc is neither stream-ordered nor capturable, and CUDA Graph capture bakes this
        // pointer into the replayed rope launch node, so the allocation has to happen exactly once
        // at construction and stay valid for the life of the Program. At device_count 2 each rank ropes its
        // own head-local q/k on its own device and stream: rank 1 cannot dereference rank 0's
        // table, so each device gets its own copy of the same 128 bytes.
        const std::vector<float> table =
            qwen3_6::detail::yarn_scale(plan_yarn_params(plan)).first;
        if (table.size() != static_cast<std::size_t>(TextConfig::rotary_dim / 2)) {
            throw std::logic_error("YaRN frequency table does not match the rotary geometry");
        }
        const std::size_t table_bytes = table.size() * sizeof(float);
        for (int rank = 0; rank < device_count; ++rank) {
            const auto slot            = static_cast<std::size_t>(rank);
            DeviceContext& rank_device = *execution.dev[slot];
            const ScopedDevice scope(rank_device.device);
            rope_frequency_storage[slot] = DeviceBuffer(table_bytes);
            CUDA_CHECK(cudaMemcpyAsync(rope_frequency_storage[slot].p, table.data(), table_bytes,
                                       cudaMemcpyHostToDevice, rank_device.stream));
            // Settled here rather than left as an ordering assumption: this copy is the only write
            // this buffer ever sees, and every later reader is a captured or eager rope launch.
            CUDA_CHECK(cudaStreamSynchronize(rank_device.stream));
            rope_frequency[slot] = ops::RopeFrequencyOverride{
                static_cast<const float*>(rope_frequency_storage[slot].p),
                static_cast<float>(yarn_mscale)};
        }
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder = std::make_unique<qwen3_6::DecoderState>(backing, plan.persistent.decoder);
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None)) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) { dflash.emplace(backing, *plan.persistent.dflash); }
    if (dflash.has_value() != plan.features.dflash()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }

    io = qwen3_6::RoundState(backing, plan.persistent.round);
    if (io.gdn_record_norm_counts.data != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(io.gdn_record_norm_counts.data, 0,
                                   io.gdn_record_norm_counts.bytes(), device.stream));
    }
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() != (speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden                  = plan.persistent.prefill_hidden.bind(backing);
    token_counts                    = plan.persistent.token_counts.bind(backing);
    sampling_config                 = plan.persistent.sampling_config.bind(backing);
    tail_hidden_store               = plan.persistent.tail_hidden.bind(backing);
    rewrite_checkpoint_hidden_store = plan.persistent.rewrite_checkpoint_hidden.bind(backing);
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        SequenceState& sequence = sequences[lane];
        sequence.lane           = lane;
        sequence.tail_hidden    = tail_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.rewrite_checkpoint_hidden =
            rewrite_checkpoint_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    }

    set_device_i32(io.text_kv_table_row, 0);
    set_device_i32(io.backend_kv_table_row, 0);

    host_tokens = static_cast<TokenId*>(round_host.data());
    // Page-locked host memory is mapped at the same address on every device under UVA, so each
    // rank reads the constraint image in place without per-round copies.
    void* constraint_device = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&constraint_device, token_constraint_host.data(), 0));
    if (constraint_device != token_constraint_host.data()) {
        throw std::runtime_error("structured-output constraints require unified host addressing");
    }
    std::memset(token_constraint_host.data(), 0, token_constraint_host.size());
    token_constraints = schedule::TokenConstraintView{
        .constrained = static_cast<const std::int32_t*>(token_constraint_host.data()),
        .bitmask     = reinterpret_cast<const std::uint32_t*>(
            static_cast<const std::int32_t*>(token_constraint_host.data()) + kTokenConstraintRows),
        .words       = runtime::token_bitmask_words(TextConfig::token_domain),
    };
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_6::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_6::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_6::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
        for (auto& peer : peers) {
            peer->ordinary_host.emplace(sizeof(qwen3_6::OrdinaryDecodeIngress));
            peer->ordinary_host_ingress =
                static_cast<qwen3_6::OrdinaryDecodeIngress*>(peer->ordinary_host->data());
            *peer->ordinary_host_ingress = {};
        }
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_6::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_6::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_6::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
        for (auto& peer : peers) {
            peer->mtp_host.emplace(sizeof(qwen3_6::MtpDecodeIngress));
            peer->mtp_host_ingress =
                static_cast<qwen3_6::MtpDecodeIngress*>(peer->mtp_host->data());
            *peer->mtp_host_ingress = {};
        }
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_6::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_6::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_6::DFlashDecodeIngress));
        *dflash_host_ingress = {};
        *dflash_host_egress  = {};
        for (auto& peer : peers) {
            peer->dflash_host.emplace(sizeof(qwen3_6::DFlashDecodeIngress));
            peer->dflash_host_ingress =
                static_cast<qwen3_6::DFlashDecodeIngress*>(peer->dflash_host->data());
            *peer->dflash_host_ingress = {};
        }
    }
    if (io.dflash_prefill) {
        CUDA_CHECK(cudaMemsetAsync(io.dflash_prefill->produced_count.data, 0,
                                   io.dflash_prefill->produced_count.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    for (auto& peer : peers) {
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(peer->device.device));
        CUDA_CHECK(cudaMemsetAsync(peer->io.rope_delta.data, 0, peer->io.rope_delta.bytes(),
                                   peer->device.stream));
        if (peer->io.mtp) {
            CUDA_CHECK(cudaMemsetAsync(peer->io.mtp->position.data, 0,
                                       peer->io.mtp->position.bytes(), peer->device.stream));
        }
        CUDA_CHECK(cudaMemsetAsync(peer->token_counts.data, 0, peer->token_counts.bytes(),
                                   peer->device.stream));
        set_peer_i32(peer->io.text_kv_table_row, 0, peer->device);
        set_peer_i32(peer->io.backend_kv_table_row, 0, peer->device);
        CUDA_CHECK(cudaSetDevice(previous));
        peer_cores[peer->rank - 1] = schedule::PeerCore{.execution        = &execution,
                                               .events           = &*peer_events,
                                               .device           = &peer->device,
                                               .model            = &peer->model,
                                               .work             = &peer->work,
                                               .linear_attention = &peer->decoder->linear_attention,
                                               .io               = &peer->io,
                                               .prefill_hidden   = &peer->prefill_hidden,
                                               .text_cache       = &peer->decoder->text_kv,
                                               .mtp_cache        = peer->decoder->mtp_cache(),
                                               .replay_records   = peer->replay_records
                                                                       ? &*peer->replay_records
                                                                       : nullptr,
                                               .ordinary_host_ingress = peer->ordinary_host_ingress,
                                               .mtp_host_ingress = peer->mtp_host_ingress,
                                               .dflash_host_ingress = peer->dflash_host_ingress,
                                               .graph_bridge = peer->graph_bridge ? &*peer->graph_bridge : nullptr};
        if (peer->replay_records.has_value() != replay_records.has_value()) {
            throw std::logic_error("peer ReplaySSM records do not match rank 0's");
        }
        if ((peer->decoder->mtp_cache() != nullptr) != (decoder->mtp_cache() != nullptr)) {
            throw std::logic_error("peer MTP KV cache does not match rank 0's");
        }
        if (peer->io.mtp.has_value() != io.mtp.has_value() ||
            peer->io.mtp_decode.has_value() != io.mtp_decode.has_value()) {
            throw std::logic_error("peer MTP round state does not match rank 0's");
        }
        set_device_i32(io.text_kv_table_row, 0);
        set_device_i32(io.backend_kv_table_row, 0);
        peer->device.synchronize();
    }
    device.synchronize();
    prepare_graphs();
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

ProgramImplCore::~ProgramImplCore() noexcept {
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
    for (auto& peer : peers) {
        if (peer->device.stream != nullptr) { (void)cudaStreamSynchronize(peer->device.stream); }
    }
}

bool ProgramImplCore::can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= max_concurrency || plan.impl_ == nullptr) { return false; }
    const RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        return false;
    }
    const SequenceState& sequence = sequences[lane];
    const auto can_replace        = [](const PagedKVPool& pool, std::uint32_t old_pages,
                                std::uint32_t new_pages) {
        return old_pages <= pool.entitled_pages() && new_pages <= pool.logical_page_capacity() &&
               new_pages <= pool.page_group_count() - (pool.entitled_pages() - old_pages);
    };
    const std::uint32_t old_text = sequence.kv ? sequence.kv->text.page_entitlement() : 0;
    if (!can_replace(decoder->text_kv.pool(), old_text, plan.impl_->text_kv_page_entitlement)) {
        return false;
    }
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(backend->pool(), old_backend, plan.impl_->backend_kv_page_entitlement);
}

bool ProgramImplCore::can_admit_lane_after_retained_eviction(
    std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= max_concurrency || plan.impl_ == nullptr) { return false; }
    const RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        return false;
    }

    std::uint32_t reclaimable_text    = 0;
    std::uint32_t reclaimable_backend = 0;
    for (std::uint32_t other = 0; other < max_concurrency; ++other) {
        if (other == lane || !sequences[other].retained || !sequences[other].kv) { continue; }
        reclaimable_text += sequences[other].kv->text.page_entitlement();
        if (sequences[other].kv->backend) {
            reclaimable_backend += sequences[other].kv->backend->page_entitlement();
        }
    }

    const auto can_replace = [](const PagedKVPool& pool, std::uint32_t old_pages,
                                std::uint32_t reclaimable_pages, std::uint32_t new_pages) {
        if (old_pages > pool.entitled_pages() ||
            reclaimable_pages > pool.entitled_pages() - old_pages ||
            new_pages > pool.logical_page_capacity()) {
            return false;
        }
        const std::uint32_t committed = pool.entitled_pages() - old_pages - reclaimable_pages;
        return new_pages <= pool.page_group_count() - committed;
    };

    const SequenceState& sequence = sequences[lane];
    const std::uint32_t old_text  = sequence.kv ? sequence.kv->text.page_entitlement() : 0;
    if (!can_replace(decoder->text_kv.pool(), old_text, reclaimable_text,
                     plan.impl_->text_kv_page_entitlement)) {
        return false;
    }

    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(backend->pool(), old_backend, reclaimable_backend,
                       plan.impl_->backend_kv_page_entitlement);
}

runtime::AdmissionResources ProgramImplCore::admission_capacity() const noexcept {
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    return runtime::AdmissionResources{
        .active_lanes     = max_concurrency,
        .main_kv_pages    = decoder->text_kv.pool().page_group_count(),
        .backend_kv_pages = backend != nullptr ? backend->pool().page_group_count() : 0U,
    };
}

runtime::PrefillStepResult ProgramImplCore::start_prefill_lane(std::uint32_t lane,
                                                               PreparedPromptData&& prompt,
                                                               RequestPlan&& plan,
                                                               runtime::TransientRegion transient) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    RequestPlanImpl& request_plan = *plan.impl_;
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("staged prefill requires a free request lane");
    }

    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (prompt_tokens != request_plan.summary.prompt_tokens ||
        (request_plan.vision.has_value() && !prompt.has_media())) {
        throw std::invalid_argument("request plan does not describe the prepared prompt");
    }
    if (prompt.identity.rewrite_checkpoint &&
        (prompt.identity.rewrite_checkpoint->frontier == 0 ||
         prompt.identity.rewrite_checkpoint->frontier > prompt_tokens)) {
        throw std::invalid_argument("prepared prompt has an invalid rewrite checkpoint");
    }
    const bool suffix_has_visual = std::any_of(
        prompt.token_types.begin() + static_cast<std::ptrdiff_t>(request_plan.reuse_base),
        prompt.token_types.end(), [](std::uint8_t type) { return type != 0; });
    if (suffix_has_visual != request_plan.vision.has_value()) {
        throw std::invalid_argument("request plan does not describe the prompt suffix modality");
    }
    if (request_plan.summary.transient_bytes != 0 &&
        (transient.data == nullptr || transient.size < request_plan.summary.transient_bytes ||
         transient.alignment < request_plan.summary.transient_alignment)) {
        throw std::invalid_argument("request transient region does not satisfy the plan");
    }
    const auto started = Clock::now();
    // Save the outgoing branch before any truncation or overwrite. A plan pins its
    // selected archive across FIFO eviction and budget trimming.
    if (sequence.retained && (request_plan.archived_prefix ||
                              request_plan.reuse_base < sequence.execution_frontier)) {
        archive_prefix(sequence);
    }
    if (request_plan.archived_prefix) {
        restore_archived_prefix(sequence, *request_plan.archived_prefix, request_plan);
    }
    if (request_plan.reuse != ReusePath::FullReset &&
        (!sequence.retained ||
         !qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          request_plan.reuse_base))) {
        throw std::logic_error("planned resident prefix is no longer reusable");
    }
    if (is_rewrite_checkpoint_restore(request_plan.reuse) &&
        (!sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.frontier != request_plan.reuse_base ||
         request_plan.reuse != restore_path(sequence.rewrite_checkpoint.kind))) {
        throw std::logic_error("planned rewrite checkpoint is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::KeepExisting &&
        (!prompt.identity.rewrite_checkpoint || !sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.kind != prompt.identity.rewrite_checkpoint->kind ||
         sequence.rewrite_checkpoint.frontier != prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.reuse == ReusePath::FullReset ||
         !qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          sequence.rewrite_checkpoint.frontier))) {
        throw std::logic_error("planned rewrite checkpoint retention is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::ReclassifyExisting &&
        (!prompt.identity.rewrite_checkpoint || !sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.kind == prompt.identity.rewrite_checkpoint->kind ||
         sequence.rewrite_checkpoint.frontier != prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.reuse == ReusePath::FullReset ||
         !qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          sequence.rewrite_checkpoint.frontier))) {
        throw std::logic_error("planned rewrite checkpoint reclassification is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::CaptureNew &&
        (!request_plan.rewrite_checkpoint_capture || !prompt.identity.rewrite_checkpoint ||
         request_plan.rewrite_checkpoint_capture->kind !=
             prompt.identity.rewrite_checkpoint->kind ||
         request_plan.rewrite_checkpoint_capture->frontier !=
             prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.rewrite_checkpoint_capture->frontier <= request_plan.reuse_base ||
         request_plan.rewrite_checkpoint_capture->frontier > prompt_tokens)) {
        throw std::logic_error("planned rewrite checkpoint capture is invalid");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::Drop &&
        prompt.identity.rewrite_checkpoint) {
        throw std::logic_error("planned rewrite checkpoint drop does not describe the prompt");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::DeferCapture &&
        (!prompt.identity.rewrite_checkpoint || request_plan.reuse == ReusePath::FullReset ||
         prompt.identity.rewrite_checkpoint->frontier > request_plan.reuse_base)) {
        throw std::logic_error("planned rewrite checkpoint deferral is invalid");
    }

    const std::uint32_t base = request_plan.reuse_base;
    const std::uint32_t initial_mtp_extent =
        speculative_backend == SpeculativeBackend::Mtp
            ? std::min({draft_window,
                        request_plan.summary.effective_output_tokens > 1
                            ? request_plan.summary.effective_output_tokens - 2
                            : 0U,
                        capacity - prompt_tokens > 0 ? capacity - prompt_tokens - 1 : 0U})
            : 0U;
    request.lifecycle = Lifecycle::Empty;
    request.exact_hit.reset();
    if (request_plan.prepare_mtp && base == prompt_tokens) {
        request.exact_hit = RequestControl::ExactHitContinuation{
            .bridge_rope = prompt_rope_position(prompt, prompt_tokens - 1),
            .proposal_extent = initial_mtp_extent};
    }
    sequence.retained = false;
    sequence.archived_snapshot.reset();
    sequence.history_enabled = request_plan.capture_prefix_checkpoints;
    sequence.shared_prefix_tokens = std::min(sequence.shared_prefix_tokens, base);
    if (base == 0) { sequence.prefix_chunks.clear(); }
    try {
        if (request_plan.reuse == ReusePath::FullReset) {
            sequence.deferred_prefix_state.reset();
            sequence.kv.reset();
            ordered_reset(sequence);
            sequence.ledger.clear();
            if (request_plan.capture_prefix_checkpoints) {
                // Invalidate the old history but recycle its bounded host storage.
                // Unrelated conversations must not repeatedly fault/zero gigabytes.
                for (auto& checkpoint : sequence.prefix_checkpoints) { checkpoint.frontier = 0; }
            } else {
                sequence.prefix_checkpoints.clear();
            }
            sequence.text_kv_valid = 0;
            sequence.mtp_kv_valid  = 0;
            reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement,
                                request_plan.backend_kv_page_entitlement);
        } else if (request_plan.reuse == ReusePath::AppendAtFrontier) {
            if (!sequence.kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (sequence.text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (sequence.deferred_prefix_state && !request.exact_hit) {
                restore_prefix_checkpoint(sequence, *sequence.deferred_prefix_state);
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash &&
                       sequence.dflash_context_frontier != base) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.text_kv_valid = base;
            sequence.ledger.resize(base);
        } else if (request_plan.reuse == ReusePath::RestorePrefixCheckpoint) {
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident prefix checkpoint has no complete KV allocation");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error("prefix-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            restore_prefix_checkpoint(sequence, base, !request.exact_hit.has_value());
            sequence.ledger.resize(base);
        } else if (is_rewrite_checkpoint_restore(request_plan.reuse)) {
            sequence.deferred_prefix_state.reset();
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident rewrite checkpoint has no complete KV allocation");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error(
                        "rewrite-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash) {
                if (!dflash || dflash->full.has_value() != sequence.kv->backend.has_value() ||
                    sequence.dflash_context_frontier < base) {
                    throw std::logic_error("planned DFlash rewrite checkpoint is unavailable");
                }
                dflash->restore_rewrite_checkpoint(static_cast<std::int32_t>(sequence.lane),
                                                   device.stream);
                sequence.dflash_context_frontier = base;
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            decoder->linear_attention.copy_slot(
                LinearStateSlots::rewrite_checkpoint_state_slot(sequence.lane, max_concurrency),
                LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                device.stream);
            for (auto& peer : peers) {
                const ScopedDevice scope(peer->device.device);
                peer->decoder->linear_attention.copy_slot(
                    LinearStateSlots::rewrite_checkpoint_state_slot(sequence.lane, max_concurrency),
                    LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                    peer->device.stream);
            }
            if (base == prompt_tokens) { copy_tail(sequence, sequence.rewrite_checkpoint_hidden); }
            sequence.ledger.resize(base);
        } else {
            throw std::logic_error("request plan has an invalid prefix reuse path");
        }

        // KV after the selected frontier will be overwritten by the new suffix. Images
        // of that old branch must not be matched against the replacement token ledger.
        for (auto& checkpoint : sequence.prefix_checkpoints) {
            if (checkpoint.frontier > base) { checkpoint.frontier = 0; }
        }
        if (sequence.rewrite_checkpoint.valid && sequence.rewrite_checkpoint.frontier > base &&
            request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::DeferCapture) {
            sequence.rewrite_checkpoint = {};
        }

        trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
        bind_sequence_kv(sequence);
        const std::uint32_t backend_materialized =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
            : backend_kv_cache() != nullptr ? prompt_tokens : 0U;
        if (ram_kv.gpu_tokens != 0) {
            const auto query = prompt.retrieval_query;
            const bool retain_query = base >= query.begin + query.count &&
                sequence.ram_query.begin == query.begin && sequence.ram_query.count == query.count;
            const std::array<int, 2> span{static_cast<int>(query.begin),
                                          static_cast<int>(query.begin + query.count)};
            const auto configure = [&](qwen3_6::PagedKVCache& cache, DeviceContext& ctx) {
                if (!retain_query) {
                    CUDA_CHECK(cudaMemsetAsync(cache.retrieval_query.data, 0, cache.retrieval_query.bytes(), ctx.stream));
                    CUDA_CHECK(cudaMemsetAsync(cache.retrieval_counts.data, 0, cache.retrieval_counts.bytes(), ctx.stream));
                }
                CUDA_CHECK(cudaMemcpyAsync(cache.retrieval_span.data, span.data(), sizeof(span), cudaMemcpyHostToDevice, ctx.stream));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            };
            configure(decoder->text_kv, device);
            for (auto& peer : peers) {
                const ScopedDevice scope(peer->device.device);
                configure(peer->decoder->text_kv, peer->device);
            }
            sequence.ram_query = query;
            sequence.ram_rank_frontier = base;
            refresh_ram_kv(sequence, base);
            // Cold prefill materializes one execution unit at a time. A retained prefix may
            // need cold pages reloaded when the query ranking changes, even without growth.
            materialize_sequence_kv(sequence, base, backend_kv_valid(sequence));
        } else { materialize_sequence_kv(sequence, prompt_tokens, backend_materialized); }
        install_sampling(sequence, request, request_plan.sampling);
        request.grammar.reset();
        if (request_plan.grammar) {
            request.grammar.emplace(request_plan.grammar, prompt.starts_in_reasoning);
        }
        (void)plan_token_constraints(request, kTokenConstraintPrefillRow, {});
        sequence.rope_delta = prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);
        for (auto& peer : peers) {
            const ScopedDevice scope(peer->device.device);
            set_peer_i32(peer->io.rope_delta, sequence.rope_delta, peer->device);
        }

        if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::Drop ||
            request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::CaptureNew) {
            sequence.rewrite_checkpoint = {};
        } else if (request_plan.rewrite_checkpoint_action ==
                   RewriteCheckpointAction::ReclassifyExisting) {
            sequence.rewrite_checkpoint.kind = prompt.identity.rewrite_checkpoint->kind;
        }
        request.timings            = {};
        request.pending            = {};
        sequence.mtp_draft_count   = 0;
        sequence.tail_hidden_valid = base == prompt_tokens && sequence.tail_hidden_valid;
        sequence.ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        sequence.prefix_identity.assign(prompt);

        if (speculative_backend == SpeculativeBackend::DFlash) {
            if (!dflash || !io.dflash_decode ||
                dflash->full.has_value() != sequence.kv->backend.has_value()) {
                throw std::logic_error("DFlash prefill state is incomplete");
            }
            *dflash_host_ingress                         = {};
            dflash_host_ingress->lanes[0]                = static_cast<std::int32_t>(sequence.lane);
            dflash_host_ingress->dflash_kv_table_rows[0] =
                sequence.kv->backend ? sequence.kv->backend->bound_row() : 0;
            CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                       sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                       device.stream));
        }

        if (request_plan.vision) {
            std::vector<bool> used(prompt.media_payloads.size(), false);
            for (const VisionUseSpan& use : request_plan.vision->uses) {
                if (use.item_index >= used.size()) {
                    throw std::logic_error("Vision plan references a missing media payload");
                }
                used[use.item_index] = true;
            }
            for (std::size_t i = 0; i < used.size(); ++i) {
                if (!used[i]) { prompt.media_payloads[i].reset(); }
            }
        }
        if (prompt.has_media() && !request_plan.vision) { prompt.release_all_media_payloads(); }

        RequestControl::Prefill prefill{
            .prompt                     = std::move(prompt),
            .vision_plan                = std::move(request_plan.vision),
            .vision                     = nullptr,
            .transient                  = transient,
            .rewrite_checkpoint_capture = request_plan.rewrite_checkpoint_capture,
            .base                       = base,
            .cursor                     = base,
            .prompt_tokens              = prompt_tokens,
            .initial_mtp_extent         = initial_mtp_extent,
            .elapsed_seconds            = 0.0,
            .prepare_mtp                = request_plan.prepare_mtp,
            .capture_prefix_checkpoints = request_plan.capture_prefix_checkpoints,
            .reuse                      = request_plan.reuse,
            .mtp_bridge                 = request_plan.mtp_bridge,
        };
        request.prefill.emplace(std::move(prefill));
        auto& staged = *request.prefill;
        if (staged.vision_plan) {
            staged.vision = std::make_unique<schedule::VisionPrefillSession>(
                device, model, work, staged.prompt, *staged.vision_plan, staged.transient);
        }
        staged.elapsed_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        request.lifecycle      = Lifecycle::Prefilling;
        return advance_prefill(sequence, request);
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        throw;
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    return advance_prefill(sequences[lane], requests[lane]);
}

void ProgramImplCore::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].pending.kind != PendingKind::Begin) {
        throw std::logic_error("resolve_prefill_lane requires a pending prefill token");
    }
    resolve_non_speculative_pending(sequences[lane], requests[lane], 1, terminal);
}

void ProgramImplCore::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                            std::span<const std::uint32_t> accepted_tokens,
                                            std::span<const std::uint8_t> terminal,
                                            std::span<const std::uint8_t> cancelled) {
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size()) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    if (speculative_backend == SpeculativeBackend::None) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
                requests[lane].pending.kind != PendingKind::Ordinary) {
                throw std::logic_error("ordinary pending batch no longer matches Program state");
            }
            if (cancelled[row]) {
                clear_lane(sequences[lane], requests[lane]);
            } else {
                resolve_non_speculative_pending(sequences[lane], requests[lane],
                                                accepted_tokens[row], terminal[row] != 0);
            }
        }
        return;
    }

    if (!replay_records) {
        throw std::logic_error("speculative pending batch has no ReplaySSM records");
    }

    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::array<std::int32_t, kMaximumConcurrency> hidden_selectors{};
    bool needs_hidden_correction = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
            requests[lane].pending.kind != PendingKind::Speculative) {
            throw std::logic_error("speculative pending batch no longer matches Program state");
        }
        const PendingCandidate& pending = requests[lane].pending;
        const SequenceState& sequence   = sequences[lane];
        if (sequence.execution_frontier != pending.base_E ||
            sequence.ledger_frontier != pending.base_S ||
            sequence.ledger.size() != pending.base_S ||
            sequence.prefix_identity.size() != pending.base_S ||
            sequence.text_kv_valid != pending.base_E ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != pending.base_E) ||
            (speculative_backend == SpeculativeBackend::DFlash &&
             sequence.dflash_context_frontier != pending.base_E)) {
            throw std::logic_error("speculative pending row is not at its recorded base");
        }
        const std::uint32_t committed = cancelled[row] ? 0U : accepted_tokens[row];
        if ((cancelled[row] && accepted_tokens[row] != 0) ||
            (!cancelled[row] && (committed == 0 || committed > pending.produced ||
                                 (!terminal[row] && committed != pending.produced)))) {
            throw std::logic_error("speculative pending row has an invalid committed prefix");
        }
        fold_rows[row] = ops::GdnReplayFoldRow{
            .linear_state_slot = LinearStateSlots::current_state_slot(lane, max_concurrency),
            .commit_columns    = static_cast<std::int32_t>(committed),
        };
        const bool partial_terminal =
            !cancelled[row] && terminal[row] && committed < pending.produced;
        hidden_selectors[row] =
            static_cast<std::int32_t>(partial_terminal ? committed - 1U : pending.produced - 1U);
        needs_hidden_correction = needs_hidden_correction || partial_terminal;
    }

    const auto tail_started = Clock::now();
    try {
        ops::gdn_replay_fold(*replay_records, decoder->linear_attention.all_layers_view(),
                             std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                             device.stream);
        // Every peer folds its own layer/head records using the same committed token prefix.
        for (auto& peer : peers) {
            if (!peer->replay_records) {
                throw std::logic_error("peer speculative round has no ReplaySSM records");
            }
            const ScopedDevice scope(peer->device.device);
            ops::gdn_replay_fold(
                *peer->replay_records, peer->decoder->linear_attention.all_layers_view(),
                std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                peer->device.stream);
        }

        if (needs_hidden_correction) {
            const auto batch = static_cast<std::int32_t>(lanes.size());
            Tensor selector_tensor;
            Tensor hidden;
            Tensor selected;
            Tensor destinations;
            if (speculative_backend == SpeculativeBackend::Mtp && io.mtp_decode) {
                qwen3_6::MtpDecodeState& frame = *io.mtp_decode;
                selector_tensor                = frame.current_extents.slice(0, 0, batch);
                hidden                         = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.lanes.slice(0, 0, batch);
            } else if (speculative_backend == SpeculativeBackend::DFlash && io.dflash_decode) {
                qwen3_6::DFlashDecodeState& frame = *io.dflash_decode;
                selector_tensor                   = frame.proposal_extents.slice(0, 0, batch);
                hidden                            = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.lanes.slice(0, 0, batch);
            } else {
                throw std::logic_error("partial speculative commit has no target frame");
            }
            CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, hidden_selectors.data(),
                                       lanes.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device.stream));
            ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected,
                                                    device.stream);
            ops::scatter(selected, destinations, tail_hidden_store, device.stream);
        }

        if (speculative_backend == SpeculativeBackend::DFlash) {
            std::array<std::uint32_t, kMaximumConcurrency> append_lanes{};
            std::array<std::uint32_t, kMaximumConcurrency> append_starts{};
            std::array<std::uint32_t, kMaximumConcurrency> append_counts{};
            std::size_t append_size = 0;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (!cancelled[row] && terminal[row]) {
                    append_lanes[append_size]  = lanes[row];
                    append_starts[append_size] = requests[lanes[row]].pending.base_E;
                    append_counts[append_size] = accepted_tokens[row];
                    ++append_size;
                }
            }
            if (append_size != 0) {
                enqueue_dflash_context_append(
                    std::span<const std::uint32_t>(append_lanes.data(), append_size),
                    std::span<const std::uint32_t>(append_starts.data(), append_size),
                    std::span<const std::uint32_t>(append_counts.data(), append_size));
            }
        }

        // Peer first, then rank 0 -- the same order every other tp2 handler uses, so that
        // `clear_lane` below cannot release KV pages rank 1's fold still references.
        for (auto& peer : peers) { peer->device.synchronize(); }
        device.synchronize();
        work.reset();
        for (auto& peer : peers) { peer->work.reset(); }
    } catch (...) {
        try {
            for (auto& peer : peers) { peer->device.synchronize(); }
            device.synchronize();
        } catch (...) {}
        work.reset();
        for (auto& peer : peers) { peer->work.reset(); }
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }

    const double tail_seconds = std::chrono::duration<double>(Clock::now() - tail_started).count();
    const std::uint32_t width = draft_window + 1U;
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            RequestControl& request = requests[lanes[row]];
            if (cancelled[row]) {
                clear_lane(sequence, request);
                continue;
            }

            const PendingCandidate pending = request.pending;
            const std::uint32_t committed  = accepted_tokens[row];
            const TokenId* token_base =
                speculative_backend == SpeculativeBackend::Mtp
                    ? mtp_host_egress->licensed_tokens.data() + row * width
                    : dflash_host_egress->licensed_tokens.data() + row * width;
            sequence.ledger.insert(sequence.ledger.end(), token_base, token_base + committed);
            if (request.grammar) { request.grammar->commit({token_base, committed}); }
            sequence.prefix_identity.append_generated(committed, sequence.rope_delta);
            sequence.execution_frontier = pending.base_E + committed;
            sequence.ledger_frontier    = pending.base_S + committed;
            sequence.text_kv_valid      = sequence.execution_frontier;
            sequence.tail_hidden_valid  = true;

            if (speculative_backend == SpeculativeBackend::Mtp) {
                sequence.mtp_kv_valid = sequence.execution_frontier;
                if (terminal[row]) {
                    sequence.mtp_draft_count = 0;
                } else {
                    const std::int32_t next  = mtp_host_egress->next_extents[row];
                    sequence.mtp_draft_count = static_cast<std::uint32_t>(next);
                    for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                        sequence.mtp_drafts[step] =
                            mtp_host_egress->next_drafts[step * max_concurrency + row];
                    }
                }
            } else {
                sequence.dflash_context_frontier =
                    terminal[row] ? sequence.execution_frontier : pending.base_E;
            }

            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            if (terminal[row]) {
                release_sequence_growth_entitlement(sequence);
                unbind_sequence_kv(sequence);
                sequence.retained = true;
                request.lifecycle = Lifecycle::Complete;
            } else {
                request.lifecycle = Lifecycle::Active;
            }
            request.pending = {};
            request.timings.decode_seconds += tail_seconds;
        }
    } catch (...) {
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

void ProgramImplCore::abort_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    clear_lane(sequences[lane], requests[lane]);
}

bool ProgramImplCore::has_retained_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency && sequences[lane].retained;
}

void ProgramImplCore::evict_retained_lane(std::uint32_t lane) {
    if (!has_retained_lane(lane)) { return; }
    archive_prefix(sequences[lane]);
    clear_lane(sequences[lane], requests[lane]);
}

GenerationTimings ProgramImplCore::generation_timings_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].timings : GenerationTimings{};
}

const SpeculativeStats& ProgramImplCore::speculative_stats_lane(std::uint32_t lane) const noexcept {
    static const SpeculativeStats empty;
    return lane < max_concurrency ? requests[lane].speculative_stats : empty;
}

std::size_t ProgramImplCore::prefix_checkpoint_bytes() const {
    std::size_t bytes = decoder->linear_attention.slot_bytes() + sequences[0].tail_hidden.bytes();
    for (const auto& peer : peers) { bytes += peer->decoder->linear_attention.slot_bytes(); }
    return bytes;
}

void ProgramImplCore::save_prefix_checkpoint(SequenceState& sequence, std::uint32_t frontier,
                                             const Tensor& hidden) {
    const std::size_t bytes = prefix_checkpoint_bytes();
    const std::size_t limit = kPrefixCheckpointBudget / max_concurrency / bytes;
    if (limit == 0) { return; }
    auto& checkpoints = sequence.prefix_checkpoints;
    if (!checkpoints.empty() && checkpoints.back().frontier == frontier) { return; }
    PrefixCheckpoint checkpoint;
    checkpoint.frontier = frontier;
    const auto unused = std::find_if(checkpoints.begin(), checkpoints.end(),
                                     [](const auto& entry) { return entry.frontier == 0; });
    if (unused != checkpoints.end()) {
        checkpoint.image = std::move(unused->image);
        checkpoints.erase(unused);
    } else if (checkpoints.size() >= limit) {
        // Keep the first image and spread the remaining images over the history.
        // Recycle the most redundant interior image, including the incoming endpoint.
        std::size_t victim = 0;
        if (limit > 1) {
            victim = 1;
            std::uint32_t closest = std::numeric_limits<std::uint32_t>::max();
            for (std::size_t i = 1; i < checkpoints.size(); ++i) {
                const auto right = i + 1 < checkpoints.size() ? checkpoints[i + 1].frontier : frontier;
                const auto span = right - checkpoints[i - 1].frontier;
                if (span < closest) { closest = span; victim = i; }
            }
        }
        checkpoint.image = std::move(checkpoints[victim].image);
        checkpoints.erase(checkpoints.begin() + static_cast<std::ptrdiff_t>(victim));
    }
    checkpoints.push_back(capture_prefix_checkpoint(sequence, frontier, hidden, std::move(checkpoint.image)));
}

PrefixCheckpoint ProgramImplCore::capture_prefix_checkpoint(SequenceState& sequence,
    std::uint32_t frontier, const Tensor& hidden, std::shared_ptr<std::byte[]> image) {
    const std::size_t bytes = prefix_checkpoint_bytes();
    PrefixCheckpoint checkpoint{frontier, std::move(image)};
    if (!checkpoint.image || checkpoint.image.use_count() != 1) {
        checkpoint.image = std::shared_ptr<std::byte[]>(new std::byte[bytes]);
    }
    std::size_t offset = 0;
    const auto capture = [&](LinearAttentionStatePool& pool, DeviceContext& ctx) {
        const ScopedDevice scope(ctx.device);
        const auto size = pool.slot_bytes();
        pool.export_slot(LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                         std::span(checkpoint.image.get(), bytes).subspan(offset, size), ctx.stream);
        offset += size;
    };
    capture(decoder->linear_attention, device);
    for (auto& peer : peers) { capture(peer->decoder->linear_attention, peer->device); }
    CUDA_CHECK(cudaMemcpyAsync(checkpoint.image.get() + offset, hidden.data, hidden.bytes(),
                               cudaMemcpyDeviceToHost, device.stream));
    device.synchronize();
    return checkpoint;
}

void ProgramImplCore::restore_prefix_checkpoint(SequenceState& sequence, std::uint32_t frontier,
                                                bool restore_linear_state) {
    const auto found = std::find_if(sequence.prefix_checkpoints.begin(), sequence.prefix_checkpoints.end(),
                                    [frontier](const auto& entry) { return entry.frontier == frontier; });
    if (found == sequence.prefix_checkpoints.end() || !found->image) {
        throw std::logic_error("planned prefix state image is unavailable");
    }
    std::size_t offset = 0;
    const auto restore = [&](LinearAttentionStatePool& pool, DeviceContext& ctx) {
        const ScopedDevice scope(ctx.device);
        const auto size = pool.slot_bytes();
        if (restore_linear_state) {
            pool.import_slot(LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                std::span<const std::byte>(found->image.get(), prefix_checkpoint_bytes())
                    .subspan(offset, size), ctx.stream);
        }
        offset += size;
    };
    restore(decoder->linear_attention, device);
    for (auto& peer : peers) { restore(peer->decoder->linear_attention, peer->device); }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, found->image.get() + offset,
                               sequence.tail_hidden.bytes(), cudaMemcpyHostToDevice, device.stream));
    device.synchronize();
    sequence.tail_hidden_valid = true;
    sequence.deferred_prefix_state = restore_linear_state ? std::nullopt
                                                          : std::optional(frontier);
}

namespace {
std::uint32_t prefix_page_count(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0 : 1 + (tokens - 1) / kPagedKVPageSize;
}
}

void ProgramImplCore::trim_prefix_history() {
    // Account unique immutable KV chunks, not repeated shared prefixes or TP ranks.
    const auto usage = [&] {
        std::vector<const PrefixKVChunk*> chunks;
        std::vector<const std::byte*> images;
        std::uint64_t tokens = 0;
        for (const auto& branch : prefix_history) {
            for (const auto& chunk : branch->chunks) {
                if (std::find(chunks.begin(), chunks.end(), chunk.get()) == chunks.end()) {
                    chunks.push_back(chunk.get());
                    tokens += std::uint64_t(chunk->pages) * kPagedKVPageSize;
                }
            }
            for (const auto& checkpoint : branch->checkpoints) {
                if (std::find(images.begin(), images.end(), checkpoint.image.get()) == images.end()) {
                    images.push_back(checkpoint.image.get());
                }
            }
        }
        return std::pair(tokens, images.size() * prefix_checkpoint_bytes());
    };
    while (!prefix_history.empty()) {
        const auto [tokens, state_bytes] = usage();
        if (tokens <= prefix_cache_tokens && state_bytes <= kPrefixCheckpointBudget) { break; }
        if (tokens <= prefix_cache_tokens) {
            // Keep branch endpoints; interior frontiers are expendable under state pressure.
            bool removed = false;
            for (auto& branch : prefix_history) {
                if (branch->checkpoints.size() > 1) {
                    branch->checkpoints.erase(branch->checkpoints.begin());
                    removed = true;
                    break;
                }
            }
            if (removed) { continue; }
        }
        prefix_history.erase(prefix_history.begin()); // FIFO oldest branch, including its private suffix.
        for (auto& sequence : sequences) {
            sequence.prefix_chunks.clear();
            sequence.shared_prefix_tokens = 0;
        }
    }
}

void ProgramImplCore::archive_prefix(SequenceState& sequence) {
    if (!sequence.archived_snapshot.expired()) { return; }
    const std::uint32_t frontier = sequence.execution_frontier;
    const auto pages = prefix_page_count(frontier);
    if (prefix_cache_tokens == 0 || !sequence.retained || !sequence.history_enabled ||
        !sequence.kv || frontier == 0 || sequence.text_kv_valid < frontier ||
        std::uint64_t(pages) * kPagedKVPageSize > prefix_cache_tokens) { return; }
    // Archive capture must not recycle the lane's checkpoints: an admission plan
    // can already name one of them, even when the lane has room for only one image.
    std::vector<PrefixCheckpoint> saved_checkpoints;
    for (const auto& checkpoint : sequence.prefix_checkpoints) {
        if (checkpoint.frontier != 0 && checkpoint.frontier <= frontier) {
            saved_checkpoints.push_back(checkpoint);
        }
    }
    if (std::none_of(saved_checkpoints.begin(), saved_checkpoints.end(),
                    [frontier](const auto& checkpoint) { return checkpoint.frontier == frontier; })) {
        if (sequence.deferred_prefix_state) {
            restore_prefix_checkpoint(sequence, *sequence.deferred_prefix_state);
        }
        if (!sequence.tail_hidden_valid) { return; }
        saved_checkpoints.push_back(capture_prefix_checkpoint(sequence, frontier, sequence.tail_hidden));
    }
    auto archive = std::make_shared<ArchivedPrefix>();
    archive->frontier = frontier;
    archive->mtp_valid = std::min(frontier, sequence.mtp_kv_valid);
    archive->ledger.assign(sequence.ledger.begin(), sequence.ledger.begin() + frontier);
    archive->identity = sequence.prefix_identity;
    archive->checkpoints = std::move(saved_checkpoints);
    std::sort(archive->checkpoints.begin(), archive->checkpoints.end(),
              [](const auto& a, const auto& b) { return a.frontier < b.frontier; });
    constexpr std::uint32_t chunk_pages = 64; // 4096 tokens, bounded tail recopy at a fork.
    const auto backend_pages = prefix_page_count(archive->mtp_valid);
    for (std::uint32_t first = 0; first < pages; first += chunk_pages) {
        const std::uint32_t count = std::min(chunk_pages, pages - first);
        const auto inherited = std::find_if(sequence.prefix_chunks.begin(), sequence.prefix_chunks.end(),
            [&](const auto& chunk) {
                return chunk->first_page == first && chunk->pages == count &&
                       std::uint64_t(first + count) * kPagedKVPageSize <= sequence.shared_prefix_tokens;
            });
        if (inherited != sequence.prefix_chunks.end()) {
            archive->chunks.push_back(*inherited);
            continue;
        }
        auto chunk = std::make_shared<PrefixKVChunk>();
        chunk->first_page = first;
        chunk->pages = count;
        chunk->backend_pages = first < backend_pages ? std::min(count, backend_pages - first) : 0;
        const auto capture = [&](const PagedKVAllocation& text,
                                  const std::optional<PagedKVAllocation>& backend,
                                  DeviceContext& ctx, int rank) {
            const ScopedDevice scope(ctx.device);
            chunk->text[rank].resize(text.snapshot_bytes(count));
            text.export_pages(first, count, chunk->text[rank], ctx.stream);
            if (backend && chunk->backend_pages != 0) {
                chunk->backend[rank].resize(backend->snapshot_bytes(chunk->backend_pages));
                backend->export_pages(first, chunk->backend_pages, chunk->backend[rank], ctx.stream);
            }
        };
        capture(sequence.kv->text, sequence.kv->backend, device, 0);
        for (auto& peer : peers) {
            capture(*sequence.kv->text_peers[peer->rank - 1],
                    sequence.kv->backend_peers[peer->rank - 1], peer->device, peer->rank);
        }
        archive->chunks.push_back(std::move(chunk));
    }
    prefix_history.push_back(archive);
    sequence.archived_snapshot = archive;
    sequence.prefix_chunks = archive->chunks;
    // The final partially valid page must never be shared with a newly written suffix.
    sequence.shared_prefix_tokens = std::min(frontier, archive->mtp_valid == 0 ? frontier : archive->mtp_valid);
    trim_prefix_history();
}

void ProgramImplCore::restore_archived_prefix(SequenceState& sequence, const ArchivedPrefix& archive,
                                               const RequestPlanImpl& plan) {
    const auto base = plan.reuse_base;
    sequence.kv.reset();
    reserve_sequence_kv(sequence, plan.text_kv_page_entitlement, plan.backend_kv_page_entitlement);
    const auto mtp_base = speculative_backend == SpeculativeBackend::Mtp ? base - 1 : 0;
    materialize_sequence_kv(sequence, base, mtp_base);
    const auto pages = prefix_page_count(base);
    const auto mtp_pages = prefix_page_count(mtp_base);
    for (const auto& chunk : archive.chunks) {
        if (chunk->first_page >= pages) { break; }
        // Chunk images are plane/head-major; a partial range must be sliced per plane
        // by the allocation transport rather than treating its byte image as a prefix.
        // Import complete chunks, including the private page tail, then trim logically.
        const auto restore = [&](PagedKVAllocation& text,
                                  std::optional<PagedKVAllocation>& backend,
                                  DeviceContext& ctx, int rank) {
            const ScopedDevice scope(ctx.device);
            const auto count = std::min(chunk->pages, pages - chunk->first_page);
            text.import_page_prefix(chunk->first_page, count, chunk->pages, chunk->text[rank], ctx.stream);
            if (backend && chunk->first_page < mtp_pages) {
                const auto bcount = std::min(chunk->backend_pages, mtp_pages - chunk->first_page);
                backend->import_page_prefix(chunk->first_page, bcount, chunk->backend_pages,
                                             chunk->backend[rank], ctx.stream);
            }
        };
        restore(sequence.kv->text, sequence.kv->backend, device, 0);
        for (auto& peer : peers) {
            restore(*sequence.kv->text_peers[peer->rank - 1],
                    sequence.kv->backend_peers[peer->rank - 1], peer->device, peer->rank);
        }
    }
    sequence.ledger = archive.ledger;
    sequence.prefix_identity = archive.identity;
    // The archive may have been captured with a larger per-lane state budget.
    // Keep only the chosen frontier resident; other frontiers stay in history.
    const auto checkpoint = std::find_if(archive.checkpoints.begin(), archive.checkpoints.end(),
        [base](const auto& entry) { return entry.frontier == base; });
    if (checkpoint == archive.checkpoints.end()) {
        throw std::logic_error("archived prefix state image is unavailable");
    }
    sequence.prefix_checkpoints = {*checkpoint};
    sequence.execution_frontier = base;
    sequence.text_kv_valid = base;
    sequence.mtp_kv_valid = mtp_base;
    sequence.rewrite_checkpoint = {};
    sequence.deferred_prefix_state.reset();
    sequence.retained = true;
    sequence.prefix_chunks = archive.chunks;
    sequence.shared_prefix_tokens = mtp_base == 0 ? base : mtp_base;
}

void ProgramImplCore::clear_lane(SequenceState& sequence, RequestControl& request) noexcept {
    request.prefill.reset();
    request.exact_hit.reset();
    request.grammar.reset();
    sequence.kv.reset();
    request.lifecycle           = Lifecycle::Empty;
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.prefix_identity.clear();
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
    sequence.mtp_draft_count         = 0;
    sequence.tail_hidden_valid       = false;
    sequence.retained                = false;
    sequence.rewrite_checkpoint      = {};
    sequence.prefix_checkpoints.clear();
    sequence.prefix_chunks.clear();
    sequence.shared_prefix_tokens = 0;
    sequence.history_enabled = false;
    sequence.archived_snapshot.reset();
    sequence.deferred_prefix_state.reset();
    request.pending                  = {};
}

qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash && dflash->full) {
        return &*dflash->full;
    }
    return nullptr;
}

const qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash && dflash->full) {
        return &*dflash->full;
    }
    return nullptr;
}

std::uint32_t ProgramImplCore::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return sequence.mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash && backend_kv_cache() != nullptr) {
        return sequence.dflash_context_frontier;
    }
    return 0;
}

void ProgramImplCore::reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                                          std::uint32_t backend_pages) {
    if (sequence.kv) { throw std::logic_error("sequence already owns a KV allocation bundle"); }
    if (text_pages == 0 || (backend_kv_cache() == nullptr) != (backend_pages == 0)) {
        throw std::invalid_argument("KV allocation entitlement does not match the active backend");
    }

    std::array<PagedKVReservation, 2> reservations{};
    std::size_t count     = 0;
    reservations[count++] = PagedKVReservation{
        .pool             = &decoder->text_kv.pool(),
        .page_entitlement = text_pages,
    };
    if (qwen3_6::PagedKVCache* backend = backend_kv_cache(); backend != nullptr) {
        reservations[count++] = PagedKVReservation{
            .pool             = &backend->pool(),
            .page_entitlement = backend_pages,
        };
    }

    std::vector<PagedKVAllocation> allocations =
        reserve_paged_kv_bundle(std::span<const PagedKVReservation>(reservations.data(), count));
    SequenceKVBundle bundle;
    bundle.text = std::move(allocations[0]);
    if (count == 2) { bundle.backend.emplace(std::move(allocations[1])); }
    for (auto& peer : peers) {
        // Same pool geometry, same call order, same entitlement: rank 1's reservation takes the
        // same page ids, so the two block tables agree without ever being compared. The MTP pool
        // rides the SAME bundle call as the text pool, so its page ids match rank 0's too.
        std::array<PagedKVReservation, 2> peer_reservations{};
        std::size_t peer_count     = 0;
        peer_reservations[peer_count++] =
            PagedKVReservation{.pool = &peer->decoder->text_kv.pool(), .page_entitlement = text_pages};
        qwen3_6::PagedKVCache* peer_backend =
            speculative_backend == SpeculativeBackend::Mtp
                ? peer->decoder->mtp_cache()
                : (peer->dflash && peer->dflash->full ? &*peer->dflash->full : nullptr);
        if (peer_backend != nullptr) {
            peer_reservations[peer_count++] =
                PagedKVReservation{.pool = &peer_backend->pool(), .page_entitlement = backend_pages};
        }
        if (peer_count != count) {
            throw std::logic_error("peer KV pool set does not match rank 0's");
        }
        std::vector<PagedKVAllocation> peer_allocations = reserve_paged_kv_bundle(
            std::span<const PagedKVReservation>(peer_reservations.data(), peer_count));
        bundle.text_peers[peer->rank - 1].emplace(std::move(peer_allocations[0]));
        if (peer_count == 2) { bundle.backend_peers[peer->rank - 1].emplace(std::move(peer_allocations[1])); }
    }
    sequence.kv.emplace(std::move(bundle));
}

void ProgramImplCore::resize_sequence_kv_entitlement(SequenceState& sequence,
                                                     std::uint32_t text_pages,
                                                     std::uint32_t backend_pages) {
    if (!sequence.kv || text_pages == 0 ||
        (sequence.kv->backend.has_value() != (backend_pages != 0))) {
        throw std::invalid_argument("KV resize entitlement does not match the sequence bundle");
    }
    std::array<PagedKVResize, 2> changes{};
    std::size_t count = 0;
    changes[count++]  = PagedKVResize{
         .allocation       = &sequence.kv->text,
         .mapped_pages     = sequence.kv->text.mapped_page_count(),
         .page_entitlement = text_pages,
    };
    if (sequence.kv->backend) {
        changes[count++] = PagedKVResize{
            .allocation       = &*sequence.kv->backend,
            .mapped_pages     = sequence.kv->backend->mapped_page_count(),
            .page_entitlement = backend_pages,
        };
    }
    resize_paged_kv_bundle(std::span<PagedKVResize>(changes.data(), count));
    for (auto& peer : peers) {
        std::array<PagedKVResize, 2> peer_changes{};
        std::size_t peer_count = 0;
        peer_changes[peer_count++] =
            PagedKVResize{.allocation       = &*sequence.kv->text_peers[peer->rank - 1],
                          .mapped_pages     = sequence.kv->text_peers[peer->rank - 1]->mapped_page_count(),
                          .page_entitlement = text_pages};
        if (sequence.kv->backend_peers[peer->rank - 1]) {
            peer_changes[peer_count++] =
                PagedKVResize{.allocation   = &*sequence.kv->backend_peers[peer->rank - 1],
                              .mapped_pages = sequence.kv->backend_peers[peer->rank - 1]->mapped_page_count(),
                              .page_entitlement = backend_pages};
        }
        resize_paged_kv_bundle(std::span<PagedKVResize>(peer_changes.data(), peer_count));
    }
}

void ProgramImplCore::bind_sequence_kv(SequenceState& sequence) {
    if (!sequence.kv || sequence.kv->text.bound_row() >= 0 ||
        (sequence.kv->backend && sequence.kv->backend->bound_row() >= 0)) {
        throw std::logic_error("KV allocation bundle is unavailable or already bound");
    }
    const std::int32_t row = static_cast<std::int32_t>(sequence.lane);
    sequence.kv->text.bind_row(row, device.stream);
    try {
        if (sequence.kv->backend) { sequence.kv->backend->bind_row(row, device.stream); }
        for (auto& peer : peers) {
            const ScopedDevice scope(peer->device.device);
            sequence.kv->text_peers[peer->rank - 1]->bind_row(row, peer->device.stream);
            set_peer_i32(peer->io.text_kv_table_row, sequence.kv->text_peers[peer->rank - 1]->bound_row(), peer->device);
            if (sequence.kv->backend_peers[peer->rank - 1]) {
                sequence.kv->backend_peers[peer->rank - 1]->bind_row(row, peer->device.stream);
            }
            set_peer_i32(peer->io.backend_kv_table_row,
                         sequence.kv->backend_peers[peer->rank - 1] ? sequence.kv->backend_peers[peer->rank - 1]->bound_row() : 0, peer->device);
        }
        set_device_i32(io.text_kv_table_row, sequence.kv->text.bound_row());
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? sequence.kv->backend->bound_row() : 0);
    } catch (...) {
        for (auto& peer : peers) {
            auto& backend = sequence.kv->backend_peers[peer->rank - 1];
            auto& text = sequence.kv->text_peers[peer->rank - 1];
            if (backend && backend->bound_row() >= 0) { backend->unbind_row(); }
            if (text && text->bound_row() >= 0) { text->unbind_row(); }
        }
        if (sequence.kv->backend && sequence.kv->backend->bound_row() >= 0) {
            sequence.kv->backend->unbind_row();
        }
        sequence.kv->text.unbind_row();
        throw;
    }
}

void ProgramImplCore::unbind_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    for (auto& peer : peers) {
        auto& backend = sequence.kv->backend_peers[peer->rank - 1];
        if (backend) { backend->unbind_row(); }
    }
    for (auto& peer : peers) { sequence.kv->text_peers[peer->rank - 1]->unbind_row(); }
    if (sequence.kv->backend) { sequence.kv->backend->unbind_row(); }
    sequence.kv->text.unbind_row();
}

void ProgramImplCore::refresh_ram_kv(SequenceState& sequence, std::uint32_t frontier) {
    if (ram_kv.gpu_tokens == 0 || frontier <= ram_kv.gpu_tokens) { return; }
    std::vector<double> combined(frontier / 128, 0);
    const auto score = [&](qwen3_6::PagedKVCache& cache, DeviceContext& ctx) {
        const ScopedDevice scope(ctx.device);
        ops::kv_retrieval_scores(cache.retrieval_keys, cache.retrieval_query,
            cache.retrieval_fallback, cache.retrieval_counts, frontier, cache.retrieval_scores, ctx.stream);
        std::vector<float> scores(combined.size());
        CUDA_CHECK(cudaMemcpyAsync(scores.data(), cache.retrieval_scores.data,
            scores.size() * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        for (std::size_t b = 0; b < scores.size(); ++b) { combined[b] += scores[b] / device_count; }
    };
    score(decoder->text_kv, device);
    for (auto& peer : peers) { score(peer->decoder->text_kv, peer->device); }
    const auto ranking = qwen3_6::detail::rank_ram_kv_pages(combined);
    sequence.kv->text.prefer_pages(ranking);
    if (sequence.kv->backend) { sequence.kv->backend->prefer_pages(ranking); }
    for (auto& peer : peers) {
        sequence.kv->text_peers[peer->rank - 1]->prefer_pages(ranking);
        if (sequence.kv->backend_peers[peer->rank - 1]) {
            sequence.kv->backend_peers[peer->rank - 1]->prefer_pages(ranking);
        }
    }
    sequence.ram_rank_frontier = frontier;
    if (std::getenv("NINFER_RAM_KV_TRACE")) {
        std::fprintf(stderr, "RAM_KV_RETRIEVAL frontier=%u blocks=%zu query=%zu:%zu\n",
            frontier, combined.size(), sequence.ram_query.begin, sequence.ram_query.count);
    }
}

void ProgramImplCore::materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                              std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    if (ram_kv.gpu_tokens != 0 || main_tokens > sequence.kv->text.mapped_token_capacity()) {
        sequence.kv->text.materialize_tokens(std::min(capacity,
            std::max(main_tokens, sequence.kv->text.mapped_token_capacity())), device.stream);
        for (auto& peer : peers) {
            const ScopedDevice scope(peer->device.device);
            sequence.kv->text_peers[peer->rank - 1]->materialize_tokens(
                std::min(capacity, std::max(main_tokens,
                    sequence.kv->text_peers[peer->rank - 1]->mapped_token_capacity())), peer->device.stream);
        }
        if (ram_kv.gpu_tokens != 0 && std::getenv("NINFER_RAM_KV_TRACE") &&
            main_tokens >= sequence.ram_query.begin) {
            std::fprintf(stderr, "RAM_KV_RESIDENT frontier=%u pages=", main_tokens);
            const auto pages = sequence.kv->text.page_ids();
            for (std::size_t p = 0; p < pages.size(); ++p) {
                if (pages[p] >= 0) { std::fprintf(stderr, "%zu,", p); }
            }
            std::fprintf(stderr, "\n");
        }
    }
    if (backend_tokens != 0 && (ram_kv.gpu_tokens != 0 ||
                               backend_tokens > sequence.kv->backend->mapped_token_capacity())) {
        sequence.kv->backend->materialize_tokens(std::min(capacity, std::max(backend_tokens,
            sequence.kv->backend->mapped_token_capacity())), device.stream);
        for (auto& peer : peers) {
            const ScopedDevice scope(peer->device.device);
            sequence.kv->backend_peers[peer->rank - 1]->materialize_tokens(std::min(capacity,
                std::max(backend_tokens, sequence.kv->backend_peers[peer->rank - 1]->mapped_token_capacity())), peer->device.stream);
        }
    }
}

void ProgramImplCore::trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                       std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    sequence.kv->text.trim_tokens(main_tokens);
    for (auto& peer : peers) { sequence.kv->text_peers[peer->rank - 1]->trim_tokens(main_tokens); }
    if (sequence.kv->backend) { sequence.kv->backend->trim_tokens(backend_tokens); }
    for (auto& peer : peers) {
        auto& backend = sequence.kv->backend_peers[peer->rank - 1];
        if (backend) { backend->trim_tokens(backend_tokens); }
    }
}

void ProgramImplCore::release_sequence_growth_entitlement(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    sequence.kv->text.cancel_unmapped_entitlement();
    for (auto& peer : peers) { sequence.kv->text_peers[peer->rank - 1]->cancel_unmapped_entitlement(); }
    if (sequence.kv->backend) { sequence.kv->backend->cancel_unmapped_entitlement(); }
    for (auto& peer : peers) {
        auto& backend = sequence.kv->backend_peers[peer->rank - 1];
        if (backend) { backend->cancel_unmapped_entitlement(); }
    }
}

qwen3_6::PagedKVCacheView ProgramImplCore::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv) { throw std::logic_error("sequence has no KV allocation bundle"); }
    return decoder->text_kv.execution_view(sequence.kv->text);
}

qwen3_6::PagedKVCacheView ProgramImplCore::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend) {
        throw std::logic_error("sequence has no MTP KV allocation");
    }
    return decoder->mtp_cache()->execution_view(*sequence.kv->backend);
}

std::array<qwen3_6::PagedKVCacheView, kMaximumExecutionDevices - 1>
ProgramImplCore::mtp_kv_views_peer(const SequenceState& sequence) const {
    std::array<qwen3_6::PagedKVCacheView, kMaximumExecutionDevices - 1> out{};
    if (speculative_backend != SpeculativeBackend::Mtp) { return out; }
    for (const auto& peer : peers) {
        const auto index = peer->rank - 1;
        if (peer->decoder->mtp_cache() == nullptr || !sequence.kv ||
            !sequence.kv->backend_peers[index]) {
            throw std::logic_error("sequence has no peer MTP KV allocation");
        }
        out[index] = peer->decoder->mtp_cache()->execution_view(*sequence.kv->backend_peers[index]);
    }
    return out;
}

void ProgramImplCore::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::set_peer_i32(Tensor& tensor, std::int32_t value, DeviceContext& rank_device) {
    CUDA_CHECK(cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice,
                               rank_device.stream));
}

void ProgramImplCore::ordered_reset(SequenceState& sequence) {
    decoder->linear_attention.zero_slot(
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency), device.stream);
    for (auto& peer : peers) {
        const ScopedDevice scope(peer->device.device);
        peer->decoder->linear_attention.zero_slot(
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
            peer->device.stream);
        peer->work.reset();
        set_peer_i32(peer->io.pos, 0, peer->device);
        set_peer_i32(peer->io.rope_pos, 0, peer->device);
        set_peer_i32(peer->io.rope_delta, 0, peer->device);
    }
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
}

void ProgramImplCore::prepare_graphs() {
    if (!use_cuda_graph) { return; }
    SequenceState& sequence = sequences[0];

    // Every helper below has to run the SAME operation on both devices at tp2, in the same order,
    // because rank 1's runtime is a structural mirror of rank 0's: same page ids, same block
    // tables, same zeroed state. `on_peer` is the one place that establishes the peer device as
    // current; the mirrors below never touch the current device themselves.
    const auto on_peer = [&](auto&& body) {
        for (auto& peer : peers) {
            const ScopedDevice scope(peer->device.device);
            body(*peer);
        }
    };
    const auto synchronize_all = [&] {
        for (auto& peer : peers) { peer->device.synchronize(); }
        device.synchronize();
    };

    std::vector<PagedKVAllocation> text_capture_allocations;
    std::array<std::vector<PagedKVAllocation>, kMaximumExecutionDevices - 1> peer_text_capture_allocations;
    std::vector<PagedKVAllocation> mtp_capture_allocations;
    std::array<std::vector<PagedKVAllocation>, kMaximumExecutionDevices - 1> peer_mtp_capture_allocations;
    std::vector<PagedKVAllocation> dflash_capture_allocations;
    const auto reserve_rows_in = [&](qwen3_6::PagedKVCache& cache,
                                     std::vector<PagedKVAllocation>& allocations,
                                     cudaStream_t stream, const char* label) {
        PagedKVPool& pool = cache.pool();
        if (pool.page_group_count() < max_concurrency) {
            throw std::invalid_argument(std::string(label) +
                                        " cannot provide one Paged KV page per concurrent request");
        }
        allocations.reserve(max_concurrency);
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            allocations.push_back(pool.reserve(1));
            PagedKVAllocation& allocation = allocations.back();
            allocation.bind_row(static_cast<std::int32_t>(row), stream);
            allocation.materialize_pages(1, stream);

            // Capture profiles exercise arbitrary context envelopes. Repeating each row's private
            // page across its temporary table keeps every dummy read/write address valid without
            // reserving C full contexts solely for graph construction.
            const std::int32_t page = allocation.page_ids().front();
            std::vector<std::int32_t> repeated(pool.logical_page_capacity(), page);
            Tensor table = pool.block_table_row(static_cast<std::int32_t>(row));
            CUDA_CHECK(cudaMemcpyAsync(table.data, repeated.data(), table.bytes(),
                                       cudaMemcpyHostToDevice, stream));
            if (pool.ram_enabled()) {
                std::vector<std::int32_t> indices(pool.logical_page_capacity());
                for (std::size_t p = 0; p < indices.size(); ++p) {
                    indices[p] = static_cast<std::int32_t>(p % pool.page_group_count());
                }
                CUDA_CHECK(cudaMemcpyAsync(pool.read_indices().data, indices.data(),
                                           indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(pool.read_tables().data, repeated.data(),
                                           repeated.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
            }
        }
    };
    const auto reserve_capture_rows = [&](qwen3_6::PagedKVCache& cache,
                                          std::vector<PagedKVAllocation>& allocations,
                                          const char* label) {
        reserve_rows_in(cache, allocations, device.stream, label);
    };
    reserve_capture_rows(decoder->text_kv, text_capture_allocations, "target KV cache");
    // The peer's pool has identical page geometry and receives the identical reserve sequence, so
    // the two allocations take the same page ids and publish identical block tables -- the same
    // by-construction lockstep the live KV bundle relies on.
    on_peer([&](PeerRuntime& p) {
        reserve_rows_in(p.decoder->text_kv, peer_text_capture_allocations[p.rank - 1], p.device.stream,
                        "peer target KV cache");
    });
    if (speculative_backend == SpeculativeBackend::Mtp) {
        reserve_capture_rows(*decoder->mtp_cache(), mtp_capture_allocations, "MTP KV cache");
        on_peer([&](PeerRuntime& p) {
            reserve_rows_in(*p.decoder->mtp_cache(), peer_mtp_capture_allocations[p.rank - 1], p.device.stream,
                            "peer MTP KV cache");
        });
    } else if (speculative_backend == SpeculativeBackend::DFlash && dflash->full) {
        reserve_capture_rows(*dflash->full, dflash_capture_allocations, "DFlash Full KV cache");
    }
    synchronize_all();

    // Graph residency is measured on EACH device: the single cross-device graph materializes
    // driver and module state on both, and the plan's allowance is a per-device budget.
    // cudaMemGetInfo insists on a total-bytes out-parameter that nothing here reads, so it is
    // consumed and discarded in one place rather than carried as a dead local.
    const auto free_device_bytes = [] {
        std::size_t free  = 0;
        std::size_t total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        return free;
    };
    std::array<std::size_t, kMaximumExecutionDevices> free_before{0, 0};
    free_before[0] = free_device_bytes();
    on_peer([&](PeerRuntime& p) { free_before[p.rank] = free_device_bytes(); });

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
        };
        if (io.mtp) {
            controls.push_back(io.mtp->position);
            controls.push_back(io.mtp->draft_tokens);
            controls.push_back(io.mtp->target_input_ids);
            controls.push_back(io.mtp->target_positions);
        }
        if (io.dflash_prefill) { controls.push_back(io.dflash_prefill->produced_count); }
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
        on_peer([&](PeerRuntime& p) {
            std::vector<Tensor> peer_controls{p.io.token, p.io.pos, p.io.rope_pos,
                                              p.io.rope_delta};
            if (p.io.mtp) {
                peer_controls.push_back(p.io.mtp->position);
                peer_controls.push_back(p.io.mtp->draft_tokens);
                peer_controls.push_back(p.io.mtp->target_input_ids);
                peer_controls.push_back(p.io.mtp->target_positions);
            }
            for (const Tensor& tensor : peer_controls) {
                CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), p.device.stream));
            }
        });
    };
    const auto zero_capture_pages = [&](qwen3_6::PagedKVCache& cache,
                                        const std::vector<PagedKVAllocation>& allocations,
                                        std::uint32_t batch_size) {
        std::vector<std::int32_t> pages;
        pages.reserve(batch_size);
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            pages.push_back(allocations[row].page_ids().front());
        }
        cache.pool().zero_pages(pages, device.stream);
    };
    const auto zero_peer_capture_pages = [&](std::uint32_t batch_size) {
        on_peer([&](PeerRuntime& p) {
            const auto zero_pool = [&](qwen3_6::PagedKVCache& cache,
                                       const std::vector<PagedKVAllocation>& allocations) {
                std::vector<std::int32_t> pages;
                pages.reserve(batch_size);
                for (std::uint32_t row = 0; row < batch_size; ++row) {
                    pages.push_back(allocations[row].page_ids().front());
                }
                cache.pool().zero_pages(pages, p.device.stream);
            };
            zero_pool(p.decoder->text_kv, peer_text_capture_allocations[p.rank - 1]);
            if (p.decoder->mtp_cache() != nullptr) {
                zero_pool(*p.decoder->mtp_cache(), peer_mtp_capture_allocations[p.rank - 1]);
            }
        });
    };
    const auto zero_cyclic_lane = [&](CyclicKVCache& cache, std::uint32_t lane) {
        for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = cache.layer_view(layer);
            const Tensor k                    = view.k.slice(3, static_cast<std::int32_t>(lane), 1);
            const Tensor v                    = view.v.slice(3, static_cast<std::int32_t>(lane), 1);
            CUDA_CHECK(cudaMemsetAsync(k.data, 0, k.bytes(), device.stream));
            CUDA_CHECK(cudaMemsetAsync(v.data, 0, v.bytes(), device.stream));
        }
    };

    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size) {
        if (batch_size == 0 || batch_size > max_concurrency) {
            throw std::logic_error("CUDA Graph representative batch is invalid");
        }
        work.reset();
        on_peer([&](PeerRuntime& p) { p.work.reset(); });
        clear_stable_controls();
        zero_capture_pages(decoder->text_kv, text_capture_allocations, batch_size);
        zero_peer_capture_pages(batch_size);
        if (decoder->mtp_cache() != nullptr) {
            zero_capture_pages(*decoder->mtp_cache(), mtp_capture_allocations, batch_size);
        }
        if (dflash && dflash->full) {
            zero_capture_pages(*dflash->full, dflash_capture_allocations, batch_size);
        }
        on_peer([&](PeerRuntime& p) {
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                p.decoder->linear_attention.zero_slot(
                    LinearStateSlots::current_state_slot(row, max_concurrency), p.device.stream);
            }
        });
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            decoder->linear_attention.zero_slot(
                LinearStateSlots::current_state_slot(row, max_concurrency), device.stream);
            if (dflash) {
                zero_cyclic_lane(dflash->local, row);
                const Tensor pending =
                    dflash->pending_features.slice(2, static_cast<std::int32_t>(row), 1);
                CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), device.stream));
            }
        }
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        on_peer([&](PeerRuntime& p) {
            set_peer_i32(p.io.pos, checked_i32(frontier, "peer graph representative position"), p.device);
            set_peer_i32(p.io.rope_pos,
                         checked_i32(frontier, "peer graph representative rope position"), p.device);
        });
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           checked_i32(frontier, "graph representative MTP position"));
        }
        on_peer([&](PeerRuntime& p) {
            if (p.io.mtp) {
                set_peer_i32(p.io.mtp->position,
                             checked_i32(frontier, "peer graph representative MTP position"), p.device);
            }
        });
        if (io.dflash_decode) {
            *dflash_host_ingress       = {};
            *dflash_host_egress        = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                dflash_host_ingress->anchors[row] = 0;
                dflash_host_ingress->execution_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash frontier");
                dflash_host_ingress->context_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash context frontier");
                dflash_host_ingress->proposal_extents[row] = static_cast<std::int32_t>(extent);
                dflash_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                dflash_host_ingress->text_kv_table_rows[row]   = static_cast<std::int32_t>(row);
                dflash_host_ingress->dflash_kv_table_rows[row] = static_cast<std::int32_t>(row);
                dflash_host_ingress->lanes[row]                = static_cast<std::int32_t>(row);
                dflash_host_ingress->sampling[row]             = {};
            }
            for (auto& peer : peers) { if (peer->dflash_host_ingress != nullptr) {
                *peer->dflash_host_ingress = *dflash_host_ingress;
            } }
        }
        if (io.mtp_decode) {
            *mtp_host_ingress          = {};
            *mtp_host_egress           = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            const std::uint32_t width  = draft_window + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                mtp_host_ingress->anchors[row] = 0;
                mtp_host_ingress->base_frontiers[row] =
                    checked_i32(frontier, "graph representative MTP frontier");
                mtp_host_ingress->remaining_budgets[row] =
                    checked_i32(capacity, "graph representative MTP budget");
                mtp_host_ingress->current_extents[row] = static_cast<std::int32_t>(extent);
                mtp_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t step = 0; step < draft_window; ++step) {
                    mtp_host_ingress->current_drafts[row * draft_window + step] = 0;
                }
                for (std::uint32_t column = 0; column < width; ++column) {
                    mtp_host_ingress->target_rope_positions[row * width + column] =
                        checked_i32(frontier + std::min(column, extent),
                                    "graph representative MTP RoPE position");
                }
                mtp_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                mtp_host_ingress->mtp_kv_table_rows[row]  = static_cast<std::int32_t>(row);
                mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(row);
                mtp_host_ingress->rope_deltas[row]        = 0;
                mtp_host_ingress->sampling[row]           = {};
            }
            // The representative's sampling configs are zeroed, so its counter pointers are null
            // and the peer copy is a plain mirror -- but it still has to exist, because the
            // captured graph bakes in the peer ingress's host ADDRESS and reads it at replay.
            for (auto& peer : peers) { if (peer->mtp_host_ingress != nullptr) { *peer->mtp_host_ingress = *mtp_host_ingress; } }
        }
        if (io.ordinary) {
            *ordinary_host_ingress = {};
            *ordinary_host_egress  = {};
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                ordinary_host_ingress->tokens[row] = 0;
                ordinary_host_ingress->cache_positions[row] =
                    checked_i32(frontier, "graph representative ordinary position");
                ordinary_host_ingress->rope_positions[row] =
                    checked_i32(frontier, "graph representative ordinary RoPE position");
                ordinary_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                ordinary_host_ingress->lanes[row]              = static_cast<std::int32_t>(row);
                ordinary_host_ingress->sampling[row]           = {};
            }
            // Same reason as the MTP mirror above: the representative's counter pointers are
            // already null, but the captured graph bakes in the peer ingress's host ADDRESS and
            // reads it at every replay, so the record has to exist and be current here.
            publish_peer_ordinary_ingress();
        }
    };
    const auto execution_core = [&] {
        return schedule::ExecutionCore{device,
                                       model,
                                       work,
                                       decoder->linear_attention,
                                       replay_records ? &*replay_records : nullptr,
                                       io,
                                       prefill_hidden,
                                       prefill_chunk,
                                       proposal_head,
                                       rope_frequency,
                                       peer_views(), pipeline_view(), pipeline_upload ? static_cast<TokenId*>(pipeline_upload->data()) : nullptr, volta_long_attention, token_constraints};
    };

    if (speculative_backend == SpeculativeBackend::None) {
        const auto ordinary_profiles = ordinary_graph_profiles(capacity);
        validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
        const std::uint32_t ordinary_batch_limit = max_concurrency;
        schedule::OrdinaryBatchContext ordinary_state{execution_core(),
                                                      decoder->text_kv,
                                                      *io.ordinary,
                                                      *ordinary_host_ingress,
                                                      *ordinary_host_egress,
                                                      tail_hidden_store};
        // One eager pass first, so every kernel's module is already loaded when capture runs: a
        // lazily loaded module inside a capture region is not capturable, and at tp2 that surface
        // exists on BOTH devices.
        const GraphExecutionProfile code_warm = ordinary_profiles.front();
        prepare_representative(code_warm.min, 1);
        synchronize_all();
        schedule::ordinary_decode_batch(ordinary_state, 1, {code_warm.min + 1, code_warm.max + 1},
                                        nullptr);
        synchronize_all();

        if (device_count > 1) {
            // Batch shape selects kernels, and a module first touched INSIDE a capture region
            // cannot be loaded there. At tp1 the code-warm pass above has always been enough in
            // practice (with one observed allowance flake, documented, consistent with a late
            // module load); at
            // tp2 the same surface exists on two devices and both are checked against the same
            // per-device allowance, so every batch size is warmed eagerly before any capture.
            for (std::uint32_t batch_size = 2; batch_size <= ordinary_batch_limit; ++batch_size) {
                prepare_representative(code_warm.min, batch_size);
                synchronize_all();
                schedule::ordinary_decode_batch(ordinary_state,
                                                static_cast<std::int32_t>(batch_size),
                                                {code_warm.min + 1, code_warm.max + 1}, nullptr);
                synchronize_all();
            }
        }

        ordinary_graphs.profiles.reserve(ordinary_profiles.size() * ordinary_batch_limit * 2U);
        for (bool constrained : {false, true}) {
            ordinary_state.execution.token_constraints =
                constrained ? token_constraints : schedule::TokenConstraintView{};
            for (std::uint32_t batch_size = 1; batch_size <= ordinary_batch_limit; ++batch_size) {
                for (const GraphExecutionProfile planned : ordinary_profiles) {
                    ordinary_graphs.profiles.emplace_back();
                    DecodeGraphProfile& profile    = ordinary_graphs.profiles.back();
                    profile.batch_size             = batch_size;
                    profile.min_execution_frontier = planned.min;
                    profile.max_execution_frontier = planned.max;
                    profile.constrained            = constrained;
                    profile.topology_class =
                        (planned.topology_class * 2U + (constrained ? 1U : 0U)) *
                            ordinary_batch_limit +
                        (batch_size - 1U);
                    const ops::GqaExecutionEnvelope envelope{planned.min + 1, planned.max + 1};
                    schedule::capture_ordinary_decode_batch(ordinary_state,
                                                            static_cast<std::int32_t>(batch_size),
                                                            envelope, profile.definition);
                }
            }
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const auto planned_profiles = mtp_graph_profiles(capacity, draft_window);
        validate_graph_profiles(planned_profiles, capacity - 1, "MTP");
        schedule::MtpBatchContext mtp_state{
            execution_core(),  decoder->text_kv, *decoder->mtp_cache(), *io.mtp_decode,
            *mtp_host_ingress, *mtp_host_egress, tail_hidden_store};
        const GraphExecutionProfile code_warm = planned_profiles.front();
        // Greedy verification exchanges shard winners; sampling exchanges full logits. Warm
        // both topologies, including TP2 batch-shape module selection, before capturing either.
        for (bool greedy_target : {false, true}) {
            mtp_state.greedy_target = greedy_target;
            const std::uint32_t warm_batches = device_count > 1 ? max_concurrency : 1U;
            for (std::uint32_t batch_size = 1; batch_size <= warm_batches; ++batch_size) {
                prepare_representative(code_warm.min, batch_size);
                synchronize_all();
                schedule::mtp_decode_batch(mtp_state, static_cast<std::int32_t>(batch_size),
                                           draft_window,
                                           mtp_gqa_envelopes(code_warm.max, draft_window,
                                                             capacity),
                                           nullptr);
                synchronize_all();
            }
        }

        mtp_graphs.profiles.reserve(planned_profiles.size() * max_concurrency * 4U);
        for (bool constrained : {false, true}) {
            mtp_state.execution.token_constraints =
                constrained ? token_constraints : schedule::TokenConstraintView{};
            for (bool greedy_target : {false, true}) {
                mtp_state.greedy_target = greedy_target;
                for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
                    for (const GraphExecutionProfile planned : planned_profiles) {
                        mtp_graphs.profiles.emplace_back();
                        DecodeGraphProfile& profile    = mtp_graphs.profiles.back();
                        profile.batch_size             = batch_size;
                        profile.min_execution_frontier = planned.min;
                        profile.max_execution_frontier = planned.max;
                        profile.greedy_target          = greedy_target;
                        profile.constrained            = constrained;
                        profile.topology_class =
                            ((planned.topology_class * 2U + (greedy_target ? 1U : 0U)) * 2U +
                             (constrained ? 1U : 0U)) *
                                max_concurrency +
                            (batch_size - 1U);
                        schedule::capture_mtp_decode_batch(
                            mtp_state, static_cast<std::int32_t>(batch_size), draft_window,
                            mtp_gqa_envelopes(planned.max, draft_window, capacity),
                            profile.definition);
                    }
                }
            }
        }
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        const auto batch_one_profiles = dflash_graph_profiles(capacity, draft_window, 1);
        validate_graph_profiles(batch_one_profiles, capacity - 1, "DFlash");
        schedule::DFlashBatchContext dflash_state{
            execution_core(),     decoder->text_kv,    *dflash,          *io.dflash_decode,
            *dflash_host_ingress, *dflash_host_egress, tail_hidden_store};
        const GraphExecutionProfile code_warm = batch_one_profiles.front();
        const ops::GqaExecutionEnvelope code_warm_target{
            1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                   capacity, static_cast<std::uint64_t>(code_warm.max) + draft_window + 1ULL))};
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        dflash_state.greedy_target = false;
        schedule::dflash_decode_batch(dflash_state, 1, draft_window,
                                      dflash_envelopes(code_warm.min, code_warm.max, draft_window),
                                      code_warm_target, nullptr);
        device.synchronize();

        // The greedy target-argmax route and the full-logit sampling route are separate CUDA
        // graph topologies. Warm both once so their modules are loaded before capture; runtime
        // request sampling selects the matching profile without changing semantics.
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        dflash_state.greedy_target = true;
        schedule::dflash_decode_batch(dflash_state, 1, draft_window,
                                      dflash_envelopes(code_warm.min, code_warm.max, draft_window),
                                      code_warm_target, nullptr);
        device.synchronize();

        dflash_graphs.profiles.reserve(batch_one_profiles.size() * max_concurrency * 2U);
        for (bool greedy_target : {false, true}) {
            dflash_state.greedy_target = greedy_target;
            for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
                const auto planned_profiles =
                    batch_size == 1 ? batch_one_profiles
                                    : dflash_graph_profiles(capacity, draft_window, batch_size);
                validate_graph_profiles(planned_profiles, capacity - 1, "DFlash");
                for (const GraphExecutionProfile planned : planned_profiles) {
                    dflash_graphs.profiles.emplace_back();
                    DecodeGraphProfile& profile    = dflash_graphs.profiles.back();
                    profile.batch_size             = batch_size;
                    profile.min_execution_frontier = planned.min;
                    profile.max_execution_frontier = planned.max;
                    profile.greedy_target          = greedy_target;
                    profile.topology_class =
                        (planned.topology_class * 2U + (greedy_target ? 1U : 0U)) *
                            max_concurrency +
                        (batch_size - 1U);
                    const ops::GqaExecutionEnvelope target_envelope{
                        1,
                        static_cast<std::uint32_t>(std::min<std::uint64_t>(
                            capacity, static_cast<std::uint64_t>(planned.max) + draft_window + 1ULL))};

                    schedule::capture_dflash_decode_batch(
                        dflash_state, static_cast<std::int32_t>(batch_size), draft_window,
                        dflash_envelopes(planned.min, planned.max, draft_window), target_envelope,
                        profile.definition);
                }
            }
        }
    }

    for (const DecodeGraphFamily* family : {&ordinary_graphs, &mtp_graphs, &dflash_graphs}) {
        if (!family->profiles.empty()) {
            graph_node_count = family->profiles.front().definition.node_count();
            break;
        }
    }

    if (!ordinary_graphs.profiles.empty()) {
        instantiate_graph_family(ordinary_graphs, "ordinary", device, prepare_representative,
                                 synchronize_all);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        instantiate_graph_family(mtp_graphs, "MTP", device, prepare_representative,
                                 synchronize_all);
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        instantiate_graph_family(dflash_graphs, "DFlash", device, prepare_representative,
                                 synchronize_all);
    }

    ordered_reset(sequence);
    clear_stable_controls();
    for (Tensor& tensor : decoder->linear_attention.conv) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    for (Tensor& tensor : decoder->linear_attention.recurrent) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    on_peer([&](PeerRuntime& p) {
        for (Tensor& tensor : p.decoder->linear_attention.conv) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), p.device.stream));
        }
        for (Tensor& tensor : p.decoder->linear_attention.recurrent) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), p.device.stream));
        }
    });
    if (dflash) {
        const auto zero_cyclic_cache = [&](CyclicKVCache& cache) {
            for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
                const CyclicKVCacheLayerView view = cache.layer_view(layer);
                CUDA_CHECK(cudaMemsetAsync(view.k.data, 0, view.k.bytes(), device.stream));
                CUDA_CHECK(cudaMemsetAsync(view.v.data, 0, view.v.bytes(), device.stream));
            }
        };
        zero_cyclic_cache(dflash->local);
        zero_cyclic_cache(dflash->rewrite_checkpoint_local);
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_features.data, 0,
                                   dflash->prefill_features.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_positions.data, 0,
                                   dflash->prefill_positions.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->pending_features.data, 0,
                                   dflash->pending_features.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    // Symmetry: capture left rank 1's counters as dirty as rank 0's, and the two lanes are only
    // meaningful while they agree.
    on_peer([&](PeerRuntime& p) {
        CUDA_CHECK(
            cudaMemsetAsync(p.token_counts.data, 0, p.token_counts.bytes(), p.device.stream));
    });
    synchronize_all();

    std::array<std::size_t, kMaximumExecutionDevices> free_after{0, 0};
    free_after[0] = free_device_bytes();
    on_peer([&](PeerRuntime& p) { free_after[p.rank] = free_device_bytes(); });
    // The allowance is a per-device budget, so each device is checked against it separately
    // rather than against a doubled or summed figure.
    const int ranks = device_count;
    for (int rank = 0; rank < ranks; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const std::size_t consumed =
            free_before[slot] > free_after[slot] ? free_before[slot] - free_after[slot] : 0;
        graph_observed_bytes[slot] = consumed;
        if (consumed > graph_allowance_bytes) {
            throw std::runtime_error(
                "CUDA Graph preparation consumed " + std::to_string(consumed) +
                " bytes on device " + std::to_string(execution.dev[rank]->device) +
                ", exceeding the planned per-device allowance of " +
                std::to_string(graph_allowance_bytes) + " bytes");
        }
    }
    for (PagedKVAllocation& allocation : dflash_capture_allocations) { allocation.unbind_row(); }
    dflash_capture_allocations.clear();
    for (PagedKVAllocation& allocation : mtp_capture_allocations) { allocation.unbind_row(); }
    mtp_capture_allocations.clear();
    on_peer([&](PeerRuntime& p) {
        for (PagedKVAllocation& allocation : peer_mtp_capture_allocations[p.rank - 1]) {
            allocation.unbind_row();
        }
        peer_mtp_capture_allocations[p.rank - 1].clear();
    });
    for (PagedKVAllocation& allocation : text_capture_allocations) { allocation.unbind_row(); }
    text_capture_allocations.clear();
    on_peer([&](PeerRuntime& p) {
        for (PagedKVAllocation& allocation : peer_text_capture_allocations[p.rank - 1]) {
            allocation.unbind_row();
        }
        peer_text_capture_allocations[p.rank - 1].clear();
    });
}

Tensor ProgramImplCore::token_counts_lane(const Tensor& storage, std::uint32_t lane) {
    return storage.slice(1, static_cast<std::int32_t>(lane), 1).view({TextConfig::token_domain});
}

void ProgramImplCore::publish_peer_token_counts(const SequenceState& sequence) {
    if (peers.empty()) { return; }
    // With penalties off the lane is never read on either rank (install_sampling leaves every
    // row's `token_counts` null), so the ~0.95 MiB peer copy per prefill is pure cost. Both lanes
    // were zeroed together in install_sampling, so skipping keeps them equal either way.
    if (requests[sequence.lane].sampling_host.token_counts == nullptr) { return; }
    const Tensor source = token_counts_lane(token_counts, sequence.lane);
    for (auto& peer : peers) {
    const Tensor target = token_counts_lane(peer->token_counts, sequence.lane);
    // The one increment rank 0 performs and rank 1 does not: prefill's bonus token is sampled on
    // rank 0 alone (the output head is vocabulary-split and sampling belongs to rank 0).
    // Every later increment is performed by `speculative_accept_greedy_drafts`, which the MTP
    // round runs on BOTH devices over bit-identical inputs, so one copy here is what makes the
    // two counter lanes agree at every point either is read.
    //
    // Same cross-device form the collectives use, and for the same reason (src/ops/common/
    // allreduce.cu's `pull_peer`): under unified virtual addressing a device pointer already names
    // its device, so `cudaMemcpyDeviceToDevice` expresses the transfer without
    // `cudaMemcpyPeerAsync`, which CUDA 13.1 rejects inside a stream capture region. This call
    // site is eager prefill, not capture, but keeping one form across every cross-device copy
    // means no future move of this code into a captured region reintroduces that failure.
    CUDA_CHECK(cudaMemcpyAsync(target.data, source.data, source.bytes(), cudaMemcpyDeviceToDevice,
                               device.stream));
    }
}

void ProgramImplCore::publish_peer_mtp_ingress(std::span<const std::uint32_t> lanes) {
    for (auto& peer : peers) {
    if (peer->mtp_host_ingress == nullptr || mtp_host_ingress == nullptr) { continue; }
    *peer->mtp_host_ingress = *mtp_host_ingress;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        ops::SamplingConfig& sampling = peer->mtp_host_ingress->sampling[row];
        if (sampling.token_counts == nullptr) { continue; }
        sampling.token_counts =
            static_cast<std::int32_t*>(token_counts_lane(peer->token_counts, lanes[row]).data);
    }
    }
}

void ProgramImplCore::publish_peer_dflash_ingress(std::span<const std::uint32_t> lanes) {
    for (auto& peer : peers) {
    if (peer->dflash_host_ingress == nullptr || dflash_host_ingress == nullptr) { continue; }
    *peer->dflash_host_ingress = *dflash_host_ingress;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        ops::SamplingConfig& sampling = peer->dflash_host_ingress->sampling[row];
        if (sampling.token_counts == nullptr) { continue; }
        sampling.token_counts =
            static_cast<std::int32_t*>(token_counts_lane(peer->token_counts, lanes[row]).data);
    }
    }
}

void ProgramImplCore::publish_peer_ordinary_ingress() {
    for (auto& peer : peers) {
    if (peer->ordinary_host_ingress == nullptr || ordinary_host_ingress == nullptr) { continue; }
    *peer->ordinary_host_ingress = *ordinary_host_ingress;
    // Rank 1 mirrors the round for its half of the weights and never samples, so its copy of the
    // sampling configs is inert. The counter pointer is the one field that is a rank-0 DEVICE
    // address; it is nulled rather than repointed at rank 1's lane (as the MTP ingress does)
    // because nothing on rank 1 advances an ordinary-round counter -- a repointed lane would be a
    // counter that never moves, which is worse than an absent one. If a future change makes rank 1
    // sample in the ordinary round, repoint here exactly as publish_peer_mtp_ingress does.
    for (ops::SamplingConfig& sampling : peer->ordinary_host_ingress->sampling) {
        sampling.token_counts = nullptr;
    }
    }
}

void ProgramImplCore::enable_peer_egress_check(bool enabled) noexcept {
    if (pp != 1) { enabled = false; }
    peer_egress_check_enabled = enabled;
}

void ProgramImplCore::check_peer_mtp_egress(std::size_t rows) {
    if (!peer_egress_check_enabled || peers.empty() || mtp_host_egress == nullptr) { return; }
    for (auto& peer : peers) {
    if (!peer->io.mtp_decode.has_value()) { return; }
    qwen3_6::MtpDecodeEgress peer_egress{};
    {
        const ScopedDevice scope(peer->device.device);
        CUDA_CHECK(cudaMemcpy(&peer_egress, peer->io.mtp_decode->egress.data,
                              sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost));
    }
    const std::uint32_t width = draft_window + 1U;
    std::uint64_t mismatches  = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        const std::int32_t count = mtp_host_egress->licensed_counts[row];
        mismatches += peer_egress.licensed_counts[row] != count ? 1U : 0U;
        mismatches +=
            peer_egress.accepted_drafts[row] != mtp_host_egress->accepted_drafts[row] ? 1U : 0U;
        mismatches += peer_egress.next_extents[row] != mtp_host_egress->next_extents[row] ? 1U : 0U;
        // Only the licensed prefix is defined; the columns past it are round scratch on both ranks.
        const std::size_t licensed =
            count > 0 ? std::min(static_cast<std::size_t>(count), static_cast<std::size_t>(width))
                      : 0U;
        for (std::size_t column = 0; column < licensed; ++column) {
            const std::size_t index = row * width + column;
            mismatches +=
                peer_egress.licensed_tokens[index] != mtp_host_egress->licensed_tokens[index] ? 1U
                                                                                              : 0U;
        }
        // `next_drafts` is deliberately NOT compared: the proposal head is vocabulary-split, so
        // TextContext::mtp_propose_batch's tp2 overload gathers both halves and writes ONE argmax
        // -- rank 0's. Rank 1's next_drafts region is never written (it reads back as zeros) and
        // is not part of its egress; next round's drafts reach rank 1 through the pinned MTP
        // ingress record, not through its own egress. Measured while writing this check: with the
        // field included, 34 of 36 rounds reported exactly `next_extents` (3) mismatches each and
        // the other two (extent 0) reported none -- 102 in total, and zero in every other field.
    }
    peer_egress_rounds += 1;
    peer_egress_mismatches += mismatches;
    }
}

schedule::TokenConstraintView
ProgramImplCore::prefill_token_constraints(const RequestControl& request) const noexcept {
    return request.grammar ? token_constraints.from_row(kTokenConstraintPrefillRow)
                           : schedule::TokenConstraintView{};
}

std::uint32_t ProgramImplCore::plan_token_constraints(RequestControl& request, std::int32_t row,
                                                     std::span<const TokenId> drafts) {
    const auto columns = static_cast<std::int32_t>(drafts.size()) + 1;
    if (row < 0 || row + columns > kTokenConstraintRows) {
        throw std::logic_error("structured-output constraint rows are out of range");
    }
    auto* flags = static_cast<std::int32_t*>(token_constraint_host.data()) + row;
    if (!request.grammar) {
        std::fill_n(flags, columns, 0);
        return static_cast<std::uint32_t>(drafts.size());
    }
    const auto words = static_cast<std::size_t>(token_constraints.words);
    auto* rows = reinterpret_cast<std::uint32_t*>(static_cast<std::int32_t*>(
                     token_constraint_host.data()) + kTokenConstraintRows) +
                 static_cast<std::size_t>(row) * words;
    return request.grammar->plan(drafts, {rows, static_cast<std::size_t>(columns) * words},
                                 {flags, static_cast<std::size_t>(columns)});
}

void ProgramImplCore::install_sampling(SequenceState& sequence, RequestControl& request,
                                       const ops::SamplingConfig& config) {
    Tensor counts = token_counts_lane(token_counts, sequence.lane);
    CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream));
    for (auto& peer : peers) {
        const ScopedDevice scope(peer->device.device);
        const Tensor peer_counts = token_counts_lane(peer->token_counts, sequence.lane);
        CUDA_CHECK(cudaMemsetAsync(peer_counts.data, 0, peer_counts.bytes(), peer->device.stream));
    }
    request.sampling_host     = config;
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
    };
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.sampling_host,
                               sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

void ProgramImplCore::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

void ProgramImplCore::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

// See logits_capture's declaration comment. io.logits is [vocab,1] BF16 regardless of device_count; this
// is the same rank-0 destination copy_round_token() reads io.token from, so no tp2-specific code
// is needed here. Inert unless the debug capture was explicitly enabled.
void ProgramImplCore::copy_round_logits() {
    if (!logits_capture_enabled) { return; }
    CUDA_CHECK(cudaMemcpyAsync(logits_capture.data(), io.logits.data,
                               logits_capture.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, device.stream));
}

void ProgramImplCore::enable_logits_capture(bool enabled) {
    logits_capture_enabled = enabled;
    if (enabled) {
        logits_capture.assign(static_cast<std::size_t>(io.logits.ne[0]), 0);
    } else {
        logits_capture.clear();
        logits_capture.shrink_to_fit();
    }
}

void ProgramImplCore::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImplCore::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                    std::span<const std::uint32_t> starts,
                                                    std::span<const std::uint32_t> counts) {
    if (speculative_backend != SpeculativeBackend::DFlash || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }

    std::uint32_t minimum_count = draft_window + 1U;
    std::uint32_t maximum_count = 0;
    *dflash_host_ingress        = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > draft_window + 1U ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = sequences[lane];
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || sequence.kv->text.bound_row() < 0 ||
            (sequence.kv->backend && sequence.kv->backend->bound_row() < 0) ||
            end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->dflash_kv_table_rows[row] =
            sequence.kv->backend ? sequence.kv->backend->bound_row() : 0;
        dflash_host_ingress->lanes[row]                = static_cast<std::int32_t>(lane);
        materialize_sequence_kv(sequence, std::max(sequence.text_kv_valid, end),
                                 sequence.kv->backend ? end : 0U);
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    qwen3_6::DFlashDecodeState& frame = *io.dflash_decode;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch     = static_cast<std::int32_t>(lanes.size());
    Tensor lane_tensor   = frame.lanes.slice(0, 0, batch);
    Tensor device_starts = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends   = frame.execution_frontiers.slice(0, 0, batch);
    Tensor table_rows    = frame.dflash_kv_table_rows.slice(0, 0, batch);
    Tensor positions     = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {DFlashConfig::feature_rows,
                                 static_cast<std::int32_t>(draft_window + 1U), batch});
    ops::prepare_ragged_prefix(dflash->pending_features, lane_tensor, device_starts, device_ends,
                               features, positions, device_counts, device.stream);

    schedule::DFlashAppendContext state{{device, model, work, decoder->linear_attention,
                                         replay_records ? &*replay_records : nullptr, io,
                                         prefill_hidden, prefill_chunk, proposal_head,
                                         rope_frequency},
                                        *dflash};
    mark_workspace_usage(workspace_plan.dflash_context);
    schedule::dflash_append_context(state, features, positions, device_counts, lane_tensor,
                                    table_rows, {minimum_count, maximum_count});
}

void ProgramImplCore::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("target returned a token outside the 248077-token domain");
        }
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill(SequenceState& sequence,
                                                            RequestControl& request) {
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base,
                                        .prefix_reuse_path    = staged.reuse};
    std::uint32_t processed_prompt_tokens = 0;
    const auto started                    = Clock::now();
    try {
        if (ram_kv.gpu_tokens != 0 && staged.cursor < staged.prompt_tokens) {
            if (staged.cursor >= sequence.ram_rank_frontier + 8192) {
                refresh_ram_kv(sequence, staged.cursor);
            }
            const auto end = std::min(staged.prompt_tokens, staged.cursor + prefill_chunk * pipeline_chunks);
            const auto require = [&](PagedKVAllocation& a) {
                a.require_pages(staged.cursor / 64, (end + 63) / 64);
            };
            require(sequence.kv->text);
            if (sequence.kv->backend) { require(*sequence.kv->backend); }
            for (auto& peer : peers) {
                require(*sequence.kv->text_peers[peer->rank - 1]);
                if (sequence.kv->backend_peers[peer->rank - 1]) { require(*sequence.kv->backend_peers[peer->rank - 1]); }
            }
            const auto backend_end = staged.prepare_mtp ? std::min(capacity, end +
                (end == staged.prompt_tokens && staged.initial_mtp_extent != 0
                    ? staged.initial_mtp_extent - 1 : 0U)) : 0U;
            materialize_sequence_kv(sequence, end, backend_end);
        }
        schedule::PrefillContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, rope_frequency, peer_views(), pipeline_view(), pipeline_upload ? static_cast<TokenId*>(pipeline_upload->data()) : nullptr, volta_long_attention, prefill_token_constraints(request)},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            mtp_kv_views_peer(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            staged.cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            &sequence.rewrite_checkpoint_hidden,
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
            LinearStateSlots::rewrite_checkpoint_state_slot(sequence.lane, max_concurrency),
            staged.initial_mtp_extent,
            dflash_host_ingress,
            sequence.rope_delta};

        if (staged.mtp_bridge == MtpBridgeMode::BeforeSuffix) {
            if (staged.cursor != staged.base || staged.base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden = is_rewrite_checkpoint_restore(staged.reuse)
                                                ? sequence.rewrite_checkpoint_hidden
                                                : sequence.tail_hidden;
            const schedule::MtpBridgeInput bridge{
                .previous_hidden = &previous_hidden,
                .position        = checked_i32(staged.base - 1, "MTP bridge position"),
                .rope_position   = prompt_rope_position(staged.prompt, staged.base - 1),
            };
            if (staged.vision) {
                schedule::mtp_bridge_multimodal(schedule_state, staged.prompt, *staged.vision,
                                                bridge);
            } else {
                Tensor bridge_token = io.mtp->target_input_ids.slice(0, 0, 1);
                const TokenId token = staged.prompt.token_ids[staged.base];
                CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                           cudaMemcpyHostToDevice, device.stream));
                schedule::mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                                 bridge.position, bridge.rope_position, false);
            }
            sequence.mtp_kv_valid = staged.base;
            staged.mtp_bridge     = MtpBridgeMode::None;
        }

        if (staged.cursor < staged.prompt_tokens) {
            const std::uint32_t quantum = prefill_chunk * pipeline_chunks;
            std::uint32_t nominal =
                std::min(quantum, staged.prompt_tokens - staged.cursor);
            std::uint32_t early_frontier = 0;
            std::uint64_t checkpoint_interval = 0;
            if (staged.capture_prefix_checkpoints) {
                const auto images = kPrefixCheckpointBudget / max_concurrency / prefix_checkpoint_bytes();
                const auto segments = std::max<std::size_t>(1, images - 1);
                const auto span = (staged.prompt_tokens + segments - 1) / segments;
                checkpoint_interval = ((span + quantum - 1) / quantum) * quantum;
                if (images > 1) {
                    // One early state covers short common prefixes. Subsequent captures
                    // are spaced to the RAM budget, not every GPU prefill chunk.
                    early_frontier = std::min(64U, std::max(1U, staged.prompt_tokens / 2));
                    if (staged.cursor < early_frontier) {
                        nominal = std::min(nominal, early_frontier - staged.cursor);
                    } else if (staged.cursor == early_frontier) {
                        // Rejoin the original chunk grid after the early split.
                        nominal = std::min(nominal, quantum - staged.cursor % quantum);
                    }
                }
            }
            const bool final_candidate = staged.cursor + nominal == staged.prompt_tokens;
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            if (speculative_backend == SpeculativeBackend::DFlash) {
                mark_workspace_usage(workspace_plan.dflash_context);
            }
            schedule::PrefillChunkResult result;
            const std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier =
                staged.rewrite_checkpoint_capture
                    ? std::optional<std::uint32_t>(staged.rewrite_checkpoint_capture->frontier)
                    : std::nullopt;
            // Capture Q once on the final suffix, then replay from the SAME recurrent state
            // with query-selected history. The preview does not sample or publish tokens.
            if (ram_kv.gpu_tokens != 0 && final_candidate && staged.cursor > ram_kv.gpu_tokens) {
                const auto copy_state = [&](bool restore) {
                    const auto copy = [&](qwen3_6::DecoderState& state, DeviceContext& ctx) {
                        const ScopedDevice scope(ctx.device);
                        state.linear_attention.copy_slot(restore ? 2 : 0, restore ? 0 : 2, ctx.stream);
                    };
                    copy(*decoder, device);
                    for (auto& peer : peers) { copy(*peer->decoder, peer->device); }
                };
                copy_state(false);
                const auto preview = schedule::prefill_text_chunk(schedule_state,
                    staged.prompt.token_ids, nominal, std::nullopt, false);
                if (preview.processed_tokens != nominal) {
                    throw std::logic_error("RAM KV query preview did not complete the suffix");
                }
                copy_state(true);
                refresh_ram_kv(sequence, staged.cursor);
                materialize_sequence_kv(sequence, staged.prompt_tokens,
                    staged.prepare_mtp ? staged.prompt_tokens : 0U);
                // Do not double-count Q on replay; the captured preview is the retrieval query.
                schedule_state.capture_retrieval_query = false;
            }
            if (staged.vision) {
                mark_workspace_usage(workspace_plan.vision_encode);
                result = schedule::prefill_multimodal_chunk(
                    schedule_state, staged.prompt, *staged.vision, nominal,
                    rewrite_checkpoint_capture_frontier, final_candidate);
            } else {
                result = schedule::prefill_text_chunk(
                    schedule_state, std::span<const TokenId>(staged.prompt.token_ids), nominal,
                    rewrite_checkpoint_capture_frontier, final_candidate);
            }
            if (result.processed_tokens == 0 || result.processed_tokens > nominal) {
                throw std::logic_error("ordinary prefill chunk made invalid progress");
            }
            processed_prompt_tokens = result.processed_tokens;
            if (staged.vision) { staged.vision->release_encoded_media_payloads(); }
            staged.cursor += result.processed_tokens;
            sequence.text_kv_valid = staged.cursor;
            if (staged.prepare_mtp) { sequence.mtp_kv_valid = staged.cursor; }
            if (speculative_backend == SpeculativeBackend::DFlash) {
                sequence.dflash_context_frontier = staged.cursor;
            }
            if (staged.capture_prefix_checkpoints &&
                (staged.cursor == early_frontier || staged.cursor == staged.prompt_tokens ||
                 staged.cursor / checkpoint_interval !=
                     (staged.cursor - result.processed_tokens) / checkpoint_interval)) {
                save_prefix_checkpoint(sequence, staged.cursor,
                    prefill_hidden.slice(1, static_cast<std::int32_t>(result.tail_column), 1));
            }

            if (!result.finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                return runtime::PrefillStepResult{
                    .summary = summary, .processed_prompt_tokens = processed_prompt_tokens};
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            copy_tail(sequence, prefill_hidden.slice(
                                    1, static_cast<std::int32_t>(result.tail_column), 1));
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!sequence.tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            schedule::sample_from_hidden(schedule_state, sequence.tail_hidden,
                                         checked_i32(staged.prompt_tokens, "sample position"),
                                         ops::kSamplePurposePrefill);
            set_device_i32(io.rope_pos, checked_i32(staged.prompt_tokens, "rope position") +
                                            sequence.rope_delta);
            if (staged.prepare_mtp) {
                if (staged.mtp_bridge != MtpBridgeMode::AfterExactHit || !request.exact_hit) {
                    throw std::logic_error("zero-suffix MTP reuse has no exact-hit bridge");
                }
                // The anchor is published before preparing its speculative continuation.
                // finish_exact_hit uses the lane ledger, not this shared io.token buffer.
                staged.mtp_bridge     = MtpBridgeMode::None;
            }
        }

        copy_round_token();
        copy_round_logits();
        // Prefill's bonus token has now been sampled on rank 0, which is the one place rank 0's
        // penalty counters advance without rank 1's doing the same. Bring rank 1's lane level
        // before any decode round reads it; a no-op at tp1 and whenever penalties are off (the
        // counter lane is then never read).
        publish_peer_token_counts(sequence);
        std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && !request.exact_hit && staged.initial_mtp_extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.mtp->draft_tokens.data,
                                       staged.initial_mtp_extent * sizeof(TokenId),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        device.synchronize();
        if (ram_kv.gpu_tokens != 0) {
            sequence.kv->text.require_pages(0, 0);
            if (sequence.kv->backend) { sequence.kv->backend->require_pages(0, 0); }
            for (auto& peer : peers) {
                sequence.kv->text_peers[peer->rank - 1]->require_pages(0, 0);
                if (sequence.kv->backend_peers[peer->rank - 1]) { sequence.kv->backend_peers[peer->rank - 1]->require_pages(0, 0); }
            }
        }
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        const double vision_seconds = staged.vision ? staged.vision->elapsed_seconds() : 0.0;
        const std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture =
            staged.rewrite_checkpoint_capture;
        const std::uint32_t prompt_tokens = staged.prompt_tokens;

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        sequence.text_kv_valid = prompt_tokens;
        if (staged.prepare_mtp) {
            if (sequence.mtp_kv_valid != prompt_tokens - (request.exact_hit ? 1U : 0U)) {
                throw std::logic_error("staged MTP prefill did not reach the prompt frontier");
            }
            sequence.mtp_draft_count = request.exact_hit ? 0U : staged.initial_mtp_extent;
            std::copy_n(initial_drafts.begin(), sequence.mtp_draft_count,
                        sequence.mtp_drafts.begin());
        } else if (speculative_backend == SpeculativeBackend::DFlash &&
                   sequence.dflash_context_frontier != prompt_tokens) {
            throw std::logic_error("staged DFlash prefill did not reach the prompt frontier");
        }
        sequence.tail_hidden_valid      = true;
        request.timings.vision_seconds  = vision_seconds;
        request.timings.prefill_seconds = std::max(0.0, staged.elapsed_seconds - vision_seconds);
        if (rewrite_checkpoint_capture) {
            const std::uint32_t frontier = rewrite_checkpoint_capture->frontier;
            if (frontier == 0 || frontier > prompt_tokens || sequence.text_kv_valid < frontier) {
                throw std::logic_error("rewrite checkpoint was not materialized by Text prefill");
            }
            if (speculative_backend == SpeculativeBackend::Mtp &&
                (!staged.prepare_mtp || sequence.mtp_kv_valid < frontier - 1)) {
                throw std::logic_error("rewrite checkpoint has no complete MTP prefix");
            }
            if (speculative_backend == SpeculativeBackend::DFlash &&
                (!dflash || !sequence.kv ||
                 dflash->full.has_value() != sequence.kv->backend.has_value() ||
                 sequence.dflash_context_frontier < frontier)) {
                throw std::logic_error("rewrite checkpoint has no complete DFlash prefix");
            }
            sequence.rewrite_checkpoint = RewriteCheckpoint{
                .valid = true, .kind = rewrite_checkpoint_capture->kind, .frontier = frontier};
        }

        staged.prompt.release_all_media_payloads();

        request.prefill.reset();
        request.pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                             .base_E        = 0,
                                             .base_S        = 0,
                                             .prompt_tokens = prompt_tokens,
                                             .produced      = 1};
        request.lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary = summary,
            .round   = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
            .processed_prompt_tokens = processed_prompt_tokens,
            .complete                = true,
        };
    } catch (...) {
        try {
            // Retire BOTH devices before tearing the lane down: rank 1 may still have enqueued
            // work referencing the KV pages and GDN slots clear_lane is about to release. Same
            // order as the decode path's handler.
            for (auto& peer : peers) { peer->device.synchronize(); }
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                                       std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("ordinary batch execution requires the ordinary backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("ordinary batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("ordinary batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            sequence.kv->text.bound_row() < 0 || sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto start = Clock::now();
    try {
        const bool constrained = std::any_of(lanes.begin(), lanes.end(), [&](std::uint32_t lane) {
            return requests[lane].grammar.has_value();
        });
        DecodeGraphExecutable* executable = nullptr;
        ops::GqaExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + 1};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(ordinary_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "ordinary batch", false, constrained);
            executable = &install_graph_profile(ordinary_graphs, profile, "ordinary batch");
            envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence            = sequences[lanes[row]];
            const RequestControl& request      = requests[lanes[row]];
            const std::uint32_t frontier       = sequence.execution_frontier;
            (void)plan_token_constraints(requests[lanes[row]], static_cast<std::int32_t>(row), {});
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "ordinary batch position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            ordinary_host_ingress->lanes[row]    = static_cast<std::int32_t>(sequence.lane);
            ordinary_host_ingress->sampling[row] = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + 1, 0);
        }
        publish_peer_ordinary_ingress();

        schedule::OrdinaryBatchContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, rope_frequency, peer_views(), pipeline_view(), pipeline_upload ? static_cast<TokenId*>(pipeline_upload->data()) : nullptr, volta_long_attention, constrained ? token_constraints : schedule::TokenConstraintView{}},
            decoder->text_kv,
            *io.ordinary,
            *ordinary_host_ingress,
            *ordinary_host_egress,
            tail_hidden_store};

        mark_workspace_usage(workspace_plan.ordinary_round);
        schedule::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                        envelope, executable);
        for (auto& peer : peers) { peer->device.synchronize(); }
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = sequences[lanes[row]];
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            sequence.text_kv_valid     = base_E + 1;
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens = std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(),
                                               lanes.size())};
    } catch (...) {
        try {
            for (auto& peer : peers) { peer->device.synchronize(); }
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

void ProgramImplCore::finish_exact_hit(SequenceState& sequence, RequestControl& request,
                                       bool decode) {
    if (!request.exact_hit) { return; }
    const auto continuation = *request.exact_hit;
    const std::uint32_t frontier = sequence.execution_frontier;
    if (decode) {
        if (sequence.deferred_prefix_state) {
            restore_prefix_checkpoint(sequence, *sequence.deferred_prefix_state);
        }
        mark_workspace_usage(workspace_plan.mtp_prefill);
        const auto extent = continuation.proposal_extent;
        materialize_sequence_kv(sequence, frontier,
            std::min(capacity, frontier + (extent == 0 ? 0U : extent - 1U)));
        set_device_i32(io.text_kv_table_row, sequence.kv->text.bound_row());
        set_device_i32(io.backend_kv_table_row, sequence.kv->backend->bound_row());
        set_device_i32(io.rope_delta, sequence.rope_delta);
        for (auto& peer : peers) {
            const ScopedDevice scope(peer->device.device);
            set_peer_i32(peer->io.text_kv_table_row,
                sequence.kv->text_peers[peer->rank - 1]->bound_row(), peer->device);
            set_peer_i32(peer->io.backend_kv_table_row,
                sequence.kv->backend_peers[peer->rank - 1]->bound_row(), peer->device);
            set_peer_i32(peer->io.rope_delta, sequence.rope_delta, peer->device);
        }
        schedule::PrefillContext context{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, rope_frequency, peer_views(), pipeline_view(),
             pipeline_upload ? static_cast<TokenId*>(pipeline_upload->data()) : nullptr,
             volta_long_attention, prefill_token_constraints(request)},
            text_kv_view(sequence), mtp_kv_view(sequence), mtp_kv_views_peer(sequence),
            decoder->text_kv, decoder->mtp_cache(), nullptr, frontier,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            &sequence.rewrite_checkpoint_hidden,
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
            LinearStateSlots::rewrite_checkpoint_state_slot(sequence.lane, max_concurrency),
            extent, nullptr, sequence.rope_delta};
        Tensor anchor = io.mtp->target_input_ids.slice(0, 0, 1);
        const TokenId token = sequence.ledger.back();
        CUDA_CHECK(cudaMemcpyAsync(anchor.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                                   device.stream));
        schedule::mtp_bridge_and_propose(context, anchor, sequence.tail_hidden,
            checked_i32(frontier - 1, "deferred MTP bridge position"),
            continuation.bridge_rope, extent != 0);
        if (extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(sequence.mtp_drafts.data(), io.mtp->draft_tokens.data,
                extent * sizeof(TokenId), cudaMemcpyDeviceToHost, device.stream));
        }
        for (auto& peer : peers) { peer->device.synchronize(); }
        device.synchronize();
        sequence.mtp_kv_valid = frontier;
        sequence.mtp_draft_count = extent;
    }
    request.exact_hit.reset();
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                                  std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    const std::uint32_t width      = draft_window + 1;
    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("MTP batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            sequence.kv->text.bound_row() < 0 || sequence.kv->backend->bound_row() < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid + (request.exact_hit ? 1U : 0U) != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto started = Clock::now();
    try {
        for (const std::uint32_t lane : lanes) {
            finish_exact_hit(sequences[lane], requests[lane], true);
        }
        const bool greedy_target = std::all_of(
            lanes.begin(), lanes.end(), [&](std::uint32_t lane) {
                return !(requests[lane].sampling_host.temperature > 0.0F);
            });
        const bool constrained = std::any_of(lanes.begin(), lanes.end(), [&](std::uint32_t lane) {
            return requests[lane].grammar.has_value();
        });
        DecodeGraphExecutable* executable = nullptr;
        schedule::MtpGqaEnvelopes envelopes =
            mtp_gqa_envelopes(maximum_frontier, draft_window, capacity);
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(mtp_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch", greedy_target, constrained);
            executable = &install_graph_profile(mtp_graphs, profile, "MTP batch");
            envelopes  = mtp_gqa_envelopes(profile.max_execution_frontier, draft_window, capacity);
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = sequences[lanes[row]];
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            // Drafts after the first one the grammar rejects can never commit; drop them.
            const std::uint32_t extent = plan_token_constraints(
                requests[lanes[row]], static_cast<std::int32_t>(row * width),
                std::span<const TokenId>(sequence.mtp_drafts.data(),
                                         std::min({sequence.mtp_draft_count, draft_window,
                                                   max_by_budget,
                                                   capacity - sequence.execution_frontier - 1})));
            std::fill_n(static_cast<std::int32_t*>(token_constraint_host.data()) + row * width +
                            extent + 1,
                        width - extent - 1, 0);
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "MTP batch frontier");
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < draft_window; ++j) {
                mtp_host_ingress->current_drafts[row * draft_window + j] =
                    j < extent ? sequence.mtp_drafts[j] : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
            }
            mtp_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            mtp_host_ingress->mtp_kv_table_rows[row]  = sequence.kv->backend->bound_row();
            mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(sequence.lane);
            mtp_host_ingress->rope_deltas[row]        = sequence.rope_delta;
            mtp_host_ingress->sampling[row]           = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1,
                                    std::min(capacity, frontier + extent + draft_window));
        }
        publish_peer_mtp_ingress(lanes);

        schedule::MtpBatchContext schedule_state{{device, model, work, decoder->linear_attention,
                                                  replay_records ? &*replay_records : nullptr, io,
                                                  prefill_hidden, prefill_chunk, proposal_head,
                                                  rope_frequency,
                                                  peer_views(), pipeline_view(), pipeline_upload ? static_cast<TokenId*>(pipeline_upload->data()) : nullptr, volta_long_attention, constrained ? token_constraints : schedule::TokenConstraintView{}},
                                                 decoder->text_kv,
                                                 *decoder->mtp_cache(),
                                                 *io.mtp_decode,
                                                 *mtp_host_ingress,
                                                 *mtp_host_egress,
                                                 tail_hidden_store,
                                                 greedy_target};

        mark_workspace_usage(workspace_plan.mtp_round);
        schedule::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                   draft_window, envelopes, executable);
        // Peer first, then rank 0: rank 1's stream carries the round's own work eagerly and the
        // graph's rank-1 nodes when captured, and the egress read below must not observe a round
        // rank 1 has not finished contributing to.
        for (auto& peer : peers) { peer->device.synchronize(); }
        device.synchronize();
        check_peer_mtp_egress(lanes.size());

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = sequences[lanes[row]];
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(mtp_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            request.pending = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width};
    } catch (...) {
        try {
            for (auto& peer : peers) { peer->device.synchronize(); }
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_dflash_batch(std::span<const std::uint32_t> lanes,
                                     std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::DFlash || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }

    const std::uint32_t width           = draft_window + 1U;
    std::uint32_t maximum_frontier      = 0;
    std::uint32_t maximum_target_tokens = 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("DFlash batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            dflash->full.has_value() != sequence.kv->backend.has_value() ||
            sequence.kv->text.bound_row() < 0 ||
            (sequence.kv->backend && sequence.kv->backend->bound_row() < 0) ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > width ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                ? budgets[row].generated_tokens_remaining - 1U
                                                : 0U;
        const std::uint32_t extent =
            std::min({draft_window, max_by_budget, capacity - sequence.execution_frontier - 1U});
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
        maximum_target_tokens =
            std::max(maximum_target_tokens, sequence.execution_frontier + extent + 1U);
    }

    const auto started = Clock::now();
    try {
        const bool greedy_target = std::all_of(
            lanes.begin(), lanes.end(), [&](std::uint32_t lane) {
                return !(requests[lane].sampling_host.temperature > 0.0F);
            });
        DecodeGraphExecutable* executable   = nullptr;
        schedule::DFlashEnvelopes envelopes = dflash_envelopes(0, maximum_frontier, draft_window);
        ops::GqaExecutionEnvelope target_envelope{1, maximum_target_tokens};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(dflash_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "DFlash batch", greedy_target);
            executable      = &install_graph_profile(dflash_graphs, profile, "DFlash batch");
            envelopes       = dflash_envelopes(profile.min_execution_frontier,
                                               profile.max_execution_frontier, draft_window);
            target_envelope = {
                1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       capacity, static_cast<std::uint64_t>(profile.max_execution_frontier) +
                                     draft_window + 1ULL))};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = sequences[lanes[row]];
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1U
                                                    : 0U;
            const std::uint32_t extent =
                std::min({draft_window, max_by_budget, capacity - frontier - 1U});
            dflash_host_ingress->anchors[row] = sequence.ledger.back();
            dflash_host_ingress->execution_frontiers[row] =
                checked_i32(frontier, "DFlash batch frontier");
            dflash_host_ingress->context_frontiers[row] =
                checked_i32(sequence.dflash_context_frontier, "DFlash context frontier");
            dflash_host_ingress->proposal_extents[row]     = static_cast<std::int32_t>(extent);
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            dflash_host_ingress->text_kv_table_rows[row]   = sequence.kv->text.bound_row();
            dflash_host_ingress->dflash_kv_table_rows[row] =
                sequence.kv->backend ? sequence.kv->backend->bound_row() : 0;
            dflash_host_ingress->lanes[row]    = static_cast<std::int32_t>(sequence.lane);
            dflash_host_ingress->sampling[row] = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1U,
                                     sequence.kv->backend ? frontier : 0U);
        }

        publish_peer_dflash_ingress(lanes);
        schedule::DFlashBatchContext schedule_state{{device, model, work, decoder->linear_attention,
                                                     replay_records ? &*replay_records : nullptr,
                                                     io, prefill_hidden, prefill_chunk,
                                                     proposal_head, rope_frequency,
                                                     peer_views(), pipeline_view(), pipeline_upload ? static_cast<TokenId*>(pipeline_upload->data()) : nullptr, volta_long_attention},
                                                    decoder->text_kv,
                                                    *dflash,
                                                    *io.dflash_decode,
                                                    *dflash_host_ingress,
                                                    *dflash_host_egress,
                                                    tail_hidden_store,
                                                    greedy_target};

        mark_workspace_usage(workspace_plan.dflash_round);
        schedule::dflash_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                      draft_window, envelopes, target_envelope, executable);
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = sequences[lanes[row]];
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = dflash_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = dflash_host_egress->accepted_drafts[row];
            const std::uint32_t extent =
                static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("DFlash batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(dflash_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (extent == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += extent;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            sequence.dflash_context_frontier = base_E;
            request.pending                  = PendingCandidate{
                                 .kind          = PendingKind::Speculative,
                                 .base_E        = base_E,
                                 .base_S        = base_S,
                                 .prompt_tokens = 0,
                                 .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(dflash_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(dflash_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width};
    } catch (...) {
        try {
            for (auto& peer : peers) { peer->device.synchronize(); }
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_batch(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend == SpeculativeBackend::None) {
        return decode_ordinary_batch(lanes, budgets);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) { return decode_mtp_batch(lanes, budgets); }
    return decode_dflash_batch(lanes, budgets);
}

void ProgramImplCore::resolve_non_speculative_pending(SequenceState& sequence,
                                                      RequestControl& request,
                                                      std::uint32_t accepted_tokens,
                                                      bool terminal) {
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    if ((request.pending.kind != PendingKind::Begin &&
         request.pending.kind != PendingKind::Ordinary) ||
        request.pending.produced != 1 || accepted_tokens != 1) {
        throw std::logic_error("non-speculative pending round must commit its single token");
    }

    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        sequence.execution_frontier = request.pending.base_E + request.pending.produced;
        sequence.ledger_frontier    = request.pending.base_S + request.pending.produced;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    if (request.grammar) { request.grammar->commit({&sequence.ledger.back(), 1}); }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) {
        // A complete owned RAM checkpoint remains a valid committed GDN frontier.
        // Neither its import nor MTP proposals are needed for a terminal anchor.
        finish_exact_hit(sequence, request, false);
        sequence.mtp_draft_count = 0;
        release_sequence_growth_entitlement(sequence);
        unbind_sequence_kv(sequence);
        sequence.retained = true;
    }
    request.lifecycle = terminal ? Lifecycle::Complete : Lifecycle::Active;
    request.pending   = {};
}

MemorySummary ProgramImplCore::memory_summary() const noexcept {
    MemorySummary out;
    out.device                = device.device;
    out.max_context           = capacity;
    out.rope_mode             = rope_mode;
    out.effective_max_context = effective_max_context;
    out.yarn_mscale           = yarn_mscale;
    out.kv_capacity           = kv_capacity;
    out.kv_cache = kv_dtype == DType::BF16 ? KvCacheStorage::BFloat16 : KvCacheStorage::Int8Group64;
    DeviceArena& weights = *model.weights_arena;
    out.weights = ArenaMemorySummary{weights.capacity(), weights.used(), weights.peak_used()};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), work.used(), work.peak_used()};
    out.workspace_logical_peak_bytes   = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes     = graph_allowance_bytes;
    out.cuda_graph_observed_bytes      = graph_observed_bytes[0];
    out.cuda_graph_peer_observed_bytes = graph_observed_bytes[1];
    out.cuda_graph_rank_observed_bytes = graph_observed_bytes;
    out.cuda_graph_node_count          = graph_node_count;
    out.kv_payload_bytes               = kv_payload_bytes;
    out.ram_kv_gpu_tokens              = ram_kv.gpu_tokens;
    if (ram_kv.gpu_tokens != 0) {
        out.ram_kv_archive_bytes = decoder->text_kv.pool().ram_archive_bytes();
        if (decoder->mtp_cache()) { out.ram_kv_archive_bytes += decoder->mtp_cache()->pool().ram_archive_bytes(); }
        out.ram_kv_archive_bytes *= device_count;
        for (const auto& sequence : sequences) {
            if (!sequence.kv) { continue; }
            out.ram_kv_transfer_bytes += sequence.kv->text.ram_transfer_bytes();
            if (sequence.kv->backend) { out.ram_kv_transfer_bytes += sequence.kv->backend->ram_transfer_bytes(); }
            for (const auto& peer : peers) {
                out.ram_kv_transfer_bytes += sequence.kv->text_peers[peer->rank - 1]->ram_transfer_bytes();
                if (sequence.kv->backend_peers[peer->rank - 1]) {
                    out.ram_kv_transfer_bytes += sequence.kv->backend_peers[peer->rank - 1]->ram_transfer_bytes();
                }
            }
        }
    }
    out.gdn_state_bytes                = gdn_state_bytes;
    return out;
}

void ProgramImplCore::reset_memory_peaks() noexcept {
    model.weights_arena->reset_peak();
    persistent.reset_peak();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
