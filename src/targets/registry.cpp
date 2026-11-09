#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "runtime/engine/kv_capacity.h"
#ifdef NINFER_AUX_VISION_ARCH
#include "targets/qwen3_6/impl/vision/sm86/backend.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
    if (options.vision_gpu &&
        (!options.enable_vision || options.vision_device != VisionDevice::Gpu)) {
        throw std::invalid_argument("vision_gpu requires enabled GPU Vision");
    }
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback = progress.callback};
}

// Save/restore the current device around a per-device query.
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

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes, int device) {
    const ScopedDevice scope(device);
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument("model weights require " + std::to_string(weight_bytes) +
                                    " bytes of device memory on device " +
                                    std::to_string(device) + ", but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

std::size_t current_free_device_bytes(int device) {
    const ScopedDevice scope(device);
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

std::size_t current_total_device_bytes(int device) {
    const ScopedDevice scope(device);
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return total_bytes;
}

void enable_weight_storage_peer(ExecutionContext& execution) {
    if (!execution.has_storage()) { return; }
    const int primary = execution.primary().device;
    const int storage = execution.storage->device;
    int can_access = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, primary, storage));
    if (!can_access) {
        throw std::invalid_argument("expert storage mode requires CUDA peer access between the two devices");
    }
    CUDA_CHECK(cudaSetDevice(primary));
    cudaError_t err = cudaDeviceEnablePeerAccess(storage, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) { CUDA_CHECK(err); }
    CUDA_CHECK(cudaSetDevice(storage));
    err = cudaDeviceEnablePeerAccess(primary, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) { CUDA_CHECK(err); }
    CUDA_CHECK(cudaSetDevice(primary));
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, ExecutionContext& execution,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {
    const auto& identity                          = reader.identity();
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);
    const int tp                                  = execution.device_count;
    DeviceContext& device                         = execution.primary();
    std::shared_ptr<DeviceContext> auxiliary;
    std::unique_ptr<ScopedDevice> restore_device;
    std::optional<int> auxiliary_slot;
    if (options.vision_gpu) {
        if constexpr (!std::is_same_v<Target, Qwen3_6_27B>) {
            throw std::invalid_argument("auxiliary GPU Vision is supported only for Qwen3.6 27B");
        }
#ifndef NINFER_AUX_VISION_ARCH
        throw std::invalid_argument("auxiliary GPU Vision is not enabled in this build");
#endif
        const int selected = *options.vision_gpu;
        if (selected < 0) { throw std::invalid_argument("invalid Vision CUDA device id"); }
        for (int rank = 0; rank < tp; ++rank) {
            if (execution.dev[static_cast<std::size_t>(rank)]->device == selected) {
                throw std::invalid_argument("Vision CUDA device must differ from execution devices");
            }
        }
        if (execution.has_storage() && execution.storage->device == selected) {
            throw std::invalid_argument("Vision CUDA device must differ from the storage device");
        }
        const int slot = tp + (execution.has_storage() ? 1 : 0);
        if (slot >= static_cast<int>(artifact::kMaximumDevices)) {
            throw std::invalid_argument("no materialization slot available for Vision CUDA device");
        }
        restore_device = std::make_unique<ScopedDevice>(device.device);
        auxiliary = std::make_shared<DeviceContext>(selected);
#ifdef NINFER_AUX_VISION_ARCH
        if (auxiliary->sm() != NINFER_AUX_VISION_ARCH) {
            throw std::invalid_argument("auxiliary GPU Vision requires an SM" +
                std::to_string(NINFER_AUX_VISION_ARCH) + " CUDA device; selected device is SM" +
                std::to_string(auxiliary->sm()));
        }
#endif
        CUDA_CHECK(cudaSetDevice(device.device));
        auxiliary_slot = slot;
    }

    const int materialization_devices = tp + (execution.has_storage() ? 1 : 0);
    artifact::Binder binder(reader, materialization_devices, execution.has_storage(), auxiliary_slot);
    auto load_plan = Target::plan_load(binder, options, weights_profile);
    // The curve describes one device's geometry: TP splits heads, PP splits layers.
    // Resolve the symmetric runtime reservation against the smallest device budget.
    auto sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    std::vector<std::size_t> preflight_runtime_bytes;
    preflight_runtime_bytes.reserve(static_cast<std::size_t>(tp));
    for (int rank = 0; rank < tp; ++rank) {
        preflight_runtime_bytes.push_back(runtime_bytes_after_planned_weights(
            load_plan.materialization().device_capacity_bytes[static_cast<std::size_t>(rank)],
            execution.dev[static_cast<std::size_t>(rank)]->device));
    }
    (void)runtime::resolve_kv_capacity_symmetric(options.kv_capacity, curve,
                                                 preflight_runtime_bytes);

    auto progress     = artifact_progress(options.load_progress);
    enable_weight_storage_peer(execution);
    artifact::MaterializedArtifact materialized;
    if (auxiliary) {
        const std::size_t available = runtime_bytes_after_planned_weights(
            load_plan.materialization().device_capacity_bytes[static_cast<std::size_t>(*auxiliary_slot)],
            auxiliary->device);
#ifdef NINFER_AUX_VISION_ARCH
        constexpr std::uint32_t kFrontendSegmentLimit = 768 / 2;
        const auto tokens = std::min(options.max_context, options.vision_max_tokens);
        const auto required = qwen3_6::vision_sm86::Backend::device_capacity_bytes(
            5120, tokens, kFrontendSegmentLimit);
        if (required > available) {
            throw std::invalid_argument("auxiliary Vision workspace and output require " +
                std::to_string(required) + " bytes, but only " + std::to_string(available) +
                " bytes remain after planned weights");
        }
#else
        (void)available;
#endif
        std::vector<DeviceContext*> devices;
        devices.reserve(static_cast<std::size_t>(materialization_devices + 1));
        for (int rank = 0; rank < tp; ++rank) {
            devices.push_back(&*execution.dev[static_cast<std::size_t>(rank)]);
        }
        if (execution.has_storage()) { devices.push_back(&*execution.storage); }
        devices.push_back(auxiliary.get());
        materialized = artifact::materialize(reader, load_plan.materialization(), devices,
                                             progress.callback ? &progress : nullptr);
    } else {
        materialized = artifact::materialize(reader, load_plan.materialization(), execution,
                                             progress.callback ? &progress : nullptr);
    }
    const artifact::MaterializationStats stats = materialized.stats();

    auto model = [&] {
        if constexpr (std::is_same_v<Target, Qwen3_6_27B>) {
            return Target::construct_loaded_model(std::move(load_plan), std::move(materialized),
                                                  auxiliary);
        } else {
            return Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
        }
    }();
    for (int rank = 0; rank < tp; ++rank) {
        execution.dev[static_cast<std::size_t>(rank)]->synchronize();
    }
    if (execution.has_storage()) { execution.storage->synchronize(); }
    std::vector<std::size_t> free_before_runtime;
    free_before_runtime.reserve(static_cast<std::size_t>(tp));
    for (int rank = 0; rank < tp; ++rank) {
        free_before_runtime.push_back(
            current_free_device_bytes(execution.dev[static_cast<std::size_t>(rank)]->device));
    }
    runtime::KvCapacityResolution capacity_resolution =
        runtime::resolve_kv_capacity_symmetric(options.kv_capacity, curve, free_before_runtime);
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    auto loaded   = std::make_unique<Loaded>(std::move(model), options);
    auto instance = std::make_unique<Instance>(std::move(loaded), capacity_resolution,
                                               std::move(sequence_plan), execution);
    for (int rank = 0; rank < tp; ++rank) {
        execution.dev[static_cast<std::size_t>(rank)]->synchronize();
    }
    if (execution.has_storage()) { execution.storage->synchronize(); }
    instance->kv_capacity_resolution.available_after_startup_bytes =
        current_free_device_bytes(device.device);

    LoadSummary summary;
    // Rotary regime, as the target runtime resolved it. `effective_max_context` and `yarn_mscale`
    // come from the constructed program (only it knows the variant's native capacity and the YaRN
    // table it built); the mode/factor/origin echo what the caller asked for.
    summary.rope_mode  = options.rope_mode;
    summary.yarn_factor  = options.rope_mode == RopeMode::Yarn ? options.yarn_factor : 0.0;
    summary.yarn_origin  = options.rope_mode == RopeMode::Yarn ? options.yarn_origin : 0U;
    // Per-device memory table. `weights_bytes` and the free/total figures are MEASURED per device;
    // the sequence, KV, GDN and workspace figures come from the finalized plan, which is symmetric
    // across devices by construction (identical page counts and stage-local state geometry).
    const MemorySummary memory = instance->program->memory_summary();
    summary.effective_max_context = memory.effective_max_context;
    summary.yarn_mscale           = memory.yarn_mscale;
    summary.tp                 = options.tp;
    summary.pp                 = options.pp;
    for (int rank = 0; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        DeviceMemoryReport& row = summary.devices[slot];
        row.device              = execution.dev[slot]->device;
        row.weights_bytes       = stats.per_device_capacity_bytes[slot];
        row.kv_pool_bytes       = memory.kv_payload_bytes;
        row.gdn_state_bytes     = memory.gdn_state_bytes;
        row.sequence_bytes      = memory.sequence.capacity_bytes;
        row.workspace_bytes     = memory.workspace.capacity_bytes;
        // Measured per device: instantiating the decode graphs materializes driver state on each
        // device the graph has nodes on, which at tp 2 is both of them.
        row.cuda_graph_bytes = memory.cuda_graph_rank_observed_bytes[rank];
        row.reserved_bytes = row.weights_bytes + capacity_resolution.runtime_reservation_bytes;
        row.free_after_startup_bytes = current_free_device_bytes(row.device);
        row.total_bytes              = current_total_device_bytes(row.device);
    }
    if (auxiliary) {
        const auto slot = static_cast<std::size_t>(*auxiliary_slot);
        DeviceMemoryReport& row = summary.devices[slot];
        row.device = auxiliary->device;
        row.weights_bytes = stats.per_device_capacity_bytes[slot];
        row.reserved_bytes = row.weights_bytes;
        row.free_after_startup_bytes = current_free_device_bytes(row.device);
        row.total_bytes = current_total_device_bytes(row.device);
    }
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults};
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         ExecutionContext& execution)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(execution.primary(), sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(loaded->model, std::move(sequence_plan), execution)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               ExecutionContext& execution)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(execution.primary(), sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan),
                                             execution)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, ExecutionContext& execution,
                                  std::shared_ptr<const std::vector<std::byte>> image) {
    validate_options(options);
    if (execution.device_count != options.tp * options.pp) {
        throw std::invalid_argument("execution context width does not match EngineOptions.tp");
    }
    const auto load_start = Clock::now();

    artifact::Reader reader = image ? artifact::Reader(std::move(image))
                                   : artifact::Reader(options.artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, execution, reader, load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, execution, reader, load_start, Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, execution, reader, load_start, Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
