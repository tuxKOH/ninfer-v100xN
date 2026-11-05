// Implements: include/ninfer/ops/allreduce.h
//
// Host-side composition: the DMA transport is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over
// UVA pointers (see pull_peer() below -- deliberately NOT cudaMemcpyPeerAsync, which stream
// capture rejects), and the local combine reuses the qualified residual_add computation body
// (x += y in BF16 with FP32 accumulation and a single round-to-nearest-even on store), which is
// exactly this Op's local step. Qualified small SM70 P2P sums instead read both operands into a
// rank-local staging output, using the same FP32-add/BF16-store arithmetic.
//
// DMA/direct-read collectives share one three-phase issue order. The phases exist because a wait must not be
// issued before the record it observes: cudaStreamWaitEvent snapshots the event's current state,
// so phase B's wait on inputs_ready[1-r] would snapshot a stale (or absent) capture point if the
// peer's phase-A record had not been issued yet.
//
//   phase A, both ranks:  record(inputs_ready[r])
//   phase B, both ranks:  wait(inputs_ready[1-r]); pull or sum peer source into own staging;
//                         record(pull_done[r])
//   phase C, both ranks:  wait(pull_done[1-r]); local combine or publish sum (allreduce_sum only)
//
// THE PULL ITSELF is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA pointers, NOT
// cudaMemcpyPeerAsync -- see pull_peer() below for why. The choreography, the streams each call
// is issued on, and the ordering proof are unchanged by that choice: it is the same transfer
// expressed through the API that CUDA graph capture accepts.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/speculative_round.h"
#include <cmath>
#include <cstring>

#include "ops/launcher/residual_add.h" // detail::residual_add_launch
#include "ops/launcher/allreduce.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <array>
#include <stdexcept>
#include <string>
#if defined(__linux__)
#include <dlfcn.h>
#include <nvml.h>
#endif

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_collective_devices(const ExecutionContext& ec, const char* message) {
    require(ec.device_count == 2 || ec.device_count == 4, message);
    for (int rank = 0; rank < ec.device_count; ++rank) {
        require(ec.dev[rank].has_value(), message);
        for (int other = 0; other < rank; ++other) {
            require(ec.dev[rank]->device != ec.dev[other]->device, message);
        }
    }
}

std::uint8_t* byte_offset(void* base, std::size_t offset) {
    return static_cast<std::uint8_t*>(base) + offset;
}

// The inbound half of a pull: `bytes` from `source` (resident on the peer device) into
// `destination` (resident on the device `stream` belongs to), issued on the DESTINATION's stream.
//
// Deliberately NOT cudaMemcpyPeerAsync. That entry point is rejected inside a stream capture
// region with cudaErrorStreamCaptureUnsupported (measured on CUDA 13.1 / driver 580.178.04, Task
// 4.2's capture probe), which would make the entire tensor-parallel decode program uncapturable
// and cost the ~40-per-layer host launch overhead that CUDA Graphs exist to remove. Under unified
// virtual addressing -- which every 64-bit Linux CUDA context has -- a device pointer already
// names its device, so cudaMemcpyAsync with cudaMemcpyDeviceToDevice expresses exactly the same
// cross-device transfer: direct over PCIe when the driver granted peer access, transparently
// staged through host memory when it did not (GeForce-class boards), identical either way in
// bytes moved and stream ordering. Verified equal to the peer form both eagerly (this file's
// qualification suite) and under capture.
cudaError_t pull_peer(void* destination, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
}

// Current-device save/restore. Both collectives issue work for each device in turn and must not
// leave the caller's current device changed.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDeviceGuard() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

void startup_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("peer transport startup: ") + operation + ": " +
                                 cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    }
}

// NVLink does not traverse the host IOMMU. Qualify this exception from the active
// links' actual PCI endpoints, not an advertised CUDA peer-access bit. NVML is
// startup-only and optional: when it cannot establish the complete mesh, retain
// the conservative PCIe/IOMMU rule. No driver setting is changed.
bool active_nvlink_mesh(const ExecutionContext& ec) {
#if defined(__linux__)
    void* library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) { return false; }
    const auto init = reinterpret_cast<decltype(&nvmlInit_v2)>(dlsym(library, "nvmlInit_v2"));
    const auto shutdown = reinterpret_cast<decltype(&nvmlShutdown)>(dlsym(library, "nvmlShutdown"));
    const auto get_device = reinterpret_cast<decltype(&nvmlDeviceGetHandleByPciBusId_v2)>(
        dlsym(library, "nvmlDeviceGetHandleByPciBusId_v2"));
    const auto state = reinterpret_cast<decltype(&nvmlDeviceGetNvLinkState)>(
        dlsym(library, "nvmlDeviceGetNvLinkState"));
    const auto remote = reinterpret_cast<decltype(&nvmlDeviceGetNvLinkRemotePciInfo_v2)>(
        dlsym(library, "nvmlDeviceGetNvLinkRemotePciInfo_v2"));
    if (!init || !shutdown || !get_device || !state || !remote || init() != NVML_SUCCESS) {
        dlclose(library);
        return false;
    }
    std::array<nvmlDevice_t, kMaximumExecutionDevices> devices{};
    std::array<std::array<unsigned, 4>, kMaximumExecutionDevices> pci{};
    bool complete = true;
    for (int rank = 0; rank < ec.device_count; ++rank) {
        char bus_id[32]{};
        if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), ec.dev[rank]->device) != cudaSuccess ||
            get_device(bus_id, &devices[rank]) != NVML_SUCCESS ||
            std::sscanf(bus_id, "%x:%x:%x.%x", &pci[rank][0], &pci[rank][1],
                        &pci[rank][2], &pci[rank][3]) != 4) {
            complete = false;
            break;
        }
    }
    for (int rank = 0; complete && rank < ec.device_count; ++rank) {
        std::array<bool, kMaximumExecutionDevices> linked{};
        linked[rank] = true;
        for (unsigned link = 0; link < NVML_NVLINK_MAX_LINKS; ++link) {
            nvmlEnableState_t enabled{};
            nvmlPciInfo_t endpoint{};
            if (state(devices[rank], link, &enabled) != NVML_SUCCESS ||
                enabled != NVML_FEATURE_ENABLED ||
                remote(devices[rank], link, &endpoint) != NVML_SUCCESS) { continue; }
            for (int other = 0; other < ec.device_count; ++other) {
                if (endpoint.domain == pci[other][0] && endpoint.bus == pci[other][1] &&
                    endpoint.device == pci[other][2]) { linked[other] = true; }
            }
        }
        for (int other = 0; other < ec.device_count; ++other) { complete &= linked[other]; }
    }
    shutdown();
    dlclose(library);
    return complete;
#else
    (void)ec;
    return false;
#endif
}

// Linux CUDA PCIe P2P is unsupported behind a translated IOMMU domain. A small
// allocation can nevertheless pass a copy probe while other mappings silently
// lose writes, so the domain restriction takes precedence over that probe.
std::string translated_iommu_domain(const ExecutionContext& ec) {
    std::string reason;
#if defined(__linux__)
    for (int rank = 0; rank < ec.device_count; ++rank) {
        char pci_bus_id[32]{};
        startup_check(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), ec.dev[rank]->device),
                      "cudaDeviceGetPCIBusId");
        // CUDA may emit uppercase hexadecimal; Linux PCI sysfs names are lowercase.
        for (char& c : pci_bus_id) {
            if (c >= 'A' && c <= 'F') { c += 'a' - 'A'; }
        }
        std::ifstream domain_file(std::string("/sys/bus/pci/devices/") + pci_bus_id +
                                  "/iommu_group/type");
        std::string domain;
        domain_file >> domain;
        if (domain == "DMA" || domain == "DMA-FQ") {
            if (!reason.empty()) { reason += "; "; }
            reason += std::string("PCI ") + pci_bus_id + " uses translated IOMMU domain " + domain;
        }
    }
#else
    (void)ec;
#endif
    return reason;
}

// Startup-only storage. The exact payload catches drivers that advertise peer access
// but silently drop DMA writes (observed with V100s behind an IOMMU). Use the same
// pull API and destination compute stream as the collectives, without GPU peer loads.
class PeerTransferProbe {
public:
    explicit PeerTransferProbe(const ExecutionContext& ec) : ec_(ec) {
        startup_check(cudaGetDevice(&previous_), "cudaGetDevice");
    }

    ~PeerTransferProbe() {
        for (int rank = 0; rank < ec_.device_count; ++rank) {
            cleanup(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
            if (source_[rank] != nullptr) { cleanup(cudaFree(source_[rank]), "cudaFree source"); }
            if (destination_[rank] != nullptr) {
                cleanup(cudaFree(destination_[rank]), "cudaFree destination");
            }
        }
        cleanup(cudaSetDevice(previous_), "restore device");
    }

    PeerTransferProbe(const PeerTransferProbe&) = delete;
    PeerTransferProbe& operator=(const PeerTransferProbe&) = delete;

    void set_device(int rank) const {
        startup_check(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
    }

    void initialize() {
        for (int rank = 0; rank < ec_.device_count; ++rank) {
            set_device(rank);
            startup_check(cudaMalloc(&source_[rank], kBytes), "cudaMalloc source");
            startup_check(cudaMalloc(&destination_[rank], kBytes), "cudaMalloc destination");
            std::array<std::uint32_t, kWords> values{};
            for (std::size_t i = 0; i < kWords; ++i) { values[i] = pattern(rank, i); }
            startup_check(cudaMemcpyAsync(source_[rank], values.data(), kBytes,
                                           cudaMemcpyHostToDevice, ec_.dev[rank]->stream),
                          "initialize source");
            startup_check(cudaStreamSynchronize(ec_.dev[rank]->stream), "retire source");
        }
    }

    // Empty means both complete copies matched their independent host patterns exactly.
    std::string qualify() {
        std::string mismatch;
        for (int rank = 0; rank < ec_.device_count; ++rank) {
            set_device(rank);
            cudaStream_t stream = ec_.dev[rank]->stream;
            for (int source_rank = 0; source_rank < ec_.device_count; ++source_rank) {
                if (rank == source_rank) { continue; }
                startup_check(cudaMemsetAsync(destination_[rank], 0xcd, kBytes, stream),
                              "clear destination");
                startup_check(pull_peer(destination_[rank], source_[source_rank], kBytes, stream),
                              "cross-device copy");
                startup_check(cudaStreamSynchronize(stream), "retire cross-device copy");
                std::array<std::uint32_t, kWords> actual{};
                startup_check(cudaMemcpy(actual.data(), destination_[rank], kBytes,
                                          cudaMemcpyDeviceToHost), "read destination");
                for (std::size_t i = 0; i < kWords; ++i) {
                    if (actual[i] != pattern(source_rank, i)) {
                        if (mismatch.empty()) {
                            mismatch = "device " + std::to_string(ec_.dev[source_rank]->device) +
                                       " -> " + std::to_string(ec_.dev[rank]->device) +
                                       " data mismatch at word " + std::to_string(i);
                        }
                        break;
                    }
                }
            }
        }
        return mismatch;
    }

    void disable_peer_access() const {
        for (int rank = 0; rank < ec_.device_count; ++rank) {
            set_device(rank);
            for (int other = 0; other < ec_.device_count; ++other) {
                if (rank == other) { continue; }
                const cudaError_t status = cudaDeviceDisablePeerAccess(ec_.dev[other]->device);
                if (status == cudaErrorPeerAccessNotEnabled) {
                    (void)cudaGetLastError();
                } else {
                    startup_check(status, "cudaDeviceDisablePeerAccess");
                }
            }
        }
    }

private:
    static constexpr std::size_t kWords = 4096;
    static constexpr std::size_t kBytes = kWords * sizeof(std::uint32_t);

    static std::uint32_t pattern(int rank, std::size_t index) {
        return 0x4f000000U ^ (std::uint32_t(rank) << 20U) ^
               (static_cast<std::uint32_t>(index) * 65537U);
    }

    static void cleanup(cudaError_t status, const char* operation) noexcept {
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during peer probe %s: %s: %s\n",
                         operation, cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    const ExecutionContext& ec_;
    int previous_ = 0;
    std::array<void*, kMaximumExecutionDevices> source_{};
    std::array<void*, kMaximumExecutionDevices> destination_{};
};

#ifndef NDEBUG
// Debug-only residency and aliasing predicates. These cost a driver round trip per pointer, so
// they are compiled out of the Release build the product ships; a wrong-device or self-overlapping
// argument is a caller bug that surfaces here during development instead of as a silently wrong
// result or an opaque cudaErrorInvalidValue later.
void require_resident_on(const void* pointer, int device, const char* message) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device, message);
}

void require_disjoint(const void* first, std::size_t first_bytes, const void* second,
                      std::size_t second_bytes, const char* message) {
    const auto* a = static_cast<const std::uint8_t*>(first);
    const auto* b = static_cast<const std::uint8_t*>(second);
    require(a + first_bytes <= b || b + second_bytes <= a, message);
}
#endif

} // namespace

bool enable_peer_access(const ExecutionContext& ec) {
    if (ec.device_count == 1) { return false; }
    require_collective_devices(ec, "peer transport requires two or four distinct devices");

    PeerTransferProbe probe(ec);
    try {
        const bool nvlink = active_nvlink_mesh(ec);
        std::string direct_failure = nvlink ? std::string{} : translated_iommu_domain(ec);
        if (direct_failure.empty()) {
            for (int rank = 0; rank < ec.device_count; ++rank) {
                for (int other = 0; other < ec.device_count; ++other) {
                    if (rank == other) { continue; }
                    int accessible = 0;
                    startup_check(cudaDeviceCanAccessPeer(&accessible, ec.dev[rank]->device,
                                                          ec.dev[other]->device),
                                  "cudaDeviceCanAccessPeer");
                    if (!accessible) { direct_failure = "peer access unavailable"; }
                }
            }
        }
        const bool supported = direct_failure.empty();
        if (supported) {
            for (int rank = 0; rank < ec.device_count; ++rank) {
                probe.set_device(rank);
                for (int other = 0; other < ec.device_count; ++other) {
                    if (rank == other) { continue; }
                    const cudaError_t status = cudaDeviceEnablePeerAccess(ec.dev[other]->device, 0);
                    if (status == cudaErrorPeerAccessAlreadyEnabled) {
                        (void)cudaGetLastError();
                    } else {
                        startup_check(status, "cudaDeviceEnablePeerAccess");
                    }
                }
            }
        } else {
            if (direct_failure.empty()) { direct_failure = "peer access unavailable"; }
            probe.disable_peer_access();
        }
        probe.initialize();
        if (supported) {
            direct_failure = probe.qualify();
            if (direct_failure.empty()) {
                if (nvlink) {
                    std::fprintf(stderr, "[ninfer] verified direct NVLink P2P across %d ranks\n", ec.device_count);
                }
                return true;
            }
            probe.disable_peer_access();
        }

        // With peer access disabled (or prohibited by the IOMMU domain), the same UVA D2D
        // operation is handled by CUDA's driver-managed staging path. Qualify that exact API
        // rather than maintaining a second explicit D2H/H2D transport in every collective.
        const std::string fallback_failure = probe.qualify();
        if (!fallback_failure.empty()) {
            throw std::runtime_error("peer transport startup: UVA D2D fallback validation failed: " +
                                     fallback_failure);
        }
        std::fprintf(stderr,
                     "[ninfer] direct P2P disabled (%s); using verified CUDA UVA D2D staging\n",
                     direct_failure.c_str());
        return false;
    } catch (...) {
        // Failed startup must not leave a partially enabled pair behind. Clear the
        // runtime's last error before cleanup; a fatal context error still prevents
        // further use, and the original startup exception remains authoritative.
        (void)cudaGetLastError();
        try { probe.disable_peer_access(); } catch (...) {}
        throw;
    }
}

PeerEvents::PeerEvents(const ExecutionContext& ec, bool direct_peer_access, HostPackets host_packets)
    : direct_peer_access_(direct_peer_access), ranks_(ec.device_count) {
    require_collective_devices(ec, "PeerEvents: requires two or four distinct devices");
    const CurrentDeviceGuard guard;
    bool packet_transport = direct_peer_access;
    for (int rank = 0; rank < ec.device_count; ++rank) {
        packet_transport &= ec.dev[rank]->sm() == 70;
    }
    std::string kernel_release;
    if (host_packets == HostPackets::Auto) {
        std::ifstream release("/proc/sys/kernel/osrelease");
        std::getline(release, kernel_release);
    }
    const bool want_host_packets = host_packets == HostPackets::Required ||
        (host_packets == HostPackets::Auto && kernel_release.find("WSL2") != std::string::npos);
    if (want_host_packets && !direct_peer_access && ec.device_count == 2 &&
        ec.dev[0]->sm() == 70 && ec.dev[1]->sm() == 70) {
        for (int rank = 0; rank < 2; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            cudaDeviceProp properties{};
            startup_check(cudaGetDeviceProperties(&properties, ec.dev[rank]->device), "host packet properties");
            require(properties.canMapHostMemory && properties.unifiedAddressing,
                    "mapped host packets require mapped host memory and UVA");
            host_packets_[rank] = std::make_unique<PinnedHostBuffer>(2 * kPacketMaxBytes);
            std::memset(host_packets_[rank]->data(), 0xff, 2 * kPacketMaxBytes);
            host_packet_epochs_[rank] = DeviceBuffer(kPacketBlocks * sizeof(unsigned));
            CUDA_CHECK(cudaMemsetAsync(host_packet_epochs_[rank].p, 0,
                kPacketBlocks * sizeof(unsigned), ec.dev[rank]->stream));
        }
        for (int rank = 0; rank < 2; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            for (int owner = 0; owner < 2; ++owner) {
                startup_check(cudaHostGetDevicePointer(&host_packet_aliases_[rank][owner],
                    host_packets_[owner]->data(), 0), "host packet mapping");
            }
        }
        std::fprintf(stderr, "[ninfer] TP2 small reductions use mapped-host packets (320 KiB pinned RAM)\n");
    } else {
        require(host_packets != HostPackets::Required,
                "mapped host packets require SM70 TP2 with direct P2P disabled");
    }
    if (packet_transport) {
        const std::size_t channel_bytes = packet_capacity_bytes();
        const std::size_t blocks = channel_bytes / (16 * 64);
        const std::size_t payload_bytes = 2 * channel_bytes * (ec.device_count == 4 ? 4 : 1);
        for (int rank = 0; rank < ec.device_count; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            packets_[rank] = DeviceBuffer(payload_bytes + blocks * sizeof(unsigned));
            CUDA_CHECK(cudaMemsetAsync(packets_[rank].p, 0xff, payload_bytes,
                                       ec.dev[rank]->stream));
            CUDA_CHECK(cudaMemsetAsync(byte_offset(packets_[rank].p, payload_bytes),
                                       0, blocks * sizeof(unsigned), ec.dev[rank]->stream));
        }
    }
    // Create through a local table so a mid-way failure destroys what was already created instead
    // of leaking it; only a fully constructed set is published into the members.
    std::array<cudaEvent_t, 2 * kMaximumExecutionDevices> created{};
    for (int slot = 0; slot < 2 * ec.device_count; ++slot) {
        const int rank             = slot % ec.device_count;
        const cudaError_t creation = cudaSetDevice(ec.dev[rank]->device);
        cudaError_t status         = creation;
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&created[slot], cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            for (int done = 0; done < slot; ++done) { cudaEventDestroy(created[done]); }
            throw std::runtime_error(std::string("PeerEvents: event creation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }
    for (int rank = 0; rank < ec.device_count; ++rank) {
        inputs_ready_[rank] = created[rank];
        pull_done_[rank] = created[ec.device_count + rank];
    }
    // Seed pull_done with completed events so the first collective may use the same inter-call
    // lifetime edge as every later one.
    for (int rank = 0; rank < ec.device_count; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaEventRecord(pull_done_[static_cast<std::size_t>(rank)],
                                   ec.dev[rank]->stream));
    }
    for (int rank = 0; rank < ec.device_count; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[rank]->stream));
    }
}

PeerEvents::~PeerEvents() {
    for (auto* group : {&inputs_ready_, &pull_done_}) {
        for (cudaEvent_t& event : *group) {
            if (event == nullptr) { continue; }
            const cudaError_t status = cudaEventDestroy(event);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "CUDA cleanup failed during cudaEventDestroy: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            event = nullptr;
        }
    }
}

PeerEvents::PeerEvents(PeerEvents&& other) noexcept
    : inputs_ready_(other.inputs_ready_), pull_done_(other.pull_done_),
      packets_(std::move(other.packets_)),
      host_packets_(std::move(other.host_packets_)),
      host_packet_aliases_(other.host_packet_aliases_),
      host_packet_epochs_(std::move(other.host_packet_epochs_)),
      direct_peer_access_(other.direct_peer_access_), ranks_(other.ranks_) {
    other.inputs_ready_ = {};
    other.pull_done_    = {};
    other.host_packet_aliases_ = {};
    other.direct_peer_access_ = false;
    other.ranks_ = 0;
}

PeerEvents& PeerEvents::operator=(PeerEvents&& other) noexcept {
    // Swap rather than destroy-then-assign: `other`'s destructor releases whatever this instance
    // held, in exactly one place.
    inputs_ready_.swap(other.inputs_ready_);
    pull_done_.swap(other.pull_done_);
    packets_.swap(other.packets_);
    host_packets_.swap(other.host_packets_);
    host_packet_aliases_.swap(other.host_packet_aliases_);
    host_packet_epochs_.swap(other.host_packet_epochs_);
    std::swap(direct_peer_access_, other.direct_peer_access_);
    std::swap(ranks_, other.ranks_);
    return *this;
}

namespace {
void validate_sum(std::span<const Tensor> buffer, std::span<const Tensor> staging,
                  const ExecutionContext& ec, const PeerEvents& events) {
    require_collective_devices(ec, "allreduce_sum: requires two or four distinct devices");
    require(buffer.size() == static_cast<std::size_t>(ec.device_count) && staging.size() == buffer.size(),
            "allreduce_sum: tensor spans must have one entry per active rank");
    for (int rank = 0; rank < ec.device_count; ++rank) {
        require(buffer[rank].dtype == DType::BF16 && staging[rank].dtype == DType::BF16,
                "allreduce_sum: buffer/staging must be BF16");
        require(buffer[rank].data != nullptr && staging[rank].data != nullptr,
                "allreduce_sum: buffer/staging data must be non-null");
        require(buffer[rank].is_contiguous() && staging[rank].is_contiguous(),
                "allreduce_sum: buffer/staging must be contiguous");
        for (int d = 0; d < 4; ++d) {
            require(buffer[rank].ne[d] == buffer[0].ne[d],
                    "allreduce_sum: buffer shapes must match on all devices");
            if (ec.device_count == 2) {
                require(staging[rank].ne[d] == buffer[0].ne[d],
                        "allreduce_sum: TP2 staging shape must match the buffer");
            }
        }
        if (ec.device_count == 4) {
            require(staging[rank].numel() >= 4 * buffer[0].numel(),
                    "allreduce_sum: TP4 staging must hold four full contributions");
        }
    }
    require(events.live() && events.ranks() == ec.device_count, "allreduce_sum: events must match active ranks");

    const std::size_t bytes = buffer[0].bytes();

#ifndef NDEBUG
    for (int rank = 0; rank < ec.device_count; ++rank) {
        require_resident_on(buffer[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: buffer[r] must be resident on ec.dev[r]");
        require_resident_on(staging[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: staging[r] must be resident on ec.dev[r]");
        require_disjoint(buffer[rank].data, bytes, staging[rank].data, staging[rank].bytes(),
                         "allreduce_sum: staging[r] must not overlap buffer[r]");
    }
#endif

}
} // namespace

void allreduce_sum(std::span<const Tensor> buffer, std::span<const Tensor> staging,
                   const ExecutionContext& ec, const PeerEvents& events) {
    validate_sum(buffer, staging, ec, events);
    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0) { return; }
    const CurrentDeviceGuard guard;

    std::uintptr_t addresses = 0;
    for (const auto& tensor : buffer) {
        addresses |= reinterpret_cast<std::uintptr_t>(tensor.data);
    }
    if (events.host_packets() && bytes <= PeerEvents::kPacketMaxBytes &&
        bytes % 16 == 0 && (addresses & 15U) == 0) {
        for (int rank = 0; rank < 2; ++rank) {
            const auto& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            detail::allreduce_host_packet_sum_launch(buffer[rank], buffer[rank],
                events.host_packet_storage(rank, rank), events.host_packet_storage(rank, 1-rank),
                events.host_packet_epochs(rank), PeerEvents::kPacketMaxBytes, local.stream);
        }
        return;
    }
    if (events.packet_storage(0) != nullptr && bytes <= events.packet_capacity_bytes() &&
        bytes % 16 == 0 && (addresses & 15U) == 0) {
        const std::array<void*, 4> storage{events.packet_storage(0), events.packet_storage(1),
                                         events.packet_storage(2), events.packet_storage(3)};
        for (int rank = 0; rank < ec.device_count; ++rank) {
            const auto& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            if (ec.device_count == 4) {
                detail::allreduce_packet_sum4_launch(buffer[rank], storage, rank,
                    events.packet_capacity_bytes(), local.stream);
            } else {
                detail::allreduce_packet_sum_launch(buffer[rank], buffer[rank],
                    storage[rank], storage[1 - rank], events.packet_capacity_bytes(), local.stream);
            }
        }
        return;
    }

    // 1Cat's wide-message reduce-scatter/allgather schedule assigns each rank one
    // quarter of the sum. Keep our canonical FP32 rank order and event-owned
    // lifetimes: all input reads retire before any original buffer is overwritten.
    // The next call's inputs-ready waits also retire this call's gathers before
    // a rank can overwrite the same staging storage.
    if (ec.device_count == 4 && events.direct_peer_access() &&
        bytes >= 262144 && bytes % 64 == 0 && (addresses & 15U) == 0 &&
        buffer[0].numel() / 4 <= std::numeric_limits<std::int32_t>::max()) {
        const std::size_t part_bytes = bytes / 4;
        const auto part_elements = static_cast<std::int32_t>(buffer[0].numel() / 4);
        for (int rank = 0; rank < 4; ++rank) {
            const auto& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
        }
        for (int rank = 0; rank < 4; ++rank) {
            const auto& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            for (int source = 0; source < 4; ++source) {
                if (source != rank) {
                    CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(source), 0));
                }
            }
            std::array<Tensor, 4> inputs;
            for (int source = 0; source < 4; ++source) {
                inputs[source] = Tensor(byte_offset(buffer[source].data, rank * part_bytes),
                                        DType::BF16, {part_elements});
            }
            Tensor partial(staging[rank].data, DType::BF16, {part_elements});
            detail::allreduce_sum4_launch(inputs, partial, local.stream);
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
        for (int rank = 0; rank < 4; ++rank) {
            const auto& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            for (int source = 0; source < 4; ++source) {
                if (source != rank) {
                    CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(source), 0));
                }
            }
            for (int source = 0; source < 4; ++source) {
                CUDA_CHECK(pull_peer(byte_offset(buffer[rank].data, source * part_bytes),
                                     staging[source].data, part_bytes, local.stream));
            }
        }
        return;
    }

    // Direct reads are legal only after startup qualification. Long payloads keep DMA: the
    // measured small-message launch-latency benefit reverses for wide prefill transfers.
    bool direct_sum = events.direct_peer_access() && bytes <= 81920;
    for (int rank = 0; rank < ec.device_count; ++rank) { direct_sum &= ec.dev[rank]->sm() == 70; }

    // Phase A: publish "my operand is complete" on each stream, before any wait observes it.
    for (int rank = 0; rank < ec.device_count; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: each rank pulls the peer operand through CUDA's UVA device-to-device path. On
    // translated IOMMU domains the driver transparently stages this copy through host memory;
    // keeping it as one captured D2D node avoids the extra D2H/H2D event chain and is materially
    // faster on the V100 PCIe bridge.
    for (int rank = 0; rank < ec.device_count; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        for (int other = 0; other < ec.device_count; ++other) {
            if (other != rank) {
                CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(other), 0));
            }
        }
        if (direct_sum) {
            if (ec.device_count == 2) {
                detail::allreduce_peer_sum_launch(buffer[rank], buffer[1 - rank], staging[rank],
                                                  local.stream);
            } else {
                const std::array<Tensor, 4> inputs{buffer[0], buffer[1], buffer[2], buffer[3]};
                Tensor sum = buffer[rank];
                sum.data = staging[rank].data;
                detail::allreduce_sum4_launch(inputs, sum, local.stream);
            }
        } else {
            if (ec.device_count == 2) {
                CUDA_CHECK(pull_peer(staging[rank].data, buffer[1 - rank].data, bytes, local.stream));
            } else {
                for (int source = 0; source < ec.device_count; ++source) {
                    CUDA_CHECK(pull_peer(byte_offset(staging[rank].data, source * bytes),
                                         buffer[source].data, bytes, local.stream));
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the in-place combine may only overwrite buffer[rank] once the peer has finished
    // reading it. That same wait is what makes the next call's phase B safe.
    for (int rank = 0; rank < ec.device_count; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        for (int other = 0; other < ec.device_count; ++other) {
            if (other != rank) {
                CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(other), 0));
            }
        }
        if (direct_sum) {
            CUDA_CHECK(cudaMemcpyAsync(buffer[rank].data, staging[rank].data, bytes,
                                       cudaMemcpyDeviceToDevice, local.stream));
        } else if (ec.device_count == 2) {
            Tensor accumulator = buffer[rank];
            detail::residual_add_launch(staging[rank], accumulator, local.stream);
        } else {
            std::array<Tensor, 4> inputs;
            for (int source = 0; source < ec.device_count; ++source) {
                inputs[source] = buffer[rank];
                inputs[source].data = byte_offset(staging[rank].data, source * bytes);
            }
            detail::allreduce_sum4_launch(inputs, buffer[rank], local.stream);
        }
    }
}

namespace {
void validate_sum_to(std::span<const Tensor> input, std::span<const Tensor> output,
                      const ExecutionContext& ec, const PeerEvents& events) {
    require_collective_devices(ec, "allreduce_sum_to: requires two distinct devices");
    require(ec.device_count == 2 && input.size() == 2 && output.size() == 2,
            "allreduce_sum_to: input/output must match TP2");
    require(events.live() && events.ranks() == 2,
            "allreduce_sum_to: events must match TP2");
    for (int rank = 0; rank < 2; ++rank) {
        require(input[rank].dtype == DType::BF16 && output[rank].dtype == DType::BF16 &&
                input[rank].data != nullptr && output[rank].data != nullptr &&
                input[rank].is_contiguous() && output[rank].is_contiguous(),
                "allreduce_sum_to: requires contiguous BF16 input/output");
        for (int d = 0; d < 4; ++d) {
            require(input[rank].ne[d] == input[0].ne[d] &&
                    output[rank].ne[d] == input[0].ne[d],
                    "allreduce_sum_to: shapes must match");
        }
#ifndef NDEBUG
        require_resident_on(input[rank].data, ec.dev[rank]->device,
                            "allreduce_sum_to: input must reside on its rank");
        require_resident_on(output[rank].data, ec.dev[rank]->device,
                            "allreduce_sum_to: output must reside on its rank");
        require_disjoint(input[rank].data, input[rank].bytes(), output[rank].data,
                         output[rank].bytes(), "allreduce_sum_to: input/output overlap");
#endif
    }
}
} // namespace

static void sum_rmsnorm(std::span<const Tensor> buffer, std::span<const Tensor> staging,
                        const ReplicatedRmsNorm& norm, const ExecutionContext& ec,
                        const PeerEvents& events, bool published) {
    validate_sum(buffer, staging, ec, events);
    require(norm.weight.size() == buffer.size() && norm.output.size() == buffer.size() &&
            std::isfinite(norm.eps) && norm.eps > 0.f, "collective norm: invalid profile");
    std::uintptr_t addresses = 0;
    for (int rank = 0; rank < ec.device_count; ++rank) {
        const auto& w = norm.weight[rank];
        const auto& n = norm.output[rank];
        require(w.dtype == DType::BF16 &&
                (n.dtype == DType::BF16 || n.dtype == DType::FP16) &&
                n.dtype == norm.output[0].dtype && w.data && n.data &&
                w.is_contiguous() && n.is_contiguous() && w.numel() == buffer[0].ne[0],
                "collective norm: invalid weight/output");
        for (int d = 0; d < 4; ++d) require(n.ne[d] == buffer[0].ne[d],
            "collective norm: output shape mismatch");
        addresses |= reinterpret_cast<std::uintptr_t>(buffer[rank].data) |
                     reinterpret_cast<std::uintptr_t>(w.data) |
                     reinterpret_cast<std::uintptr_t>(n.data);
#ifndef NDEBUG
        require_resident_on(w.data, ec.dev[rank]->device, "collective norm weight residency");
        require_resident_on(n.data, ec.dev[rank]->device, "collective norm output residency");
        require_disjoint(n.data, n.bytes(), buffer[rank].data, buffer[rank].bytes(),
                         "collective norm sum overlap");
        require_disjoint(n.data, n.bytes(), staging[rank].data, staging[rank].bytes(),
                         "collective norm staging overlap");
#endif
    }
    if (ec.device_count == 4 && events.packet_storage(0) && buffer[0].ne[0] == 5120 &&
        buffer[0].ne[1] >= 1 && buffer[0].ne[1] <= 64 && buffer[0].ne[2] == 1 &&
        buffer[0].ne[3] == 1 && (addresses & 15U) == 0 &&
        (norm.output[0].dtype == DType::BF16 || (published && buffer[0].ne[1] <= 8))) {
        const CurrentDeviceGuard guard;
        const std::array<void*, 4> storage{events.packet_storage(0), events.packet_storage(1),
                                         events.packet_storage(2), events.packet_storage(3)};
        for (int rank = 0; rank < 4; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            detail::allreduce_packet_sum4_rmsnorm_launch(buffer[rank], norm.weight[rank],
                norm.output[rank], norm.eps, norm.unit_offset, storage, rank,
                events.packet_capacity_bytes(), ec.dev[rank]->stream, published);
        }
        return;
    }
    require(!published, "published TP4 collective requires qualified packet norm geometry");
    require(norm.output[0].dtype == DType::BF16,
            "collective FP16 norm image requires published TP4 D5120 T1..8");
    allreduce_sum(buffer, staging, ec, events);
    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < ec.device_count; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        Tensor normalized = norm.output[rank];
        rmsnorm(buffer[rank], norm.weight[rank], norm.eps, norm.unit_offset, normalized,
                ec.dev[rank]->stream);
    }
}

void allreduce_sum_rmsnorm(std::span<const Tensor> buffer, std::span<const Tensor> staging,
                          const ReplicatedRmsNorm& norm,
                          const ExecutionContext& ec, const PeerEvents& events) {
    sum_rmsnorm(buffer, staging, norm, ec, events, false);
}

void detail::allreduce_published_sum_rmsnorm(std::span<const Tensor> buffer,
    std::span<const Tensor> staging, const ReplicatedRmsNorm& norm,
    const ExecutionContext& ec, const PeerEvents& events) {
    sum_rmsnorm(buffer, staging, norm, ec, events, true);
}

void allreduce_sum_to(std::span<const Tensor> input, std::span<const Tensor> output,
                      const ExecutionContext& ec, const PeerEvents& events) {
    validate_sum_to(input, output, ec, events);
    const auto bytes = input[0].bytes();
    if (bytes == 0) { return; }
    const CurrentDeviceGuard guard;
    const auto addresses = reinterpret_cast<std::uintptr_t>(input[0].data) |
                           reinterpret_cast<std::uintptr_t>(input[1].data) |
                           reinterpret_cast<std::uintptr_t>(output[0].data) |
                           reinterpret_cast<std::uintptr_t>(output[1].data);
    if (events.host_packets() && bytes <= PeerEvents::kPacketMaxBytes &&
        bytes % 16 == 0 && (addresses & 15U) == 0) {
        for (int rank = 0; rank < 2; ++rank) {
            const auto& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            detail::allreduce_host_packet_sum_launch(input[rank], output[rank],
                events.host_packet_storage(rank, rank), events.host_packet_storage(rank, 1-rank),
                events.host_packet_epochs(rank), PeerEvents::kPacketMaxBytes, local.stream);
        }
        return;
    }
    if (events.packet_storage(0) != nullptr && bytes <= events.packet_capacity_bytes() &&
        bytes % 16 == 0 && (addresses & 15U) == 0) {
        for (int rank = 0; rank < 2; ++rank) {
            const auto& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            detail::allreduce_packet_sum_launch(input[rank], output[rank],
                events.packet_storage(rank), events.packet_storage(1 - rank),
                events.packet_capacity_bytes(), local.stream);
        }
        return;
    }
    const bool direct = events.direct_peer_access() && bytes <= 81920 &&
                        ec.dev[0]->sm() == 70 && ec.dev[1]->sm() == 70;
    for (int rank = 0; rank < 2; ++rank) {
        const auto& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
        if (direct) {
            detail::allreduce_peer_sum_launch(input[rank], input[1 - rank], output[rank],
                                              local.stream);
        } else {
            CUDA_CHECK(pull_peer(output[rank].data, input[1 - rank].data, bytes, local.stream));
            Tensor accumulator = output[rank];
            detail::residual_add_launch(input[rank], accumulator, local.stream);
        }
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
    }
}

static void sum_rmsnorm_to(std::span<const Tensor> input, std::span<const Tensor> output,
                              const ReplicatedRmsNorm& norm, const ExecutionContext& ec,
                              const PeerEvents& events, bool published) {
    validate_sum_to(input, output, ec, events);
    require(norm.weight.size() == 2 && norm.output.size() == 2 &&
            std::isfinite(norm.eps) && norm.eps > 0.0F,
            "allreduce_sum_rmsnorm_to: invalid RMSNorm profile");
    std::uintptr_t addresses = 0;
    for (int rank = 0; rank < 2; ++rank) {
        const auto& w = norm.weight[rank];
        const auto& n = norm.output[rank];
        require(w.dtype == DType::BF16 && n.dtype == DType::BF16 && w.data && n.data &&
                w.is_contiguous() && n.is_contiguous() && w.numel() == input[0].ne[0],
                "allreduce_sum_rmsnorm_to: invalid weight/output");
        for (int d = 0; d < 4; ++d) require(n.ne[d] == input[0].ne[d],
            "allreduce_sum_rmsnorm_to: normalized output shape mismatch");
        addresses |= reinterpret_cast<std::uintptr_t>(input[rank].data) |
                     reinterpret_cast<std::uintptr_t>(output[rank].data) |
                     reinterpret_cast<std::uintptr_t>(w.data) |
                     reinterpret_cast<std::uintptr_t>(n.data);
#ifndef NDEBUG
        require_resident_on(w.data, ec.dev[rank]->device, "collective norm weight residency");
        require_resident_on(n.data, ec.dev[rank]->device, "collective norm output residency");
        require_disjoint(n.data, n.bytes(), input[rank].data, input[rank].bytes(), "collective norm input overlap");
        require_disjoint(n.data, n.bytes(), output[rank].data, output[rank].bytes(), "collective norm sum overlap");
#endif
    }
    if (events.packet_storage(0) && input[0].ne[0] == 5120 && input[0].ne[1] >= 1 &&
        input[0].bytes() <= events.packet_capacity_bytes() &&
        input[0].ne[2] == 1 && input[0].ne[3] == 1 && (addresses & 15U) == 0) {
        const CurrentDeviceGuard guard;
        for (int rank = 0; rank < 2; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            detail::allreduce_packet_rmsnorm_launch(input[rank], output[rank], norm.weight[rank],
                norm.output[rank], norm.eps, norm.unit_offset, events.packet_storage(rank),
                events.packet_storage(1-rank), events.packet_capacity_bytes(), ec.dev[rank]->stream, published);
        }
        return;
    }
    require(!published, "published collective requires qualified packet norm geometry");
    allreduce_sum_to(input, output, ec, events);
    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        Tensor normalized = norm.output[rank];
        rmsnorm(output[rank], norm.weight[rank], norm.eps, norm.unit_offset, normalized, ec.dev[rank]->stream);
    }
}

void allreduce_sum_rmsnorm_to(std::span<const Tensor> input, std::span<const Tensor> output,
                              const ReplicatedRmsNorm& norm, const ExecutionContext& ec,
                              const PeerEvents& events) {
    sum_rmsnorm_to(input, output, norm, ec, events, false);
}

void detail::allreduce_published_sum_rmsnorm_to(
    std::span<const Tensor> input, std::span<const Tensor> output,
    const ReplicatedRmsNorm& norm, const ExecutionContext& ec, const PeerEvents& events) {
    sum_rmsnorm_to(input, output, norm, ec, events, true);
}

void allgather_rows(std::span<const Tensor> destination, std::span<const Tensor> part,
                    const ExecutionContext& ec, const PeerEvents& events) {
    require_collective_devices(ec, "allgather_rows: requires two or four distinct devices");
    require(destination.size() == static_cast<std::size_t>(ec.device_count) && part.size() == destination.size(),
            "allgather_rows: tensor spans must have one entry per active rank");
    const DType dtype             = destination[0].dtype;
    const std::int32_t row_length = destination[0].ne[0];
    const std::int32_t total_rows = destination[0].ne[1];
    std::int64_t contributed_rows = 0;
    for (int rank = 0; rank < ec.device_count; ++rank) {
        require(destination[rank].dtype == dtype && part[rank].dtype == dtype,
                "allgather_rows: destination/part must share one dtype");
        require(destination[rank].data != nullptr && part[rank].data != nullptr,
                "allgather_rows: destination/part data must be non-null");
        require(destination[rank].is_contiguous() && part[rank].is_contiguous(),
                "allgather_rows: destination/part must be contiguous");
        require(destination[rank].ne[0] == row_length && part[rank].ne[0] == row_length,
                "allgather_rows: destination/part must agree on row length ne[0]");
        require(destination[rank].ne[1] == total_rows,
                "allgather_rows: both destinations must have the same row count");
        require(destination[rank].ne[2] == 1 && destination[rank].ne[3] == 1 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "allgather_rows: destination/part must be two-dimensional [C, R]");
        contributed_rows += part[rank].ne[1];
    }
    require(contributed_rows == total_rows,
            "allgather_rows: owned row counts must sum to the destination row count");
    require(events.live() && events.ranks() == ec.device_count, "allgather_rows: events must match active ranks");

    const std::size_t row_bytes = static_cast<std::size_t>(row_length) * dtype_size(dtype);
    std::array<std::size_t, kMaximumExecutionDevices> block{}, offset{};
    std::size_t cursor = 0;
    for (int rank = 0; rank < ec.device_count; ++rank) {
        offset[rank] = cursor;
        block[rank] = row_bytes * static_cast<std::size_t>(part[rank].ne[1]);
        cursor += block[rank];
    }

#ifndef NDEBUG
    for (int rank = 0; rank < ec.device_count; ++rank) {
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "allgather_rows: destination[r] must be resident on ec.dev[r]");
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "allgather_rows: part[r] must be resident on ec.dev[r]");
        require_disjoint(destination[rank].data, destination[rank].bytes(), part[rank].data,
                         block[rank], "allgather_rows: part[r] must not overlap destination[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my block is complete".
    for (int rank = 0; rank < ec.device_count; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: rank r writes its own block locally and pulls the peer block, both on its stream.
    for (int rank = 0; rank < ec.device_count; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        for (int other = 0; other < ec.device_count; ++other) {
            if (other != rank) {
                CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(other), 0));
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                   part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                   local.stream));
        for (int source = 0; source < ec.device_count; ++source) {
            if (source == rank) { continue; }
            CUDA_CHECK(pull_peer(byte_offset(destination[rank].data, offset[source]),
                                 part[source].data, block[source], local.stream));
        }
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: the Op writes nothing else, but the caller (or the next call) will overwrite
    // part[rank]. Ordering each stream after the peer's read is what makes that safe without a
    // host synchronization.
    for (int rank = 0; rank < ec.device_count; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        for (int other = 0; other < ec.device_count; ++other) {
            if (other != rank) {
                CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(other), 0));
            }
        }
    }
}

void gather_columns_rank0(const Tensor& destination, std::span<const Tensor> part,
                          const ExecutionContext& ec, const PeerEvents& events) {
    require_collective_devices(ec, "gather_columns_rank0: requires two or four distinct devices");
    require(part.size() == static_cast<std::size_t>(ec.device_count),
            "gather_columns_rank0: tensor span must have one entry per active rank");
    const DType dtype             = destination.dtype;
    const std::int32_t full_width = destination.ne[0];
    const std::int32_t columns    = destination.ne[1];
    require(full_width > 0 && columns > 0,
            "gather_columns_rank0: dimensions must be positive");
    require(destination.data != nullptr && destination.is_contiguous(),
            "gather_columns_rank0: destination must be contiguous and non-null");
    require(destination.ne[2] == 1 && destination.ne[3] == 1,
            "gather_columns_rank0: destination must be two-dimensional [C,T]");
    std::int64_t contributed_width = 0;
    for (int rank = 0; rank < ec.device_count; ++rank) {
        require(part[rank].dtype == dtype && part[rank].data != nullptr &&
                    part[rank].is_contiguous(),
                "gather_columns_rank0: parts must share dtype and be contiguous/non-null");
        require(part[rank].ne[1] == columns && part[rank].ne[0] > 0 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "gather_columns_rank0: parts must be two-dimensional [C_r,T]");
        contributed_width += part[rank].ne[0];
    }
    require(contributed_width == full_width,
            "gather_columns_rank0: owned widths must sum to destination width");
    require(events.live() && events.ranks() == ec.device_count,
            "gather_columns_rank0: events must match active ranks");

    const std::size_t element_bytes = dtype_size(dtype);
    const std::size_t destination_pitch =
        static_cast<std::size_t>(full_width) * element_bytes;
    std::array<std::size_t, kMaximumExecutionDevices> block{}, offset{};
    std::size_t cursor = 0;
    for (int rank = 0; rank < ec.device_count; ++rank) {
        offset[rank] = cursor;
        block[rank] = static_cast<std::size_t>(part[rank].ne[0]) * element_bytes;
        cursor += block[rank];
    }

#ifndef NDEBUG
    require_resident_on(destination.data, ec.dev[0]->device,
                        "gather_columns_rank0: destination must be resident on rank 0");
    for (int rank = 0; rank < ec.device_count; ++rank) {
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "gather_columns_rank0: part must be resident on its rank");
        require_disjoint(destination.data, destination.bytes(), part[rank].data, part[rank].bytes(),
                         "gather_columns_rank0: part must not overlap destination");
    }
#endif

    const CurrentDeviceGuard guard;
    const DeviceContext& rank0 = *ec.dev[0];
    for (int rank = 0; rank < ec.device_count; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), ec.dev[rank]->stream));
    }
    CurrentDeviceGuard::set(rank0.device);
    for (int source = 0; source < ec.device_count; ++source) {
        if (source != 0) {
            CUDA_CHECK(cudaStreamWaitEvent(rank0.stream, events.inputs_ready(source), 0));
        }
        CUDA_CHECK(cudaMemcpy2DAsync(byte_offset(destination.data, offset[source]), destination_pitch,
                                     part[source].data, block[source], block[source],
                                     static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice,
                                     rank0.stream));
    }
    CUDA_CHECK(cudaEventRecord(events.pull_done(0), rank0.stream));

    // Rank 1 may overwrite its proposal shard only after rank 0 has consumed it.  No reciprocal
    // destination write is needed because the selector and all subsequent DFlash logic run on
    // rank 0.
    for (int rank = 1; rank < ec.device_count; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaStreamWaitEvent(ec.dev[rank]->stream, events.pull_done(0), 0));
    }
}

void allreduce_argmax(std::span<const Tensor> values, std::span<const Tensor> indices,
                       std::span<const Tensor> output, Tensor& gathered_values,
                       Tensor& gathered_indices, std::int32_t shard_rows,
                       const ExecutionContext& ec, const PeerEvents& events,
                       const std::int32_t* id_map) {
    require_collective_devices(ec, "allreduce_argmax: requires two or four distinct devices");
    const auto ranks = static_cast<std::size_t>(ec.device_count);
    require(values.size() == ranks && indices.size() == ranks && output.size() == ranks &&
                shard_rows > 0 && events.live() && events.ranks() == ec.device_count,
            "allreduce_argmax: invalid domain");
    const int columns = output[0].ne[0];
    const auto shape_ok = [](const Tensor& t, DType dtype, int width, int height) {
        return t.data && t.dtype == dtype && t.is_contiguous() && t.ne[0] == width &&
               t.ne[1] == height && t.ne[2] == 1 && t.ne[3] == 1;
    };
    require(columns > 0 && shape_ok(gathered_values, DType::FP32, ec.device_count, columns) &&
                shape_ok(gathered_indices, DType::I32, ec.device_count, columns),
            "allreduce_argmax: invalid gather scratch");
    for (int rank = 0; rank < ec.device_count; ++rank) {
        require(shape_ok(values[rank], DType::FP32, 1, columns) &&
                    shape_ok(indices[rank], DType::I32, 1, columns) &&
                    shape_ok(output[rank], DType::I32, columns, 1),
                "allreduce_argmax: invalid tensors");
    }
    if (events.direct_peer_access()) {
        const CurrentDeviceGuard guard;
        for (int rank = 0; rank < ec.device_count; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), ec.dev[rank]->stream));
        }
        CurrentDeviceGuard::set(ec.dev[0]->device);
        for (int rank = 1; rank < ec.device_count; ++rank) {
            CUDA_CHECK(cudaStreamWaitEvent(ec.dev[0]->stream, events.inputs_ready(rank), 0));
        }
        if (ec.device_count == 2) {
            detail::allreduce_argmax_tp2_launch(values, indices, output, shard_rows, id_map, ec.dev[0]->stream);
        } else {
            detail::allreduce_argmax_tp4_launch(values, indices, output, shard_rows, id_map, ec.dev[0]->stream);
        }
        CUDA_CHECK(cudaEventRecord(events.pull_done(0), ec.dev[0]->stream));
        for (int rank = 1; rank < ec.device_count; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            CUDA_CHECK(cudaStreamWaitEvent(ec.dev[rank]->stream, events.pull_done(0), 0));
        }
        return;
    }
    gather_columns_rank0(gathered_values, values, ec, events);
    gather_columns_rank0(gathered_indices, indices, ec, events);
    Tensor result = output[0];
    {
        const CurrentDeviceGuard guard;
        CurrentDeviceGuard::set(ec.dev[0]->device);
        merge_argmax_shards(gathered_values, gathered_indices, result, shard_rows, ec.dev[0]->stream);
        if (id_map != nullptr) {
            proposal_remap_token_ids(result, id_map, ec.device_count * shard_rows, ec.dev[0]->stream);
        }
    }
    broadcast_rank0(result, output.subspan(1), ec, events);
}

void broadcast_rank0(const Tensor& source, std::span<const Tensor> destinations,
                     const ExecutionContext& ec, const PeerEvents& events) {
    require_collective_devices(ec, "broadcast_rank0: requires two or four devices");
    require(events.live() && events.ranks() == ec.device_count, "broadcast_rank0: events must match ranks");
    require(destinations.size() == static_cast<std::size_t>(ec.device_count - 1),
            "broadcast_rank0: destinations must name every non-origin rank");
    for (int rank = 1; rank < ec.device_count; ++rank) {
        const auto& destination = destinations[static_cast<std::size_t>(rank - 1)];
        require(source.data != nullptr && destination.data != nullptr &&
                    source.dtype == destination.dtype && source.is_contiguous() &&
                    destination.is_contiguous(), "broadcast_rank0: invalid tensors");
        for (int d = 0; d < 4; ++d) {
            require(source.ne[d] == destination.ne[d], "broadcast_rank0: shapes must match");
        }
#ifndef NDEBUG
        require_resident_on(source.data, ec.dev[0]->device, "broadcast_rank0: source is not on rank 0");
        require_resident_on(destination.data, ec.dev[rank]->device,
                            "broadcast_rank0: destination is not on its owning rank");
#endif
    }
    const std::size_t bytes = source.bytes();
    if (bytes == 0) { return; }
    const CurrentDeviceGuard guard;
    const DeviceContext& origin = *ec.dev[0];
    if (ec.device_count == 4 && events.direct_peer_access() &&
        source.dtype == DType::I32 && bytes <= 4096 &&
        ec.dev[0]->sm() == 70 && ec.dev[1]->sm() == 70 &&
        ec.dev[2]->sm() == 70 && ec.dev[3]->sm() == 70) {
        // The origin writes the destinations directly. Join their preceding consumers
        // before the write, then join publication before any rank reuses its storage.
        for (int rank = 1; rank < 4; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), ec.dev[rank]->stream));
        }
        CurrentDeviceGuard::set(origin.device);
        for (int rank = 1; rank < 4; ++rank) {
            CUDA_CHECK(cudaStreamWaitEvent(origin.stream, events.inputs_ready(rank), 0));
        }
        detail::broadcast_i32_tp4_launch(source, destinations, origin.stream);
        CUDA_CHECK(cudaEventRecord(events.pull_done(0), origin.stream));
        for (int rank = 1; rank < 4; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            CUDA_CHECK(cudaStreamWaitEvent(ec.dev[rank]->stream, events.pull_done(0), 0));
        }
        return;
    }
    CurrentDeviceGuard::set(origin.device);
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), origin.stream));
    for (int rank = 1; rank < ec.device_count; ++rank) {
        const auto& peer = *ec.dev[rank];
        CurrentDeviceGuard::set(peer.device);
        CUDA_CHECK(cudaStreamWaitEvent(peer.stream, events.inputs_ready(0), 0));
        CUDA_CHECK(pull_peer(destinations[static_cast<std::size_t>(rank - 1)].data,
                             source.data, bytes, peer.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), peer.stream));
    }
    CurrentDeviceGuard::set(origin.device);
    for (int rank = 1; rank < ec.device_count; ++rank) {
        CUDA_CHECK(cudaStreamWaitEvent(origin.stream, events.pull_done(rank), 0));
    }
}

} // namespace ninfer::ops
