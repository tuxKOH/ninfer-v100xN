#include "artifact/materializer.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <span>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSlotCount = 4;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw ArtifactError(label); }
    return a + b;
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value / alignment * alignment;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* label) {
    return checked_add(value, alignment - 1, label) / alignment * alignment;
}

class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }
    ~CurrentDeviceGuard() { (void)cudaSetDevice(previous_); }
private:
    int previous_ = 0;
};

class LoadStreamGuard {
public:
    explicit LoadStreamGuard(std::span<DeviceContext* const> devices) : devices_(devices) {}
    ~LoadStreamGuard() {
        for (DeviceContext* device : devices_) {
            (void)cudaSetDevice(device->device);
            (void)cudaStreamSynchronize(device->load_stream);
        }
    }
private:
    std::span<DeviceContext* const> devices_;
};

// One staging buffer plus a completion event per destination device: a slot may only be refilled
// once every device has finished reading it.
class Slot {
public:
    Slot(std::size_t bytes, std::span<DeviceContext* const> devices) : buffer(bytes) {
        try {
            for (std::size_t i = 0; i < devices.size(); ++i) {
                devices_[i] = devices[i];
                CUDA_CHECK(cudaSetDevice(devices[i]->device));
                CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Slot() { cleanup(); }

    void cleanup() noexcept {
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (devices_[i] == nullptr) { continue; }
            (void)cudaSetDevice(devices_[i]->device);
            // A copy can be queued before its completion event is recorded.
            (void)cudaStreamSynchronize(devices_[i]->load_stream);
            if (events[i] != nullptr) { (void)cudaEventDestroy(events[i]); }
            events[i] = nullptr;
        }
    }

    void wait() {
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (pending[i]) {
                CUDA_CHECK(cudaEventSynchronize(events[i]));
                pending[i] = false;
            }
        }
    }

    PinnedHostBuffer buffer;
    std::array<cudaEvent_t, kMaximumDevices> events{};
    std::array<bool, kMaximumDevices> pending{};
    std::array<DeviceContext*, kMaximumDevices> devices_{};
};

struct CopyRange {
    std::uint64_t source_begin = 0;
    std::uint64_t source_end   = 0;
    std::byte* destination     = nullptr;
    int device                 = 0;
};

struct ReadSpan {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

} // namespace

struct MaterializedArtifact::VirtualMapping {
    CUdeviceptr base = 0;
    std::size_t virtual_bytes = 0;
    struct Segment {
        CUmemGenericAllocationHandle handle = 0;
        std::size_t virtual_offset = 0;
        std::size_t bytes = 0;
    };
    std::vector<Segment> segments;

    ~VirtualMapping() {
        int previous = 0;
        (void)cudaGetDevice(&previous);
        // The runtime primary context remains current for the owning engine; VMM teardown does not
        // require a device-side synchronization because materialization waits on both load streams
        // before the mapping can be destroyed.
        for (const Segment& segment : segments) {
            if (base != 0 && segment.bytes != 0) {
                (void)cuMemUnmap(base + segment.virtual_offset, segment.bytes);
            }
            if (segment.handle != 0) { (void)cuMemRelease(segment.handle); }
        }
        if (base != 0) { (void)cuMemAddressFree(base, virtual_bytes); }
        (void)cudaSetDevice(previous);
    }
};

MaterializedArtifact::MaterializedArtifact() = default;
MaterializedArtifact::~MaterializedArtifact() {
    int previous = 0;
    const bool restore = cudaGetDevice(&previous) == cudaSuccess;
    virtual_mappings_.clear();
    for (std::size_t i = 0; i < device_arena_.size(); ++i) {
        if (device_arena_[i]) {
            (void)cudaSetDevice(physical_devices_[i]);
            device_arena_[i].reset();
        }
    }
    if (restore) { (void)cudaSetDevice(previous); }
}
MaterializedArtifact::MaterializedArtifact(MaterializedArtifact&&) noexcept = default;
MaterializedArtifact& MaterializedArtifact::operator=(MaterializedArtifact&& other) noexcept {
    if (this != &other) {
        this->~MaterializedArtifact();
        new (this) MaterializedArtifact(std::move(other));
    }
    return *this;
}

std::string driver_error(CUresult result) {
    const char* name = nullptr;
    const char* message = nullptr;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &message);
    return std::string(name != nullptr ? name : "CUDA_ERROR_UNKNOWN") + ": " +
           (message != nullptr ? message : "unknown driver error");
}

void driver_check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        throw ArtifactError(std::string(operation) + " failed: " + driver_error(result));
    }
}

std::unique_ptr<MaterializedArtifact::VirtualMapping>
create_virtual_mapping(const VirtualDeviceMaterialization& placement,
                        std::span<DeviceContext* const> devices) {
    if (devices.size() != kVirtualStorageDevices || placement.virtual_bytes == 0 ||
        placement.segments.empty()) {
        throw ArtifactError("virtual materialization requires two nonempty device slices");
    }
    driver_check(cuInit(0), "cuInit");
    std::size_t granularity = 0;
    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = devices[0]->device;
    driver_check(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                 "cuMemGetAllocationGranularity");
    auto mapping = std::make_unique<MaterializedArtifact::VirtualMapping>();
    if (placement.virtual_bytes > std::numeric_limits<std::size_t>::max() - (granularity - 1)) {
        throw ArtifactError("virtual materialization address size overflows size_t");
    }
    mapping->virtual_bytes = static_cast<std::size_t>(
        (placement.virtual_bytes + granularity - 1) / granularity * granularity);
    try {
        driver_check(cuMemAddressReserve(&mapping->base, mapping->virtual_bytes, granularity, 0, 0),
                     "cuMemAddressReserve");
        for (const VirtualDeviceSegment& segment : placement.segments) {
            if (segment.device < 0 || segment.device >= static_cast<int>(kVirtualStorageDevices) ||
                segment.bytes == 0 || segment.virtual_offset % granularity != 0 ||
                segment.bytes % granularity != 0 ||
                segment.virtual_offset > mapping->virtual_bytes ||
                segment.bytes > mapping->virtual_bytes - segment.virtual_offset) {
                throw ArtifactError("virtual materialization segment is not VMM-aligned");
            }
            for (const auto& mapped : mapping->segments) {
                const auto begin = static_cast<std::uint64_t>(segment.virtual_offset);
                const auto end = begin + segment.bytes;
                const auto mapped_begin = static_cast<std::uint64_t>(mapped.virtual_offset);
                const auto mapped_end = mapped_begin + mapped.bytes;
                if (begin < mapped_end && mapped_begin < end) {
                    throw ArtifactError("virtual materialization segments overlap");
                }
            }
            prop.location.id = devices[static_cast<std::size_t>(segment.device)]->device;
            MaterializedArtifact::VirtualMapping::Segment mapped;
            mapped.virtual_offset = static_cast<std::size_t>(segment.virtual_offset);
            mapped.bytes          = static_cast<std::size_t>(segment.bytes);
            driver_check(cuMemCreate(&mapped.handle, mapped.bytes, &prop, 0), "cuMemCreate");
            mapping->segments.push_back(mapped);
            driver_check(cuMemMap(mapping->base + mapped.virtual_offset, mapped.bytes, 0,
                                  mapped.handle, 0),
                         "cuMemMap(segment)");
        }
        std::array<CUmemAccessDesc, kVirtualStorageDevices> access{};
        for (int device = 0; device < static_cast<int>(kVirtualStorageDevices); ++device) {
            access[static_cast<std::size_t>(device)].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access[static_cast<std::size_t>(device)].location.id =
                devices[static_cast<std::size_t>(device)]->device;
            access[static_cast<std::size_t>(device)].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            driver_check(cuMemSetAccess(mapping->base, mapping->virtual_bytes,
                                        &access[static_cast<std::size_t>(device)], 1),
                         "cuMemSetAccess");
        }
    } catch (...) {
        throw;
    }
    return mapping;
}

void* MaterializedArtifact::device_data(ObjectHandle handle) const { return device_data(handle, 0); }

void* MaterializedArtifact::device_data(ObjectHandle handle, int device) const {
    if (device < 0 || device >= stats_.device_count) {
        throw ArtifactError("materialized artifact does not cover that device");
    }
    if (handle.index >= objects_.size() ||
        objects_[handle.index].device[static_cast<std::size_t>(device)] == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device[static_cast<std::size_t>(device)];
}

std::uint64_t MaterializedArtifact::device_bytes(ObjectHandle handle, int device) const {
    if (device < 0 || device >= stats_.device_count) {
        throw ArtifactError("materialized artifact does not cover that device");
    }
    if (handle.index >= objects_.size() ||
        objects_[handle.index].device[static_cast<std::size_t>(device)] == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device_bytes[static_cast<std::size_t>(device)];
}

std::span<const std::byte> MaterializedArtifact::host_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized host object");
    }
    return objects_[handle.index].resource;
}

std::vector<std::byte> MaterializedArtifact::take_host_bytes(ObjectHandle handle) {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized host object");
    }
    auto& resource = objects_[handle.index].resource;
    stats_.retained_host_bytes -= resource.size();
    return std::move(resource);
}

DeviceArena& MaterializedArtifact::device_arena() { return device_arena(0); }

DeviceArena& MaterializedArtifact::device_arena(int device) {
    if (device < 0 || device >= stats_.device_count ||
        !device_arena_[static_cast<std::size_t>(device)]) {
        throw ArtifactError("artifact has no device tensor backing");
    }
    return *device_arena_[static_cast<std::size_t>(device)];
}

int MaterializedArtifact::physical_device(int device) const {
    if (device < 0 || device >= stats_.device_count) {
        throw ArtifactError("materialized artifact does not cover that device");
    }
    return physical_devices_[static_cast<std::size_t>(device)];
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 std::span<DeviceContext* const> devices, LoadProgress* progress) {
    const int device_count = plan.device_count;
    if (device_count < 1 || device_count > static_cast<int>(kMaximumDevices) ||
        devices.size() != static_cast<std::size_t>(device_count) ||
        plan.execution_storage_device_count < 1 ||
        plan.execution_storage_device_count > device_count ||
        (plan.auxiliary_device
             ? (*plan.auxiliary_device != plan.execution_storage_device_count ||
                device_count != plan.execution_storage_device_count + 1)
             : device_count != plan.execution_storage_device_count)) {
        throw ArtifactError("materialization plan and device slots disagree");
    }
    for (std::size_t i = 0; i < devices.size(); ++i) {
        if (devices[i] == nullptr) { throw ArtifactError("null materialization device"); }
        for (std::size_t j = 0; j < i; ++j) {
            if (devices[i]->device == devices[j]->device) {
                throw ArtifactError("materialization devices must be distinct");
            }
        }
    }
    CurrentDeviceGuard current_device;
    MaterializedArtifact out;
    LoadStreamGuard load_streams(devices);
    out.objects_.resize(plan.object_count);
    out.stats_.device_count = device_count;
    for (int index = 0; index < device_count; ++index) {
        const auto slot              = static_cast<std::size_t>(index);
        out.physical_devices_[slot]  = devices[slot]->device;
        const std::uint64_t capacity = plan.device_arena_capacity_bytes[slot];
        if (capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
            throw ArtifactError("artifact tensor backing size is invalid");
        }
        CUDA_CHECK(cudaSetDevice(devices[slot]->device));
        if (capacity != 0) {
            out.device_arena_[slot] =
                std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
        }
        if (device_count > 1 && out.device_arena_[slot] != nullptr) {
            // A shard's plane offsets are recomputed for its own row/column count, so the
            // alignment gaps between its planes are not covered by any copy. The layouts define
            // those gap bytes as zero; make them so rather than leaving allocator residue.
            //
            // This MUST be stream-ordered on the same stream the copies use. load_stream is
            // created with cudaStreamNonBlocking, so it does not synchronize with the legacy
            // default stream that synchronous-API cudaMemset would run on -- a multi-GiB fill
            // could still be in flight when the first weight copies land and would silently zero
            // them. cudaMemsetAsync on load_stream orders the fill strictly before every copy.
            CUDA_CHECK(cudaMemsetAsync(out.device_arena_[slot]->base(), 0,
                                       static_cast<std::size_t>(capacity),
                                       devices[slot]->load_stream));
        }
        out.stats_.per_device_capacity_bytes[slot] = plan.device_capacity_bytes[slot];
        out.stats_.device_capacity_bytes =
            checked_add(out.stats_.device_capacity_bytes, plan.device_capacity_bytes[slot],
                        "device capacity overflows u64");
    }
    std::vector<bool> counted(plan.object_count, false);
    for (const auto& placement : plan.device_objects) { counted.at(placement.object.index) = true; }
    for (const auto& placement : plan.virtual_objects) { counted.at(placement.object.index) = true; }
    out.stats_.tensor_count = static_cast<std::size_t>(std::count(counted.begin(), counted.end(), true));
    out.stats_.resource_count = 0;

    for (const HostMaterialization& placement : plan.host_objects) {
        if (std::holds_alternative<ResourceDescriptor>(reader.objects().at(placement.object.index))) {
            ++out.stats_.resource_count;
        } else if (!counted.at(placement.object.index)) { ++out.stats_.tensor_count; }
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_host_bytes += resource.size();
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, resource.size(), "artifact read bytes overflow u64");
    }

    std::vector<CopyRange> ranges;
    std::uint64_t copied         = 0;
    std::uint64_t last_published = 0;
    std::uint64_t total          = 0;

    // Virtual expert banks are mapped before any H2D ranges are built. The primary device sees
    // one full-shape pointer with the original plane offsets; each physical plane slice is fed by
    // its own device load stream, so the normal direct-I/O pipeline below remains unchanged.
    if (!plan.virtual_objects.empty()) {
        if (plan.execution_storage_device_count != static_cast<int>(kVirtualStorageDevices)) {
            throw ArtifactError("virtual materialization requires two execution/storage slots");
        }
        CUDA_CHECK(cudaSetDevice(devices[0]->device));
    }
    for (const VirtualDeviceMaterialization& placement : plan.virtual_objects) {
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        auto mapping = create_virtual_mapping(placement, devices.first(kVirtualStorageDevices));
        out.virtual_mappings_.push_back(std::move(mapping));
        const CUdeviceptr base = out.virtual_mappings_.back()->base;
        auto& object = out.objects_.at(placement.object.index);
        object.device[0]       = reinterpret_cast<void*>(base);
        object.device_bytes[0] = placement.virtual_bytes;
        object.device[1]       = reinterpret_cast<void*>(base);
        object.device_bytes[1] = placement.virtual_bytes;
        for (const VirtualDeviceSegment& segment : placement.segments) {
            if (segment.device < 0 || segment.device >= static_cast<int>(kVirtualStorageDevices) ||
                segment.bytes == 0 || segment.virtual_offset > placement.virtual_bytes ||
                segment.bytes > placement.virtual_bytes - segment.virtual_offset) {
                throw ArtifactError("virtual materialization segment exceeds its payload");
            }
            const auto slot = static_cast<std::size_t>(segment.device);
            auto* const destination_base =
                reinterpret_cast<std::byte*>(base + segment.virtual_offset);
            if (!segment.prefix.empty()) {
                if (segment.prefix.size() > segment.bytes) {
                    throw ArtifactError("virtual materialization prefix exceeds its allocation");
                }
                CUDA_CHECK(cudaSetDevice(devices[slot]->device));
                CUDA_CHECK(cudaMemcpyAsync(destination_base, segment.prefix.data(),
                                            segment.prefix.size(), cudaMemcpyHostToDevice,
                                            devices[slot]->load_stream));
            }
            for (const PlaneCopy& copy : segment.copies) {
                if (copy.source_offset > payload.data.size() ||
                    payload.data.size() - copy.source_offset < copy.bytes ||
                    copy.dest_offset > segment.bytes ||
                    segment.bytes - copy.dest_offset < copy.bytes) {
                    throw ArtifactError("virtual materialization range exceeds its payload");
                }
                ranges.push_back(CopyRange{
                    .source_begin = checked_add(payload.absolute_offset, copy.source_offset,
                                                "artifact tensor source range overflows u64"),
                    .source_end = checked_add(payload.absolute_offset + copy.source_offset,
                                              copy.bytes,
                                              "artifact tensor source range overflows u64"),
                    .destination = destination_base + copy.dest_offset,
                    .device = segment.device,
                });
                total = checked_add(total, copy.bytes, "artifact tensor byte count overflows u64");
            }
        }
    }

    std::size_t range_count = ranges.size();
    for (const DeviceMaterialization& placement : plan.device_objects) {
        range_count += placement.copies.empty() ? 1 : placement.copies.size();
    }
    ranges.reserve(range_count);
    for (const DeviceMaterialization& placement : plan.device_objects) {
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        const auto slot           = static_cast<std::size_t>(placement.device);
        if (placement.device < 0 || placement.device >= device_count) {
            throw ArtifactError("materialization placement names a device outside the plan");
        }
        if (!out.device_arena_[slot]) {
            throw ArtifactError("materialization placement has no device arena");
        }
        DeviceArena& arena = *out.device_arena_[slot];
        DeviceSpan storage = arena.alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                               static_cast<std::size_t>(placement.alignment));
        const auto actual_offset = static_cast<std::uint64_t>(
            static_cast<std::byte*>(storage.data) - static_cast<std::byte*>(arena.base()));
        if (actual_offset != placement.offset) {
            throw ArtifactError("materialization plan does not match artifact payload");
        }
        out.objects_.at(placement.object.index).device[slot]       = storage.data;
        out.objects_.at(placement.object.index).device_bytes[slot] = placement.bytes;
        auto* const base                                     = static_cast<std::byte*>(storage.data);
        if (!placement.prefix.empty()) {
            if (placement.prefix.size() > placement.bytes) {
                throw ArtifactError("materialization prefix exceeds its allocation");
            }
            CUDA_CHECK(cudaSetDevice(devices[slot]->device));
            CUDA_CHECK(cudaMemcpyAsync(base, placement.prefix.data(), placement.prefix.size(),
                                        cudaMemcpyHostToDevice, devices[slot]->load_stream));
        }
        const auto add_range = [&](std::uint64_t source_offset, std::uint64_t dest_offset,
                                   std::uint64_t bytes) {
            if (source_offset > payload.data.size() ||
                payload.data.size() - source_offset < bytes || dest_offset > placement.bytes ||
                placement.bytes - dest_offset < bytes) {
                throw ArtifactError("materialization plan does not match artifact payload");
            }
            ranges.push_back(CopyRange{
                .source_begin = checked_add(payload.absolute_offset, source_offset,
                                            "artifact tensor source range overflows u64"),
                .source_end   = checked_add(payload.absolute_offset + source_offset, bytes,
                                            "artifact tensor source range overflows u64"),
                .destination  = base + dest_offset,
                .device       = placement.device,
            });
            total = checked_add(total, bytes, "artifact tensor byte count overflows u64");
        };
        if (placement.copies.empty()) {
            if (payload.data.size() != placement.bytes) {
                throw ArtifactError("materialization plan does not match artifact payload");
            }
            add_range(0, 0, placement.bytes);
        } else {
            for (const PlaneCopy& copy : placement.copies) {
                add_range(copy.source_offset, copy.dest_offset, copy.bytes);
            }
        }
    }
    if (ranges.empty()) { throw ArtifactError("materialization plan has no device tensors"); }
    // Destination ranges must stay disjoint *within* a device. The single-device path used to get
    // this from the source-range disjointness check below, which sharding had to drop (a
    // replicated object legitimately reads the same source twice). Check the destinations
    // directly instead: two copies landing on the same arena byte would make the result depend on
    // staging-chunk order.
    {
        std::vector<std::pair<const std::byte*, std::uint64_t>> written;
        written.reserve(ranges.size());
        for (int index = 0; index < device_count; ++index) {
            written.clear();
            for (const CopyRange& range : ranges) {
                if (range.device == index) {
                    written.emplace_back(range.destination, range.source_end - range.source_begin);
                }
            }
            std::sort(written.begin(), written.end());
            for (std::size_t i = 1; i < written.size(); ++i) {
                if (written[i].first < written[i - 1].first + written[i - 1].second) {
                    throw ArtifactError("materialization destination ranges overlap on a device");
                }
            }
        }
    }
    // Source ranges may overlap: a replicated object feeds the same source bytes to both devices, and
    // a shard reads interleaved sub-ranges. The copy loop below intersects every still-live range
    // with each staging chunk independently, so overlap (and even containment) is safe.
    std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
        return a.source_begin < b.source_begin;
    });

    constexpr std::uint64_t alignment = Reader::direct_io_alignment;
    std::vector<ReadSpan> read_spans;
    read_spans.reserve(ranges.size());
    std::uint64_t aligned_read_bytes = 0;
    for (const CopyRange& range : ranges) {
        const std::uint64_t begin = align_down(range.source_begin, alignment);
        if (read_spans.empty() || begin > align_up(read_spans.back().end, alignment,
                                                   "artifact direct I/O span overflows u64")) {
            read_spans.push_back(ReadSpan{begin, range.source_end});
        } else {
            read_spans.back().end = std::max(read_spans.back().end, range.source_end);
        }
    }
    for (const ReadSpan& span : read_spans) {
        aligned_read_bytes = checked_add(
            aligned_read_bytes,
            align_up(span.end - span.begin, alignment, "artifact direct I/O span overflows u64"),
            "artifact direct I/O byte count overflows u64");
    }
    const std::size_t slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_read_bytes));
    const std::size_t slot_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_read_bytes - 1) / slot_bytes));
    std::vector<std::unique_ptr<Slot>> slots;
    slots.reserve(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes, devices.first(
                                                               static_cast<std::size_t>(device_count))));
    }
    out.stats_.peak_staging_bytes = static_cast<std::uint64_t>(slot_bytes) * slot_count;

    std::size_t next_slot        = 0;
    std::size_t first_unfinished = 0;
    const auto start             = std::chrono::steady_clock::now();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }
    for (const ReadSpan& span : read_spans) {
        for (std::uint64_t source = span.begin; source < span.end; source += slot_bytes) {
            Slot& slot = *slots[next_slot++ % slot_count];
            slot.wait();

            const std::uint64_t remaining = span.end - source;
            const std::size_t request     = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes,
                align_up(remaining, alignment, "artifact direct I/O request overflows u64")));
            auto destination =
                std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), request);
            const std::size_t bytes_read = reader.read_direct(source, destination);
            const std::uint64_t required = std::min<std::uint64_t>(request, remaining);
            if (bytes_read < required) {
                throw ArtifactError("direct artifact read ended before the planned tensor range");
            }
            out.stats_.file_bytes =
                checked_add(out.stats_.file_bytes, bytes_read, "artifact read bytes overflow u64");
            const std::uint64_t chunk_end =
                checked_add(source, bytes_read, "artifact direct I/O result overflows u64");

            // Retire the leading run of ranges this chunk has already left behind. Ranges that
            // still straddle `source` stay in the scan window; they are cheap to re-test.
            while (first_unfinished < ranges.size() &&
                   ranges[first_unfinished].source_end <= source) {
                ++first_unfinished;
            }
            // One pass per destination device: a stream belongs to its device, so the copies are
            // grouped rather than interleaved, and only devices this chunk actually fed record an
            // event on it.
            for (int index = 0; index < device_count; ++index) {
                const auto device_slot = static_cast<std::size_t>(index);
                if (device_count > 1) { CUDA_CHECK(cudaSetDevice(devices[device_slot]->device)); }
                bool fed = false;
                for (std::size_t range_index = first_unfinished;
                     range_index < ranges.size() &&
                     ranges[range_index].source_begin < chunk_end;
                     ++range_index) {
                    const CopyRange& range = ranges[range_index];
                    if (range.device != index) { continue; }
                    const std::uint64_t copy_begin = std::max(source, range.source_begin);
                    const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
                    if (copy_begin >= copy_end) { continue; }
                    const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                    CUDA_CHECK(cudaMemcpyAsync(
                        range.destination +
                            static_cast<std::size_t>(copy_begin - range.source_begin),
                        static_cast<std::byte*>(slot.buffer.data()) +
                            static_cast<std::size_t>(copy_begin - source),
                        amount, cudaMemcpyHostToDevice, devices[device_slot]->load_stream));
                    copied =
                        checked_add(copied, amount, "artifact copied byte count overflows u64");
                    out.stats_.per_device_h2d_bytes[device_slot] += amount;
                    fed = true;
                }
                if (fed) {
                    CUDA_CHECK(cudaEventRecord(slot.events[device_slot],
                                               devices[device_slot]->load_stream));
                    slot.pending[device_slot] = true;
                }
            }

            if (progress != nullptr && progress->callback && copied != last_published &&
                copied < total) {
                last_published = copied;
                progress->callback("weights", copied, total);
            }
        }
    }
    for (const auto& slot : slots) { slot->wait(); }
    for (int index = 0; index < device_count; ++index) {
        CUDA_CHECK(cudaStreamSynchronize(devices[static_cast<std::size_t>(index)]->load_stream));
    }
    if (copied != total) {
        throw ArtifactError("direct materialization did not cover every tensor byte");
    }
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (progress != nullptr && progress->callback) { progress->callback("weights", copied, total); }
    return out;
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 ExecutionContext& execution, LoadProgress* progress) {
    std::array<DeviceContext*, kMaximumDevices> devices{};
    std::size_t count = 0;
    for (int rank = 0; rank < execution.device_count; ++rank) {
        if (count >= kMaximumDevices) {
            throw ArtifactError("artifact materialization supports at most two execution devices");
        }
        devices[count++] = &*execution.dev[static_cast<std::size_t>(rank)];
    }
    if (execution.has_storage()) {
        if (count >= kMaximumDevices) {
            throw ArtifactError("artifact materialization supports at most two devices");
        }
        devices[count++] = &*execution.storage;
    }
    return materialize(reader, plan, std::span<DeviceContext* const>(devices.data(), count),
                       progress);
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress) {
    if (plan.device_count != 1) {
        throw ArtifactError("a multi-device materialization plan needs an ExecutionContext");
    }
    DeviceContext* one = &device;
    return materialize(reader, plan, std::span<DeviceContext* const>(&one, 1), progress);
}

} // namespace ninfer::artifact
