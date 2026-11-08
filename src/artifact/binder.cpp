#include "artifact/binder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace ninfer::artifact {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    const std::uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    return (value + mask) & ~mask;
}

} // namespace

Binder::Binder(const Reader& reader, int device_count, bool storage_only,
               std::optional<int> auxiliary_device)
    : reader_(reader), consumed_(reader.objects().size(), false),
      planned_(reader.objects().size(), false), storage_only_(storage_only) {
    if (device_count < 1 || device_count > static_cast<int>(kMaximumDevices) ||
        (auxiliary_device && (*auxiliary_device != device_count ||
                              device_count == static_cast<int>(kMaximumDevices)))) {
        throw ArtifactError("materialization requires 1 to 4 slots with auxiliary appended");
    }
    materialization_.object_count = reader.objects().size();
    materialization_.execution_storage_device_count = device_count;
    materialization_.auxiliary_device = auxiliary_device;
    materialization_.device_count = device_count + (auxiliary_device ? 1 : 0);
}

bool Binder::has_tensor(std::string_view name) const noexcept { return reader_.find(name) != nullptr; }

void Binder::set_shard_resolver(ShardResolver resolver) {
    if (!materialization_.device_objects.empty() || device_mask_resolver_) {
        throw ArtifactError("the shard resolver must be installed before any tensor is placed");
    }
    shard_resolver_ = std::move(resolver);
}

void Binder::set_device_mask_resolver(DeviceMaskResolver resolver) {
    if (!materialization_.device_objects.empty() || shard_resolver_ || storage_only_) {
        throw ArtifactError("device placement must be selected before materialization");
    }
    device_mask_resolver_ = std::move(resolver);
}

ObjectHandle Binder::find_unconsumed(std::string_view name) {
    const auto& objects            = reader_.objects();
    const ObjectDescriptor* object = reader_.find(name);
    if (object == nullptr) {
        throw ArtifactError("required artifact object is missing: " + std::string(name));
    }
    const auto index = static_cast<std::size_t>(object - objects.data());
    if (consumed_[index]) {
        throw ArtifactError("artifact object was bound more than once: " + std::string(name));
    }
    consumed_[index] = true;
    return ObjectHandle{index};
}

ObjectHandle Binder::require_tensor(std::string_view name, NumericFormat format,
                                    StorageLayout layout, std::span<const std::uint64_t> shape) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* tensor        = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("required tensor is a resource: " + std::string(name));
    }
    if (tensor->format != format || tensor->layout != layout ||
        !std::equal(tensor->shape.begin(), tensor->shape.end(), shape.begin(), shape.end())) {
        throw ArtifactError("tensor descriptor does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

ObjectHandle Binder::require_resource(std::string_view name, ResourceEncoding encoding) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* resource      = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("required resource is a tensor: " + std::string(name));
    }
    if (resource->encoding != encoding) {
        throw ArtifactError("resource encoding does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

const ObjectDescriptor& Binder::descriptor(ObjectHandle handle) const {
    if (handle.index >= reader_.objects().size()) {
        throw ArtifactError("artifact object handle is out of range");
    }
    return reader_.objects()[handle.index];
}

PayloadSpan Binder::payload(ObjectHandle handle) const {
    return reader_.payload(descriptor(handle));
}

void Binder::place(ObjectHandle handle, int device, std::uint64_t bytes, std::uint64_t alignment,
                   std::vector<PlaneCopy> copies) {
    std::uint64_t& capacity =
        materialization_.device_arena_capacity_bytes[static_cast<std::size_t>(device)];
    const std::uint64_t offset = align_up(capacity, alignment);
    if (bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    materialization_.device_objects.push_back(
        DeviceMaterialization{handle, device, offset, bytes, alignment, std::move(copies)});
    capacity = offset + bytes;
}

void Binder::materialize_on_device(ObjectHandle handle) {
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("resource cannot be materialized as a device tensor");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(tensor->name));
    }
    const std::uint64_t alignment = tensor_alignment(tensor->layout);
    if (device_mask_resolver_) {
        const std::uint32_t mask = device_mask_resolver_(tensor->name);
        if (mask == 0 || (mask >> materialization_.execution_storage_device_count) != 0) {
            throw ArtifactError("invalid device placement mask: " + std::string(tensor->name));
        }
        for (int device = 0; device < materialization_.execution_storage_device_count; ++device) {
            if ((mask & (1U << device)) != 0) { place_on_device(handle, device); }
        }
        planned_[handle.index] = true;
        return;
    }
    const ShardPlacement placement =
        shard_resolver_ ? shard_resolver_(tensor->name) : ShardPlacement{};

    if (storage_only_ && placement.axis == ShardAxis::Replicated) {
        place_on_device(handle, 0);
        planned_[handle.index] = true;
        return;
    }

    for (int device = 0; device < materialization_.execution_storage_device_count; ++device) {
        const std::vector<SliceRange>& ranges =
            placement.device_ranges[static_cast<std::size_t>(device)];
        if (placement.axis == ShardAxis::Replicated) {
            if (!ranges.empty()) {
                throw ArtifactError("a replicated placement must carry no shard ranges: " +
                                    std::string(tensor->name));
            }
            // Whole object. `copies` stays empty: the materializer copies the payload verbatim,
            // which is byte-for-byte what the single-device path has always done.
            place(handle, device, tensor->bytes, alignment, {});
            continue;
        }
        if (ranges.empty()) {
            throw ArtifactError("sharded object names no range for device " +
                                std::to_string(device) + ": " + std::string(tensor->name));
        }
        TensorSlice slice;
        if (placement.axis == ShardAxis::Rows) {
            slice = tensor_row_slice(tensor->layout, tensor->format, tensor->shape, ranges, payload(handle).data);
        } else {
            // Multiple column ranges are legal only where the layout allows them
            // (contiguous-le-v1 and whole-block ggml-k256-v1);
            // tensor_column_slice enforces that per layout rather than this call site guessing.
            slice = tensor_column_slice(tensor->layout, tensor->format, tensor->shape, ranges, payload(handle).data);
        }
        std::uint64_t covered = 0;
        for (const PlaneCopy& copy : slice.copies) {
            if (copy.bytes == 0 || copy.source_offset > tensor->bytes ||
                tensor->bytes - copy.source_offset < copy.bytes ||
                copy.dest_offset > slice.encoded_bytes ||
                slice.encoded_bytes - copy.dest_offset < copy.bytes) {
                throw ArtifactError("shard slice range is outside its tensor: " +
                                    std::string(tensor->name));
            }
            covered += copy.bytes;
        }
        if (covered > slice.encoded_bytes) {
            throw ArtifactError("shard slice copies overlap: " + std::string(tensor->name));
        }
        // Two copies writing the same shard byte would make the result depend on staging-chunk
        // order. The single-device path got this for free (one copy per object); reinstate it
        // explicitly now that a shard is many copies. Ordering by destination is cheap here --
        // every layout emits its copies plane by plane, so this is nearly sorted already.
        {
            std::vector<const PlaneCopy*> ordered;
            ordered.reserve(slice.copies.size());
            for (const PlaneCopy& copy : slice.copies) { ordered.push_back(&copy); }
            std::sort(ordered.begin(), ordered.end(),
                      [](const PlaneCopy* a, const PlaneCopy* b) {
                          return a->dest_offset < b->dest_offset;
                      });
            for (std::size_t i = 1; i < ordered.size(); ++i) {
                if (ordered[i]->dest_offset <
                    ordered[i - 1]->dest_offset + ordered[i - 1]->bytes) {
                    throw ArtifactError("shard slice writes the same byte twice: " +
                                        std::string(tensor->name));
                }
            }
        }
        place(handle, device, slice.encoded_bytes, alignment, std::move(slice.copies));
        materialization_.device_objects.back().prefix = std::move(slice.prefix);
    }
    planned_[handle.index] = true;
}

void Binder::place_on_device(ObjectHandle handle, int device) {
    if (device < 0 || device >= materialization_.device_count) {
        throw ArtifactError("artifact materialization device is outside the plan");
    }
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) { throw ArtifactError("resource cannot be materialized as a device tensor"); }
    place(handle, device, tensor->bytes, tensor_alignment(tensor->layout), {});
}

void Binder::materialize_on_device(ObjectHandle handle, int device) {
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(object_name(descriptor(handle))));
    }
    place_on_device(handle, device);
    planned_[handle.index] = true;
}

void Binder::materialize_virtual_rows(ObjectHandle handle,
                                       std::array<SliceRange, kVirtualStorageDevices> ranges) {
    if (materialization_.execution_storage_device_count != static_cast<int>(kVirtualStorageDevices)) {
        throw ArtifactError("virtual row placement requires exactly two materialization devices");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(object_name(descriptor(handle))));
    }
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) { throw ArtifactError("resource cannot be materialized as a device tensor"); }
    if (tensor->layout != StorageLayout::RowSplitK128V1 || tensor->shape.size() != 2) {
        throw ArtifactError("virtual row placement requires row-split rank-two tensors");
    }
    const auto payload_span = payload(handle).data;
    const auto parent = row_split_geometry(tensor->format, tensor->shape);
    VirtualDeviceMaterialization placement;
    placement.object       = handle;
    placement.virtual_bytes = tensor->bytes;
    for (int device = 0; device < static_cast<int>(kVirtualStorageDevices); ++device) {
        const auto range = ranges[static_cast<std::size_t>(device)];
        if (range.count == 0) {
            throw ArtifactError("virtual row placement names an empty device range");
        }
        if (range.begin > tensor->shape[0] || tensor->shape[0] - range.begin < range.count) {
            throw ArtifactError("virtual row placement range exceeds the tensor");
        }
        const std::array<std::uint64_t, 2> shard_shape = {range.count, tensor->shape[1]};
        const auto shard = row_split_geometry(tensor->format, shard_shape);
        const auto slice = tensor_row_slice(
            tensor->layout, tensor->format, tensor->shape,
            std::span<const SliceRange>(&ranges[static_cast<std::size_t>(device)], 1),
            payload_span);
        if (slice.encoded_bytes == 0) {
            throw ArtifactError("virtual row placement produced an empty slice");
        }

        struct Plane {
            std::uint64_t parent_offset;
            std::uint64_t parent_row_bytes;
            std::uint64_t shard_offset;
            std::uint64_t shard_bytes;
        };
        const std::array<Plane, 3> planes = {
            Plane{0, parent.low_bytes_per_group * parent.groups_per_row,
                  0, shard.low_plane_bytes},
            Plane{parent.high_plane_offset,
                  parent.high_bytes_per_group * parent.groups_per_row,
                  shard.high_plane_offset, shard.high_plane_bytes},
            Plane{parent.scale_plane_offset, 2 * parent.groups_per_row,
                  shard.scale_plane_offset, shard.scale_plane_bytes},
        };
        for (const Plane& plane : planes) {
            if (plane.shard_bytes == 0) { continue; }
            if (plane.parent_row_bytes == 0 ||
                range.count > std::numeric_limits<std::uint64_t>::max() /
                                   plane.parent_row_bytes) {
                throw ArtifactError("virtual row placement plane size overflows u64");
            }
            const auto bytes = range.count * plane.parent_row_bytes;
            if (bytes != plane.shard_bytes) {
                throw ArtifactError("virtual row placement plane geometry mismatch");
            }
            if (range.begin > std::numeric_limits<std::uint64_t>::max() /
                                  plane.parent_row_bytes) {
                throw ArtifactError("virtual row placement offset overflows u64");
            }
            const auto row_offset = range.begin * plane.parent_row_bytes;
            if (plane.parent_offset > std::numeric_limits<std::uint64_t>::max() - row_offset) {
                throw ArtifactError("virtual row placement offset overflows u64");
            }
            VirtualDeviceSegment segment;
            segment.device         = device;
            segment.virtual_offset = plane.parent_offset + row_offset;
            segment.bytes          = bytes;
            for (const PlaneCopy& copy : slice.copies) {
                if (copy.dest_offset < plane.shard_offset ||
                    copy.dest_offset - plane.shard_offset >= plane.shard_bytes) {
                    continue;
                }
                const auto local = copy.dest_offset - plane.shard_offset;
                if (copy.bytes > plane.shard_bytes - local) {
                    throw ArtifactError("virtual row placement copy exceeds its plane");
                }
                segment.copies.push_back(
                    PlaneCopy{copy.source_offset, local, copy.bytes});
            }
            if (segment.copies.empty()) {
                throw ArtifactError("virtual row placement plane has no source copies");
            }
            placement.segments.push_back(std::move(segment));
            auto& capacity = materialization_.virtual_capacity_bytes[static_cast<std::size_t>(device)];
            if (bytes > std::numeric_limits<std::uint64_t>::max() - capacity) {
                throw ArtifactError("virtual row placement capacity overflows u64");
            }
            capacity += bytes;
        }
    }
    materialization_.virtual_objects.push_back(std::move(placement));
    planned_[handle.index] = true;
}

void Binder::retain_on_host(ObjectHandle handle) {
    const auto* resource = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("tensor cannot be retained as a host resource");
    }
    materialize_on_host(handle);
}

void Binder::materialize_on_host(ObjectHandle handle) {
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::visit([](const auto& object) { return std::string(object.name); }, descriptor(handle)));
    }
    materialization_.host_objects.push_back(HostMaterialization{handle});
    planned_[handle.index] = true;
}

void Binder::validate_only(ObjectHandle handle) {
    const ObjectDescriptor& object = descriptor(handle);
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(object_name(object)));
    }
    consumed_[handle.index] = true;
    planned_[handle.index] = true;
}

void Binder::validate_unconsumed_matching(std::string_view prefix) {
    for (std::size_t i = 0; i < reader_.objects().size(); ++i) {
        if (!consumed_[i]) {
            std::string_view name = object_name(reader_.objects()[i]);
            if (prefix.empty() || name.starts_with(prefix)) {
                consumed_[i] = true;
                planned_[i]  = true;
            }
        }
    }
}

MaterializationPlan Binder::finish() {
    const auto it = std::find(consumed_.begin(), consumed_.end(), false);
    if (it != consumed_.end()) {
        const auto index = static_cast<std::size_t>(it - consumed_.begin());
        throw ArtifactError("artifact object was not consumed by the selected target: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    const auto unplanned = std::find(planned_.begin(), planned_.end(), false);
    if (unplanned != planned_.end()) {
        const auto index = static_cast<std::size_t>(unplanned - planned_.begin());
        throw ArtifactError("artifact object has no materialization placement: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    for (int device = 0; device < materialization_.device_count; ++device) {
        auto& total = materialization_.device_capacity_bytes[static_cast<std::size_t>(device)];
        const auto arena = materialization_.device_arena_capacity_bytes[static_cast<std::size_t>(device)];
        const auto virtual_bytes = materialization_.virtual_capacity_bytes[static_cast<std::size_t>(device)];
        if (virtual_bytes > std::numeric_limits<std::uint64_t>::max() - arena) {
            throw ArtifactError("materialization device capacity overflows u64");
        }
        total = arena + virtual_bytes;
    }
    return std::move(materialization_);
}

} // namespace ninfer::artifact
