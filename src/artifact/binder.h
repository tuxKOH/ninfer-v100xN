#pragma once

#include "artifact/reader.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

// Capacity includes execution, storage and auxiliary slots.
inline constexpr std::size_t kMaximumDevices = 4;
inline constexpr std::size_t kVirtualStorageDevices = 2;

enum class TensorPlacement : std::uint8_t {
    Device,
    PrimaryDevice,
    AuxiliaryDevice,
    Host,
    ValidateOnly,
};

// Which logical axis of a tensor a shard map splits. `Rows` narrows axis 0 (the output/row
// dimension: column-parallel ops), `Columns` narrows axis 1 (the input dimension: row-parallel
// ops). `Replicated` means every device holds the whole object.
enum class ShardAxis : std::uint8_t {
    Replicated,
    Rows,
    Columns,
    // The object is presented through one full logical virtual address range. Its row-split
    // base/high/scale planes are mapped at their original offsets, while the backing pages live
    // on both GPUs.
    VirtualRows,
};

struct ObjectHandle {
    std::size_t index = 0;
};

// The per-device shard map for one object. `device_ranges[d]` lists the ranges of `axis` that
// device d owns, in the order they are concatenated into that device's shard. `Replicated` means
// every device holds the complete object and carries no ranges at all; under `Rows` or `Columns`
// every device in the plan must name at least one range -- an empty list there is rejected rather
// than quietly promoted to a full copy, since "this device happens to own nothing" is far more
// likely to be a shard-map bug than an intent.
struct ShardPlacement {
    ShardAxis axis = ShardAxis::Replicated;
    std::array<std::vector<SliceRange>, kMaximumDevices> device_ranges;
};

struct DeviceMaterialization {
    ObjectHandle object;
    int device              = 0;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 0;
    // Contiguous parent-payload ranges that fill this device's copy. Empty means the whole
    // payload lands at `offset` verbatim -- the only case that exists at tp1.
    std::vector<PlaneCopy> copies;
    std::vector<std::byte> prefix;
};

// A two-device virtual placement.  Segments describe physical allocations mapped at their
// original offsets in one full logical payload.  This matters for row-split tensors: their base,
// high-bit, and scale planes are not laid out as two standalone row shards concatenated together.
// The public object view is therefore the original full payload, while each segment's backing
// pages may live on either GPU. This is intentionally separate from TP sharding: it does not
// change the logical shape or execution width of the tensor.
struct VirtualDeviceSegment {
    int device = 0;
    std::uint64_t virtual_offset = 0;
    std::uint64_t bytes = 0;
    std::vector<PlaneCopy> copies;
    std::vector<std::byte> prefix;
};

struct VirtualDeviceMaterialization {
    ObjectHandle object;
    std::uint64_t virtual_bytes = 0;
    std::vector<VirtualDeviceSegment> segments;
};

struct HostMaterialization {
    ObjectHandle object;
};

struct MaterializationPlan {
    std::size_t object_count = 0;
    // Devices this plan targets; device_capacity_bytes[d] is valid for d < device_count.
    int device_count = 1;
    int execution_storage_device_count = 1;
    std::optional<int> auxiliary_device;
    std::array<std::uint64_t, kMaximumDevices> device_capacity_bytes{};
    // Bytes assigned to ordinary DeviceArena placements.  Virtual placements contribute only to
    // device_capacity_bytes because their physical backing is allocated by the VMM mapper.
    std::array<std::uint64_t, kMaximumDevices> device_arena_capacity_bytes{};
    std::array<std::uint64_t, kMaximumDevices> virtual_capacity_bytes{};
    // One entry per (object, device) pair that receives bytes; ascending by device within object.
    std::vector<DeviceMaterialization> device_objects;
    std::vector<VirtualDeviceMaterialization> virtual_objects;
    std::vector<HostMaterialization> host_objects;
};

class Binder {
public:
    // Resolves an object's shard map from its artifact name. Installed by a target that binds for
    // tp > 1; when absent (the tp1 path) every device tensor is placed whole on device 0.
    using ShardResolver = std::function<ShardPlacement(std::string_view)>;
    using DeviceMaskResolver = std::function<std::uint32_t(std::string_view)>;

    // device_count excludes the optional auxiliary slot, which must immediately follow it.
    explicit Binder(const Reader& reader, int device_count = 1, bool storage_only = false,
                    std::optional<int> auxiliary_device = std::nullopt);

    // Total materialization slots, including the optional auxiliary slot.
    [[nodiscard]] int device_count() const noexcept { return materialization_.device_count; }
    [[nodiscard]] int execution_storage_device_count() const noexcept {
        return materialization_.execution_storage_device_count;
    }
    [[nodiscard]] std::optional<int> auxiliary_device() const noexcept {
        return materialization_.auxiliary_device;
    }

    // Optional companion packages (such as DFlash2) are allowed to live in the same artifact;
    // targets use this read-only probe to bind them only when the startup feature selects them.
    [[nodiscard]] bool has_tensor(std::string_view name) const noexcept;

    void set_shard_resolver(ShardResolver resolver);
    void set_device_mask_resolver(DeviceMaskResolver resolver);

    ObjectHandle require_tensor(std::string_view name, NumericFormat format, StorageLayout layout,
                                std::span<const std::uint64_t> shape);
    ObjectHandle require_resource(std::string_view name, ResourceEncoding encoding);

    const ObjectDescriptor& descriptor(ObjectHandle handle) const;
    PayloadSpan payload(ObjectHandle handle) const;
    void materialize_on_device(ObjectHandle handle);
    // Places one complete tensor only on the selected materialization device.  Used by storage
    // mode for weights that remain primary-device resident while the virtual expert banks carry
    // the cross-device capacity.
    void materialize_on_device(ObjectHandle handle, int device);
    // Places a row-splittable tensor as a full-shape virtual view backed by one set of physical
    // plane slices per device. The two ranges must cover the complete row domain in order.
    void materialize_virtual_rows(ObjectHandle handle,
                                  std::array<SliceRange, kVirtualStorageDevices> ranges);
    void retain_on_host(ObjectHandle handle);
    void materialize_on_host(ObjectHandle handle);
    void validate_only(ObjectHandle handle);
    void validate_unconsumed_matching(std::string_view prefix = "");
    MaterializationPlan finish();

private:
    ObjectHandle find_unconsumed(std::string_view name);
    void place(ObjectHandle handle, int device, std::uint64_t bytes, std::uint64_t alignment,
               std::vector<PlaneCopy> copies);
    void place_on_device(ObjectHandle handle, int device);

    const Reader& reader_;
    std::vector<bool> consumed_;
    std::vector<bool> planned_;
    ShardResolver shard_resolver_;
    DeviceMaskResolver device_mask_resolver_;
    bool storage_only_ = false;
    MaterializationPlan materialization_;
};

} // namespace ninfer::artifact
