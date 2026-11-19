#pragma once

#include "artifact/binder.h"
#include "core/arena.h"
#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

struct LoadProgress {
    std::function<void(std::string_view, std::uint64_t, std::uint64_t)> callback;
};

struct MaterializationStats {
    std::uint64_t file_bytes              = 0;
    std::uint64_t h2d_bytes               = 0; // summed over every device
    std::uint64_t device_capacity_bytes   = 0; // summed over every device
    std::uint64_t retained_host_bytes = 0;
    std::uint64_t peak_staging_bytes      = 0;
    std::size_t tensor_count              = 0;
    std::size_t resource_count            = 0;
    double upload_seconds                 = 0.0;
    int device_count                      = 1;
    std::array<std::uint64_t, kMaximumDevices> per_device_h2d_bytes{};
    std::array<std::uint64_t, kMaximumDevices> per_device_capacity_bytes{};
};

class MaterializedArtifact {
public:
    struct VirtualMapping;
    MaterializedArtifact();
    ~MaterializedArtifact();
    MaterializedArtifact(MaterializedArtifact&&) noexcept;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    // Device 0's copy: the only copy at tp1, and the local shard on the primary device at tp2.
    void* device_data(ObjectHandle handle) const;
    void* device_data(ObjectHandle handle, int device) const;
    // Bytes actually placed for (object, device). Consumers that wrap `device_data` in a Tensor or
    // a Weight MUST check the shape they claim against this: at tp > 1 an object's placement is
    // its SHARD, so a caller that keeps a whole-model literal shape would read past the end of a
    // half-sized allocation with nothing else noticing.
    std::uint64_t device_bytes(ObjectHandle handle, int device = 0) const;
    std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    std::vector<std::byte> take_host_bytes(ObjectHandle handle);

    const MaterializationStats& stats() const noexcept { return stats_; }

    DeviceArena& device_arena();
    DeviceArena& device_arena(int device);
    [[nodiscard]] int device_count() const noexcept { return stats_.device_count; }
    [[nodiscard]] int physical_device(int device) const;

private:
    friend MaterializedArtifact materialize(const Reader&, const MaterializationPlan&,
                                            std::span<DeviceContext* const>, LoadProgress*);

    struct ObjectStorage {
        std::array<void*, kMaximumDevices> device{};
        std::array<std::uint64_t, kMaximumDevices> device_bytes{};
        std::vector<std::byte> resource;
        std::unique_ptr<PinnedHostBuffer> mapped_host;
    };

    std::array<std::unique_ptr<DeviceArena>, kMaximumDevices> device_arena_;
    std::array<int, kMaximumDevices> physical_devices_{};
    std::vector<ObjectStorage> objects_;
    std::vector<std::unique_ptr<VirtualMapping>> virtual_mappings_;
    MaterializationStats stats_;
};

// Materializes `plan` across `devices[0 .. plan.device_count)`. This is the implementation; the
// two overloads below are the call forms.
MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 std::span<DeviceContext* const> devices,
                                 LoadProgress* progress = nullptr);

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 ExecutionContext& execution, LoadProgress* progress = nullptr);

// Single-device call form for the tp1 path; requires plan.device_count == 1.
MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress = nullptr);

} // namespace ninfer::artifact
