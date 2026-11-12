#pragma once

#include <ninfer/targets/qwen3_6/vision.h>
#include <ninfer/targets/qwen3_6/vision_control.h>
#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>
#include <cuda_runtime_api.h>

namespace ninfer::targets::qwen3_6::vision_sm86 {

// The owner retains every allocation referenced by weights until backend destruction.
// Weights must be canonical row-split allocations on device, immutable after binding.
// The caller accounts for the returned completed host buffers.
class Backend {
public:
    Backend(int device, const VisionWeights& weights, std::shared_ptr<const void> weight_owner,
            int output_hidden, std::uint32_t max_merged_tokens, std::uint32_t max_segments);
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    [[nodiscard]] static std::size_t workspace_bytes(const VisionItemControl& item);
    [[nodiscard]] static std::size_t workspace_capacity_bytes(std::uint32_t max_merged_tokens,
                                                               std::uint32_t max_segments);
    // Peak device allocation, excluding materialized weights and CUDA runtime overhead.
    [[nodiscard]] static std::size_t device_capacity_bytes(int output_hidden,
                                                           std::uint32_t max_merged_tokens,
                                                           std::uint32_t max_segments);
    [[nodiscard]] std::shared_ptr<const std::vector<std::uint16_t>>
    encode(std::span<const std::uint16_t> patches, const VisionItemControl& item,
           const PreparationControl& control);

private:
    int device_;
    VisionWeights weights_;
    std::shared_ptr<const void> weight_owner_;
    int output_hidden_;
    std::uint32_t max_merged_tokens_;
    std::uint32_t max_segments_;
    cudaStream_t stream_ = nullptr;
    std::timed_mutex mutex_;
};

} // namespace ninfer::targets::qwen3_6::vision_sm86
