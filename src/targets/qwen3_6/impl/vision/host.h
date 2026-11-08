#pragma once

#include "ninfer/ops/host_vision.h"
#include "targets/qwen3_6/impl/frontend/preparation_control.h"
#include <ninfer/targets/qwen3_6/vision.h>
#include <ninfer/targets/qwen3_6/vision_control.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6 {

struct HostVisionLayerWeights {
    ops::HostLinearWeight qkv, output, fc1, fc2;
    std::vector<float> qkv_bias, output_bias, fc1_bias, fc2_bias;
    std::vector<float> norm1_weight, norm1_bias, norm2_weight, norm2_bias;
};

struct HostVisionWeights {
    ops::HostLinearWeight patch_embedding, merger_fc1, merger_fc2;
    std::vector<float> patch_embedding_bias, position_embedding;
    std::vector<float> merger_fc1_bias, merger_fc2_bias, merger_norm_weight, merger_norm_bias;
    std::array<HostVisionLayerWeights, VisionBackboneConfig::layers> layers;
};

// Model-specific embeddings belong to one Program, never to the identity-free frontend.
// Immutable entries are FIFO-bounded to 128 MiB; request references can outlive eviction.
class HostVisionCache {
public:
    double prepare(const HostVisionWeights& weights, PreparedPromptData& prompt,
                   const PreparationControl& control);

    // The encoder returns a completed immutable host buffer. Its owner must account for
    // live outputs and finish all device work before returning or throwing.
    template <class Encoder>
    double prepare_with(PreparedPromptData& prompt, const PreparationControl& preparation,
                        std::size_t output_hidden, Encoder&& encode) {
        constexpr std::size_t capacity = 128ULL << 20;
        const auto controls = build_vision_control(prompt);
        std::vector<std::shared_ptr<const std::vector<std::uint16_t>>> prepared;
        prepared.reserve(controls.items.size());
        double encoded_seconds = 0.0;
        for (std::size_t i = 0; i < controls.items.size(); ++i) {
            frontend_internal::check_preparation_control(preparation, "Vision");
            const auto& item = prompt.vision_items.at(i);
            const auto& control = controls.items[i];
            if (output_hidden == 0 ||
                control.merged_count > std::numeric_limits<std::size_t>::max() / output_hidden) {
                throw std::overflow_error("Vision embedding extent is invalid");
            }
            const std::size_t elements = output_hidden * control.merged_count;
            if (elements > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
                throw std::overflow_error("Vision embedding byte extent is invalid");
            }
            const auto matches = [&](const Entry& entry) {
                return entry.digest == item.content_digest && entry.modality == item.modality &&
                       entry.grid.temporal == item.grid.temporal &&
                       entry.grid.height == item.grid.height && entry.grid.width == item.grid.width;
            };
            std::shared_ptr<const std::vector<std::uint16_t>> embeddings;
            {
                std::lock_guard lock(mutex_);
                const auto found = std::find_if(entries_.begin(), entries_.end(), matches);
                if (found != entries_.end()) { embeddings = found->embeddings; }
            }
            if (!embeddings) {
                if (i >= prompt.media_payloads.size() || !prompt.media_payloads[i]) {
                    throw std::invalid_argument("Vision request has missing prepared patches");
                }
                const auto start = std::chrono::steady_clock::now();
                embeddings = encode(prompt.media_payloads[i]->span(), control, preparation);
                frontend_internal::check_preparation_control(preparation, "Vision");
                encoded_seconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                if (!embeddings || embeddings->size() != elements) {
                    throw std::invalid_argument("Vision encoder returned an invalid embedding extent");
                }
                const auto bytes = elements * sizeof(std::uint16_t);
                if (bytes <= capacity) {
                    std::lock_guard lock(mutex_);
                    const auto found = std::find_if(entries_.begin(), entries_.end(), matches);
                    if (found != entries_.end()) { embeddings = found->embeddings; }
                    else {
                        while (retained_bytes_ > capacity - bytes) {
                            retained_bytes_ -= entries_.front().embeddings->size() * sizeof(std::uint16_t);
                            entries_.pop_front();
                        }
                        entries_.push_back({item.content_digest, item.modality, item.grid, embeddings});
                        retained_bytes_ += bytes;
                    }
                }
            }
            if (embeddings->size() != elements) {
                throw std::invalid_argument("Cached Vision embedding extent mismatch");
            }
            prepared.push_back(std::move(embeddings));
        }
        frontend_internal::check_preparation_control(preparation, "Vision");
        prompt.host_vision_embeddings = std::move(prepared);
        return encoded_seconds;
    }
private:
    struct Entry {
        std::array<std::uint8_t, 32> digest;
        PromptModality modality;
        VisionGrid grid;
        std::shared_ptr<const std::vector<std::uint16_t>> embeddings;
    };
    std::mutex mutex_;
    std::deque<Entry> entries_;
    std::size_t retained_bytes_ = 0;
};

// Same patch ordering, temporal segments, 2-D RoPE, 27 residual blocks and 2x2 merger as
// the CUDA schedule. No model identity or CUDA graph/device allocation belongs here.
std::vector<std::uint16_t> encode_host_vision(const HostVisionWeights& weights,
    std::span<const std::uint16_t> patches, const VisionItemControl& control,
    const PreparationControl& preparation = {});

} // namespace ninfer::targets::qwen3_6
