#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include "core/device.h"
#include "core/layout.h"
#include "targets/qwen3_6/impl/vision/host.h"
#include "targets/qwen3_6/impl/vision/schedule.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/vision_attention.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <algorithm>
#include <cstddef>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

using vision_schedule::checked_mul;
using VisionWorkspaceLayout = vision_schedule::Layout;
constexpr std::size_t kWorkspaceAlignment = vision_schedule::alignment;

VisionWorkspaceLayout build_workspace_layout(std::size_t patches, std::size_t tokens,
                                             std::size_t segments) {
    return vision_schedule::build_layout(patches, tokens, segments, 1,
        [](std::int32_t p, std::int32_t s) {
            return ops::vision_attention_workspace_capacity_bytes(p, p, s, s);
        });
}

struct PrimaryVisionOps {
    cudaStream_t stream;
    void copy(const void* src, Tensor& dst) const {
        if (dst.bytes()) CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
    }
    void check_layer() const {}
    void linear(const Tensor& x, const Weight& w, Tensor& y) const { ops::linear(x, w, y, stream); }
    void add_bias(const Tensor& bias, Tensor& x) const { ops::add_bias(bias, x, stream); }
    void layer_norm(const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y) const {
        ops::layer_norm(x, w, b, VisionScheduleConfig::norm_eps, y, stream);
    }
    void position_add(const Tensor& table, const Tensor& indices, const Tensor& weights, Tensor& x) const {
        ops::vision_pos_embed_add(table, indices, weights, x, stream);
    }
    void rope(const Tensor& positions, Tensor& q, Tensor& k) const {
        ops::rope(positions, VisionScheduleConfig::rotary_dim, VisionScheduleConfig::rope_theta, q, k, stream);
    }
    void attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& segments,
                   DeviceSpan scratch, Tensor& out) const {
        WorkspaceArena arena(scratch);
        ops::vision_attention(q, k, v, segments, arena, out, stream);
    }
    void residual_add(const Tensor& x, Tensor& y) const { ops::residual_add(x, y, stream); }
    void gelu_tanh(Tensor& x) const { ops::gelu(x, ops::GeluMode::Tanh, stream); }
    void gelu_exact(Tensor& x) const { ops::gelu(x, ops::GeluMode::Exact, stream); }
};

} // namespace

VisionContext::VisionContext(DeviceContext& ctx, const LoadedModelData& weights, bool auxiliary) : ctx_(ctx) {
    if (auxiliary) { return; }
    if (!weights.vision) {
        throw std::invalid_argument("Vision execution was requested without materialized weights");
    }
    const auto& vision = *weights.vision;
    if (vision.host) {
        host_ = vision.host.get();
        return;
    }
    patch_embed_       = &vision.common.patch_embedding;
    patch_embed_bias_  = &vision.common.patch_embedding_bias;
    position_embed_    = &vision.common.position_embedding;
    for (std::uint32_t layer = 0; layer < blocks_.size(); ++layer) {
        const auto& source  = vision.common.layers[layer];
        BlockW& out         = blocks_[layer];
        out.norm1_weight    = &source.norm1_weight;
        out.norm1_bias      = &source.norm1_bias;
        out.qkv             = &source.qkv;
        out.qkv_bias        = &source.qkv_bias;
        out.projection      = &source.output;
        out.projection_bias = &source.output_bias;
        out.norm2_weight    = &source.norm2_weight;
        out.norm2_bias      = &source.norm2_bias;
        out.fc1             = &source.fc1;
        out.fc1_bias        = &source.fc1_bias;
        out.fc2             = &source.fc2;
        out.fc2_bias        = &source.fc2_bias;
    }
    merger_.norm_weight = &vision.common.merger_norm_weight;
    merger_.norm_bias   = &vision.common.merger_norm_bias;
    merger_.fc1         = &vision.common.merger_fc1;
    merger_.fc1_bias    = &vision.common.merger_fc1_bias;
    merger_.fc2         = &vision.merger_fc2;
    merger_.fc2_bias    = &vision.merger_fc2_bias;
}

std::size_t VisionContext::workspace_bytes(const qwen3_6::VisionItemControl& item) {
    return build_workspace_layout(item.patch_count, item.merged_count,
                                  static_cast<std::size_t>(item.segment_count))
        .bytes;
}

std::size_t VisionContext::output_transient_bytes(std::size_t merged_tokens) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision output transient extent must fit positive int32");
    }
    LayoutBuilder layout;
    (void)layout.add_tensor(
        DType::BF16, {VisionScheduleConfig::out_hidden, static_cast<std::int32_t>(merged_tokens)},
        kWorkspaceAlignment, "Vision item output transient");
    return layout.finish(kWorkspaceAlignment, "Vision item output transient layout");
}

std::size_t VisionContext::workspace_capacity_bytes(std::uint32_t max_merged_tokens,
                                                    std::uint32_t max_segments) {
    if (max_merged_tokens == 0 || max_segments == 0) {
        throw std::invalid_argument("Vision workspace capacity bounds must be positive");
    }
    const std::uint32_t segments = std::min(max_merged_tokens, max_segments);
    return build_workspace_layout(checked_mul(max_merged_tokens, VisionScheduleConfig::merge_unit,
                                              "capacity patch count"),
                                  max_merged_tokens, segments)
        .bytes;
}

void VisionContext::encode(const VisionItemView& item, Tensor& output,
                           WorkspaceArena& workspace) const {
    if (item.control == nullptr) { throw std::invalid_argument("Vision item control is null"); }
    const qwen3_6::VisionItemControl& control = *item.control;
    const auto patches64                      = control.patch_count;
    const auto tokens64                       = control.merged_count;
    if (item.patches.size() !=
        checked_mul(patches64, VisionScheduleConfig::patch_dim, "patch elements")) {
        throw std::invalid_argument("Vision processor patch buffer has invalid shape");
    }
    if (output.dtype != DType::BF16 || output.ne[0] != VisionScheduleConfig::out_hidden ||
        output.ne[1] != static_cast<std::int32_t>(tokens64) || output.ne[2] != 1 ||
        output.ne[3] != 1 || !output.is_contiguous() || output.data == nullptr) {
        throw std::invalid_argument("Vision output must be contiguous BF16 [H,V]");
    }
    if (host_ != nullptr) {
        throw std::logic_error("CPU Vision must be encoded before GPU admission");
    }
    const VisionWorkspaceLayout layout = build_workspace_layout(
        patches64, tokens64, static_cast<std::size_t>(control.segment_count));
    if (workspace.capacity() < layout.bytes) {
        throw std::invalid_argument("Vision workspace capacity is too small for request");
    }
    workspace.reset();
    const DeviceSpan backing = workspace.alloc_bytes(layout.bytes, kWorkspaceAlignment);
    vision_schedule::Weights weights{};
    weights.patch = patch_embed_;
    weights.patch_bias = patch_embed_bias_;
    weights.position = position_embed_;
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        const auto& b = blocks_[i];
        weights.blocks[i] = {b.norm1_weight, b.norm1_bias, b.qkv, b.qkv_bias,
                             b.projection, b.projection_bias, b.norm2_weight, b.norm2_bias,
                             b.fc1, b.fc1_bias, b.fc2, b.fc2_bias};
    }
    weights.merger_norm_weight = merger_.norm_weight;
    weights.merger_norm_bias = merger_.norm_bias;
    weights.merger_fc1 = merger_.fc1;
    weights.merger_fc1_bias = merger_.fc1_bias;
    weights.merger_fc2 = merger_.fc2;
    weights.merger_fc2_bias = merger_.fc2_bias;
    PrimaryVisionOps operations{ctx_.stream};
    vision_schedule::execute(layout, backing, item.patches, control, weights, output, operations);
}

VisionPrefillSession::VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                                           WorkspaceArena& workspace,
                                           qwen3_6::PreparedPromptData& prompt,
                                           const VisionPrefillPlan& plan,
                                           runtime::TransientRegion transient, bool auxiliary)
    : device_(device), workspace_(workspace), prompt_(prompt), plan_(plan), transient_(transient),
      auxiliary_(auxiliary), context_(device, model, auxiliary) {
    if (plan_.control == nullptr || plan_.control->items.empty() || plan_.uses.empty()) {
        throw std::invalid_argument("Vision prefill plan has no suffix item spans");
    }
    if (transient_.data == nullptr || transient_.alignment < kWorkspaceAlignment) {
        throw std::invalid_argument("Vision item output transient is missing or misaligned");
    }
    encoded_payloads_pending_release_.reserve(plan_.uses.size());
    timers_.reserve(plan_.uses.size());
}

VisionChunk VisionPrefillSession::prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length) {
    if (nominal_length == 0 || begin >= prompt_.token_ids.size()) {
        throw std::invalid_argument("Vision chunk range is empty or outside the prompt");
    }
    const std::uint64_t nominal_end64 =
        static_cast<std::uint64_t>(begin) + static_cast<std::uint64_t>(nominal_length);
    std::uint32_t end = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(nominal_end64, prompt_.token_ids.size()));

    const VisionUseSpan* active = nullptr;
    for (const VisionUseSpan& use : plan_.uses) {
        if (use.end <= begin) { continue; }
        if (use.begin >= end) { break; }
        if (active == nullptr) {
            active = &use;
        } else {
            end = std::min(end, use.begin);
            break;
        }
    }
    if (end <= begin) { throw std::logic_error("Vision chunk cap made no forward progress"); }
    if (active == nullptr) {
        return VisionChunk{static_cast<std::int32_t>(end - begin), nullptr, {}};
    }
    if (active->item_index >= plan_.control->items.size() ||
        active->item_index >= prompt_.vision_items.size() ||
        active->item_index >= prompt_.media_payloads.size()) {
        throw std::logic_error("Vision prefill item index is out of range");
    }
    const qwen3_6::VisionItemControl& control = plan_.control->items[active->item_index];
    const qwen3_6::VisionItem& source         = prompt_.vision_items[active->item_index];
    if (source.modality != control.modality || source.grid.temporal != control.grid.temporal ||
        source.grid.height != control.grid.height || source.grid.width != control.grid.width ||
        source.patch_begin != control.patch_begin || source.patch_count != control.patch_count) {
        throw std::invalid_argument("Vision prefill plan does not describe the prepared item");
    }
    if (control.merged_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision item output columns exceed int32");
    }
    const std::size_t output_bytes =
        checked_mul(checked_mul(static_cast<std::size_t>(VisionScheduleConfig::out_hidden),
                                control.merged_count, "item output elements"),
                    dtype_size(DType::BF16), "item output bytes");
    if (output_bytes > transient_.size) {
        throw std::invalid_argument("Vision item output transient is too small");
    }
    Tensor output(
        transient_.data, DType::BF16,
        {VisionScheduleConfig::out_hidden, static_cast<std::int32_t>(control.merged_count)});

    if (!active_item_ || *active_item_ != active->item_index) {
        if (active_item_ && active->item_index <= *active_item_) {
            throw std::logic_error("Vision items are not consumed in strictly increasing order");
        }
        const std::size_t patch_elements = checked_mul(
            control.patch_count, static_cast<std::size_t>(VisionScheduleConfig::patch_dim),
            "item patch elements");
        const auto& payload = prompt_.media_payloads[active->item_index];
        if (!payload || payload->patch_elements != patch_elements) {
            throw std::invalid_argument("Vision item patch payload has an invalid shape");
        }
        if (auxiliary_ || context_.is_cpu() || !prompt_.host_vision_embeddings.empty()) {
            const auto start = std::chrono::steady_clock::now();
            if (active->item_index >= prompt_.host_vision_embeddings.size()) {
                throw std::logic_error("Vision request was not staged before GPU admission");
            }
            const auto& encoded = prompt_.host_vision_embeddings[active->item_index];
            if (!encoded || encoded->size() != output_bytes / sizeof(std::uint16_t)) {
                throw std::logic_error("Vision embedding extent mismatch");
            }
            CUDA_CHECK(cudaMemcpyAsync(output.data, encoded->data(), output.bytes(),
                                       cudaMemcpyHostToDevice, device_.stream));
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            host_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        } else {
            timers_.emplace_back(device_);
            timers_.back().start();
            context_.encode(VisionItemView{payload->span(), &control}, output, workspace_);
            timers_.back().record_stop();
        }
        workspace_.reset();
        active_item_ = active->item_index;
        encoded_payloads_pending_release_.push_back(active->item_index);
    }
    return VisionChunk{static_cast<std::int32_t>(end - begin), &control, output};
}

void VisionPrefillSession::release_encoded_media_payloads() noexcept {
    for (const std::uint32_t item_index : encoded_payloads_pending_release_) {
        if (item_index >= prompt_.media_payloads.size()) { std::terminate(); }
        prompt_.media_payloads[item_index].reset();
        if (item_index < prompt_.host_vision_embeddings.size()) {
            prompt_.host_vision_embeddings[item_index].reset();
        }
    }
    encoded_payloads_pending_release_.clear();
}

double VisionPrefillSession::elapsed_seconds() const {
    double milliseconds = 0.0;
    for (const CudaEventTimer& timer : timers_) { milliseconds += timer.elapsed_ms(); }
    return milliseconds / 1000.0 + host_seconds_;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
