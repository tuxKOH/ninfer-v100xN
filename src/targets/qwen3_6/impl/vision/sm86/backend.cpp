#include "targets/qwen3_6/impl/vision/sm86/backend.h"
#include "targets/qwen3_6/impl/vision/sm86/attention.h"
#include "targets/qwen3_6/impl/vision/sm86/elementwise.h"
#include "targets/qwen3_6/impl/vision/sm86/linear.h"
#include "targets/qwen3_6/impl/frontend/preparation_control.h"
#include "targets/qwen3_6/impl/vision/schedule.h"
#include "core/device.h"
#include "core/layout.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_6::vision_sm86 {
namespace {
using C = VisionBackboneConfig;
constexpr std::size_t alignment = 256;

std::size_t mul(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("SM86 Vision size overflow");
    return a * b;
}

class DeviceGuard {
public:
    explicit DeviceGuard(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        if (previous_ != device) CUDA_CHECK(cudaSetDevice(device));
    }
    ~DeviceGuard() { (void)cudaSetDevice(previous_); }
private:
    int previous_ = 0;
};

vision_schedule::Layout layout(std::size_t patches, std::size_t tokens, std::size_t segments) {
    if (segments > tokens) throw std::invalid_argument("SM86 Vision invalid workspace dimensions");
    return vision_schedule::build_layout(patches, tokens, segments, alignment,
        [](std::int32_t p, std::int32_t s) {
            const auto bytes = attention_workspace_bytes(p, s);
            if (bytes != 0) throw std::logic_error("SM86 Vision attention workspace contract changed");
            return bytes;
        });
}

struct AuxiliaryVisionOps {
    cudaStream_t stream;
    const PreparationControl& control;
    void copy(const void* src, Tensor& dst) const {
        CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
    }
    void check_layer() const { frontend_internal::check_preparation_control(control, "SM86 Vision"); }
    void linear(const Tensor& x, const Weight& w, Tensor& y) const { vision_sm86::linear(x, w, y, stream); }
    void add_bias(const Tensor& b, Tensor& x) const { vision_sm86::add_bias(b, x, stream); }
    void layer_norm(const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y) const {
        vision_sm86::layer_norm(x, w, b, C::norm_epsilon, y, stream);
    }
    void position_add(const Tensor& table, const Tensor& indices, const Tensor& weights, Tensor& x) const {
        vision_sm86::position_add(table, indices, weights, x, stream);
    }
    void rope(const Tensor& positions, Tensor& q, Tensor& k) const {
        vision_sm86::rope_2d(positions, q, k, stream);
    }
    void attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& segments,
                   DeviceSpan, Tensor& out) const {
        vision_sm86::attention_cu_seqlens(q, k, v, segments, out, stream);
    }
    void residual_add(const Tensor& x, Tensor& y) const { vision_sm86::residual_add(x, y, stream); }
    void gelu_tanh(Tensor& x) const { vision_sm86::gelu_tanh(x, stream); }
    void gelu_exact(Tensor& x) const { vision_sm86::gelu_exact(x, stream); }
};

void device_pointer(const void* ptr, int device) {
    if (!ptr) throw std::invalid_argument("SM86 Vision null weight pointer");
    cudaPointerAttributes attr{};
    CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
    if (attr.type != cudaMemoryTypeDevice || attr.device != device)
        throw std::invalid_argument("SM86 Vision weight is not on the auxiliary device");
}

void tensor(const Tensor& t, int elements, int device) {
    if (t.dtype != DType::BF16 || !t.is_contiguous() || t.numel() != elements)
        throw std::invalid_argument("SM86 Vision invalid BF16 weight shape");
    device_pointer(t.data, device);
}

void weight(const Weight& w, QType type, int n, int k, int device) {
    const int group = type == QType::W8G32_F16S ? 32 : 64;
    const std::size_t high_per_group = type == QType::Q5G64_F16S ? 8 :
                                       type == QType::Q6G64_F16S ? 16 : 0;
    const std::size_t padded = (static_cast<std::size_t>(k) + 127) / 128 * 128;
    const std::size_t groups = mul(static_cast<std::size_t>(n), padded / group);
    const std::size_t low = mul(groups, 32);
    const std::size_t high = mul(groups, high_per_group);
    const auto align = [](std::size_t value) {
        if (value > std::numeric_limits<std::size_t>::max() - 255)
            throw std::overflow_error("SM86 Vision weight size overflow");
        return (value + 255) / 256 * 256;
    };
    const std::size_t high_offset = align(low);
    const std::size_t high_padded = align(high);
    if (high_padded > std::numeric_limits<std::size_t>::max() - high_offset)
        throw std::overflow_error("SM86 Vision weight size overflow");
    const std::size_t scale_offset = high_offset + high_padded;
    const std::size_t scale_bytes = mul(groups, 2);
    if (scale_bytes > std::numeric_limits<std::size_t>::max() - scale_offset)
        throw std::overflow_error("SM86 Vision weight size overflow");
    const auto base = reinterpret_cast<std::uintptr_t>(w.payload);
    if (base > std::numeric_limits<std::uintptr_t>::max() - scale_offset ||
        w.qtype != type || w.layout != QuantLayout::RowSplit || w.n != n || w.k != k ||
        w.ndim != 2 || w.shape[0] != n || w.shape[1] != k || w.shape[2] != 1 ||
        w.shape[3] != 1 || w.padded_shape[0] != n || w.padded_shape[1] != padded ||
        w.padded_shape[2] != 1 || w.padded_shape[3] != 1 || w.group != group ||
        w.group_size != static_cast<std::uint32_t>(group) || w.scale_dtype != DType::FP16 ||
        base % 16 != 0 || w.payload_bytes < scale_offset + scale_bytes ||
        w.high_plane_bytes != high || w.qdata != w.payload ||
        reinterpret_cast<std::uintptr_t>(w.scales) != base + scale_offset ||
        (high ? reinterpret_cast<std::uintptr_t>(w.qhigh) != base + high_offset : w.qhigh != nullptr))
        throw std::invalid_argument("SM86 Vision invalid canonical weight");
    device_pointer(w.payload, device);
}

void validate_weights(const VisionWeights& w, int out, int device) {
    const auto& c = w.common;
    weight(c.patch_embedding, QType::Q6G64_F16S, C::hidden, C::patch_dim, device);
    tensor(c.patch_embedding_bias, C::hidden, device);
    tensor(c.position_embedding, C::hidden * C::position_embeddings, device);
    for (const auto& l : c.layers) {
        tensor(l.norm1_weight, C::hidden, device); tensor(l.norm1_bias, C::hidden, device);
        weight(l.qkv, QType::Q4G64_F16S, 3 * C::hidden, C::hidden, device);
        tensor(l.qkv_bias, 3 * C::hidden, device);
        weight(l.output, QType::Q5G64_F16S, C::hidden, C::hidden, device);
        tensor(l.output_bias, C::hidden, device);
        tensor(l.norm2_weight, C::hidden, device); tensor(l.norm2_bias, C::hidden, device);
        weight(l.fc1, QType::Q4G64_F16S, C::intermediate, C::hidden, device);
        tensor(l.fc1_bias, C::intermediate, device);
        weight(l.fc2, QType::Q5G64_F16S, C::hidden, C::intermediate, device);
        tensor(l.fc2_bias, C::hidden, device);
    }
    tensor(c.merger_norm_weight, C::hidden, device); tensor(c.merger_norm_bias, C::hidden, device);
    weight(c.merger_fc1, QType::W8G32_F16S, C::merger_hidden, C::merger_hidden, device);
    tensor(c.merger_fc1_bias, C::merger_hidden, device);
    weight(w.merger_fc2, QType::W8G32_F16S, out, C::merger_hidden, device);
    tensor(w.merger_fc2_bias, out, device);
}

void validate_item(std::span<const std::uint16_t> patches, const VisionItemControl& item) {
    (void)layout(item.patch_count, item.merged_count, static_cast<std::size_t>(item.segment_count));
    const auto p = item.patch_count;
    if (patches.size() != mul(p, C::patch_dim) || item.position_ids.size() != mul(p, 2) ||
        item.position_table_indices.size() != mul(p, 4) ||
        item.position_table_weights.size() != mul(p, 4) ||
        item.cu_seqlens.size() != std::size_t(item.segment_count) + 1 ||
        item.cu_seqlens.front() != 0 || item.cu_seqlens.back() != static_cast<std::int32_t>(p))
        throw std::invalid_argument("SM86 Vision invalid prepared item shape");
    for (std::size_t i = 1; i < item.cu_seqlens.size(); ++i)
        if (item.cu_seqlens[i] <= item.cu_seqlens[i - 1])
            throw std::invalid_argument("SM86 Vision invalid segment bounds");
    for (auto i : item.position_table_indices)
        if (i < 0 || i >= C::position_embeddings)
            throw std::invalid_argument("SM86 Vision position index out of bounds");
}



struct Drain {
    cudaStream_t stream;
    ~Drain() { if (stream) (void)cudaStreamSynchronize(stream); }
};
} // namespace

Backend::Backend(int device, const VisionWeights& weights, std::shared_ptr<const void> owner,
                 int output_hidden, std::uint32_t max_tokens, std::uint32_t max_segments)
    : device_(device), weights_(weights), weight_owner_(std::move(owner)), output_hidden_(output_hidden),
      max_merged_tokens_(max_tokens), max_segments_(max_segments) {
    if (!weight_owner_ || output_hidden <= 0)
        throw std::invalid_argument("SM86 Vision requires weight ownership and output shape");
    (void)workspace_capacity_bytes(max_tokens, max_segments);
    DeviceGuard guard(device_);
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_));
    if (properties.major * 10 + properties.minor != NINFER_AUX_VISION_ARCH)
        throw std::invalid_argument("auxiliary Vision requires an SM" +
            std::to_string(NINFER_AUX_VISION_ARCH) + " device; selected device is SM" +
            std::to_string(properties.major * 10 + properties.minor));
    validate_weights(weights_, output_hidden_, device_);
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

Backend::~Backend() {
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device_) == cudaSuccess) {
        if (stream_) { (void)cudaStreamSynchronize(stream_); (void)cudaStreamDestroy(stream_); }
        (void)cudaSetDevice(previous);
    }
}

std::size_t Backend::workspace_bytes(const VisionItemControl& item) {
    return layout(item.patch_count, item.merged_count, static_cast<std::size_t>(item.segment_count)).bytes;
}

std::size_t Backend::device_capacity_bytes(int output_hidden, std::uint32_t tokens,
                                           std::uint32_t segments) {
    if (output_hidden <= 0) throw std::invalid_argument("SM86 Vision output shape must be positive");
    const auto workspace = workspace_capacity_bytes(tokens, segments);
    const auto output = mul(mul(tokens, static_cast<std::size_t>(output_hidden)), sizeof(std::uint16_t));
    if (output > std::numeric_limits<std::size_t>::max() - workspace)
        throw std::overflow_error("SM86 Vision device capacity overflow");
    return workspace + output;
}

std::size_t Backend::workspace_capacity_bytes(std::uint32_t tokens, std::uint32_t segments) {
    if (!tokens || !segments) throw std::invalid_argument("SM86 Vision capacity must be positive");
    return layout(mul(tokens, C::merge_unit), tokens, std::min(tokens, segments)).bytes;
}

std::shared_ptr<const std::vector<std::uint16_t>> Backend::encode(
    std::span<const std::uint16_t> patches, const VisionItemControl& item,
    const PreparationControl& control) {
    frontend_internal::check_preparation_control(control, "SM86 Vision");
    validate_item(patches, item);
    if (item.merged_count > max_merged_tokens_ || item.segment_count > static_cast<std::int64_t>(max_segments_))
        throw std::invalid_argument("SM86 Vision item exceeds configured capacity");
    std::unique_lock lock(mutex_, std::defer_lock);
    while (!lock.try_lock_for(std::chrono::milliseconds(10)))
        frontend_internal::check_preparation_control(control, "SM86 Vision");
    frontend_internal::check_preparation_control(control, "SM86 Vision");
    DeviceGuard guard(device_);
    const auto l = layout(item.patch_count, item.merged_count, item.segment_count);
    DeviceBuffer workspace(l.bytes);
    const auto elements = mul(item.merged_count, static_cast<std::size_t>(output_hidden_));
    DeviceBuffer output_buffer(mul(elements, sizeof(std::uint16_t)));
    auto result = std::make_shared<std::vector<std::uint16_t>>(elements);
    Drain drain{stream_};
    DeviceSpan backing{workspace.p, workspace.bytes};
    const auto& w = weights_.common;
    vision_schedule::Weights weights{};
    weights.patch = &w.patch_embedding;
    weights.patch_bias = &w.patch_embedding_bias;
    weights.position = &w.position_embedding;
    for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto& b = w.layers[i];
        weights.blocks[i] = {&b.norm1_weight, &b.norm1_bias, &b.qkv, &b.qkv_bias,
                             &b.output, &b.output_bias, &b.norm2_weight, &b.norm2_bias,
                             &b.fc1, &b.fc1_bias, &b.fc2, &b.fc2_bias};
    }
    weights.merger_norm_weight = &w.merger_norm_weight;
    weights.merger_norm_bias = &w.merger_norm_bias;
    weights.merger_fc1 = &w.merger_fc1;
    weights.merger_fc1_bias = &w.merger_fc1_bias;
    weights.merger_fc2 = &weights_.merger_fc2;
    weights.merger_fc2_bias = &weights_.merger_fc2_bias;
    Tensor output(output_buffer.p, DType::BF16, {output_hidden_, static_cast<int>(item.merged_count)});
    AuxiliaryVisionOps operations{stream_, control};
    vision_schedule::execute(l, backing, patches, item, weights, output, operations);
    CUDA_CHECK(cudaMemcpyAsync(result->data(), output.data, output.bytes(), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    drain.stream = nullptr;
    frontend_internal::check_preparation_control(control, "SM86 Vision");
    return result;
}

} // namespace ninfer::targets::qwen3_6::vision_sm86
