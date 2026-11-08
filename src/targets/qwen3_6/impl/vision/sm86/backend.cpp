#include "targets/qwen3_6/impl/vision/sm86/backend.h"
#include "targets/qwen3_6/impl/vision/sm86/attention.h"
#include "targets/qwen3_6/impl/vision/sm86/elementwise.h"
#include "targets/qwen3_6/impl/vision/sm86/linear.h"
#include "targets/qwen3_6/impl/frontend/preparation_control.h"
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

struct Layout {
    TensorRegion positions, segments, indices, coefficients, x, patches;
    TensorRegion attended, qkv, norm1, projected, up, norm2, down, normalized, merger;
    std::size_t bytes;
};

Layout layout(std::size_t patches, std::size_t tokens, std::size_t segments) {
    if (!tokens || patches != mul(tokens, C::merge_unit) || !segments || segments > tokens ||
        patches > std::size_t(std::numeric_limits<std::int32_t>::max()) ||
        segments >= std::size_t(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("SM86 Vision invalid workspace dimensions");
    auto p = static_cast<std::int32_t>(patches);
    auto t = static_cast<std::int32_t>(tokens);
    auto s = static_cast<std::int32_t>(segments);
    LayoutBuilder b;
    Layout l{};
    auto add = [&](DType type, std::initializer_list<std::int32_t> shape, const char* name) {
        return b.add_tensor(type, shape, alignment, name);
    };
    l.positions = add(DType::I32, {p, 2}, "vision positions");
    l.segments = add(DType::I32, {s + 1}, "vision segments");
    l.indices = add(DType::I32, {4, p}, "vision indices");
    l.coefficients = add(DType::FP32, {4, p}, "vision coefficients");
    l.x = add(DType::BF16, {C::hidden, p}, "vision residual");
    l.patches = add(DType::BF16, {C::patch_dim, p}, "vision patches");
    {
        auto scope = b.scope();
        l.attended = add(DType::BF16, {C::hidden, p}, "vision attended");
        {
            auto inner = b.scope();
            l.qkv = add(DType::BF16, {3 * C::hidden, p}, "vision qkv");
            {
                auto norm = b.scope();
                l.norm1 = add(DType::BF16, {C::hidden, p}, "vision norm1");
            }
            if (attention_workspace_bytes(p, s) != 0)
                throw std::logic_error("SM86 Vision attention workspace contract changed");
        }
        l.projected = add(DType::BF16, {C::hidden, p}, "vision projected");
    }
    {
        auto scope = b.scope();
        l.up = add(DType::BF16, {C::intermediate, p}, "vision up");
        {
            auto inner = b.scope();
            l.norm2 = add(DType::BF16, {C::hidden, p}, "vision norm2");
        }
        l.down = add(DType::BF16, {C::hidden, p}, "vision down");
    }
    l.normalized = add(DType::BF16, {C::hidden, p}, "vision merger norm");
    l.merger = add(DType::BF16, {C::merger_hidden, t}, "vision merger hidden");
    l.bytes = b.finish(alignment, "vision workspace");
    return l;
}

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

void copy(const void* src, Tensor& dst, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
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
    if (properties.major != 8 || properties.minor != 6)
        throw std::invalid_argument("SM86 Vision requires an SM86 device");
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
    const int p = static_cast<int>(item.patch_count), t = static_cast<int>(item.merged_count);
    Tensor positions = l.positions.bind(backing), segments = l.segments.bind(backing);
    Tensor indices = l.indices.bind(backing), coefficients = l.coefficients.bind(backing);
    Tensor x = l.x.bind(backing), input = l.patches.bind(backing);
    copy(item.position_ids.data(), positions, stream_); copy(item.cu_seqlens.data(), segments, stream_);
    copy(item.position_table_indices.data(), indices, stream_);
    copy(item.position_table_weights.data(), coefficients, stream_); copy(patches.data(), input, stream_);
    const auto& w = weights_.common;
    linear(input, w.patch_embedding, x, stream_); add_bias(w.patch_embedding_bias, x, stream_);
    Tensor table = w.position_embedding.reshape({C::hidden, C::position_embeddings});
    position_add(table, indices, coefficients, x, stream_);
    for (const auto& block : w.layers) {
        frontend_internal::check_preparation_control(control, "SM86 Vision");
        Tensor attended = l.attended.bind(backing), qkv = l.qkv.bind(backing), norm = l.norm1.bind(backing);
        layer_norm(x, block.norm1_weight, block.norm1_bias, C::norm_epsilon, norm, stream_);
        linear(norm, block.qkv, qkv, stream_); add_bias(block.qkv_bias, qkv, stream_);
        const std::size_t plane = C::hidden * sizeof(std::uint16_t);
        Tensor q(qkv.data, DType::BF16, {C::head_dim, C::heads, p});
        Tensor k(static_cast<unsigned char*>(qkv.data) + plane, DType::BF16, {C::head_dim, C::heads, p});
        Tensor v(static_cast<unsigned char*>(qkv.data) + 2 * plane, DType::BF16, {C::head_dim, C::heads, p});
        q.nb[2] = k.nb[2] = v.nb[2] = qkv.nb[1];
        rope_2d(positions, q, k, stream_);
        Tensor heads = attended.view({C::head_dim, C::heads, p});
        attention_cu_seqlens(q, k, v, segments, heads, stream_);
        Tensor projected = l.projected.bind(backing);
        linear(attended, block.output, projected, stream_); add_bias(block.output_bias, projected, stream_);
        residual_add(projected, x, stream_);
        Tensor up = l.up.bind(backing), down = l.down.bind(backing), norm2 = l.norm2.bind(backing);
        layer_norm(x, block.norm2_weight, block.norm2_bias, C::norm_epsilon, norm2, stream_);
        linear(norm2, block.fc1, up, stream_); add_bias(block.fc1_bias, up, stream_);
        gelu_tanh(up, stream_); linear(up, block.fc2, down, stream_);
        add_bias(block.fc2_bias, down, stream_); residual_add(down, x, stream_);
    }
    Tensor normalized = l.normalized.bind(backing), hidden = l.merger.bind(backing);
    layer_norm(x, w.merger_norm_weight, w.merger_norm_bias, C::norm_epsilon, normalized, stream_);
    Tensor merged = normalized.view({C::merger_hidden, t});
    linear(merged, w.merger_fc1, hidden, stream_); add_bias(w.merger_fc1_bias, hidden, stream_);
    gelu_exact(hidden, stream_);
    Tensor output(output_buffer.p, DType::BF16, {output_hidden_, t});
    linear(hidden, weights_.merger_fc2, output, stream_); add_bias(weights_.merger_fc2_bias, output, stream_);
    CUDA_CHECK(cudaMemcpyAsync(result->data(), output.data, output.bytes(), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    drain.stream = nullptr;
    frontend_internal::check_preparation_control(control, "SM86 Vision");
    return result;
}

} // namespace ninfer::targets::qwen3_6::vision_sm86
