#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "core/weight.h"
#include <ninfer/targets/qwen3_6/vision.h>
#include <ninfer/targets/qwen3_6/vision_control.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::vision_schedule {
using C = VisionBackboneConfig;
constexpr std::size_t alignment = 256;

inline std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    return a * b;
}

struct Layout {
    TensorRegion position_ids, cu_seqlens, pos_indices, pos_weights, x, patch_bf16;
    TensorRegion attended, qkv, attention_norm;
    std::optional<LayoutRegion> attention_workspace;
    TensorRegion projected, mlp_down, mlp_up, mlp_norm, normalized, merger_hidden;
    std::size_t bytes = 0;
};

template<class AttentionBytes>
Layout build_layout(std::size_t patches, std::size_t tokens, std::size_t segments,
                    std::size_t finish_alignment, AttentionBytes attention_bytes) {
    if (!patches || !tokens || patches != checked_mul(tokens, C::merge_unit, "patch/token relation"))
        throw std::invalid_argument("Vision workspace requires P=4V>0");
    if (patches > std::size_t(std::numeric_limits<std::int32_t>::max()) ||
        tokens > std::size_t(std::numeric_limits<std::int32_t>::max()) || !segments ||
        segments >= std::size_t(std::numeric_limits<std::int32_t>::max()))
        throw std::overflow_error("Vision request dimensions exceed int32");
    const auto p = static_cast<std::int32_t>(patches);
    const auto t = static_cast<std::int32_t>(tokens);
    const auto s = static_cast<std::int32_t>(segments);
    LayoutBuilder b;
    Layout l;
    auto add = [&](DType type, std::initializer_list<std::int32_t> shape, const char* name) {
        return b.add_tensor(type, shape, alignment, name);
    };
    l.position_ids = add(DType::I32, {p, 2}, "vision position ids");
    l.cu_seqlens = add(DType::I32, {s + 1}, "vision segment bounds");
    l.pos_indices = add(DType::I32, {4, p}, "vision position indices");
    l.pos_weights = add(DType::FP32, {4, p}, "vision position weights");
    l.x = add(DType::BF16, {C::hidden, p}, "vision residual");
    l.patch_bf16 = add(DType::BF16, {C::patch_dim, p}, "vision BF16 patches");
    {
        auto scope = b.scope();
        l.attended = add(DType::BF16, {C::hidden, p}, "vision attended");
        {
            auto qkv_scope = b.scope();
            l.qkv = add(DType::BF16, {3 * C::hidden, p}, "vision QKV");
            {
                auto norm_scope = b.scope();
                l.attention_norm = add(DType::BF16, {C::hidden, p}, "vision attention norm");
            }
            const auto bytes = attention_bytes(p, s);
            if (bytes) l.attention_workspace = b.add(bytes, alignment, "vision attention workspace");
        }
        l.projected = add(DType::BF16, {C::hidden, p}, "vision projected");
    }
    {
        auto scope = b.scope();
        l.mlp_up = add(DType::BF16, {C::intermediate, p}, "vision MLP up");
        {
            auto norm_scope = b.scope();
            l.mlp_norm = add(DType::BF16, {C::hidden, p}, "vision MLP norm");
        }
        l.mlp_down = add(DType::BF16, {C::hidden, p}, "vision MLP down");
    }
    l.normalized = add(DType::BF16, {C::hidden, p}, "vision merger norm");
    l.merger_hidden = add(DType::BF16, {C::merger_hidden, t}, "vision merger hidden");
    l.bytes = b.finish(finish_alignment, "vision workspace");
    return l;
}

struct Block {
    const Tensor *norm1_weight, *norm1_bias;
    const Weight* qkv;
    const Tensor* qkv_bias;
    const Weight* projection;
    const Tensor *projection_bias, *norm2_weight, *norm2_bias;
    const Weight* fc1;
    const Tensor* fc1_bias;
    const Weight* fc2;
    const Tensor* fc2_bias;
};
struct Weights {
    const Weight* patch;
    const Tensor *patch_bias, *position;
    std::array<Block, C::layers> blocks;
    const Tensor *merger_norm_weight, *merger_norm_bias;
    const Weight* merger_fc1;
    const Tensor* merger_fc1_bias;
    const Weight* merger_fc2;
    const Tensor* merger_fc2_bias;
};

template<class Ops>
void execute(const Layout& l, DeviceSpan backing, std::span<const std::uint16_t> patches,
             const VisionItemControl& item, const Weights& w, Tensor& output, Ops& ops) {
    const auto p = static_cast<std::int32_t>(item.patch_count);
    const auto t = static_cast<std::int32_t>(item.merged_count);
    Tensor positions = l.position_ids.bind(backing), segments = l.cu_seqlens.bind(backing);
    Tensor indices = l.pos_indices.bind(backing), coefficients = l.pos_weights.bind(backing);
    Tensor x = l.x.bind(backing), input = l.patch_bf16.bind(backing);
    ops.copy(item.position_ids.data(), positions);
    ops.copy(item.cu_seqlens.data(), segments);
    ops.copy(item.position_table_indices.data(), indices);
    ops.copy(item.position_table_weights.data(), coefficients);
    ops.copy(patches.data(), input);
    ops.linear(input, *w.patch, x);
    ops.add_bias(*w.patch_bias, x);
    // The artifact's row-major [rows, hidden] table is a zero-copy [hidden, rows] view.
    Tensor table = w.position->reshape({C::hidden, C::position_embeddings});
    ops.position_add(table, indices, coefficients, x);
    for (const auto& block : w.blocks) {
        ops.check_layer();
        Tensor attended = l.attended.bind(backing), qkv = l.qkv.bind(backing);
        Tensor norm = l.attention_norm.bind(backing);
        ops.layer_norm(x, *block.norm1_weight, *block.norm1_bias, norm);
        ops.linear(norm, *block.qkv, qkv);
        ops.add_bias(*block.qkv_bias, qkv);
        const std::size_t plane = C::hidden * sizeof(std::uint16_t);
        Tensor q(qkv.data, DType::BF16, {C::head_dim, C::heads, p});
        Tensor k(static_cast<unsigned char*>(qkv.data) + plane, DType::BF16, {C::head_dim, C::heads, p});
        Tensor v(static_cast<unsigned char*>(qkv.data) + 2 * plane, DType::BF16, {C::head_dim, C::heads, p});
        q.nb[2] = k.nb[2] = v.nb[2] = qkv.nb[1];
        ops.rope(positions, q, k);
        Tensor heads = attended.view({C::head_dim, C::heads, p});
        const DeviceSpan scratch = l.attention_workspace ? l.attention_workspace->bind(backing) : backing;
        ops.attention(q, k, v, segments, scratch, heads);
        Tensor projected = l.projected.bind(backing);
        ops.linear(attended, *block.projection, projected);
        ops.add_bias(*block.projection_bias, projected);
        ops.residual_add(projected, x);
        Tensor up = l.mlp_up.bind(backing), down = l.mlp_down.bind(backing);
        Tensor norm2 = l.mlp_norm.bind(backing);
        ops.layer_norm(x, *block.norm2_weight, *block.norm2_bias, norm2);
        ops.linear(norm2, *block.fc1, up);
        ops.add_bias(*block.fc1_bias, up);
        ops.gelu_tanh(up);
        ops.linear(up, *block.fc2, down);
        ops.add_bias(*block.fc2_bias, down);
        ops.residual_add(down, x);
    }
    Tensor normalized = l.normalized.bind(backing), hidden = l.merger_hidden.bind(backing);
    ops.layer_norm(x, *w.merger_norm_weight, *w.merger_norm_bias, normalized);
    Tensor merged = normalized.view({C::merger_hidden, t});
    ops.linear(merged, *w.merger_fc1, hidden);
    ops.add_bias(*w.merger_fc1_bias, hidden);
    ops.gelu_exact(hidden);
    ops.linear(hidden, *w.merger_fc2, output);
    ops.add_bias(*w.merger_fc2_bias, output);
}
} // namespace ninfer::targets::qwen3_6::vision_schedule
