#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_batch.cuh"

#include <algorithm>

namespace ninfer::ops::detail {
namespace {

// Narrow column-shard record route. Projection still materializes the same BF16
// values, but its shared column tile also supplies the chronological convolution.
template <class Geometry>
struct Nvfp4GdnConvRecordOutput {
    __nv_bfloat16* record;
    __nv_bfloat16* z;
    const __nv_bfloat16* conv_weight;
    const __nv_bfloat16* state;
    const std::int32_t* initial_slots;
    const std::int32_t* valid_columns;
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* value;
    int width;

    template <int Tokens>
    __device__ __forceinline__ void store(int row, const float (&projected)[Tokens]) const {
        using Sections = Nvfp4GdnInputSections<Geometry>;
        constexpr int QueryRows = Sections::kKeyRows, ValueRows = Sections::kValueRows;
        constexpr int Channels = 2 * QueryRows + ValueRows;
        if (row >= Channels) {
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                if (token < width) {
                    z[token * ValueRows + row - Channels] = __float2bfloat16_rn(projected[token]);
                }
            }
            return;
        }
        const auto base = static_cast<std::int64_t>(initial_slots[0]) * Channels * 3;
        float s0 = __bfloat162float(state[base + row]);
        float s1 = __bfloat162float(state[base + Channels + row]);
        float s2 = __bfloat162float(state[base + 2 * Channels + row]);
        const float w0 = __bfloat162float(conv_weight[row]);
        const float w1 = __bfloat162float(conv_weight[Channels + row]);
        const float w2 = __bfloat162float(conv_weight[2 * Channels + row]);
        const float w3 = __bfloat162float(conv_weight[3 * Channels + row]);
        const int valid = valid_columns == nullptr ? width : min(width, max(0, valid_columns[0]));
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if (token >= width) { break; }
            const auto represented = __float2bfloat16_rn(projected[token]);
            record[token * Channels + row] = represented;
            __nv_bfloat16 output = __float2bfloat16_rn(0.0F);
            if (token < valid) {
                const float p = __bfloat162float(represented);
                float conv = fmaf(w0, s0, 0.0F);
                conv = fmaf(w1, s1, conv);
                conv = fmaf(w2, s2, conv);
                conv = fmaf(w3, p, conv);
                output = __float2bfloat16_rn(silu(conv));
                s0 = s1;
                s1 = s2;
                s2 = p;
            }
            if (row < QueryRows) { query[token * QueryRows + row] = output; }
            else if (row < 2 * QueryRows) {
                key[token * QueryRows + row - QueryRows] = output;
            } else { value[token * ValueRows + row - 2 * QueryRows] = output; }
        }
    }
};

template <class Geometry>
__global__ void split_gdn_output(const __nv_bfloat16* projected,
                                 Nvfp4GdnInputShardOutput<Geometry> output,
                                 std::int64_t count) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int row = static_cast<int>(i % Geometry::kOutputRows);
    const int token = static_cast<int>(i / Geometry::kOutputRows);
    *output.destination(row, token) = projected[i];
}

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
            WorkspaceArena* workspace, cudaStream_t stream) {
    Nvfp4GdnInputShardOutput<Geometry> output{static_cast<__nv_bfloat16*>(qkv.data),
                                              static_cast<__nv_bfloat16*>(z.data)};
    if (workspace != nullptr && weight.layout == QuantLayout::VoltaQpnPrepacked &&
        nvfp4_volta_batch_supported(weight.n, weight.k, x.ne[1])) {
        launch_nvfp4_volta_batch_output(x, weight, output, weight.n, *workspace, stream);
        return;
    }
    if (workspace != nullptr &&
        nvfp4_volta_qpn_precast_supported(weight.n, weight.k, x.ne[1])) {
        launch_nvfp4_volta_qpn_with_precast_output(x, weight, output, weight.n, *workspace, stream);
        return;
    }
    if (x.ne[1] >= 128 && workspace != nullptr) {
        auto scope = workspace->scope();
        auto projected = workspace->alloc(DType::BF16, {weight.n, x.ne[1]});
        nvfp4_cutlass_sm70_launch(x, weight, projected, *workspace, stream);
        const auto count = static_cast<std::int64_t>(weight.n) * x.ne[1];
        split_gdn_output<Geometry><<<static_cast<unsigned>((count + 255) / 256), 256,
                                     0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data), output, count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    for (int begin = 0; begin < x.ne[1]; begin += kNvfp4VoltaQpnMaxTokens) {
        const int active = std::min(kNvfp4VoltaQpnMaxTokens, x.ne[1] - begin);
        auto chunk = x.slice(1, begin, active);
        Nvfp4GdnInputShardOutput<Geometry> chunk_output{
            output.qkv + static_cast<std::int64_t>(begin) * output.kQkvRows,
            output.z + static_cast<std::int64_t>(begin) * output.kZRows};
        launch_nvfp4_volta_qpn_with_output(chunk, weight, chunk_output, weight.n,
                                           1.0F / weight.weight_scale_divisor, stream);
    }
}

template <class Geometry>
void launch_record(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, const Tensor& state,
    const Tensor& valid_columns, const Tensor& initial_slots, Tensor& record, Tensor& query,
    Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    using Output = Nvfp4GdnConvRecordOutput<Geometry>;
    const Output output{
        static_cast<__nv_bfloat16*>(record.data), static_cast<__nv_bfloat16*>(z.data),
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(state.data),
        static_cast<const std::int32_t*>(initial_slots.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data), x.ne[1]};
    auto launch_profile = [&]<class Activation, bool Fp32Scale>(const Activation* input) {
        constexpr int GroupUnroll = Geometry::kOutputRows == 4096 ? 4 : 2;
        nvfp4_volta_qpn_prepacked_kernel<1, 8, 2, Output,
                                        Activation, false, true, true, Fp32Scale, GroupUnroll>
            <<<Geometry::kOutputRows / 32, 256, 0, stream>>>(
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), input, weight.n, weight.k,
                x.ne[1], 1.0F / weight.weight_scale_divisor, output);
    };
    auto launch_record = [&]<class Activation>(const Activation* input) {
        if (weight.volta_fp32_scale) {
            launch_profile.template operator()<Activation, true>(input);
        } else {
            launch_profile.template operator()<Activation, false>(input);
        }
    };
    if (nvfp4_volta_qpn_precast_supported(weight.n, weight.k, x.ne[1])) {
        auto activation = workspace.alloc(DType::FP16, {x.ne[0], x.ne[1]});
        nvfp4_volta_qpn_cast_activation(x, activation, stream);
        launch_record(static_cast<const half*>(activation.data));
    } else {
        launch_record(static_cast<const __nv_bfloat16*>(x.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_gdn_input_sm70_record_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, const Tensor& state,
    const Tensor& valid_columns, const Tensor& initial_slots, Tensor& record, Tensor& query,
    Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    if (weight.n == Nvfp4GdnInputTp4ColumnGeometry::kOutputRows) {
        launch_record<Nvfp4GdnInputTp4ColumnGeometry>(x, weight, conv_weight, state,
            valid_columns, initial_slots, record, query, key, value, z, workspace, stream);
    } else {
        launch_record<Nvfp4GdnInputTp2ColumnGeometry>(x, weight, conv_weight, state,
            valid_columns, initial_slots, record, query, key, value, z, workspace, stream);
    }
}

std::size_t nvfp4_gdn_input_sm70_workspace_bytes(int rows, int tokens) {
    if (tokens < 128) {
        return std::max(nvfp4_volta_qpn_precast_workspace_bytes(rows, 5120, tokens),
                        nvfp4_volta_batch_workspace_bytes(rows, 5120, tokens));
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {rows, tokens});
    (void)layout.alloc_bytes(nvfp4_cutlass_sm70_workspace_bytes(rows, 5120, tokens));
    return layout.peak_bytes(1);
}

void nvfp4_gdn_input_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                Tensor& z, WorkspaceArena* workspace, cudaStream_t stream) {
    if (weight.n == Nvfp4GdnInputGeometry::kOutputRows) {
        launch<Nvfp4GdnInputGeometry>(x, weight, qkv, z, workspace, stream);
    } else if (weight.n == Nvfp4GdnInputTp4ColumnGeometry::kOutputRows) {
        launch<Nvfp4GdnInputTp4ColumnGeometry>(x, weight, qkv, z, workspace, stream);
    } else {
        launch<Nvfp4GdnInputTp2ColumnGeometry>(x, weight, qkv, z, workspace, stream);
    }
}

} // namespace ninfer::ops::detail
