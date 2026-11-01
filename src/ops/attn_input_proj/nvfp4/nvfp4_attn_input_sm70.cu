#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_volta_batch.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <algorithm>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
struct AttentionOutput {
    using Sections = Nvfp4AttnInputSections<Geometry>;
    static constexpr int Q = Sections::kQueryRows;
    static constexpr int K = Sections::kKeyRows;
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ void store(int row, int token, float result) const {
        const auto offset = static_cast<std::int64_t>(token);
        const auto rounded = __float2bfloat16_rn(result);
        if (row < Q) { query[offset * Q + row] = rounded; }
        else if (row < Q + K) { key[offset * K + row - Q] = rounded; }
        else if (row < 2 * Q + K) { gate[offset * Q + row - Q - K] = rounded; }
        else { value[offset * K + row - 2 * Q - K] = rounded; }
    }
};

template <class Geometry>
__global__ void split_attention_output(const __nv_bfloat16* projected,
                                       AttentionOutput<Geometry> output, std::int64_t count) {
    // Copy already-rounded BF16 values, rather than creating a second precision boundary.
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    using O = AttentionOutput<Geometry>;
    const int row = static_cast<int>(i % Geometry::kOutputRows);
    const auto token = i / Geometry::kOutputRows;
    if (row < O::Q) { output.query[token * O::Q + row] = projected[i]; }
    else if (row < O::Q + O::K) { output.key[token * O::K + row - O::Q] = projected[i]; }
    else if (row < 2 * O::Q + O::K) {
        output.gate[token * O::Q + row - O::Q - O::K] = projected[i];
    } else { output.value[token * O::K + row - 2 * O::Q - O::K] = projected[i]; }
}

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
            Tensor& v, WorkspaceArena* workspace, cudaStream_t stream) {
    AttentionOutput<Geometry> output{static_cast<__nv_bfloat16*>(q.data),
                                      static_cast<__nv_bfloat16*>(k.data),
                                      static_cast<__nv_bfloat16*>(gate.data),
                                      static_cast<__nv_bfloat16*>(v.data)};
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
        split_attention_output<Geometry><<<static_cast<unsigned>((count + 255) / 256), 256,
                                           0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data), output, count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // QPN writes directly to the four final planes, including all MTP verifier widths.
    for (int begin = 0; begin < x.ne[1]; begin += kNvfp4VoltaQpnMaxTokens) {
        const int active = std::min(kNvfp4VoltaQpnMaxTokens, x.ne[1] - begin);
        auto chunk = x.slice(1, begin, active);
        AttentionOutput<Geometry> chunk_output{
            output.query + static_cast<std::int64_t>(begin) * output.Q,
            output.key + static_cast<std::int64_t>(begin) * output.K,
            output.gate + static_cast<std::int64_t>(begin) * output.Q,
            output.value + static_cast<std::int64_t>(begin) * output.K};
        launch_nvfp4_volta_qpn_with_output(chunk, weight, chunk_output, weight.n,
                                           1.0F / weight.weight_scale_divisor, stream);
    }
}

} // namespace

std::size_t nvfp4_attn_input_sm70_workspace_bytes(int rows, int tokens) {
    if (tokens < 128) {
        return std::max(nvfp4_volta_qpn_precast_workspace_bytes(rows, 5120, tokens),
                        nvfp4_volta_batch_workspace_bytes(rows, 5120, tokens));
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {rows, tokens});
    (void)layout.alloc_bytes(nvfp4_cutlass_sm70_workspace_bytes(rows, 5120, tokens));
    return layout.peak_bytes(1);
}

void nvfp4_attn_input_sm70_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                 Tensor& gate, Tensor& k, Tensor& v,
                                 WorkspaceArena* workspace, cudaStream_t stream) {
    if (weight.n == Nvfp4AttnInputGeometry::kOutputRows) {
        launch<Nvfp4AttnInputGeometry>(x, weight, q, gate, k, v, workspace, stream);
    } else {
        launch<Nvfp4AttnInputTp2ColumnGeometry>(x, weight, q, gate, k, v, workspace, stream);
    }
}

} // namespace ninfer::ops::detail
