#include "ops/attn_input_proj/volta_tp4.h"

#include "core/layout.h"
#include "ninfer/ops/mtp_pack.h"
#include "ops/linear/fp8/fp8_cutlass_sm70.h"
#include "ops/linear/fp8/fp8_block.h"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"
#include "ops/linear/nvfp4/nvfp4_volta_batch.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kRows = 3584;
constexpr int kQueryRows = 1536;
constexpr int kKvRows = 256;
constexpr int kWideTokens = 128;

struct HeadLocalOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ void store(int row, int token, float result) const {
        __nv_bfloat16* destination;
        if (row < 1536) { destination = query + token * kQueryRows + row; }
        else if (row < 1792) { destination = key + token * kKvRows + row - 1536; }
        else if (row < 3328) { destination = gate + token * kQueryRows + row - 1792; }
        else { destination = value + token * kKvRows + row - 3328; }
        *destination = __float2bfloat16_rn(result);
    }
};
} // namespace

std::size_t attn_input_volta_tp4_workspace_bytes(QType type, std::int32_t tokens) {
    if (tokens <= 0 || (type != QType::NVFP4 && type != QType::FP8_E4M3FN_ROW_BF16S &&
                       type != QType::FP8_E4M3FN_BLOCK128_BF16S)) {
        throw std::invalid_argument("SM70 TP4 attention: invalid profile");
    }
    if (tokens < kWideTokens) {
        return type == QType::NVFP4
            ? std::max(nvfp4_volta_qpn_precast_workspace_bytes(kRows, 5120, tokens),
                       nvfp4_volta_batch_workspace_bytes(kRows, 5120, tokens)) : 0;
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {kRows, tokens}, 256);
    const auto bytes = type == QType::NVFP4
        ? nvfp4_cutlass_sm70_workspace_bytes(kRows, 5120, tokens)
        : fp8_cutlass_sm70_workspace_bytes(kRows, 5120, tokens);
    if (bytes) { (void)layout.alloc_bytes(bytes, 256); }
    return layout.peak_bytes(1);
}

void attn_input_volta_tp4_launch(const Tensor& x, const Weight& weight, Tensor& query,
                                Tensor& gate, Tensor& key, Tensor& value,
                                WorkspaceArena* workspace, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens >= kWideTokens) {
        if (!workspace) { throw std::invalid_argument("SM70 TP4 attention requires workspace"); }
        auto scope = workspace->scope();
        Tensor projected = workspace->alloc(DType::BF16, {kRows, tokens}, 256);
        if (weight.qtype == QType::NVFP4) {
            nvfp4_cutlass_sm70_launch(x, weight, projected, *workspace, stream);
        } else {
            fp8_cutlass_sm70_launch(x, weight, projected, *workspace, stream);
        }
        Tensor q = query.view({256, 6, tokens});
        Tensor k = key.view({256, 1, tokens});
        Tensor g = gate.view({256, 6, tokens});
        Tensor v = value.view({256, 1, tokens});
        mtp_split_attn_in(projected, q, k, g, v, stream);
        return;
    }
    if (weight.qtype == QType::NVFP4 && weight.layout == QuantLayout::VoltaQpnPrepacked &&
        nvfp4_volta_batch_supported(kRows, 5120, tokens)) {
        if (!workspace) { throw std::invalid_argument("SM70 TP4 batch attention requires workspace"); }
        const HeadLocalOutput output{static_cast<__nv_bfloat16*>(query.data),
            static_cast<__nv_bfloat16*>(key.data), static_cast<__nv_bfloat16*>(gate.data),
            static_cast<__nv_bfloat16*>(value.data)};
        launch_nvfp4_volta_batch_output(x, weight, output, kRows, *workspace, stream);
        return;
    }
    // One pass over each small-T weight tile, with a fused head-local store. No BF16
    // intermediate or separate section-copy kernels, and no private device allocation.
    for (int begin = 0; begin < tokens; begin += 32) {
        const int width = std::min(32, tokens - begin);
        const Tensor input = x.slice(1, begin, width);
        const HeadLocalOutput output{
            static_cast<__nv_bfloat16*>(query.data) + begin * kQueryRows,
            static_cast<__nv_bfloat16*>(key.data) + begin * kKvRows,
            static_cast<__nv_bfloat16*>(gate.data) + begin * kQueryRows,
            static_cast<__nv_bfloat16*>(value.data) + begin * kKvRows};
        if (weight.qtype == QType::NVFP4) {
            if (workspace && nvfp4_volta_qpn_precast_supported(kRows, 5120, width)) {
                launch_nvfp4_volta_qpn_with_precast_output(
                    input, weight, output, kRows, *workspace, stream);
            } else {
                launch_nvfp4_volta_qpn_with_output(input, weight, output, kRows,
                                                   1.0F / weight.weight_scale_divisor, stream);
            }
        } else if (weight.qtype == QType::FP8_E4M3FN_BLOCK128_BF16S) {
            launch_fp8_volta_qpn_with_output<HeadLocalOutput, true>(input, weight, output, kRows, stream);
        } else {
            launch_fp8_volta_qpn_with_output(input, weight, output, kRows, stream);
        }
    }
}
} // namespace ninfer::ops::detail
