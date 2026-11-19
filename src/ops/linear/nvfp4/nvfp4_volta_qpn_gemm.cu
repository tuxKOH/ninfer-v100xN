#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>
#include <algorithm>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Nvfp4VoltaQpnSchedule;
static_assert(kNvfp4VoltaQpnRowsPerTile == S::kRowsPerTile,
              "dispatch mirrors the tile height for host code");

__global__ void qpn_cast_activation(const __nv_bfloat16* input, half* output, int count) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { output[index] = __float2half_rn(__bfloat162float(input[index])); }
}
} // namespace

bool nvfp4_volta_qpn_precast_supported(std::int32_t n, std::int32_t k,
                                      std::int32_t tokens) noexcept {
    const bool gdn_output = n == 5120 && k == 3072 && tokens >= 1 && tokens <= 8;
    return gdn_output || ((tokens >= 4 && tokens <= 6) &&
           ((k == 5120 && (n == 3584 || n == 4096 || n == 7168 || n == 8192 ||
                           n == 14336 || n == 16384)) ||
            (n == 5120 && (k == 1536 || k == 4352 || k == 8704 || k == 6144 || k == 17408))));
}

std::size_t nvfp4_volta_qpn_precast_workspace_bytes(std::int32_t n, std::int32_t k,
                                                   std::int32_t max_tokens) {
    const bool gdn_output = n == 5120 && k == 3072;
    const int first = gdn_output ? 1 : 4;
    const int last = gdn_output ? 8 : 6;
    if (max_tokens < first || !nvfp4_volta_qpn_precast_supported(n, k, first)) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP16, {k, std::min(max_tokens, last)});
    return layout.peak_bytes(1);
}

void nvfp4_volta_qpn_cast_activation(const Tensor& input, Tensor& output, cudaStream_t stream) {
    const int count = static_cast<int>(input.numel());
    qpn_cast_activation<<<(count + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<half*>(output.data), count);
    CUDA_CHECK(cudaGetLastError());
}

void launch_nvfp4_volta_qpn_precast(const Tensor& x, const Weight& w, Tensor& out,
                                    WorkspaceArena& workspace, cudaStream_t stream) {
    if (w.layout == QuantLayout::VoltaQpnPrepacked && w.n == 5120 &&
        (w.k == 6144 || w.k == 17408) && x.ne[1] >= 4 && x.ne[1] <= 6) {
        auto scope = workspace.scope();
        auto image = workspace.alloc(DType::FP16, {w.k, x.ne[1]});
        nvfp4_volta_qpn_cast_activation(x, image, stream);
        const auto launch = [&]<int Split, bool Fast, int Unroll>() {
            nvfp4_volta_qpn_prepacked_kernel<1, Split, 2, Nvfp4ContiguousOutput, half,
                                             false, true, false, Fast, Unroll>
                <<<w.n / 32, Split * 32, 0, stream>>>(
                    static_cast<const std::uint8_t*>(w.qdata),
                    static_cast<const std::uint8_t*>(w.scales),
                    static_cast<const half*>(image.data), w.n, w.k, x.ne[1],
                    1.0F / w.weight_scale_divisor,
                    {static_cast<__nv_bfloat16*>(out.data), w.n});
        };
        if (w.volta_fp32_scale) { launch.template operator()<16, true, 4>(); }
        else { launch.template operator()<8, false, 8>(); }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    launch_nvfp4_volta_qpn_with_precast_output(
        x, w, Nvfp4ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), out.ne[0]}, out.ne[0],
        workspace, stream);
}

bool nvfp4_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // The scale plane's BlockScaleK16M128x4 swizzle tiles on 128 output rows.
    if (n % 128 != 0) { return false; }
    // A lane consumes whole 64-k chunks (four NVFP4 groups, matching the swizzle's contiguous
    // scale-byte run), and the CTA's SPLITK warps split K by whole chunks. SPLITK=8 is the
    // production floor across every kTiles bucket (see launch_nvfp4_volta_qpn_with_output).
    if (k % S::kChunkK != 0) { return false; }
    if (k / S::kChunkK < 8) { return false; }
    // Prepacked runtime weights also use QPN2 at T=1; checkpoint-native weights keep the existing
    // SIMT decode route through the semantic-op plan.
    return t >= 1 && t <= 4 * S::kRowsPerTile;
}

void launch_nvfp4_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n                   = out.ne[0];
    const float inverse_weight_divisor = 1.0F / w.weight_scale_divisor;
    launch_nvfp4_volta_qpn_with_output(
        x, w, Nvfp4ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), n}, n,
        inverse_weight_divisor, stream);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
