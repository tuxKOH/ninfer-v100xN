#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/nvfp4/nvfp4_volta_batch.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"
#endif

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule        = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    auto* output          = static_cast<__nv_bfloat16*>(residual.data);
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse,
        Nvfp4AddResidualEpilogue{output, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

#ifdef NINFER_VOLTA_BUILD
namespace {
struct Nvfp4RoundedAddOutput {
    const __nv_bfloat16* residual;
    __nv_bfloat16* output;
    int rows;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        const auto index = static_cast<std::int64_t>(token) * rows + row;
        const float projected = __bfloat162float(__float2bfloat16_rn(value));
        output[index] = __float2bfloat16_rn(projected + __bfloat162float(residual[index]));
    }
};

template <bool AddResidual>
struct Nvfp4PacketOutput {
    const __nv_bfloat16* residual;
    __nv_bfloat16* output;
    unsigned* peer;
    const unsigned* epochs;
    int epoch_pairs;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        const int index = token * 5120 + row;
        auto represented = __float2bfloat16_rn(value);
        if constexpr (AddResidual) {
            represented = __float2bfloat16_rn(
                __bfloat162float(represented) + __bfloat162float(residual[index]));
        }
        output[index] = represented;
        unsigned bits = *reinterpret_cast<const unsigned short*>(&represented);
        if (bits == 0xffffU) { bits = 0x7fc0U; }
        const unsigned high = __shfl_down_sync(0xffffffffU, bits, 1);
        bits |= high << 16;
        // Every QPN output warp owns one contiguous 32-column row, so pairs
        // share the same live warp, packet domain, and aligned 32-bit address.
        if ((row & 1) == 0) {
            const int pair = index / 2;
            auto* address = peer + (epochs[pair / 256] & 1U) * epoch_pairs + pair;
            asm volatile("st.volatile.global.u32 [%0],%1;"
                         : : "l"(address), "r"(bits) : "memory");
        }
    }
};
template <int Rank, bool AddResidual>
struct Nvfp4Packet4Output {
    const __nv_bfloat16* residual;
    __nv_bfloat16* output;
    unsigned* storage[4];
    const unsigned* epochs;
    int channel_pairs;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        const int index = token * 5120 + row;
        auto represented = __float2bfloat16_rn(value);
        if constexpr (AddResidual) {
            represented = __float2bfloat16_rn(
                __bfloat162float(represented) + __bfloat162float(residual[index]));
        }
        output[index] = represented;
        unsigned bits = *reinterpret_cast<const unsigned short*>(&represented);
        if (bits == 0xffffU) bits = 0x7fc0U;
        const unsigned high = __shfl_down_sync(0xffffffffU, bits, 1);
        bits |= high << 16;
        const unsigned b1 = __shfl_down_sync(0xffffffffU, bits, 2);
        const unsigned b2 = __shfl_down_sync(0xffffffffU, bits, 4);
        const unsigned b3 = __shfl_down_sync(0xffffffffU, bits, 6);
        if ((row & 7) == 0) {
            const int pair = index / 2;
            // One vector publishes all eight represented BF16 values.
            const int offset = (epochs[pair / 256] & 1U) * 4 * channel_pairs +
                               Rank * channel_pairs + pair;
#pragma unroll
            for (int destination = 0; destination < 4; ++destination) {
                if (destination != Rank) {
                    auto* address = storage[destination] + offset;
                    asm volatile("st.volatile.global.v4.b32 [%4],{%0,%1,%2,%3};"
                                 : : "r"(bits), "r"(b1), "r"(b2), "r"(b3), "l"(address)
                                 : "memory");
                }
            }
        }
    }
};
} // namespace

void nvfp4_linear_add_qpn_to(const Tensor& x, const Weight& weight, const Tensor& residual,
                             const Tensor& output, WorkspaceArena& workspace, cudaStream_t stream) {
    const Nvfp4RoundedAddOutput epilogue{static_cast<const __nv_bfloat16*>(residual.data),
                                       static_cast<__nv_bfloat16*>(output.data), weight.n};
    if (nvfp4_volta_batch_supported(weight.n, weight.k, x.ne[1])) {
        launch_nvfp4_volta_batch_output(x, weight, epilogue, weight.n, workspace, stream);
    } else if (nvfp4_volta_qpn_precast_supported(weight.n, weight.k, x.ne[1])) {
        launch_nvfp4_volta_qpn_with_precast_output(x, weight, epilogue, weight.n, workspace, stream);
    } else {
        launch_nvfp4_volta_qpn_with_output(x, weight, epilogue, weight.n,
                                          1.0F / weight.weight_scale_divisor, stream);
    }
}
void nvfp4_linear_add_qpn_publish4(const Tensor& x, const Weight& weight,
    const Tensor& residual, const Tensor& output, WorkspaceArena& workspace,
    const std::array<void*, 4>& storage, int rank, std::size_t channel_bytes, cudaStream_t stream) {
    const auto launch = [&]<int Rank>() {
        constexpr bool AddResidual = Rank == 0;
        const Nvfp4Packet4Output<Rank, AddResidual> epilogue{
            static_cast<const __nv_bfloat16*>(residual.data),
            static_cast<__nv_bfloat16*>(output.data),
            {static_cast<unsigned*>(storage[0]), static_cast<unsigned*>(storage[1]),
             static_cast<unsigned*>(storage[2]), static_cast<unsigned*>(storage[3])},
            reinterpret_cast<const unsigned*>(static_cast<const std::uint8_t*>(storage[Rank]) +
                                                8 * channel_bytes),
            static_cast<int>(channel_bytes / sizeof(__nv_bfloat162))};
        if (nvfp4_volta_batch_supported(weight.n, weight.k, x.ne[1])) {
            launch_nvfp4_volta_batch_output(x, weight, epilogue, weight.n, workspace, stream);
        } else if (nvfp4_volta_qpn_precast_supported(weight.n, weight.k, x.ne[1])) {
            launch_nvfp4_volta_qpn_with_precast_output(x, weight, epilogue, weight.n,
                                                       workspace, stream);
        } else {
            launch_nvfp4_volta_qpn_with_output(x, weight, epilogue, weight.n,
                                              1.0F / weight.weight_scale_divisor, stream);
        }
    };
    switch (rank) {
    case 0: launch.template operator()<0>(); break;
    case 1: launch.template operator()<1>(); break;
    case 2: launch.template operator()<2>(); break;
    case 3: launch.template operator()<3>(); break;
    default: throw std::invalid_argument("NVFP4 TP4 packet output: invalid rank");
    }
}

void nvfp4_linear_add_qpn_publish(const Tensor& x, const Weight& weight,
    const Tensor& residual, const Tensor& output, WorkspaceArena& workspace,
    void* local_storage, void* peer_storage, std::size_t epoch_bytes, cudaStream_t stream) {
    auto launch = [&]<bool AddResidual>() {
        const Nvfp4PacketOutput<AddResidual> epilogue{
            static_cast<const __nv_bfloat16*>(residual.data),
            static_cast<__nv_bfloat16*>(output.data), static_cast<unsigned*>(peer_storage),
            reinterpret_cast<const unsigned*>(static_cast<const std::uint8_t*>(local_storage) +
                                                2 * epoch_bytes),
            static_cast<int>(epoch_bytes / sizeof(__nv_bfloat162))};
        if (nvfp4_volta_batch_supported(weight.n, weight.k, x.ne[1])) {
            launch_nvfp4_volta_batch_output(x, weight, epilogue, weight.n, workspace, stream);
        } else if (nvfp4_volta_qpn_precast_supported(weight.n, weight.k, x.ne[1])) {
            launch_nvfp4_volta_qpn_with_precast_output(x, weight, epilogue, weight.n,
                                                       workspace, stream);
        } else {
            launch_nvfp4_volta_qpn_with_output(x, weight, epilogue, weight.n,
                                              1.0F / weight.weight_scale_divisor, stream);
        }
    };
    if (residual.data) { launch.template operator()<true>(); }
    else { launch.template operator()<false>(); }
}

#endif

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream) {
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::Residual6144:
        launch<Nvfp4Residual6144Geometry>(x, weight, residual, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch<Nvfp4Residual17408Geometry>(x, weight, residual, stream);
        return;
    case Nvfp4Problem::Residual6144Tp2Row:
        launch<Nvfp4Residual6144Tp2RowGeometry>(x, weight, residual, stream);
        return;
    case Nvfp4Problem::Residual17408Tp2Row:
        launch<Nvfp4Residual17408Tp2RowGeometry>(x, weight, residual, stream);
        return;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
    case Nvfp4Problem::AttnInputTp2Column:
    case Nvfp4Problem::GdnInputTp2Column:
    case Nvfp4Problem::MlpGateUpTp2Column:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
