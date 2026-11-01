#pragma once

// Batch activation supply / phased K reduction inspired by 1CatAI/1Cat-vLLM,
// csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu (Apache-2.0). NInfer retains its
// original QPN weight image, normal/fast scale profiles and BF16 Op boundaries.
// The QPN layout/MMA mapping derives from dnv2003/v100-skinny (MIT);
// see third_party/licenses/v100-skinny.txt.
#include "core/arena.h"
#include "core/device.h"
#include "ops/common/volta_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail {

Tensor nvfp4_volta_batch_pack(const Tensor& input, WorkspaceArena& workspace,
                             cudaStream_t stream);

inline int nvfp4_volta_batch_tiles(int tokens) {
    return tokens <= 16 ? 2 : tokens <= 32 ? 4 : tokens <= 40 ? 5 : 8;
}

// Keep eight independent logical K partitions, with the same per-partition MMA
// chain and ascending FP32 reduction as the existing T9..32 QPN route. Physical
// phases reduce shared memory and thread count without reassociating that sum.
// K-major, zero-padded FP16 activation supply removes repeated BF16 conversions
// from every weight CTA. No model weight is expanded or copied.
template <int Tiles, int PhysicalWarps, bool Paired, bool Fp32Scale, class Output>
__global__ __launch_bounds__(PhysicalWarps * 32 * (Paired ? 2 : 1), 2)
void nvfp4_volta_batch_kernel(const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ scales,
                              const half* __restrict__ input, int n, int k, int tokens,
                              int padded, float inverse_divisor, Output output) {
    constexpr int kSplits = 8;
    constexpr int kElements = Tiles * 8 * 32;
    constexpr int kProjections = Paired ? 2 : 1;
    // The five-tile paired verifier needs a fourth physical K warp without
    // exceeding the 48 KiB block limit. Its output owners keep the inter-phase
    // carry in registers; other shapes retain the qualified shared carry.
    constexpr bool kRegisterCarry = Paired && Tiles == 5;
    constexpr int kThreads = PhysicalWarps * 32 * kProjections;
    constexpr int kSlots = (kElements + kThreads - 1) / kThreads;
    __shared__ float partial[kProjections][PhysicalWarps + (kRegisterCarry ? 0 : 1)][kElements];
    float carry[kRegisterCarry ? kSlots : 1] = {};
    float carry_up[kRegisterCarry ? kSlots : 1] = {};
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int projection = warp / PhysicalWarps;
    const int physical = warp % PhysicalWarps;
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups = k / 16;
    const int quotient = groups / kSplits;
    const auto base = (static_cast<std::int64_t>(blockIdx.x) + projection * (n / 32)) *
                      groups * 32 + lane;
    const half2 rebias = __float2half2_rn(16384.0F);
    const half2 divisor = __float2half2_rn(inverse_divisor * 256.0F);
    const float epilogue_scale = Fp32Scale ? 256.0F * inverse_divisor : 1.0F;

#pragma unroll
    for (int phase = 0; phase < kSplits / PhysicalWarps; ++phase) {
        float accum[Tiles][8] = {};
        const int logical = physical + phase * PhysicalWarps;
        const int end = logical == kSplits - 1 ? groups : (logical + 1) * quotient;
#pragma unroll 2
        for (int group = logical * quotient; group < end; ++group) {
            const auto index = base + static_cast<std::int64_t>(group) * 32;
            const uint2 q = __ldg(reinterpret_cast<const uint2*>(codes) + index);
            const std::uint8_t scale_byte = scales[index];
            const std::uint16_t scale_bits = ((scale_byte & 0x80U) << 8) |
                                             ((scale_byte & 0x7FU) << 7);
            const half2 scale = __hmul2(__half2half2(__ushort_as_half(scale_bits)),
                                        Fp32Scale ? rebias : divisor);
            half2 weights[8];
            const std::uint32_t words[2] = {q.x, q.y};
#pragma unroll
            for (int word_index = 0; word_index < 2; ++word_index) {
                const std::uint32_t word = words[word_index];
                const std::uint32_t bits[4] = {
                    ((word << 12) & 0x80008000U) | ((word << 9) & 0x0E000E00U),
                    ((word << 8) & 0x80008000U) | ((word << 5) & 0x0E000E00U),
                    ((word << 4) & 0x80008000U) | ((word << 1) & 0x0E000E00U),
                    (word & 0x80008000U) | ((word >> 3) & 0x0E000E00U)};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    half2 value = *reinterpret_cast<const half2*>(&bits[j]);
                    if constexpr (!Fp32Scale) { value = __hmul2(value, rebias); }
                    weights[word_index * 4 + j] = __hmul2(value, scale);
                }
            }
            const auto* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
            for (int tile = 0; tile < Tiles; ++tile) {
                const half* a = input +
                    (static_cast<std::int64_t>(group) * padded + tile * 8 + r) * 16;
                const uint4 first = *reinterpret_cast<const uint4*>(a);
                const uint4 second = *reinterpret_cast<const uint4*>(a + 8);
                const auto* a0 = reinterpret_cast<const unsigned*>(&first);
                const auto* a1 = reinterpret_cast<const unsigned*>(&second);
                volta_mma_qp_n(accum[tile], a0[0], a0[1], b[0], b[1]);
                volta_mma_qp_n(accum[tile], a0[2], a0[3], b[2], b[3]);
                volta_mma_qp_n(accum[tile], a1[0], a1[1], b[4], b[5]);
                volta_mma_qp_n(accum[tile], a1[2], a1[3], b[6], b[7]);
            }
        }
#pragma unroll
        for (int tile = 0; tile < Tiles; ++tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partial[projection][physical][row * 32 + qp * 8 + col] = accum[tile][i];
            }
        }
        __syncthreads();
        if constexpr (kRegisterCarry) {
#pragma unroll
            for (int slot = 0; slot < kSlots; ++slot) {
                const int e = threadIdx.x + slot * kThreads;
                if (e >= kElements) { continue; }
                float value = carry[slot];
                float up = carry_up[slot];
#pragma unroll
                for (int w = 0; w < PhysicalWarps; ++w) {
                    value += partial[0][w][e];
                    up += partial[1][w][e];
                }
                if (phase + 1 < kSplits / PhysicalWarps) {
                    carry[slot] = value;
                    carry_up[slot] = up;
                } else if (e / 32 < tokens) {
                    output.store(blockIdx.x * 32 + e % 32, e / 32,
                                 value * epilogue_scale, up * epilogue_scale);
                }
            }
        } else {
            for (int e = threadIdx.x; e < kElements; e += blockDim.x) {
                float value = phase > 0 ? partial[0][PhysicalWarps][e] : 0.0F;
                float up = 0.0F;
                if constexpr (Paired) {
                    up = phase > 0 ? partial[1][PhysicalWarps][e] : 0.0F;
                }
#pragma unroll
                for (int w = 0; w < PhysicalWarps; ++w) {
                    value += partial[0][w][e];
                    if constexpr (Paired) { up += partial[1][w][e]; }
                }
                if (phase + 1 < kSplits / PhysicalWarps) {
                    partial[0][PhysicalWarps][e] = value;
                    if constexpr (Paired) { partial[1][PhysicalWarps][e] = up; }
                } else if (e / 32 < tokens) {
                    const int channel = blockIdx.x * 32 + e % 32;
                    if constexpr (Paired) {
                        output.store(channel, e / 32, value * epilogue_scale, up * epilogue_scale);
                    } else {
                        output.store(channel, e / 32, value * epilogue_scale);
                    }
                }
            }
        }
        // Partial planes may be reused only after every output owner has read them.
        __syncthreads();
    }
}

template <bool Paired = false, class Output>
void launch_nvfp4_volta_batch_output(const Tensor& input, const Weight& weight, Output output,
                                     int output_rows, WorkspaceArena& workspace,
                                     cudaStream_t stream) {
    auto scope = workspace.scope();
    const Tensor packed = nvfp4_volta_batch_pack(input, workspace, stream);
    const int tokens = input.ne[1];
    const int k = input.ne[0];
    const int padded = packed.ne[1];
    const auto launch = [&]<bool Fp32Scale>() {
        const auto issue = [&]<int Tiles, int PhysicalWarps>() {
            nvfp4_volta_batch_kernel<Tiles, PhysicalWarps, Paired, Fp32Scale>
                <<<output_rows / 32, PhysicalWarps * 32 * (Paired ? 2 : 1), 0, stream>>>(
                    static_cast<const std::uint8_t*>(weight.qdata),
                    static_cast<const std::uint8_t*>(weight.scales),
                    static_cast<const half*>(packed.data), output_rows, k, tokens, padded,
                    1.0F / weight.weight_scale_divisor, output);
        };
        // The two row-parallel contractions have only 160 output CTAs. Eight
        // physical K warps fill their SMs better without changing logical sums.
        if (tokens <= 16) {
            if constexpr (Paired) { issue.template operator()<2, 8>(); }
            else if (output_rows == 8192) { issue.template operator()<2, 4>(); }
            else { issue.template operator()<2, 8>(); }
        } else if (tokens <= 32) {
            if constexpr (Paired) { issue.template operator()<4, 4>(); }
            else if (output_rows == 5120) { issue.template operator()<4, 8>(); }
            else { issue.template operator()<4, 4>(); }
        } else if (tokens <= 40) {
            if constexpr (Paired) { issue.template operator()<5, 4>(); }
            else if (output_rows == 7168) { issue.template operator()<5, 4>(); }
            else { issue.template operator()<5, 8>(); }
        }
        else { issue.template operator()<8, Paired ? 2 : 4>(); }
    };
    if (weight.volta_fp32_scale) { launch.template operator()<true>(); }
    else { launch.template operator()<false>(); }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
