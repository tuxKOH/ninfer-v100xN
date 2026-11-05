#pragma once

// Implements the sum stage of include/ninfer/ops/allreduce.h. SM70 packets use
// qualified device P2P or WSL2 mapped host memory; direct peer-read variants
// keep both operands immutable until the caller's cross-device completion edge.
#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops {

// A small metadata image is published by one origin kernel after joining every
// destination's prior consumer. Completion is joined before any source/output reuse.
__global__ void broadcast_i32_tp4_kernel(const int* source, int* first, int* second,
                                         int* third, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int value = source[i];
    first[i] = value;
    second[i] = value;
    third[i] = value;
    __threadfence_system();
}

__global__ void allreduce_argmax_tp2_kernel(const float* first_values, const float* second_values,
    const int* first_indices, const int* second_indices, int* first_output, int* second_output,
    int columns, int shard_rows, const int* id_map) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= columns) { return; }
    const float a = first_values[t], b = second_values[t];
    const int ia = first_indices[t], ib = second_indices[t] + shard_rows;
    const int winner = b > a || (b == a && ib < ia) ? ib : ia;
    const int token = id_map == nullptr ? winner : id_map[winner];
    first_output[t] = token;
    second_output[t] = token;
    __threadfence_system();
}

// Rank zero reads immutable peer shards and publishes the selected ID to all ranks.
// The caller joins each producer and orders every next source/output reuse after completion.
__global__ void allreduce_argmax_tp4_kernel(
    const float* v0, const float* v1, const float* v2, const float* v3,
    const int* i0, const int* i1, const int* i2, const int* i3,
    int* o0, int* o1, int* o2, int* o3, int columns, int shard_rows, const int* id_map) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= columns) { return; }
    float best = v0[t];
    int winner = i0[t];
    const float scores[3] = {v1[t], v2[t], v3[t]};
    const int indices[3] = {i1[t] + shard_rows, i2[t] + 2 * shard_rows,
                            i3[t] + 3 * shard_rows};
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        if (scores[r] > best || (scores[r] == best && indices[r] < winner)) {
            best = scores[r];
            winner = indices[r];
        }
    }
    const int token = id_map == nullptr ? winner : id_map[winner];
    o0[t] = token;
    o1[t] = token;
    o2[t] = token;
    o3[t] = token;
    __threadfence_system();
}

__device__ __forceinline__ Bf16x8Pack allreduce_packet_load(const Bf16x8Pack* address) {
    uint4 bits;
    asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3},[%4];"
                 : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                 : "l"(address));
    return *reinterpret_cast<const Bf16x8Pack*>(&bits);
}

__device__ __forceinline__ void allreduce_packet_store(Bf16x8Pack* address,
                                                       const Bf16x8Pack& value) {
    const auto& bits = *reinterpret_cast<const uint4*>(&value);
    asm volatile("st.volatile.global.v4.b32 [%4],{%0,%1,%2,%3};"
                 : : "r"(bits.x), "r"(bits.y), "r"(bits.z), "r"(bits.w), "l"(address)
                 : "memory");
}

// TP2 specialization of 1Cat's two-epoch push mechanism: retain the local
// contribution in registers and send only the remote copy. A reserved BF16 NaN
// marks each empty lane; every finite bit pattern (including signed zero) passes
// unchanged. Incoming NaNs remain NaNs. Each offset has one producer/consumer.
// Completing epoch e requires the peer to have produced e, so neither rank can
// recycle e+2 before the peer has consumed/cleared e. The pitch and block-to-offset
// mapping are fixed across calls, including different message sizes.
template <bool HostMapped = false>
__global__ void allreduce_packet_sum_kernel(const Bf16x8Pack* source, Bf16x8Pack* output,
                                            Bf16x8Pack* local, Bf16x8Pack* peer,
                                            unsigned* epochs, int packs, int epoch_packs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned epoch = epochs[blockIdx.x] & 1U;
    if (i < packs) {
        auto a = load_vec<Bf16x8Pack>(source + i);
        Bf16x8Pack empty;
        auto* a_bits = reinterpret_cast<std::uint16_t*>(&a);
        auto* empty_bits = reinterpret_cast<std::uint16_t*>(&empty);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (a_bits[j] == 0xffffU) { a_bits[j] = 0x7fc0U; }
            empty_bits[j] = 0xffffU;
        }
        const auto offset = epoch * epoch_packs + i;
        allreduce_packet_store(peer + offset, a);
        // Mapped RAM has two GPU writers and no CPU participant. Complete our
        // posted system writes before polling, and retire the empty marker
        // before advancing this block's epoch. Device P2P keeps its qualified path.
        if constexpr (HostMapped) { __threadfence_system(); }
        Bf16x8Pack b;
        unsigned long long started = 0;
        if constexpr (HostMapped) { started = clock64(); }
        while (true) {
            b = allreduce_packet_load(local + offset);
            const auto* bits = reinterpret_cast<const std::uint16_t*>(&b);
            bool missing = false;
#pragma unroll
            for (int j = 0; j < 8; ++j) { missing |= bits[j] == 0xffffU; }
            if (!missing) { break; }
            // Fail a lost peer rather than leaving a GPU kernel polling forever.
            if constexpr (HostMapped) {
                if (clock64() - started > 1000000000ULL) { __trap(); }
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            a.pair[j] = __floats2bfloat162_rn(__low2float(a.pair[j]) + __low2float(b.pair[j]),
                                            __high2float(a.pair[j]) + __high2float(b.pair[j]));
        }
        store_vec(output + i, a);
        allreduce_packet_store(local + offset, empty);
        if constexpr (HostMapped) { __threadfence_system(); }
    }
    __syncthreads();
    if (threadIdx.x == 0) { epochs[blockIdx.x] = epoch ^ 1U; }
}

// Adapted from 1Cat's TP4 two-epoch push. Each destination owns a mailbox
// channel per source rank. Keep our own input in registers, publish to the
// other three ranks, then reduce in global rank order with one BF16 rounding.
// Offsets and epoch domains are independent of message size. A rank cannot
// produce e+2 until every peer has produced e+1, which in turn requires those
// peers to have consumed/cleared e. No peer reads the source being overwritten.
template <int Rank, int Source>
__device__ __forceinline__ Bf16x8Pack allreduce_packet_sum4_receive(
    Bf16x8Pack value, const Bf16x8Pack* local_storage, int base, int channel_packs) {
    if constexpr (Rank == Source) {
        return value;
    } else {
        Bf16x8Pack received;
        const auto* address = local_storage + base + Source * channel_packs;
        while (true) {
            received = allreduce_packet_load(address);
            const auto* bits = reinterpret_cast<const std::uint16_t*>(&received);
            bool missing = false;
#pragma unroll
            for (int j = 0; j < 8; ++j) { missing |= bits[j] == 0xffffU; }
            if (!missing) { break; }
        }
        return received;
    }
}

inline constexpr int kAllreduceTp4PacketThreads = 64;

template<int Rank>
__device__ __forceinline__ void allreduce_packet_sum4_publish(
    Bf16x8Pack value, Bf16x8Pack* storage0, Bf16x8Pack* storage1,
    Bf16x8Pack* storage2, Bf16x8Pack* storage3, int i, unsigned epoch, int channel_packs) {
    auto* bits = reinterpret_cast<std::uint16_t*>(&value);
#pragma unroll
    for (int j = 0; j < 8; ++j) if (bits[j] == 0xffffU) bits[j] = 0x7fc0U;
    const int base = epoch * 4 * channel_packs + i;
    if constexpr (Rank != 0) {
        allreduce_packet_store(storage0 + base + Rank * channel_packs, value);
    }
    if constexpr (Rank != 1) {
        allreduce_packet_store(storage1 + base + Rank * channel_packs, value);
    }
    if constexpr (Rank != 2) {
        allreduce_packet_store(storage2 + base + Rank * channel_packs, value);
    }
    if constexpr (Rank != 3) {
        allreduce_packet_store(storage3 + base + Rank * channel_packs, value);
    }
}

template <int Rank, bool Published = false>
__device__ __forceinline__ Bf16x8Pack allreduce_packet_sum4_value(
    const Bf16x8Pack* buffer, Bf16x8Pack* storage0, Bf16x8Pack* storage1,
    Bf16x8Pack* storage2, Bf16x8Pack* storage3, Bf16x8Pack* local_storage,
    int i, unsigned epoch, int channel_packs) {

    Bf16x8Pack empty;
    auto value = load_vec<Bf16x8Pack>(buffer + i);
    auto* bits = reinterpret_cast<std::uint16_t*>(&value);
    auto* empty_bits = reinterpret_cast<std::uint16_t*>(&empty);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (bits[j] == 0xffffU) { bits[j] = 0x7fc0U; }
        empty_bits[j] = 0xffffU;
    }
    const int base = epoch * 4 * channel_packs + i;
    if constexpr (!Published) {
        allreduce_packet_sum4_publish<Rank>(value, storage0, storage1, storage2, storage3,
                                          i, epoch, channel_packs);
    }
    // Consume ranks in the public 0,1,2,3 order while retaining only one
    // incoming pack. This is the same FP32 association as the four-pack
    // reduction, with materially lower register pressure on SM70.
    const Bf16x8Pack first =
        allreduce_packet_sum4_receive<Rank, 0>(value, local_storage, base, channel_packs);
    float sums[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sums[2 * j] = __low2float(first.pair[j]);
        sums[2 * j + 1] = __high2float(first.pair[j]);
    }
    const auto add_source = [&](const Bf16x8Pack next) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sums[2 * j] = __fadd_rn(sums[2 * j], __low2float(next.pair[j]));
            sums[2 * j + 1] = __fadd_rn(sums[2 * j + 1], __high2float(next.pair[j]));
        }
    };
    add_source(allreduce_packet_sum4_receive<Rank, 1>(value, local_storage, base,
                                                       channel_packs));
    add_source(allreduce_packet_sum4_receive<Rank, 2>(value, local_storage, base,
                                                       channel_packs));
    add_source(allreduce_packet_sum4_receive<Rank, 3>(value, local_storage, base,
                                                       channel_packs));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        value.pair[j] = __floats2bfloat162_rn(sums[2 * j], sums[2 * j + 1]);
    }
    if constexpr (Rank != 0) {
        allreduce_packet_store(local_storage + base, empty);
    }
    if constexpr (Rank != 1) {
        allreduce_packet_store(local_storage + base + channel_packs, empty);
    }
    if constexpr (Rank != 2) {
        allreduce_packet_store(local_storage + base + 2 * channel_packs, empty);
    }
    if constexpr (Rank != 3) {
        allreduce_packet_store(local_storage + base + 3 * channel_packs, empty);
    }
    return value;
}

template <int Rank, bool Published = false>
__global__ __launch_bounds__(kAllreduceTp4PacketThreads, 2) void allreduce_packet_sum4_kernel(Bf16x8Pack* buffer,
    Bf16x8Pack* storage0, Bf16x8Pack* storage1, Bf16x8Pack* storage2,
    Bf16x8Pack* storage3, unsigned* epochs, int packs, int channel_packs) {
    static_assert(Rank >= 0 && Rank < 4);
    Bf16x8Pack* const local_storage = Rank == 0 ? storage0 :
                                      Rank == 1 ? storage1 :
                                      Rank == 2 ? storage2 : storage3;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned epoch = epochs[blockIdx.x] & 1U;
    if (i < packs) {
        const auto value = allreduce_packet_sum4_value<Rank, Published>(buffer, storage0, storage1,
            storage2, storage3, local_storage, i, epoch, channel_packs);
        store_vec(buffer + i, value);
    }
    __syncthreads();
    if (threadIdx.x == 0) { epochs[blockIdx.x] = epoch ^ 1U; }
}

// One complete D5120 row owns ten 64-pack epoch domains.
template <int Rank, RmsEpilogue Epilogue, bool Published, bool Fp16Image, int Threads = 640>
__global__ __launch_bounds__(Threads, 1) void allreduce_packet_sum4_rmsnorm_kernel(
    Bf16x8Pack* buffer, const Bf16x8Pack* weight, Bf16x8Pack* normalized,
    Bf16x8Pack* storage0, Bf16x8Pack* storage1, Bf16x8Pack* storage2,
    Bf16x8Pack* storage3, unsigned* epochs, int rows, int channel_packs, float eps) {
    constexpr int PacksPerRow = 640;
    static_assert(Threads == 128 || Threads == PacksPerRow);
    constexpr int Passes = PacksPerRow / Threads;
    const int tid = threadIdx.x, row = blockIdx.x;
    Bf16x8Pack* const local_storage = Rank == 0 ? storage0 : Rank == 1 ? storage1 :
                                      Rank == 2 ? storage2 : storage3;
    Bf16x8Pack values[Passes];
    __shared__ float warp_sums[20], inverse;
    // Batch rows supply the grid. Reuse a smaller CTA over five pack panels,
    // keeping all twenty logical warp sums and their final association intact.
    // This follows 1Cat's small-CTA batch sum/norm scheduling, without changing
    // NInfer's BF16 contribution/residual or fixed global-rank reduction order.
#pragma unroll
    for (int pass = 0; pass < Passes; ++pass) {
        const int pack = tid + pass * Threads, i = row * PacksPerRow + pack;
        values[pass] = allreduce_packet_sum4_value<Rank, Published>(buffer, storage0, storage1,
            storage2, storage3, local_storage, i, epochs[i / 64] & 1U, channel_packs);
        store_vec(buffer + i, values[pass]);
        float square_sum = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const auto v = __bfloat1622float2(values[pass].pair[j]);
            square_sum += v.x * v.x + v.y * v.y;
        }
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
            square_sum += __shfl_xor_sync(0xffffffffU, square_sum, delta);
        if ((tid & 31) == 0) warp_sums[pack / 32] = square_sum;
    }
    __syncthreads();
    if (tid < 32) {
        float sum = tid < 20 ? warp_sums[tid] : 0.f;
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
            sum += __shfl_down_sync(0xffffffffU, sum, delta);
        if (tid == 0) inverse = rsqrtf(sum / 5120.f + eps);
    }
    if (tid < 10) epochs[row * 10 + tid] ^= 1U;
    __syncthreads();
#pragma unroll
    for (int pass = 0; pass < Passes; ++pass) {
        const int pack = tid + pass * Threads;
        const auto w = load_vec<Bf16x8Pack>(weight + pack);
        auto value = values[pass];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const auto v = __bfloat1622float2(value.pair[j]);
            const auto ww = __bfloat1622float2(w.pair[j]);
            const auto rounded = __floats2bfloat162_rn(
                rmsnorm_epilogue<Epilogue>(v.x, inverse, ww.x, 0.f),
                rmsnorm_epilogue<Epilogue>(v.y, inverse, ww.y, 0.f));
            if constexpr (Fp16Image) {
                const auto represented = __bfloat1622float2(rounded);
                const auto image = __floats2half2_rn(represented.x, represented.y);
                value.pair[j] = *reinterpret_cast<const __nv_bfloat162*>(&image);
            } else {
                value.pair[j] = rounded;
            }
        }
        store_vec(normalized + row * PacksPerRow + pack, value);
    }
}

// Shares the packet route's fixed 256-pair epoch domains. Each complete D5120
// row owns ten domains, so fused/unfused and different-width calls may interleave.
template <RmsEpilogue Epilogue, bool Published = false>
__global__ void allreduce_packet_rmsnorm_5120_kernel(
    const __nv_bfloat162* source, __nv_bfloat162* sum_output,
    const __nv_bfloat162* weight, __nv_bfloat162* norm_output,
    unsigned* local, unsigned* peer, unsigned* epochs, int epoch_pairs, float eps) {
    constexpr int Block = 512, Pairs = 2560, Width = 5120;
    const int row = blockIdx.x, tid = threadIdx.x;
    __nv_bfloat162 values[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int i = row * Pairs + tid + k * Block;
        const unsigned epoch = epochs[i / 256] & 1U;
        values[k] = source[i];
        if constexpr (!Published) {
            unsigned bits = *reinterpret_cast<unsigned*>(&values[k]);
            if ((bits & 65535) == 65535) { bits = (bits & 0xffff0000) | 0x7fc0; }
            if ((bits >> 16) == 65535) { bits = (bits & 0xffff) | 0x7fc00000; }
            auto* address = peer + epoch * epoch_pairs + i;
            asm volatile("st.volatile.global.u32 [%0],%1;"
                         : : "l"(address), "r"(bits) : "memory");
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int i = row * Pairs + tid + k * Block;
        const unsigned epoch = epochs[i / 256] & 1U;
        unsigned bits;
        const auto* address = local + epoch * epoch_pairs + i;
        do {
            asm volatile("ld.volatile.global.u32 %0,[%1];" : "=r"(bits) : "l"(address));
        } while ((bits & 65535) == 65535 || (bits >> 16) == 65535);
        const auto a = __bfloat1622float2(values[k]);
        const auto b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&bits));
        values[k] = __floats2bfloat162_rn(a.x + b.x, a.y + b.y);
        sum_output[i] = values[k];
        const auto v = __bfloat1622float2(values[k]);
        sum += v.x * v.x + v.y * v.y;
        auto* empty = local + epoch * epoch_pairs + i;
        asm volatile("st.volatile.global.u32 [%0],%1;"
                     : : "l"(empty), "r"(0xffffffff) : "memory");
    }
    __shared__ float warp_sums[16], inv_shared;
    const float reduced = ops::block_reduce_sum<Block>(sum, warp_sums);
    if (tid == 0) { inv_shared = rsqrtf(reduced / float(Width) + eps); }
    __syncthreads();
    if (tid < 10) { epochs[row * 10 + tid] ^= 1U; }
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int i = tid + k * Block;
        const auto v = __bfloat1622float2(values[k]);
        const auto w = __bfloat1622float2(weight[i]);
        norm_output[row * Pairs + i] = __floats2bfloat162_rn(
            rmsnorm_epilogue<Epilogue>(v.x, inv_shared, w.x, 0.0F),
            rmsnorm_epilogue<Epilogue>(v.y, inv_shared, w.y, 0.0F));
    }
}

__global__ void allreduce_peer_sum_scalar_kernel(const __nv_bfloat16* local,
                                                const __nv_bfloat16* peer,
                                                __nv_bfloat16* output, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = __float2bfloat16_rn(__bfloat162float(local[i]) + __bfloat162float(peer[i]));
    }
}

__global__ void allreduce_peer_sum_packed_kernel(const Bf16x8Pack* local,
                                                const Bf16x8Pack* peer,
                                                Bf16x8Pack* output, std::int64_t packs) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= packs) { return; }
    auto a = load_vec<Bf16x8Pack>(local + i);
    const auto b = load_vec<Bf16x8Pack>(peer + i);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a.pair[j] = __floats2bfloat162_rn(__low2float(a.pair[j]) + __low2float(b.pair[j]),
                                        __high2float(a.pair[j]) + __high2float(b.pair[j]));
    }
    store_vec(output + i, a);
}

// Fixed global rank order is essential: rotating the association per destination
// would produce different replicated residuals once there are more than two ranks.
__global__ void allreduce_sum4_scalar_kernel(const __nv_bfloat16* a,
                                            const __nv_bfloat16* b,
                                            const __nv_bfloat16* c,
                                            const __nv_bfloat16* d,
                                            __nv_bfloat16* output, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float sum = __bfloat162float(a[i]) + __bfloat162float(b[i]);
        sum += __bfloat162float(c[i]);
        sum += __bfloat162float(d[i]);
        output[i] = __float2bfloat16_rn(sum);
    }
}

__global__ void allreduce_sum4_packed_kernel(const Bf16x8Pack* a, const Bf16x8Pack* b,
                                            const Bf16x8Pack* c, const Bf16x8Pack* d,
                                            Bf16x8Pack* output, std::int64_t packs) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= packs) { return; }
    auto av = load_vec<Bf16x8Pack>(a + i);
    const auto bv = load_vec<Bf16x8Pack>(b + i);
    const auto cv = load_vec<Bf16x8Pack>(c + i);
    const auto dv = load_vec<Bf16x8Pack>(d + i);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float low = __low2float(av.pair[j]) + __low2float(bv.pair[j]);
        float high = __high2float(av.pair[j]) + __high2float(bv.pair[j]);
        low += __low2float(cv.pair[j]);
        high += __high2float(cv.pair[j]);
        low += __low2float(dv.pair[j]);
        high += __high2float(dv.pair[j]);
        av.pair[j] = __floats2bfloat162_rn(low, high);
    }
    store_vec(output + i, av);
}

} // namespace ninfer::ops
