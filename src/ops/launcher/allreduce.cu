#include "ops/launcher/allreduce.h"
#include "ops/kernel/allreduce.cuh"
#include "core/device.h"
#include "ops/common/math.h"

#include <stdexcept>

namespace ninfer::ops::detail {
void broadcast_i32_tp4_launch(const Tensor& source, std::span<const Tensor> destinations,
                              cudaStream_t stream) {
    const int count = static_cast<int>(source.numel());
    broadcast_i32_tp4_kernel<<<div_up(count, 128), 128, 0, stream>>>(
        static_cast<const int*>(source.data), static_cast<int*>(destinations[0].data),
        static_cast<int*>(destinations[1].data), static_cast<int*>(destinations[2].data), count);
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_argmax_tp2_launch(std::span<const Tensor> values, std::span<const Tensor> indices,
    std::span<const Tensor> output, std::int32_t shard_rows, const std::int32_t* id_map, cudaStream_t stream) {
    const int columns = output[0].ne[0];
    allreduce_argmax_tp2_kernel<<<div_up(columns, 128), 128, 0, stream>>>(
        static_cast<const float*>(values[0].data), static_cast<const float*>(values[1].data),
        static_cast<const int*>(indices[0].data), static_cast<const int*>(indices[1].data),
        static_cast<int*>(output[0].data), static_cast<int*>(output[1].data), columns, shard_rows, id_map);
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_argmax_tp4_launch(std::span<const Tensor> values, std::span<const Tensor> indices,
    std::span<const Tensor> output, std::int32_t shard_rows, const std::int32_t* id_map, cudaStream_t stream) {
    const int columns = output[0].ne[0];
    allreduce_argmax_tp4_kernel<<<div_up(columns, 128), 128, 0, stream>>>(
        static_cast<const float*>(values[0].data), static_cast<const float*>(values[1].data),
        static_cast<const float*>(values[2].data), static_cast<const float*>(values[3].data),
        static_cast<const int*>(indices[0].data), static_cast<const int*>(indices[1].data),
        static_cast<const int*>(indices[2].data), static_cast<const int*>(indices[3].data),
        static_cast<int*>(output[0].data), static_cast<int*>(output[1].data),
        static_cast<int*>(output[2].data), static_cast<int*>(output[3].data), columns, shard_rows, id_map);
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_packet_sum_launch(const Tensor& local, const Tensor& output,
                                 void* local_storage, void* peer_storage,
                                 std::size_t epoch_bytes, cudaStream_t stream) {
    const int packs = static_cast<int>(local.numel() / 8);
    auto* epochs = reinterpret_cast<unsigned*>(static_cast<std::uint8_t*>(local_storage) +
                                                2 * epoch_bytes);
    allreduce_packet_sum_kernel<false><<<div_up(packs, 64), 64, 0, stream>>>(
        static_cast<const Bf16x8Pack*>(local.data), static_cast<Bf16x8Pack*>(output.data),
        static_cast<Bf16x8Pack*>(local_storage), static_cast<Bf16x8Pack*>(peer_storage),
        epochs, packs, static_cast<int>(epoch_bytes / sizeof(Bf16x8Pack)));
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_host_packet_sum_launch(const Tensor& local, const Tensor& output,
    void* local_storage, void* peer_storage, unsigned* epochs,
    std::size_t epoch_bytes, cudaStream_t stream) {
    const int packs = static_cast<int>(local.numel() / 8);
    allreduce_packet_sum_kernel<true><<<div_up(packs, 64), 64, 0, stream>>>(
        static_cast<const Bf16x8Pack*>(local.data), static_cast<Bf16x8Pack*>(output.data),
        static_cast<Bf16x8Pack*>(local_storage), static_cast<Bf16x8Pack*>(peer_storage),
        epochs, packs, static_cast<int>(epoch_bytes / sizeof(Bf16x8Pack)));
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_packet_sum4_launch(const Tensor& buffer, const std::array<void*, 4>& storage,
                                  int rank, std::size_t channel_bytes, cudaStream_t stream, bool published) {
    const int packs = static_cast<int>(buffer.numel() / 8);
    auto* epochs = reinterpret_cast<unsigned*>(static_cast<std::uint8_t*>(storage[rank]) +
                                              8 * channel_bytes);
    const auto launch = [&]<int Rank, bool Published>() {
        allreduce_packet_sum4_kernel<Rank, Published>
            <<<div_up(packs, kAllreduceTp4PacketThreads), kAllreduceTp4PacketThreads, 0, stream>>>(
            static_cast<Bf16x8Pack*>(buffer.data),
            static_cast<Bf16x8Pack*>(storage[0]), static_cast<Bf16x8Pack*>(storage[1]),
            static_cast<Bf16x8Pack*>(storage[2]), static_cast<Bf16x8Pack*>(storage[3]),
            epochs, packs, static_cast<int>(channel_bytes / sizeof(Bf16x8Pack)));
    };
    const auto by_rank = [&]<int Rank>() {
        if (published) launch.template operator()<Rank, true>();
        else launch.template operator()<Rank, false>();
    };
    switch (rank) {
    case 0: by_rank.template operator()<0>(); break;
    case 1: by_rank.template operator()<1>(); break;
    case 2: by_rank.template operator()<2>(); break;
    case 3: by_rank.template operator()<3>(); break;
    default: throw std::invalid_argument("TP4 packet all-reduce rank must be in [0, 3]");
    }
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_packet_sum4_rmsnorm_launch(const Tensor& buffer, const Tensor& weight,
    const Tensor& normalized, float eps, bool unit_offset, const std::array<void*, 4>& storage,
    int rank, std::size_t channel_bytes, cudaStream_t stream, bool published) {
    auto* epochs = reinterpret_cast<unsigned*>(static_cast<std::uint8_t*>(storage[rank]) +
                                              8 * channel_bytes);
    const auto launch = [&]<int Rank, RmsEpilogue Epilogue, bool Published, bool Fp16Image>() {
        const auto issue = [&]<int Threads>() {
            allreduce_packet_sum4_rmsnorm_kernel<Rank, Epilogue, Published, Fp16Image, Threads>
            <<<buffer.ne[1], Threads, 0, stream>>>(
            static_cast<Bf16x8Pack*>(buffer.data), static_cast<const Bf16x8Pack*>(weight.data),
            static_cast<Bf16x8Pack*>(normalized.data), static_cast<Bf16x8Pack*>(storage[0]),
            static_cast<Bf16x8Pack*>(storage[1]), static_cast<Bf16x8Pack*>(storage[2]),
            static_cast<Bf16x8Pack*>(storage[3]), epochs, buffer.ne[1],
            static_cast<int>(channel_bytes / sizeof(Bf16x8Pack)), eps);
        };
        if (buffer.ne[1] > 8) issue.template operator()<128>();
        else issue.template operator()<640>();
    };
    const auto rank_launch = [&]<int Rank>() {
        const auto profile = [&]<bool Published, bool Fp16Image>() {
            if (unit_offset) launch.template operator()<Rank, RmsEpilogue::Offset, Published, Fp16Image>();
            else launch.template operator()<Rank, RmsEpilogue::Plain, Published, Fp16Image>();
        };
        if (published) {
            if (normalized.dtype == DType::FP16) profile.template operator()<true, true>();
            else profile.template operator()<true, false>();
        } else profile.template operator()<false, false>();
    };
    switch (rank) {
    case 0: rank_launch.template operator()<0>(); break;
    case 1: rank_launch.template operator()<1>(); break;
    case 2: rank_launch.template operator()<2>(); break;
    case 3: rank_launch.template operator()<3>(); break;
    default: throw std::invalid_argument("TP4 packet norm rank must be in [0, 3]");
    }
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_packet_rmsnorm_launch(const Tensor& input, const Tensor& output,
    const Tensor& weight, const Tensor& normalized, float eps, bool unit_offset,
    void* local_storage, void* peer_storage, std::size_t epoch_bytes, cudaStream_t stream,
    bool published) {
    auto* epochs = reinterpret_cast<unsigned*>(static_cast<std::uint8_t*>(local_storage) + 2 * epoch_bytes);
    const auto launch = [&]<RmsEpilogue Epilogue, bool Published>() {
        allreduce_packet_rmsnorm_5120_kernel<Epilogue, Published><<<input.ne[1], 512, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(input.data), static_cast<__nv_bfloat162*>(output.data),
            static_cast<const __nv_bfloat162*>(weight.data), static_cast<__nv_bfloat162*>(normalized.data),
            static_cast<unsigned*>(local_storage), static_cast<unsigned*>(peer_storage), epochs,
            static_cast<int>(epoch_bytes / sizeof(__nv_bfloat162)), eps);
    };
    if (published) {
        if (unit_offset) { launch.template operator()<RmsEpilogue::Offset, true>(); }
        else { launch.template operator()<RmsEpilogue::Plain, true>(); }
    } else {
        if (unit_offset) { launch.template operator()<RmsEpilogue::Offset, false>(); }
        else { launch.template operator()<RmsEpilogue::Plain, false>(); }
    }
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_peer_sum_launch(const Tensor& local, const Tensor& peer, const Tensor& output,
                               cudaStream_t stream) {
    const auto count = local.numel();
    const auto addresses = reinterpret_cast<std::uintptr_t>(local.data) |
                           reinterpret_cast<std::uintptr_t>(peer.data) |
                           reinterpret_cast<std::uintptr_t>(output.data);
    if ((addresses & (alignof(Bf16x8Pack) - 1)) == 0 && count % 8 == 0) {
        const auto packs = count / 8;
        allreduce_peer_sum_packed_kernel<<<div_up(packs, std::int64_t{64}), 64, 0, stream>>>(
            static_cast<const Bf16x8Pack*>(local.data), static_cast<const Bf16x8Pack*>(peer.data),
            static_cast<Bf16x8Pack*>(output.data), packs);
    } else {
        allreduce_peer_sum_scalar_kernel<<<div_up(count, std::int64_t{128}), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(local.data), static_cast<const __nv_bfloat16*>(peer.data),
            static_cast<__nv_bfloat16*>(output.data), count);
    }
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_sum4_launch(const std::array<Tensor, 4>& inputs, const Tensor& output,
                          cudaStream_t stream) {
    const auto count = output.numel();
    auto addresses = reinterpret_cast<std::uintptr_t>(output.data);
    for (const auto& input : inputs) {
        addresses |= reinterpret_cast<std::uintptr_t>(input.data);
    }
    if ((addresses & (alignof(Bf16x8Pack) - 1)) == 0 && count % 8 == 0) {
        const auto packs = count / 8;
        allreduce_sum4_packed_kernel<<<div_up(packs, std::int64_t{64}), 64, 0, stream>>>(
            static_cast<const Bf16x8Pack*>(inputs[0].data),
            static_cast<const Bf16x8Pack*>(inputs[1].data),
            static_cast<const Bf16x8Pack*>(inputs[2].data),
            static_cast<const Bf16x8Pack*>(inputs[3].data),
            static_cast<Bf16x8Pack*>(output.data), packs);
    } else {
        allreduce_sum4_scalar_kernel<<<div_up(count, std::int64_t{128}), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(inputs[0].data),
            static_cast<const __nv_bfloat16*>(inputs[1].data),
            static_cast<const __nv_bfloat16*>(inputs[2].data),
            static_cast<const __nv_bfloat16*>(inputs[3].data),
            static_cast<__nv_bfloat16*>(output.data), count);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
