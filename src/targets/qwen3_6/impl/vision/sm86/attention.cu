#include "targets/qwen3_6/impl/vision/sm86/attention.h"

#include "core/device.h"
#include "ops/kernel/vision_attention.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::vision_sm86 {
namespace {

constexpr int kDim = 72;
constexpr int kHeads = 16;
constexpr int kTile = ops::kVisionAttentionBr;
constexpr int kVisionAttentionPaddedD = ops::kVisionAttentionPaddedD;
constexpr int kThreads = 128;
constexpr std::int64_t kTokenBytes = kDim * kHeads * 2;

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

AddressRange checked_range(const void* data, std::uint64_t bytes, const char* name) {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (data == nullptr || bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument(std::string("vision attention: invalid ") + name + " extent");
    }
    return {begin, begin + static_cast<std::uintptr_t>(bytes)};
}

bool overlaps(AddressRange a, AddressRange b) {
    return a.begin < b.end && b.begin < a.end;
}

AddressRange validate_activation(const Tensor& tensor, std::int32_t patches, const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kDim || tensor.ne[1] != kHeads ||
        tensor.ne[2] != patches || tensor.ne[3] != 1 || tensor.nb[0] != 2 ||
        tensor.nb[1] != kDim * 2 || tensor.nb[2] < kTokenBytes || tensor.nb[2] % 2 != 0 ||
        reinterpret_cast<std::uintptr_t>(tensor.data) % 2 != 0) {
        throw std::invalid_argument(std::string("vision attention: invalid ") + name +
                                    " shape, strides or alignment");
    }
    const auto stride = static_cast<std::uint64_t>(tensor.nb[2]);
    const auto last = static_cast<std::uint64_t>(patches - 1);
    const auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (last != 0 && stride > (max_offset - kTokenBytes) / last) {
        throw std::overflow_error(std::string("vision attention: ") + name + " extent overflows");
    }
    return checked_range(tensor.data, last * stride + kTokenBytes, name);
}


} // namespace

std::size_t attention_workspace_bytes(std::int32_t patches, std::int32_t segments) {
    if (patches <= 0 || segments <= 0 || segments > patches) {
        throw std::invalid_argument("vision attention: invalid patch or segment count");
    }
    return 0;
}

void attention_cu_seqlens(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& cu_seqlens, Tensor& out, cudaStream_t stream) {
    const std::int32_t patches = q.ne[2];
    const std::int64_t segments64 = static_cast<std::int64_t>(cu_seqlens.ne[0]) - 1;
    if (segments64 <= 0 || segments64 > patches) {
        throw std::invalid_argument("vision attention: invalid patch or segment count");
    }
    const auto segments = static_cast<std::int32_t>(segments64);
    (void)attention_workspace_bytes(patches, segments);
    const auto q_range = validate_activation(q, patches, "q");
    const auto k_range = validate_activation(k, patches, "k");
    const auto v_range = validate_activation(v, patches, "v");
    const auto out_range = validate_activation(out, patches, "out");
    if (!out.is_contiguous()) {
        throw std::invalid_argument("vision attention: output must be contiguous");
    }
    if (cu_seqlens.dtype != DType::I32 || cu_seqlens.ne[1] != 1 ||
        cu_seqlens.ne[2] != 1 || cu_seqlens.ne[3] != 1 || !cu_seqlens.is_contiguous() ||
        reinterpret_cast<std::uintptr_t>(cu_seqlens.data) % alignof(std::int32_t) != 0) {
        throw std::invalid_argument("vision attention: bounds must be contiguous I32 [S+1]");
    }
    const auto bounds_bytes = static_cast<std::size_t>(cu_seqlens.ne[0]) * sizeof(std::int32_t);
    const auto bounds_range = checked_range(cu_seqlens.data, bounds_bytes, "bounds");
    if (overlaps(out_range, q_range) || overlaps(out_range, k_range) ||
        overlaps(out_range, v_range) || overlaps(out_range, bounds_range)) {
        throw std::invalid_argument("vision attention: output must not overlap inputs");
    }
    std::vector<std::int32_t> bounds(static_cast<std::size_t>(segments) + 1);
    CUDA_CHECK(cudaMemcpyAsync(bounds.data(), cu_seqlens.data, bounds_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (bounds.front() != 0 || bounds.back() != patches) {
        throw std::invalid_argument("vision attention: bounds must span [0, P]");
    }
    for (std::size_t i = 1; i < bounds.size(); ++i) {
        if (bounds[i] <= bounds[i - 1]) {
            throw std::invalid_argument("vision attention: bounds must be strictly increasing");
        }
    }
    const dim3 grid(static_cast<unsigned>((static_cast<std::int64_t>(patches) + kTile - 1) / kTile), kHeads);
    constexpr int shared_bytes = (kTile + 2 * kTile) * kVisionAttentionPaddedD * sizeof(__nv_bfloat16);
    ops::vision_attention_flash_kernel<kTile, kTile, true><<<grid, kThreads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), nullptr, patches, 0,
        static_cast<__nv_bfloat16*>(out.data), static_cast<const std::int32_t*>(cu_seqlens.data),
        segments, 1, kDim, q.nb[2] / 2, 1, kDim, k.nb[2] / 2, 1, kDim, v.nb[2] / 2);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_6::vision_sm86
