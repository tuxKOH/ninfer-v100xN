#include "targets/qwen3_6/impl/vision/sm86/elementwise.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

namespace vision_aux_kernels {
namespace ninfer::ops {
using ::ninfer::ops::Bf16x8Pack;
using ::ninfer::ops::load_vec;
using ::ninfer::ops::store_vec;
}
#include "ops/kernel/add_bias.cuh"
#include "ops/kernel/gelu.cuh"
#include "ops/kernel/layer_norm.cuh"
#include "ops/kernel/residual_add.cuh"
#include "ops/kernel/rope.cuh"
#include "ops/kernel/vision_pos_embed.cuh"
}

namespace vision_kernels = vision_aux_kernels::ninfer::ops;

namespace ninfer::targets::qwen3_6::vision_sm86 {
namespace {

constexpr int kBlock = 256;
constexpr int kMaxBlocks = 65535;

std::int64_t elements(const Tensor& t) {
    std::int64_t n = 1;
    bool empty = false;
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] < 0) { throw std::invalid_argument("vision elementwise: negative dimension"); }
        if (t.ne[i] == 0) { empty = true; continue; }
        if (n > std::numeric_limits<std::int64_t>::max() / t.ne[i]) {
            throw std::overflow_error("vision elementwise: tensor size overflow");
        }
        n *= t.ne[i];
    }
    return empty ? 0 : n;
}

void storage(const Tensor& t, DType dtype) {
    if (t.dtype != dtype || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument("vision elementwise: invalid tensor storage");
    }
}

bool vector_of(const Tensor& t, int d) {
    return t.ne[0] == d && t.ne[1] == 1 && t.ne[2] == 1 && t.ne[3] == 1;
}

void same_shape(const Tensor& a, const Tensor& b) {
    for (int i = 0; i < 4; ++i) {
        if (a.ne[i] != b.ne[i]) {
            throw std::invalid_argument("vision elementwise: shapes differ");
        }
    }
}

int grid_for(std::int64_t n, int per_block = kBlock) {
    const std::int64_t blocks = (n - 1) / per_block + 1;
    return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}


template <bool Tanh>
void gelu_impl(Tensor& x, cudaStream_t stream) {
    if (x.dtype != DType::BF16) { throw std::invalid_argument("vision gelu: expected BF16"); }
    const auto n = elements(x);
    if (!n) { return; }
    storage(x, DType::BF16);
    vision_kernels::gelu_kernel<Tanh><<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void add_bias(const Tensor& bias, Tensor& x, cudaStream_t stream) {
    if (bias.dtype != DType::BF16 || x.dtype != DType::BF16 || !vector_of(bias, x.ne[0])) {
        throw std::invalid_argument("vision add_bias: invalid shape or dtype");
    }
    const auto n = elements(x);
    (void)elements(bias);
    if (!n) { return; }
    storage(bias, DType::BF16);
    storage(x, DType::BF16);
    vision_kernels::add_bias_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bias.data), static_cast<__nv_bfloat16*>(x.data),
        x.ne[0], n);
    CUDA_CHECK(cudaGetLastError());
}

void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps,
                Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || bias.dtype != DType::BF16 ||
        out.dtype != DType::BF16 || !vector_of(weight, x.ne[0]) || !vector_of(bias, x.ne[0]) ||
        !(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("vision layer_norm: invalid dtype, parameters or shape");
    }
    same_shape(x, out);
    const auto n = elements(x);
    (void)elements(weight); (void)elements(bias); (void)elements(out);
    if (!n) { return; }
    storage(x, DType::BF16); storage(weight, DType::BF16);
    storage(bias, DType::BF16); storage(out, DType::BF16);
    const auto rows = n / x.ne[0];
    const bool packed = x.ne[0] == 1152 &&
        reinterpret_cast<std::uintptr_t>(x.data) % alignof(__nv_bfloat162) == 0 &&
        reinterpret_cast<std::uintptr_t>(weight.data) % alignof(__nv_bfloat162) == 0 &&
        reinterpret_cast<std::uintptr_t>(bias.data) % alignof(__nv_bfloat162) == 0 &&
        reinterpret_cast<std::uintptr_t>(out.data) % alignof(__nv_bfloat162) == 0;
    const std::int64_t rows_per_launch = packed ? kMaxBlocks * (kBlock / 32) : kMaxBlocks;
    for (std::int64_t first = 0; first < rows;) {
        const std::int64_t count = rows - first < rows_per_launch ? rows - first : rows_per_launch;
        const auto* input = static_cast<const __nv_bfloat16*>(x.data) + first * x.ne[0];
        auto* output = static_cast<__nv_bfloat16*>(out.data) + first * x.ne[0];
        if (packed) {
            vision_kernels::layer_norm_d1152_warp_kernel<kBlock>
                <<<static_cast<unsigned>((count - 1) / (kBlock / 32) + 1), kBlock, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat162*>(input),
                    static_cast<const __nv_bfloat162*>(weight.data),
                    static_cast<const __nv_bfloat162*>(bias.data),
                    reinterpret_cast<__nv_bfloat162*>(output), count, eps);
        } else {
            vision_kernels::layer_norm_kernel<kBlock>
                <<<static_cast<unsigned>(count), kBlock, 0, stream>>>(
                    input, static_cast<const __nv_bfloat16*>(weight.data),
                    static_cast<const __nv_bfloat16*>(bias.data), output, x.ne[0], count, eps);
        }
        CUDA_CHECK(cudaGetLastError());
        first += count;
    }
}

void position_add(const Tensor& table, const Tensor& indices, const Tensor& weights,
                  Tensor& x, cudaStream_t stream) {
    if (table.dtype != DType::BF16 || x.dtype != DType::BF16 || indices.dtype != DType::I32 ||
        weights.dtype != DType::FP32 || table.ne[0] != x.ne[0] || table.ne[1] <= 0 ||
        table.ne[2] != 1 || table.ne[3] != 1 || x.ne[2] != 1 || x.ne[3] != 1 ||
        indices.ne[0] != 4 || indices.ne[1] != x.ne[1] || indices.ne[2] != 1 || indices.ne[3] != 1 ||
        weights.ne[0] != 4 || weights.ne[1] != x.ne[1] || weights.ne[2] != 1 || weights.ne[3] != 1) {
        throw std::invalid_argument("vision position_add: invalid shape or dtype");
    }
    const auto n = elements(x);
    (void)elements(table); (void)elements(indices); (void)elements(weights);
    if (!n) { return; }
    storage(table, DType::BF16); storage(indices, DType::I32);
    storage(weights, DType::FP32); storage(x, DType::BF16);
    vision_kernels::vision_pos_embed_add_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(table.data), static_cast<const std::int32_t*>(indices.data),
        static_cast<const float*>(weights.data), static_cast<__nv_bfloat16*>(x.data),
        x.ne[0], x.ne[1], table.ne[1], n);
    CUDA_CHECK(cudaGetLastError());
}

void rope_2d(const Tensor& positions, Tensor& q, Tensor& k, cudaStream_t stream) {
    const int tokens = q.ne[2];
    if (positions.dtype != DType::I32 || q.dtype != DType::BF16 || k.dtype != DType::BF16 ||
        tokens < 0 || positions.ne[0] != tokens || positions.ne[1] != 2 ||
        positions.ne[2] != 1 || positions.ne[3] != 1) {
        throw std::invalid_argument("vision rope_2d: invalid positions or dtype");
    }
    for (const Tensor* t : {&q, &k}) {
        if (t->ne[0] != 72 || t->ne[1] != 16 || t->ne[2] != tokens || t->ne[3] != 1 ||
            t->nb[0] != 2 || t->nb[1] != 144 || t->nb[2] < 2304 || t->nb[2] % 2 != 0) {
            throw std::invalid_argument("vision rope_2d: invalid q/k shape or strides");
        }
    }
    if (tokens == 0) { return; }
    storage(positions, DType::I32);
    if (q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("vision rope_2d: null q/k data");
    }
    const bool aligned =
        reinterpret_cast<std::uintptr_t>(q.data) % alignof(__nv_bfloat162) == 0 &&
        reinterpret_cast<std::uintptr_t>(k.data) % alignof(__nv_bfloat162) == 0 &&
        q.nb[2] % 4 == 0 && k.nb[2] % 4 == 0;
    if (aligned) {
        vision_kernels::rope_fixed_kernel<vision_kernels::RopeKernelMode::Vision2D, 16, 16, true>
            <<<tokens, kBlock, 0, stream>>>(
                static_cast<const std::int32_t*>(positions.data),
                static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                tokens, q.nb[2] / 2, k.nb[2] / 2);
    } else {
        vision_kernels::rope_fixed_kernel<vision_kernels::RopeKernelMode::Vision2D, 16, 16, false>
            <<<tokens, kBlock, 0, stream>>>(
                static_cast<const std::int32_t*>(positions.data),
                static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                tokens, q.nb[2] / 2, k.nb[2] / 2);
    }
    CUDA_CHECK(cudaGetLastError());
}

void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || y.dtype != DType::BF16) {
        throw std::invalid_argument("vision residual_add: expected BF16");
    }
    same_shape(x, y);
    const auto n = elements(x);
    if (!n) { return; }
    storage(x, DType::BF16); storage(y, DType::BF16);
    vision_kernels::residual_add_scalar_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(y.data), static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void gelu_tanh(Tensor& x, cudaStream_t stream) { gelu_impl<true>(x, stream); }
void gelu_exact(Tensor& x, cudaStream_t stream) { gelu_impl<false>(x, stream); }

} // namespace ninfer::targets::qwen3_6::vision_sm86
