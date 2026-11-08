#include "targets/qwen3_6/impl/vision/sm86/elementwise.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

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

__global__ void bias_kernel(const __nv_bfloat16* bias, __nv_bfloat16* x,
                            std::int64_t n, int d) {
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(bias[i % d]));
    }
}

struct Moments { float mean; float m2; int count; };

__device__ __forceinline__ void update(Moments& m, float v) {
    ++m.count;
    const float delta = v - m.mean;
    m.mean += delta / static_cast<float>(m.count);
    m.m2 += delta * (v - m.mean);
}

__device__ __forceinline__ Moments merge(Moments a, Moments b) {
    if (b.count == 0) { return a; }
    if (a.count == 0) { return b; }
    const float delta = b.mean - a.mean;
    const int count = a.count + b.count;
    return {a.mean + delta * (static_cast<float>(b.count) / static_cast<float>(count)),
            a.m2 + b.m2 + delta * delta *
                (static_cast<float>(a.count) * static_cast<float>(b.count) /
                 static_cast<float>(count)), count};
}

// One warp owns a column. Pair order and Welford reduction match the Vision norm path.
__global__ void norm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                            const __nv_bfloat16* bias, __nv_bfloat16* out,
                            std::int64_t rows, int d, float eps) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int warps = blockDim.x / 32;
    for (std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * warps + warp;
         row < rows; row += static_cast<std::int64_t>(gridDim.x) * warps) {
        const std::int64_t base = row * d;
        Moments m{0.0f, 0.0f, 0};
        for (int pair = lane; pair < (d + 1) / 2; pair += 32) {
            const int i = pair * 2;
            update(m, __bfloat162float(x[base + i]));
            if (i + 1 < d) { update(m, __bfloat162float(x[base + i + 1])); }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            Moments other{__shfl_down_sync(0xffffffffu, m.mean, offset),
                          __shfl_down_sync(0xffffffffu, m.m2, offset),
                          __shfl_down_sync(0xffffffffu, m.count, offset)};
            m = merge(m, other);
        }
        const float mean = __shfl_sync(0xffffffffu, m.mean, 0);
        const float m2 = __shfl_sync(0xffffffffu, m.m2, 0);
        const float inv = rsqrtf(m2 / static_cast<float>(d) + eps);
        for (int i = lane; i < d; i += 32) {
            const float v = (__bfloat162float(x[base + i]) - mean) * inv;
            out[base + i] = __float2bfloat16_rn(v * __bfloat162float(weight[i]) +
                                               __bfloat162float(bias[i]));
        }
    }
}

__global__ void position_kernel(const __nv_bfloat16* table, const std::int32_t* indices,
                                const float* weights, __nv_bfloat16* x,
                                std::int64_t n, int d, int table_rows) {
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const int channel = static_cast<int>(i % d);
        const std::int64_t control = (i / d) * 4;
        float value = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const int index = indices[control + j];
            if (index >= 0 && index < table_rows) {
                value += __bfloat162float(table[static_cast<std::int64_t>(index) * d + channel]) *
                         weights[control + j];
            }
        }
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + value);
    }
}

__device__ __constant__ float kInvFreq[18] = {
    1.000000000e+00F, 5.994842503e-01F, 3.593813664e-01F, 2.154434690e-01F,
    1.291549665e-01F, 7.742636827e-02F, 4.641588834e-02F, 2.782559402e-02F,
    1.668100537e-02F, 1.000000000e-02F, 5.994842503e-03F, 3.593813664e-03F,
    2.154434690e-03F, 1.291549665e-03F, 7.742636827e-04F, 4.641588834e-04F,
    2.782559402e-04F, 1.668100537e-04F,
};

__global__ void rope_kernel(const std::int32_t* positions, __nv_bfloat16* q,
                            __nv_bfloat16* k, int tokens, std::int64_t q_stride,
                            std::int64_t k_stride) {
    for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
        __shared__ float sine[36], cosine[36];
        if (threadIdx.x < 36) {
            const int pair = threadIdx.x;
            const float angle = static_cast<float>(positions[static_cast<std::int64_t>(pair / 18) *
                                                              tokens + token]) * kInvFreq[pair % 18];
            sincosf(angle, &sine[pair], &cosine[pair]);
        }
        __syncthreads();
        for (int index = threadIdx.x; index < 32 * 36; index += blockDim.x) {
            const int head = index / 36;
            const int pair = index % 36;
            __nv_bfloat16* data = head < 16 ? q + static_cast<std::int64_t>(token) * q_stride + head * 72
                                            : k + static_cast<std::int64_t>(token) * k_stride + (head - 16) * 72;
            const float a = __bfloat162float(data[pair]);
            const float b = __bfloat162float(data[pair + 36]);
            data[pair] = __float2bfloat16_rn(a * cosine[pair] - b * sine[pair]);
            data[pair + 36] = __float2bfloat16_rn(b * cosine[pair] + a * sine[pair]);
        }
        __syncthreads();
    }
}

__global__ void residual_kernel(const __nv_bfloat16* y, __nv_bfloat16* x, std::int64_t n) {
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(y[i]));
    }
}

template <bool Tanh>
__global__ void gelu_kernel(__nv_bfloat16* x, std::int64_t n) {
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const float v = __bfloat162float(x[i]);
        float result;
        if constexpr (Tanh) {
            result = 0.5f * v * (1.0f + tanhf(0.79788456080286535588f *
                                                (v + 0.044715f * v * v * v)));
        } else {
            result = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
        }
        x[i] = __float2bfloat16_rn(result);
    }
}

template <bool Tanh>
void gelu_impl(Tensor& x, cudaStream_t stream) {
    if (x.dtype != DType::BF16) { throw std::invalid_argument("vision gelu: expected BF16"); }
    const auto n = elements(x);
    if (!n) { return; }
    storage(x, DType::BF16);
    gelu_kernel<Tanh><<<grid_for(n), kBlock, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n);
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
    bias_kernel<<<grid_for(n), kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(bias.data),
                                                    static_cast<__nv_bfloat16*>(x.data), n, x.ne[0]);
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
    norm_kernel<<<grid_for(rows, kBlock / 32), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(bias.data), static_cast<__nv_bfloat16*>(out.data),
        rows, x.ne[0], eps);
    CUDA_CHECK(cudaGetLastError());
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
    position_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(table.data), static_cast<const std::int32_t*>(indices.data),
        static_cast<const float*>(weights.data), static_cast<__nv_bfloat16*>(x.data), n,
        x.ne[0], table.ne[1]);
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
    rope_kernel<<<grid_for(tokens, 1), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), tokens, q.nb[2] / 2, k.nb[2] / 2);
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
    residual_kernel<<<grid_for(n), kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(y.data),
                                                        static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void gelu_tanh(Tensor& x, cudaStream_t stream) { gelu_impl<true>(x, stream); }
void gelu_exact(Tensor& x, cudaStream_t stream) { gelu_impl<false>(x, stream); }

} // namespace ninfer::targets::qwen3_6::vision_sm86
