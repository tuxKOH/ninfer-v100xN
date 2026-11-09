#include "targets/qwen3_6/impl/vision/sm86/attention.h"
#include "targets/qwen3_6/impl/vision/sm86/elementwise.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace vision = ninfer::targets::qwen3_6::vision_sm86;

namespace {
constexpr int dim = 72;
constexpr int heads = 16;
constexpr int plane = dim * heads;
constexpr PointwiseCriterion tolerance{0.0001, 0.0045};

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) result[i] = f32_to_bf16(values[i]);
    return result;
}

template<class F> int rejects(const char* label, F f) {
    try { f(); } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int attention(bool isolation, bool increasing = false) {
    const std::vector<int> bounds{0, 1, 8, 41};
    constexpr int patches = 41;
    const std::size_t count = patches * plane;
    std::vector<float> q(count), k(count), v(count);
    fill_uniform(q, 41, -1.5f, 1.5f);
    fill_uniform(k, 42, -1.5f, 1.5f);
    fill_uniform(v, 43, -2.0f, 2.0f);
    if (isolation) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(k.begin(), k.end(), 0.0f);
        for (std::size_t s = 0; s + 1 < bounds.size(); ++s)
            std::fill(v.begin() + bounds[s] * plane, v.begin() + bounds[s + 1] * plane,
                      static_cast<float>(int(s) * 5 - 4));
    }
    if (increasing) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(k.begin(), k.end(), 0.0f);
        for (int t = 0; t < patches; ++t) {
            for (int h = 0; h < heads; ++h) {
                q[t * plane + h * dim] = 1.0f;
                k[t * plane + h * dim] = static_cast<float>(t);
            }
        }
    }
    round_to_bf16(q); round_to_bf16(k); round_to_bf16(v);
    std::vector<double> ref(count);
    for (std::size_t s = 0; s + 1 < bounds.size(); ++s) {
        for (int t = bounds[s]; t < bounds[s + 1]; ++t) {
            for (int h = 0; h < heads; ++h) {
                std::vector<double> scores(bounds[s + 1] - bounds[s]);
                double maximum = -std::numeric_limits<double>::infinity();
                for (int u = bounds[s]; u < bounds[s + 1]; ++u) {
                    double dot = 0;
                    for (int d = 0; d < dim; ++d)
                        dot += double(q[t * plane + h * dim + d]) * k[u * plane + h * dim + d];
                    scores[u - bounds[s]] = dot / std::sqrt(double(dim));
                    maximum = std::max(maximum, scores[u - bounds[s]]);
                }
                double sum = 0;
                for (double& score : scores) { score = std::exp(score - maximum); sum += score; }
                for (int d = 0; d < dim; ++d) {
                    double value = 0;
                    for (int u = bounds[s]; u < bounds[s + 1]; ++u)
                        value += scores[u - bounds[s]] * v[u * plane + h * dim + d];
                    ref[t * plane + h * dim + d] = value / sum;
                }
            }
        }
    }
    const auto qb = bits(q), kb = bits(k), vb = bits(v);
    std::vector<std::uint16_t> packed(count * 3);
    for (int t = 0; t < patches; ++t) {
        std::copy_n(qb.begin() + t * plane, plane, packed.begin() + t * plane * 3);
        std::copy_n(kb.begin() + t * plane, plane, packed.begin() + t * plane * 3 + plane);
        std::copy_n(vb.begin() + t * plane, plane, packed.begin() + t * plane * 3 + 2 * plane);
    }
    auto d_qkv = to_device(packed);
    auto d_bounds = to_device_i32(bounds);
    GuardedDeviceBuffer d_out(count * 2);
    d_out.fill(0x7f);
    Tensor qt(d_qkv.p, DType::BF16, {dim, heads, patches});
    qt.nb[2] = plane * 3 * 2;
    Tensor kt = qt, vt = qt;
    kt.data = static_cast<std::uint16_t*>(d_qkv.p) + plane;
    vt.data = static_cast<std::uint16_t*>(d_qkv.p) + 2 * plane;
    Tensor bt(d_bounds.p, DType::I32, {int(bounds.size())});
    Tensor out(d_out.data(), DType::BF16, {dim, heads, patches});
    vision::attention_cu_seqlens(qt, kt, vt, bt, out, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(isolation ? "attention isolation" :
                                    increasing ? "attention increasing maxima" : "attention strided",
                                    from_device_bf16(d_out.data(), count), ref, tolerance);
    failures += d_out.verify_guards("attention output");
    failures += verify_exact("attention preserves qkv", from_device<std::uint16_t>(d_qkv, packed.size()), packed);
    failures += verify_exact("attention preserves bounds", from_device<int>(d_bounds, bounds.size()), bounds);
    if (!isolation) {
        failures += rejects("attention output alias", [&] { vision::attention_cu_seqlens(qt, kt, vt, bt, qt, nullptr); });
        Tensor bad = qt; bad.nb[2] = 2;
        failures += rejects("attention stride", [&] { vision::attention_cu_seqlens(bad, kt, vt, bt, out, nullptr); });
        for (const std::vector<int>& invalid : {std::vector<int>{1, 2, 8, 41}, {0, 1, 1, 41}, {0, 1, 8, 40}}) {
            auto d_invalid = to_device_i32(invalid);
            Tensor invalid_t(d_invalid.p, DType::I32, {int(invalid.size())});
            failures += rejects("attention bounds", [&] { vision::attention_cu_seqlens(qt, kt, vt, invalid_t, out, nullptr); });
        }
    }
    return failures;
}

int attention_tiles(const std::vector<int>& bounds, bool isolation, bool extreme, bool identical = false) {
    const int patches = bounds.back();
    const std::size_t count = std::size_t(patches) * plane;
    // An odd BF16 stride exercises both aligned and unaligned row loads.
    constexpr int stride = 3 * plane + 33;
    std::vector<float> q(count), k(count), v(count);
    fill_uniform(q, 71, -1.0f, 1.0f);
    fill_uniform(k, 72, -1.0f, 1.0f);
    fill_uniform(v, 73, -2.0f, 2.0f);
    if (identical) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(k.begin(), k.end(), 0.0f);
    }
    for (std::size_t s = 0; s + 1 < bounds.size(); ++s) {
        for (int t = bounds[s]; t < bounds[s + 1]; ++t) {
            for (int h = 0; h < heads; ++h) {
                if (extreme) {
                    // Large common offsets require max subtraction. Alternate heads
                    // reverse the maxima, and repeated maxima span tile boundaries.
                    const int pos = t - bounds[s];
                    const float key = 64.0f + float((pos % 17) - 8);
                    for (int d = 0; d < dim; ++d) {
                        q[t * plane + h * dim + d] = 0.0f;
                        k[t * plane + h * dim + d] = 0.0f;
                    }
                    q[t * plane + h * dim] = h % 2 ? -16.0f : 16.0f;
                    k[t * plane + h * dim] = key;
                }
                if (isolation) {
                    for (int d = 0; d < dim; ++d)
                        v[t * plane + h * dim + d] = float(int(s) * 4 - 12);
                }
            }
        }
    }
    round_to_bf16(q); round_to_bf16(k); round_to_bf16(v);
    std::vector<double> ref(count);
    for (std::size_t s = 0; s + 1 < bounds.size(); ++s) {
        const int begin = bounds[s], end = bounds[s + 1];
        std::vector<double> weights(end - begin);
        for (int t = begin; t < end; ++t) {
            for (int h = 0; h < heads; ++h) {
                double maximum = -std::numeric_limits<double>::infinity();
                for (int u = begin; u < end; ++u) {
                    double dot = 0.0;
                    for (int d = 0; d < dim; ++d)
                        dot += double(q[t * plane + h * dim + d]) * double(k[u * plane + h * dim + d]);
                    weights[u - begin] = dot / std::sqrt(double(dim));
                    maximum = std::max(maximum, weights[u - begin]);
                }
                double sum = 0.0;
                for (double& weight : weights) {
                    weight = std::exp(weight - maximum);
                    sum += weight;
                }
                for (int d = 0; d < dim; ++d) {
                    double value = 0.0;
                    for (int u = begin; u < end; ++u)
                        value += weights[u - begin] * double(v[u * plane + h * dim + d]);
                    ref[t * plane + h * dim + d] = value / sum;
                }
            }
        }
    }
    const auto qb = bits(q), kb = bits(k), vb = bits(v);
    std::vector<std::uint16_t> packed(std::size_t(patches) * stride + 1, f32_to_bf16(31.0f));
    for (int t = 0; t < patches; ++t) {
        std::copy_n(qb.begin() + t * plane, plane, packed.begin() + 1 + t * stride);
        std::copy_n(kb.begin() + t * plane, plane, packed.begin() + 1 + t * stride + plane);
        std::copy_n(vb.begin() + t * plane, plane, packed.begin() + 1 + t * stride + 2 * plane);
    }
    auto d_qkv = to_device(packed);
    auto d_bounds = to_device_i32(bounds);
    GuardedDeviceBuffer d_out(count * sizeof(std::uint16_t));
    d_out.fill(0x7f);
    Tensor qt(static_cast<std::uint16_t*>(d_qkv.p) + 1, DType::BF16, {dim, heads, patches});
    qt.nb[2] = stride * sizeof(std::uint16_t);
    Tensor kt = qt, vt = qt;
    kt.data = static_cast<std::uint16_t*>(d_qkv.p) + 1 + plane;
    vt.data = static_cast<std::uint16_t*>(d_qkv.p) + 1 + 2 * plane;
    Tensor bt(d_bounds.p, DType::I32, {int(bounds.size())});
    Tensor out(d_out.data(), DType::BF16, {dim, heads, patches});
    vision::attention_cu_seqlens(qt, kt, vt, bt, out, nullptr);
    cuda_synchronize();
    const char* label = extreme ? "attention extreme softmax" :
                        identical ? "attention identical logits" :
                        isolation ? "attention ragged isolation" : "attention tile boundaries";
    int failures = verify_pointwise(label, from_device_bf16(d_out.data(), count), ref, tolerance);
    failures += d_out.verify_guards(label);
    failures += verify_exact("attention tiled preserves qkv", from_device<std::uint16_t>(d_qkv, packed.size()), packed);
    failures += verify_exact("attention tiled preserves bounds", from_device<int>(d_bounds, bounds.size()), bounds);
    return failures;
}

int attention_long() {
    const std::vector<int> bounds{0, 2049, 3074};
    constexpr int patches = 3074;
    constexpr int stride = 3 * plane;
    const std::size_t count = std::size_t(patches) * plane;
    std::vector<std::uint16_t> packed(std::size_t(patches) * stride, f32_to_bf16(0.0f));
    std::vector<double> ref(count);
    for (std::size_t s = 0; s + 1 < bounds.size(); ++s) {
        const int begin = bounds[s], end = bounds[s + 1];
        for (int c = 0; c < plane; ++c) {
            double sum = 0;
            for (int t = begin; t < end; ++t) {
                const float value = float(int(s) * 4 + (t - begin) % 17 - 8) +
                                    float(c % 4) * 0.25f;
                packed[std::size_t(t) * stride + 2 * plane + c] = f32_to_bf16(value);
                sum += value;
            }
            for (int t = begin; t < end; ++t) ref[std::size_t(t) * plane + c] = sum / (end - begin);
        }
    }
    GuardedDeviceBuffer d_qkv(packed.size() * sizeof(std::uint16_t));
    d_qkv.copy_from_host(packed.data(), d_qkv.bytes());
    GuardedDeviceBuffer d_bounds(bounds.size() * sizeof(int));
    d_bounds.copy_from_host(bounds.data(), d_bounds.bytes());
    GuardedDeviceBuffer d_out(count * sizeof(std::uint16_t));
    d_out.fill(0x7f);
    Tensor qt(d_qkv.data(), DType::BF16, {dim, heads, patches});
    qt.nb[2] = stride * sizeof(std::uint16_t);
    Tensor kt = qt, vt = qt;
    kt.data = static_cast<std::uint16_t*>(d_qkv.data()) + plane;
    vt.data = static_cast<std::uint16_t*>(d_qkv.data()) + 2 * plane;
    Tensor bt(d_bounds.data(), DType::I32, {int(bounds.size())});
    Tensor out(d_out.data(), DType::BF16, {dim, heads, patches});
    vision::attention_cu_seqlens(qt, kt, vt, bt, out, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise("attention long identical logits",
                                    from_device_bf16(d_out.data(), count), ref, tolerance);
    failures += d_out.verify_guards("attention long output");
    failures += d_qkv.verify_guards("attention long qkv");
    failures += d_bounds.verify_guards("attention long bounds");
    failures += verify_exact("attention long preserves qkv",
                             from_device<std::uint16_t>(d_qkv.data(), packed.size()), packed);
    failures += verify_exact("attention long preserves bounds",
                             from_device<int>(d_bounds.data(), bounds.size()), bounds);
    return failures;
}

int attention_extent_checks() {
    constexpr std::int64_t patches = 65536;
    int failures = 0;
    failures += rejects("negative patches", [] { (void)vision::attention_workspace_bytes(-1, 1); });
    failures += rejects("zero segments", [] { (void)vision::attention_workspace_bytes(1, 0); });
    Tensor q(reinterpret_cast<void*>(std::uintptr_t(2)), DType::BF16, {dim, heads, int(patches)});
    Tensor bounds(reinterpret_cast<void*>(std::uintptr_t(4)), DType::I32, {2});
    Tensor out = q;
    q.nb[2] = ((std::numeric_limits<std::int64_t>::max() - plane * 2) / (patches - 1) + 2) & ~std::int64_t(1);
    try {
        vision::attention_cu_seqlens(q, q, q, bounds, out, nullptr);
        std::cerr << "attention extent overflow: expected overflow_error\n";
        ++failures;
    } catch (const std::overflow_error&) {}
    return failures;
}

int elementwise() {
    constexpr int d = 37, rows = 5, n = d * rows;
    std::vector<float> x(n), y(n), weight(d), bias(d);
    fill_uniform(x, 11, -3, 3); fill_uniform(y, 12, -2, 2);
    fill_uniform(weight, 13, -1, 1); fill_uniform(bias, 14, -1, 1);
    round_to_bf16(x); round_to_bf16(y); round_to_bf16(weight); round_to_bf16(bias);
    auto dx = to_device_bf16(x), dy = to_device_bf16(y);
    auto dw = to_device_bf16(weight), db = to_device_bf16(bias);
    GuardedDeviceBuffer dout(n * 2);
    Tensor xt(dx.p, DType::BF16, {d, rows}), yt(dy.p, DType::BF16, {d, rows});
    Tensor wt(dw.p, DType::BF16, {d}), bt(db.p, DType::BF16, {d});
    Tensor out(dout.data(), DType::BF16, {d, rows});
    std::vector<double> ref(n);
    for (int r = 0; r < rows; ++r) {
        double mean = 0, variance = 0;
        for (int i = 0; i < d; ++i) mean += x[r * d + i];
        mean /= d;
        for (int i = 0; i < d; ++i) variance += std::pow(x[r * d + i] - mean, 2);
        variance /= d;
        for (int i = 0; i < d; ++i)
            ref[r * d + i] = (x[r * d + i] - mean) / std::sqrt(variance + 1e-5f) * weight[i] + bias[i];
    }
    vision::layer_norm(xt, wt, bt, 1e-5f, out, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise("layer norm", from_device_bf16(dout.data(), n), ref, tolerance);
    failures += dout.verify_guards("layer norm output");
    failures += rejects("layer norm epsilon", [&] { vision::layer_norm(xt, wt, bt, 0, out, nullptr); });
    failures += rejects("layer norm nan epsilon", [&] { vision::layer_norm(xt, wt, bt, std::numeric_limits<float>::quiet_NaN(), out, nullptr); });
    Tensor short_bias(db.p, DType::BF16, {d - 1});
    failures += rejects("bias shape", [&] { vision::add_bias(short_bias, xt, nullptr); });
    for (int i = 0; i < n; ++i) ref[i] = double(x[i]) + bias[i % d];
    vision::add_bias(bt, xt, nullptr);
    cuda_synchronize();
    failures += verify_pointwise("bias", from_device_bf16(dx, n), ref, tolerance);
    dx = to_device_bf16(x); xt.data = dx.p;
    for (int i = 0; i < n; ++i) ref[i] = double(x[i]) + y[i];
    vision::residual_add(yt, xt, nullptr);
    cuda_synchronize();
    failures += verify_pointwise("residual", from_device_bf16(dx, n), ref, tolerance);
    Tensor wrong(dy.p, DType::BF16, {d, rows - 1});
    failures += rejects("residual shape", [&] { vision::residual_add(wrong, xt, nullptr); });
    for (bool approximate : {false, true}) {
        dx = to_device_bf16(x); xt.data = dx.p;
        for (int i = 0; i < n; ++i) {
            const double z = x[i];
            ref[i] = approximate ? 0.5 * z * (1 + std::tanh(std::sqrt(2 / std::acos(-1.0)) * (z + 0.044715 * z * z * z)))
                                 : 0.5 * z * (1 + std::erf(z / std::sqrt(2.0)));
        }
        if (approximate) vision::gelu_tanh(xt, nullptr); else vision::gelu_exact(xt, nullptr);
        cuda_synchronize();
        failures += verify_pointwise(approximate ? "gelu tanh" : "gelu exact", from_device_bf16(dx, n), ref, tolerance);
    }
    Tensor invalid = xt; invalid.dtype = DType::FP32;
    failures += rejects("gelu dtype", [&] { vision::gelu_exact(invalid, nullptr); });
    return failures;
}

int position() {
    constexpr int d = 37, rows = 7, patches = 3;
    std::vector<float> table(d * rows), x(d * patches);
    fill_uniform(table, 20, -2, 2); fill_uniform(x, 21, -2, 2);
    round_to_bf16(table); round_to_bf16(x);
    const std::vector<int> indices{0, 1, 3, 6, 6, 2, 2, 0, 4, 3, 1, 5};
    const std::vector<float> weights{0.1f, 0.2f, 0.3f, 0.4f, 1, 0, 0, 0, 0.25f, 0.25f, 0.25f, 0.25f};
    std::vector<double> ref(x.size());
    for (int p = 0; p < patches; ++p) for (int c = 0; c < d; ++c) {
        double v = x[p * d + c];
        for (int j = 0; j < 4; ++j) v += double(table[indices[p * 4 + j] * d + c]) * weights[p * 4 + j];
        ref[p * d + c] = v;
    }
    auto dt = to_device_bf16(table), dx = to_device_bf16(x), di = to_device_i32(indices), dw = to_device(weights);
    Tensor tt(dt.p, DType::BF16, {d, rows}), xt(dx.p, DType::BF16, {d, patches});
    Tensor it(di.p, DType::I32, {4, patches}), wt(dw.p, DType::FP32, {4, patches});
    vision::position_add(tt, it, wt, xt, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise("position interpolation", from_device_bf16(dx, x.size()), ref, tolerance);
    failures += verify_exact("position table unchanged", from_device<std::uint16_t>(dt, table.size()), bits(table));
    Tensor bad = it; bad.ne[0] = 3;
    failures += rejects("position control shape", [&] { vision::position_add(tt, bad, wt, xt, nullptr); });
    return failures;
}

int rope() {
    constexpr int patches = 3, qs = plane + 11, ks = plane + 19;
    const std::vector<int> positions{0, 3, 17, 0, 11, 2};
    std::vector<float> q(patches * qs), k(patches * ks);
    fill_uniform(q, 30, -2, 2); fill_uniform(k, 31, -2, 2);
    round_to_bf16(q); round_to_bf16(k);
    auto reference = [&](const std::vector<float>& input, int stride) {
        std::vector<double> result(input.begin(), input.end());
        for (int p = 0; p < patches; ++p) for (int h = 0; h < heads; ++h)
            for (int pair = 0; pair < 36; ++pair) {
                const double angle = positions[(pair / 18) * patches + p] * std::pow(10000.0, -double(pair % 18) / 18.0);
                const int i = p * stride + h * dim + pair;
                result[i] = input[i] * std::cos(angle) - input[i + 36] * std::sin(angle);
                result[i + 36] = input[i + 36] * std::cos(angle) + input[i] * std::sin(angle);
            }
        return result;
    };
    auto qr = reference(q, qs), kr = reference(k, ks);
    auto dq = to_device_bf16(q), dk = to_device_bf16(k), dp = to_device_i32(positions);
    Tensor qt(dq.p, DType::BF16, {dim, heads, patches}), kt(dk.p, DType::BF16, {dim, heads, patches});
    qt.nb[2] = qs * 2; kt.nb[2] = ks * 2;
    Tensor pt(dp.p, DType::I32, {patches, 2});
    vision::rope_2d(pt, qt, kt, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise("rope q", from_device_bf16(dq, q.size()), qr, tolerance);
    failures += verify_pointwise("rope k", from_device_bf16(dk, k.size()), kr, tolerance);
    failures += verify_exact("rope positions unchanged", from_device<int>(dp, positions.size()), positions);
    Tensor bad = kt; bad.nb[2] = plane * 2 - 2;
    failures += rejects("rope stride", [&] { vision::rope_2d(pt, qt, bad, nullptr); });
    return failures;
}
} // namespace

int main() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice) { std::cout << "SKIP: CUDA device unavailable\n"; return 77; }
    cuda_check(status, "cudaGetDeviceCount");
    int device = -1;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties");
        if (prop.major * 10 + prop.minor == NINFER_AUX_VISION_ARCH) { device = i; break; }
    }
    if (device < 0) { std::cout << "SKIP: SM" << NINFER_AUX_VISION_ARCH << " required\n"; return 77; }
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
    int failures = 0;
    if (vision::attention_workspace_bytes(9, 3) != 0) { std::cerr << "attention workspace nonzero\n"; ++failures; }
    failures += rejects("zero patches", [] { (void)vision::attention_workspace_bytes(0, 1); });
    failures += rejects("too many segments", [] { (void)vision::attention_workspace_bytes(2, 3); });
    failures += attention(false) + attention(true) + attention(false, true) + elementwise() + position() + rope();
    failures += attention_tiles({0, 63, 127, 192, 319, 447, 576, 833}, false, false);
    failures += attention_tiles({0, 63, 127, 192, 321}, true, false);
    failures += attention_tiles({0, 65, 194, 451}, false, true);
    failures += attention_tiles({0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 66, 129}, false, false, true);
    failures += attention_long() + attention_extent_checks();
    std::cout << (failures ? "FAIL" : "PASS") << " vision_sm86_ops\n";
    return failures ? 1 : 0;
}
