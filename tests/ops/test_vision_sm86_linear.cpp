#include "ninfer/ops/linear.h"
#include "ops/quantized_weight.h"
#include "targets/qwen3_6/impl/vision/sm86/linear.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
namespace fixture = ninfer::test::quantized_weight;
namespace vision = ninfer::targets::qwen3_6::vision_sm86;

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Device allocations have 256-byte guards, preserving the alignment of the payload.
class Guarded {
public:
    static constexpr std::size_t guard = 256;
    static constexpr std::uint8_t marker = 0xa5;
    explicit Guarded(std::size_t size) : size_(size) {
        check(cudaMalloc(&base_, size + 2 * guard));
        check(cudaMemset(base_, marker, size + 2 * guard));
    }
    ~Guarded() { cudaFree(base_); }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    void* data() const { return static_cast<std::uint8_t*>(base_) + guard; }
    void upload(const void* source) { check(cudaMemcpy(data(), source, size_, cudaMemcpyHostToDevice)); }
    std::vector<std::uint8_t> download() const {
        std::vector<std::uint8_t> result(size_);
        check(cudaMemcpy(result.data(), data(), size_, cudaMemcpyDeviceToHost));
        return result;
    }
    void check_guards() const {
        std::vector<std::uint8_t> before(guard), after(guard);
        check(cudaMemcpy(before.data(), base_, guard, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(after.data(), static_cast<std::uint8_t*>(data()) + size_, guard,
                         cudaMemcpyDeviceToHost));
        require(std::all_of(before.begin(), before.end(), [](auto b) { return b == marker; }) &&
                std::all_of(after.begin(), after.end(), [](auto b) { return b == marker; }),
                "device guard overwritten");
    }
private:
    void* base_ = nullptr;
    std::size_t size_;
};

std::uint16_t bf16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
float from_bf16(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}

template<class F> void rejects(const char* label, F change, const Tensor& input,
                               const Weight& weight, const Tensor& output, cudaStream_t stream) {
    auto x = input;
    auto w = weight;
    auto y = output;
    change(x, w, y);
    try { vision::linear(x, w, y, stream); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(std::string("linear accepted invalid ") + label);
}

void invalid_cases(const Tensor& x, const Weight& w, const Tensor& y, cudaStream_t stream) {
    rejects("layout", [](auto&, auto& a, auto&) { a.layout = QuantLayout::Contiguous; }, x,w,y,stream);
    rejects("qtype", [](auto&, auto& a, auto&) { a.qtype = QType::BF16_CTRL; }, x,w,y,stream);
    rejects("group", [](auto&, auto& a, auto&) { ++a.group; }, x,w,y,stream);
    rejects("group size", [](auto&, auto& a, auto&) { ++a.group_size; }, x,w,y,stream);
    rejects("scale dtype", [](auto&, auto& a, auto&) { a.scale_dtype = DType::FP32; }, x,w,y,stream);
    rejects("shape", [](auto&, auto& a, auto&) { ++a.shape[0]; }, x,w,y,stream);
    rejects("rank", [](auto&, auto& a, auto&) { a.ndim = 1; }, x,w,y,stream);
    rejects("padded K", [](auto&, auto& a, auto&) { a.padded_shape[1] += 128; }, x,w,y,stream);
    rejects("payload extent", [](auto&, auto& a, auto&) { --a.payload_bytes; }, x,w,y,stream);
    rejects("high extent", [](auto&, auto& a, auto&) { ++a.high_plane_bytes; }, x,w,y,stream);
    rejects("low plane", [](auto&, auto& a, auto&) { a.qdata = nullptr; }, x,w,y,stream);
    rejects("high plane", [](auto&, auto& a, auto&) { a.qhigh = a.payload; }, x,w,y,stream);
    rejects("scale plane", [](auto&, auto& a, auto&) { a.scales = a.payload; }, x,w,y,stream);
    rejects("input extent", [](auto& a, auto&, auto&) { --a.ne[0]; }, x,w,y,stream);
    rejects("zero tokens", [](auto& a, auto&, auto&) { a.ne[1] = 0; }, x,w,y,stream);
    rejects("input stride", [](auto& a, auto&, auto&) { a.nb[0] += 2; }, x,w,y,stream);
    rejects("output extent", [](auto&, auto&, auto& a) { --a.ne[0]; }, x,w,y,stream);
    rejects("output dtype", [](auto&, auto&, auto& a) { a.dtype = DType::FP32; }, x,w,y,stream);
    rejects("output alignment", [](auto&, auto&, auto& a) {
        a.data = static_cast<std::uint8_t*>(a.data) + 2;
    }, x,w,y,stream);
    rejects("input/output overlap", [](auto& a, auto&, auto& b) { b.data = a.data; }, x,w,y,stream);
    rejects("weight/output overlap", [](auto&, auto& a, auto& b) {
        b.data = const_cast<void*>(a.payload);
    }, x,w,y,stream);
}

struct Shape { QType type; int n, k, tokens; };

template<class Linear> void run_case(Shape shape, cudaStream_t stream, Linear&& linear) {
    auto packed = fixture::make_patterned_weight(shape.type, shape.n, shape.k, 33,
        {.row_split_scale = fixture::RowSplitScalePattern::Small,
         .row_split_codes = fixture::RowSplitCodePattern::Hashed});
    // Halfway between BF16 values, so the oracle must use the stored FP16 scales.
    for (std::size_t offset = packed.scale_plane_offset; offset < packed.payload.size(); offset += 2) {
        if (packed.payload[offset] != 0 || packed.payload[offset + 1] != 0)
            packed.payload[offset] += 4;
    }
    const std::size_t input_count = static_cast<std::size_t>(shape.k) * shape.tokens;
    const std::size_t output_count = static_cast<std::size_t>(shape.n) * shape.tokens;
    std::vector<std::uint16_t> input(input_count);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto v = static_cast<int>((i * 7919 + (i / shape.k) * 101 + 37) % 509) - 254;
        input[i] = bf16(static_cast<float>(v) / 1024.0f);
    }
    Guarded weights(packed.payload.size()), inputs(input.size() * sizeof(input[0]));
    Guarded outputs(output_count * sizeof(std::uint16_t));
    weights.upload(packed.payload.data());
    inputs.upload(input.data());
    auto w = packed.device_weight(weights.data());
    Tensor x(inputs.data(), DType::BF16, {shape.k, shape.tokens});
    Tensor y(outputs.data(), DType::BF16, {shape.n, shape.tokens});
    invalid_cases(x, w, y, stream);
    linear(x, w, y, stream);
    check(cudaStreamSynchronize(stream));
    const auto first = outputs.download();
    linear(x, w, y, stream);
    check(cudaStreamSynchronize(stream));
    const auto second = outputs.download();
    require(first == second, "linear is not repeatable");
    weights.check_guards(); inputs.check_guards(); outputs.check_guards();
    require(weights.download() == packed.payload, "linear modified weights");
    const auto input_after = inputs.download();
    require(std::memcmp(input_after.data(), input.data(), input_after.size()) == 0,
            "linear modified input");

    // Decode represented values from the stored planes, independently of the GPU kernel.
    std::vector<double> sums(static_cast<std::size_t>(shape.tokens));
    for (int row = 0; row < shape.n; ++row) {
        std::fill(sums.begin(), sums.end(), 0.0);
        for (int k = 0; k < shape.k; ++k) {
            const double value = fixture::logical_weight_fp64(packed, row, k);
            for (int t = 0; t < shape.tokens; ++t) {
                sums[t] += value * static_cast<double>(from_bf16(input[static_cast<std::size_t>(t) * shape.k + k]));
            }
        }
        for (int t = 0; t < shape.tokens; ++t) {
            const auto index = static_cast<std::size_t>(t) * shape.n + row;
            std::uint16_t bits;
            std::memcpy(&bits, first.data() + index * sizeof(bits), sizeof(bits));
            const double actual = from_bf16(bits);
            const double expected = sums[t];
            if (!std::isfinite(actual) || std::abs(actual - expected) > 0.004 * std::abs(expected) + 0.00003) {
                std::cerr << "linear type=" << static_cast<int>(shape.type) << " N=" << shape.n
                          << " K=" << shape.k << " T=" << shape.tokens << " row=" << row
                          << " token=" << t << " actual=" << actual << " expected=" << expected << '\n';
                throw std::runtime_error("linear FP64 oracle mismatch");
            }
        }
    }
}
} // namespace

int main() {
    cudaStream_t stream = nullptr;
    try {
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice) return 77;
        check(status);
        int auxiliary = -1;
        int text = -1;
        for (int device = 0; device < count; ++device) {
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, device));
            const int arch = properties.major * 10 + properties.minor;
            if (arch == NINFER_AUX_VISION_ARCH) auxiliary = device;
            if (arch == 70) text = device;
        }
        if (auxiliary < 0 || text < 0) return 77;
        check(cudaSetDevice(text));
        check(cudaStreamCreate(&stream));
        const auto text_linear = [](const Tensor& x, const Weight& w, Tensor& y, cudaStream_t s) {
            ops::linear(x, w, y, s);
        };
        for (const auto shape : {Shape{QType::Q5G64_F16S, 1152, 1152, 4},
                                 {QType::W8G32_F16S, 2048, 4096, 5}}) {
            run_case(shape, stream, text_linear);
        }
        check(cudaStreamDestroy(stream));
        stream = nullptr;
        check(cudaSetDevice(auxiliary));
        check(cudaStreamCreate(&stream));
        for (const auto shape : {Shape{QType::Q6G64_F16S,1152,1536,3},
                {QType::Q4G64_F16S,3456,1152,1}, {QType::Q5G64_F16S,1152,1152,5},
                {QType::Q4G64_F16S,4304,1152,3}, {QType::Q5G64_F16S,1152,4304,3},
                {QType::W8G32_F16S,4608,4608,1}, {QType::W8G32_F16S,5120,4608,3},
                {QType::W8G32_F16S,2048,4608,5},
                {QType::Q4G64_F16S,5,33,52429},
                {QType::Q5G64_F16S,7,65,3}, {QType::Q6G64_F16S,7,65,3},
                {QType::W8G32_F16S,7,33,3},
                {QType::Q4G64_F16S,16,64,16}, {QType::Q4G64_F16S,19,65,21},
                {QType::Q5G64_F16S,16,64,16}, {QType::Q5G64_F16S,19,65,21},
                {QType::Q6G64_F16S,16,64,16}, {QType::Q6G64_F16S,19,65,21},
                {QType::W8G32_F16S,16,64,16}, {QType::W8G32_F16S,19,65,21}}) {
            run_case(shape, stream, [](const Tensor& x, const Weight& w, Tensor& y, cudaStream_t s) {
                vision::linear(x, w, y, s);
            });
        }
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& error) {
        if (stream) cudaStreamDestroy(stream);
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
