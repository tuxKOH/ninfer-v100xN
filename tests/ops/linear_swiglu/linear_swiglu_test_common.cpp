#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include "core/arena.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::test::linear_swiglu {
namespace {

// The criterion belongs to the activation-compute profile, not the weight storage format or a
// private materialized/fused implementation.
constexpr ReductionCriterion tolerance_for(ActivationCompute activation_compute) {
    switch (activation_compute) {
    case ActivationCompute::A16:
        return {3.3e-3, 5.0e-3, 6.3e-3};
    case ActivationCompute::A8:
        // Both independently A8-quantized projections feed the nonlinear product, so this profile
        // allows twice Linear's relative-L2 quantization allowance plus a bounded gross tail.
        return {8.0e-2, 1.0e-2, 1.2e-1};
    case ActivationCompute::A4:
        return {1.6e-1, 1.0e-2, 1.6e-1};
    }
    throw std::invalid_argument("linear_swiglu test: unknown activation compute profile");
}

bool cuda_available() { return !test::cuda_unavailable(); }

std::size_t checked_elements(std::int32_t first, std::int32_t second, const char* label) {
    if (first <= 0 || second <= 0) {
        throw std::invalid_argument(std::string("linear_swiglu test: invalid ") + label);
    }
    const std::size_t a = static_cast<std::size_t>(first);
    const std::size_t b = static_cast<std::size_t>(second);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("linear_swiglu test: ") + label + " overflows");
    }
    return a * b;
}

std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::vector<std::uint16_t> make_activation(const Profile& profile, std::int32_t tokens) {
    std::vector<std::uint16_t> activation(
        checked_elements(profile.input_rows, tokens, "activation size"), 0);

    // Token zero is dense, so every logical K contributes to every production implementation.
    // Later tokens use exact zeros plus a rotating set of nonzeros. This keeps a full-output,
    // full-formula oracle practical at large registered T boundaries without adopting any
    // production staging or reduction behavior.
    const bool native_float_weight =
        profile.qtype == QType::NVFP4 || profile.qtype == QType::FP8_E4M3FN_ROW_BF16S ||
        profile.qtype == QType::FP8_E4M3FN_BLOCK128_BF16S;
    const float dense_scale =
        profile.qtype == QType::Q4G64_F16S ? 1.25e-4F : (native_float_weight ? 1.0e-3F : 1.0e-5F);
    for (std::int32_t column = 0; column < profile.input_rows; ++column) {
        const std::uint64_t mixed = mix64((static_cast<std::uint64_t>(profile.seed) << 32) |
                                          static_cast<std::uint32_t>(column));
        int numerator             = static_cast<int>((mixed >> 48) % 63U) - 31;
        if (numerator == 0) { numerator = (column & 1) == 0 ? 1 : -1; }
        const float value                            = static_cast<float>(numerator) * dense_scale;
        activation[static_cast<std::size_t>(column)] = test::f32_to_bf16(value);
    }

    constexpr std::int32_t kNonzerosPerSparseToken = 4;
    const float sparse_scale =
        profile.qtype == QType::Q4G64_F16S ? 1.5e-2F : (native_float_weight ? 2.0e-2F : 1.5e-3F);
    for (std::int32_t token = 1; token < tokens; ++token) {
        for (std::int32_t lane = 0; lane < kNonzerosPerSparseToken; ++lane) {
            const std::uint64_t mixed =
                mix64((static_cast<std::uint64_t>(profile.seed + 1U) << 32) |
                      static_cast<std::uint32_t>(token * kNonzerosPerSparseToken + lane));
            const std::int32_t column =
                static_cast<std::int32_t>(mixed % static_cast<std::uint64_t>(profile.input_rows));
            int numerator = static_cast<int>((mixed >> 48) % 31U) - 15;
            if (numerator == 0) { numerator = (lane & 1) == 0 ? 1 : -1; }
            const float value = static_cast<float>(numerator) * sparse_scale;
            activation[static_cast<std::size_t>(token) * profile.input_rows + column] =
                test::f32_to_bf16(value);
        }
    }
    return activation;
}

struct ActiveValue {
    std::int32_t token;
    double value;
};

std::vector<std::vector<ActiveValue>>
index_nonzero_activations(const std::vector<std::uint16_t>& activation, std::int32_t input_rows,
                          std::int32_t tokens) {
    std::vector<std::vector<ActiveValue>> by_column(static_cast<std::size_t>(input_rows));
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < input_rows; ++column) {
            const std::uint16_t bits =
                activation[static_cast<std::size_t>(token) * input_rows + column];
            if ((bits & 0x7fffU) == 0U) { continue; }
            by_column[static_cast<std::size_t>(column)].push_back(
                {token, static_cast<double>(test::bf16_to_f32(bits))});
        }
    }
    return by_column;
}

double silu_fp64(double value) {
    if (value >= 0.0) { return value / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

std::vector<double> linear_swiglu_oracle_fp64(const Profile& profile,
                                              const quantized_weight::PackedWeight& weight,
                                              const std::vector<std::uint16_t>& activation,
                                              std::int32_t tokens) {
    const auto active_by_column = index_nonzero_activations(activation, profile.input_rows, tokens);
    std::vector<double> output(checked_elements(profile.output_rows, tokens, "oracle output size"));

    const unsigned hardware_threads = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t thread_count =
        std::min(profile.output_rows, static_cast<std::int32_t>(hardware_threads));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));
    for (std::int32_t thread = 0; thread < thread_count; ++thread) {
        const std::int32_t row_begin = static_cast<std::int32_t>(
            static_cast<std::int64_t>(profile.output_rows) * thread / thread_count);
        const std::int32_t row_end = static_cast<std::int32_t>(
            static_cast<std::int64_t>(profile.output_rows) * (thread + 1) / thread_count);
        workers.emplace_back([&, row_begin, row_end] {
            std::vector<double> gate(static_cast<std::size_t>(tokens));
            std::vector<double> up(static_cast<std::size_t>(tokens));
            for (std::int32_t row = row_begin; row < row_end; ++row) {
                std::fill(gate.begin(), gate.end(), 0.0);
                std::fill(up.begin(), up.end(), 0.0);
                const std::int32_t up_row = profile.output_rows + row;
                for (std::int32_t column = 0; column < profile.input_rows; ++column) {
                    const double gate_weight =
                        quantized_weight::logical_weight_fp64(weight, row, column);
                    const double up_weight =
                        quantized_weight::logical_weight_fp64(weight, up_row, column);
                    for (const ActiveValue active :
                         active_by_column[static_cast<std::size_t>(column)]) {
                        gate[static_cast<std::size_t>(active.token)] += gate_weight * active.value;
                        up[static_cast<std::size_t>(active.token)] += up_weight * active.value;
                    }
                }
                for (std::int32_t token = 0; token < tokens; ++token) {
                    const double fused = silu_fp64(gate[static_cast<std::size_t>(token)]) *
                                         up[static_cast<std::size_t>(token)];
                    output[static_cast<std::size_t>(token) * profile.output_rows + row] = fused;
                }
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
    return output;
}

std::vector<double> read_bf16_output(const test::GuardedDeviceBuffer& output,
                                     std::size_t elements) {
    std::vector<std::uint16_t> bits(elements);
    cuda_check(cudaMemcpy(bits.data(), output.data(), elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy LinearSwiGLU output");
    std::vector<double> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        values[index] = static_cast<double>(test::bf16_to_f32(bits[index]));
    }
    return values;
}

int compare_output(std::string_view label, const std::vector<double>& actual,
                   const double* reference, ActivationCompute activation_compute) {
    if (actual.empty() || reference == nullptr) {
        std::cerr << label << ": invalid numerical comparison\n";
        return 1;
    }

    return verify_reduction(label, actual, std::span<const double>(reference, actual.size()),
                            tolerance_for(activation_compute));
}

int verify_unchanged(std::string_view label, const test::GuardedDeviceBuffer& device,
                     const void* expected, std::size_t bytes) {
    if (device.bytes() != bytes) {
        std::cerr << label << ": read-only input size mismatch\n";
        return 1;
    }
    std::vector<std::uint8_t> actual(bytes);
    device.copy_to_host(actual.data(), actual.size());
    if (std::memcmp(actual.data(), expected, bytes) == 0) { return 0; }
    std::cerr << label << ": read-only input was modified\n";
    return 1;
}

void validate_profile(const Profile& profile) {
    const bool q4 = profile.qtype == QType::Q4G64_F16S && profile.gate_up_rows == 34816 &&
                    profile.input_rows == 5120 && profile.output_rows == 17408;
    const bool w8 = profile.qtype == QType::W8G32_F16S && profile.gate_up_rows == 12288 &&
                    profile.input_rows == 2048 && profile.output_rows == 6144;
    bool native_shape = profile.gate_up_rows == 34816 && profile.output_rows == 17408;
#ifdef NINFER_VOLTA_BUILD
    native_shape = native_shape || (profile.gate_up_rows == 8704 && profile.output_rows == 4352);
#endif
    const bool nvfp4 = profile.qtype == QType::NVFP4 && native_shape && profile.input_rows == 5120;
    const bool fp8 = (profile.qtype == QType::FP8_E4M3FN_ROW_BF16S ||
                      profile.qtype == QType::FP8_E4M3FN_BLOCK128_BF16S) &&
                     native_shape && profile.input_rows == 5120;
    if ((!q4 && !w8 && !nvfp4 && !fp8) || profile.gate_up_rows != 2 * profile.output_rows) {
        throw std::invalid_argument("linear_swiglu test: profile is not registered");
    }
    if ((nvfp4 && profile.activation_compute != ActivationCompute::A16 &&
         profile.activation_compute != ActivationCompute::A4) ||
        (fp8 && profile.activation_compute != ActivationCompute::A16 &&
         profile.activation_compute != ActivationCompute::A8) ||
        (!nvfp4 && !fp8 && profile.activation_compute != ActivationCompute::A16)) {
        throw std::invalid_argument("linear_swiglu test: invalid activation-compute profile");
    }
}

} // namespace

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases,
                bool volta_prepacked, bool volta_fp32_scale) {
    validate_profile(profile);
    if (token_cases.empty()) { throw std::invalid_argument("linear_swiglu test: no token cases"); }
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    for (std::size_t index = 0; index < token_cases.size(); ++index) {
        if (token_cases[index] <= 0 ||
            (index != 0 && token_cases[index] <= token_cases[index - 1])) {
            throw std::invalid_argument(
                "linear_swiglu test: token cases must be positive and increasing");
        }
    }
    const std::int32_t maximum_tokens = token_cases.back();

    quantized_weight::PatternedWeightOptions weight_options;
    if (profile.qtype == QType::NVFP4) {
        weight_options.weight_scale_divisor = 0.125F;
        weight_options.input_scale_divisor  = 3.5F;
    }
    quantized_weight::PackedWeight host_weight = quantized_weight::make_patterned_weight(
        profile.qtype, profile.gate_up_rows, profile.input_rows, profile.seed, weight_options);
    const std::vector<std::uint16_t> host_activation = make_activation(profile, maximum_tokens);
    const std::vector<double> reference =
        linear_swiglu_oracle_fp64(profile, host_weight, host_activation, maximum_tokens);

    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    Weight weight = host_weight.device_weight(device_weight.data());
    std::vector<std::uint8_t> expected_device_weight = host_weight.payload;
#ifdef NINFER_VOLTA_BUILD
    if (volta_prepacked) {
        ops::detail::nvfp4_prepack_qpn_sm70(weight);
        weight.volta_fp32_scale = volta_fp32_scale;
        device_weight.copy_to_host(expected_device_weight.data(), expected_device_weight.size());
    }
#else
    if (volta_prepacked || volta_fp32_scale) {
        throw std::invalid_argument("linear_swiglu: Volta profile requires an SM70 build");
    }
#endif

    test::GuardedDeviceBuffer device_activation(host_activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(host_activation.data(),
                                     host_activation.size() * sizeof(std::uint16_t));

    const ops::LinearPolicy policy =
        profile.activation_compute == ActivationCompute::A4
            ? ops::LinearPolicy::AllowA4
            : (profile.activation_compute == ActivationCompute::A8 ? ops::LinearPolicy::AllowA8
                                                                   : ops::LinearPolicy::A16Only);
    const std::size_t workspace_bytes = ops::linear_swiglu_workspace_capacity_bytes(
        profile.qtype, profile.gate_up_rows, profile.input_rows, policy, 1, maximum_tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    int failures = 0;
    for (const std::int32_t tokens : token_cases) {
        const std::size_t output_elements =
            checked_elements(profile.output_rows, tokens, "output size");
        test::GuardedDeviceBuffer output(output_elements * sizeof(std::uint16_t));
        output.fill(0xff);

        Tensor x(device_activation.data(), DType::BF16, {profile.input_rows, tokens});
        Tensor destination(output.data(), DType::BF16, {profile.output_rows, tokens});
        workspace.reset();
        workspace.reset_peak();
        const std::string case_label = std::string(label) + " T=" + std::to_string(tokens);
        try {
            ops::linear_swiglu(x, weight, destination, policy, workspace, nullptr);
            test::cuda_check(cudaDeviceSynchronize(), "synchronize LinearSwiGLU");
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            continue;
        }
        const std::size_t exact_workspace = ops::linear_swiglu_workspace_capacity_bytes(
            profile.qtype, profile.gate_up_rows, profile.input_rows, policy, tokens, tokens);
        if (workspace.used() != 0 || workspace.peak_used() != exact_workspace) {
            std::cerr << case_label << ": exact workspace query/execution high-water mismatch\n";
            ++failures;
        }
        failures += output.verify_guards(case_label);
        const std::vector<double> actual = read_bf16_output(output, output_elements);
        failures +=
            compare_output(case_label, actual, reference.data(), profile.activation_compute);
    }
    failures += device_activation.verify_guards(std::string(label) + " activation");
    failures += device_weight.verify_guards(std::string(label) + " weight");
    failures +=
        verify_unchanged(std::string(label) + " activation", device_activation,
                         host_activation.data(), host_activation.size() * sizeof(std::uint16_t));
    failures += verify_unchanged(std::string(label) + " weight", device_weight,
                                 expected_device_weight.data(), expected_device_weight.size());
    return failures;
}

int run_column_parallel_profile(std::string_view label, const Profile& shard_profile,
                                std::span<const std::int32_t> token_cases,
                                bool volta_prepacked, bool volta_fp32_scale, bool fp16_image,
                                bool fp16_output_image) {
    Profile parent_profile = shard_profile;
    const int tp = 34816 / shard_profile.gate_up_rows;
    if (tp != 2 && tp != 4) { throw std::invalid_argument("LinearSwiGLU split: invalid width"); }
    parent_profile.gate_up_rows *= tp;
    parent_profile.output_rows *= tp;
    validate_profile(parent_profile);
    if (token_cases.empty()) { throw std::invalid_argument("LinearSwiGLU split: no token cases"); }
    if (!cuda_available()) { return 77; }
    int devices = 0;
    cuda_check(cudaGetDeviceCount(&devices), "count devices");
    if (devices < tp) { return 77; }
    const std::array ids{0, 1, 2, 3};
    const ExecutionContext ec{std::vector<int>(ids.begin(), ids.begin() + tp)};
    const std::int32_t maximum_tokens = token_cases.back();
    const auto policy = ops::LinearPolicy::A16Only;
    if (shard_profile.activation_compute != ActivationCompute::A16) {
        throw std::invalid_argument("LinearSwiGLU split oracle fixture requires A16");
    }
    struct Rank {
        quantized_weight::PackedWeight packed;
        std::vector<std::uint16_t> activation, image;
        std::vector<std::uint8_t> expected_weight;
        Weight weight_view;
        std::vector<double> expected;
        std::unique_ptr<test::GuardedDeviceBuffer> weight;
        std::unique_ptr<test::GuardedDeviceBuffer> x;
        std::unique_ptr<test::GuardedDeviceBuffer> output;
        std::unique_ptr<WorkspaceArena> workspace;
    };
    std::array<Rank, 4> ranks;
    const std::size_t capacity = ops::linear_swiglu_column_parallel_workspace_capacity_bytes(
        shard_profile.qtype, policy, 1, maximum_tokens, tp);
    for (int rank = 0; rank < tp; ++rank) {
        cuda_check(cudaSetDevice(ec.dev[rank]->device), "select split device");
        auto& data = ranks[rank];
        Profile profile = shard_profile;
        profile.seed += 73U * rank;
        quantized_weight::PatternedWeightOptions options;
        if (profile.qtype == QType::NVFP4) {
            options.weight_scale_divisor = 0.125F;
            options.input_scale_divisor = 3.5F;
        }
        data.packed = quantized_weight::make_patterned_weight(
            profile.qtype, profile.gate_up_rows, profile.input_rows, profile.seed, options);
        // Both ranks consume the same replicated activation and distinct gate/up shards.
        data.activation = make_activation(shard_profile, maximum_tokens);
        data.expected = linear_swiglu_oracle_fp64(
            profile, data.packed, data.activation, maximum_tokens);
        data.weight = std::make_unique<test::GuardedDeviceBuffer>(data.packed.payload.size());
        data.weight->copy_from_host(data.packed.payload.data(), data.packed.payload.size());
        data.weight_view = data.packed.device_weight(data.weight->data());
#ifdef NINFER_VOLTA_BUILD
        if (volta_prepacked) {
            ops::detail::nvfp4_prepack_qpn_sm70(data.weight_view);
            data.weight_view.volta_fp32_scale = volta_fp32_scale;
        }
#endif
        data.expected_weight.resize(data.packed.payload.size());
        data.weight->copy_to_host(data.expected_weight.data(), data.expected_weight.size());
        data.image = data.activation;
        if (fp16_image) for (auto& v : data.image) {
            const float represented = test::bf16_to_f32(v);
            v = quantized_weight::detail::f32_to_f16(represented);
            if (quantized_weight::detail::f16_to_f32(v) != represented) {
                throw std::logic_error("FP16 image fixture must preserve its BF16 oracle input");
            }
        }
        data.x = std::make_unique<test::GuardedDeviceBuffer>(data.image.size() * 2);
        data.x->copy_from_host(data.image.data(), data.image.size() * 2);
        data.workspace = std::make_unique<WorkspaceArena>(std::max<std::size_t>(capacity, 256));
    }
    int failures = 0;
    for (const auto tokens : token_cases) {
        std::array<Tensor, 4> x, output;
        std::array<Weight, 4> weights;
        std::array<WorkspaceArena*, 4> workspace;
        const std::size_t elements = checked_elements(shard_profile.output_rows, tokens, "split");
        for (int rank = 0; rank < tp; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select split device");
            auto& data = ranks[rank];
            data.output = std::make_unique<test::GuardedDeviceBuffer>(elements * 2);
            data.output->fill(0xff);
            x[rank] = Tensor(data.x->data(), fp16_image ? DType::FP16 : DType::BF16,
                             {shard_profile.input_rows, tokens});
            output[rank] = Tensor(data.output->data(), DType::BF16,
                                  {shard_profile.output_rows, tokens});
            weights[rank] = data.weight_view;
            data.workspace->reset();
            data.workspace->reset_peak();
            workspace[rank] = data.workspace.get();
            cuda_check(cudaDeviceSynchronize(), "retire split staging");
        }
        std::array<std::vector<std::uint16_t>, 4> expected_image;
        if (fp16_output_image) {
            ops::linear_swiglu_column_parallel(std::span(x).first(tp), std::span(weights).first(tp),
                                              std::span(output).first(tp), policy,
                                              std::span(workspace).first(tp), ec);
            for (int rank = 0; rank < tp; ++rank) {
                cuda_check(cudaSetDevice(ec.dev[rank]->device), "select BF16 image reference rank");
                cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "complete BF16 reference");
                expected_image[rank].resize(elements);
                ranks[rank].output->copy_to_host(expected_image[rank].data(), elements * 2);
                for (auto& value : expected_image[rank]) {
                    value = quantized_weight::detail::f32_to_f16(test::bf16_to_f32(value));
                }
                output[rank].dtype = DType::FP16;
                ranks[rank].output->fill(0xff);
                ranks[rank].workspace->reset_peak();
                cuda_check(cudaDeviceSynchronize(), "retire FP16 output staging");
            }
        }
        ops::linear_swiglu_column_parallel(std::span(x).first(tp), std::span(weights).first(tp),
                                          std::span(output).first(tp), policy,
                                          std::span(workspace).first(tp), ec);
        const std::size_t exact = ops::linear_swiglu_column_parallel_workspace_capacity_bytes(
            shard_profile.qtype, policy, tokens, tokens, tp);
        for (int rank = 0; rank < tp; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select split device");
            cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "complete split SwiGLU");
            auto& data = ranks[rank];
            const std::string name = std::string(label) + " T=" + std::to_string(tokens) +
                                     " rank=" + std::to_string(rank);
            auto actual = read_bf16_output(*data.output, elements);
            if (fp16_output_image) {
                std::vector<std::uint16_t> bits(elements);
                data.output->copy_to_host(bits.data(), elements * 2);
                if (bits != expected_image[rank]) {
                    std::cerr << name << ": BF16-rounded FP16 output image differs\n";
                    ++failures;
                }
                for (std::size_t i = 0; i < elements; ++i) {
                    actual[i] = quantized_weight::detail::f16_to_f32(bits[i]);
                }
            }
            failures += compare_output(name, actual, data.expected.data(), ActivationCompute::A16);
            failures += data.output->verify_guards(name);
            // The public capacity query is an upper bound across weight layouts. A
            // prepacked batch can need less than the portable materialization route.
            if (data.workspace->used() != 0 || data.workspace->peak_used() > exact) {
                std::cerr << name << ": workspace exceeds capacity or escapes its scope\n";
                ++failures;
            }
        }
    }
    for (int rank = 0; rank < tp; ++rank) {
        cuda_check(cudaSetDevice(ec.dev[rank]->device), "select split device");
        auto& data = ranks[rank];
        failures += verify_unchanged(label, *data.weight, data.expected_weight.data(),
                                     data.expected_weight.size());
        failures += verify_unchanged(label, *data.x, data.image.data(), data.image.size() * 2);
        data.output.reset();
        data.workspace.reset();
        data.x.reset();
        data.weight.reset();
    }
    cuda_check(cudaSetDevice(0), "restore primary device");
    return failures;
}

} // namespace ninfer::test::linear_swiglu
