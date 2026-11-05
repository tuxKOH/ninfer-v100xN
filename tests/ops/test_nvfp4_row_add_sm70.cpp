#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"

#include "core/decode_graph.h"
#include "ops/input_projection_test_common.h"

#include <array>
#include <memory>

using namespace ninfer;
using namespace ninfer::test;
namespace fixture = ninfer::test::input_projection;
namespace qw = ninfer::test::quantized_weight;

namespace {
constexpr int N = 5120;
bool fp32_scale = false;

qw::PackedWeight make_weight(int k, int rank) {
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    options.weight_scale_divisor = .125F;
    options.input_scale_divisor = 3.5F;
    return qw::make_patterned_weight(QType::NVFP4, N, k, 581u + rank, options);
}

struct Rank {
    fixture::DevicePackedWeight weight;
    std::vector<float> activation;
    DeviceBuffer input;
    DeviceBuffer input_image;
    GuardedDeviceBuffer residual, contribution, normalized;
    std::vector<float> norm_weight;
    DeviceBuffer norm_weight_device;
    WorkspaceArena workspace;
    Rank(int k, int tokens, int rank, int width)
        : weight(make_weight(k, rank), true),
          activation(fixture::make_bf16_activation(k, tokens, 601u + rank)),
          input(to_device_bf16(activation)), residual(2ULL * N * tokens),
          contribution(2ULL * N * tokens * (width == 4 ? 4 : 1)), normalized(2ULL * N * tokens),
          norm_weight(N),
          workspace(std::max<std::size_t>(256,
              ops::linear_add_workspace_capacity_bytes(QType::NVFP4, N, k,
                  ops::LinearPolicy::A16Only, tokens, tokens))) {
        fill_uniform(norm_weight, 613u + rank, -.75F, .75F); round_to_bf16(norm_weight);
        norm_weight_device = to_device_bf16(norm_weight);
    }
};

template<int Width>
int run(const ExecutionContext& ec, const ops::PeerEvents& events, int k, int tokens,
        bool fp16_image = false, bool fp16_input = false) {
    std::array<std::unique_ptr<Rank>, Width> ranks;
    std::array<Tensor, Width> input, residual, contribution, norm_weight, normalized;
    std::array<Weight, Width> weight;
    std::array<WorkspaceArena*, Width> ws;
    for (int rank = 0; rank < Width; ++rank) {
        cuda_check(cudaSetDevice(ec.dev[rank]->device), "select rank");
        ranks[rank] = std::make_unique<Rank>(k, tokens, rank, Width);
        auto& r = *ranks[rank];
        input[rank] = Tensor(r.input.p, DType::BF16, {k, tokens});
        if (fp16_input) {
            std::vector<std::uint16_t> image(r.activation.size());
            for (std::size_t i = 0; i < image.size(); ++i) {
                image[i] = qw::detail::f32_to_f16(r.activation[i]);
                // The oracle consumes the represented public FP16 input, including
                // subnormal rounding. This value also has an exact BF16 image for
                // the composed reference because the source was already BF16.
                r.activation[i] = qw::detail::f16_to_f32(image[i]);
            }
            r.input = to_device_bf16(r.activation);
            r.input_image = to_device(image);
            input[rank] = Tensor(r.input_image.p, DType::FP16, {k, tokens});
        }
        residual[rank] = Tensor(r.residual.data(), DType::BF16, {N, tokens});
        contribution[rank] = Tensor(r.contribution.data(), DType::BF16, {N, tokens, Width == 4 ? 4 : 1});
        norm_weight[rank] = Tensor(r.norm_weight_device.p, DType::BF16, {N});
        normalized[rank] = Tensor(r.normalized.data(), fp16_image ? DType::FP16 : DType::BF16, {N, tokens});
        weight[rank] = r.weight.view();
        weight[rank].volta_fp32_scale = fp32_scale;
        ws[rank] = &r.workspace;
        cuda_synchronize();
    }
    const ops::ReplicatedRmsNorm norm{norm_weight, normalized, 1e-6F, true};
    std::array<std::vector<std::uint16_t>, Width> reference_norm;
    std::vector<float> initial(N * tokens);
    fill_uniform(initial, 607u, -.05F, .05F); round_to_bf16(initial);
    const auto sync = [&] {
        for (int rank = 0; rank < Width; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select sync rank");
            cuda_synchronize(ec.dev[rank]->stream);
        }
    };
    const auto reset = [&] {
        for (int rank = 0; rank < Width; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select reset rank");
            auto staged = to_device_bf16(initial);
            cuda_check(cudaMemcpy(residual[rank].data, staged.p, staged.bytes,
                                   cudaMemcpyDeviceToDevice), "stage residual");
            cuda_synchronize();
        }
    };
    const auto compose = [&] {
        for (int rank = 0; rank < Width; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select compose rank");
            Tensor out = rank == 0 ? Tensor(contribution[rank].data, DType::BF16, {N, tokens}) : residual[rank];
            const Tensor bf16_input(ranks[rank]->input.p, DType::BF16, {k, tokens});
            ops::linear(bf16_input, weight[rank], out, ops::LinearPolicy::A16Only,
                         *ws[rank], ec.dev[rank]->stream);
            if (rank == 0) {
                Tensor destination = residual[rank];
                ops::residual_add(out, destination, ec.dev[rank]->stream);
            }
        }
        ops::allreduce_sum(residual, contribution, ec, events);
        for (int rank = 0; rank < Width; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select norm reference rank");
            Tensor bf16 = normalized[rank];
            bf16.dtype = DType::BF16;
            ops::rmsnorm(residual[rank], norm_weight[rank], norm.eps, norm.unit_offset,
                         bf16, ec.dev[rank]->stream);
        }
    };
    const auto invoke = [&] {
        ops::linear_add_row_parallel(input, weight, residual, contribution,
                                     ops::LinearPolicy::A16Only, ws, ec, events, &norm);
    };
    const auto reference_bits = [&] {
        reset(); compose(); sync();
        if (fp16_image || fp16_input) {
            reset();
            const auto saved_input = input;
            for (int rank = 0; rank < Width; ++rank) {
                input[rank] = Tensor(ranks[rank]->input.p, DType::BF16, {k, tokens});
            }
            for (auto& output : normalized) output.dtype = DType::BF16;
            invoke(); sync();
            input = saved_input;
            for (auto& output : normalized) output.dtype = fp16_image ? DType::FP16 : DType::BF16;
        }
        for (int rank = 0; rank < Width; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select norm reference rank");
            reference_norm[rank] = from_device<std::uint16_t>(normalized[rank].data, N * tokens);
        }
        cuda_check(cudaSetDevice(ec.dev[0]->device), "select reference rank");
        return from_device<std::uint16_t>(residual[0].data, N * tokens);
    };
    const auto label = "NVFP4 TP" + std::to_string(Width) + " row add K=" + std::to_string(k) + " T=" + std::to_string(tokens) +
        (fp16_image ? " BF16-to-FP16 norm image" : "") + (fp16_input ? " FP16 input image" : "");
    int failures = 0;
    const auto verify = [&](const std::vector<std::uint16_t>& expected) {
        std::vector<double> oracle;
        for (int row : fixture::sampled_rows(N, 17)) {
            for (int token = 0; token < tokens; ++token) {
                double sum = initial[token * N + row];
                for (int rank = 0; rank < Width; ++rank) {
                    sum += qw::dot_fp64(ranks[rank]->weight.host, row,
                        ranks[rank]->activation.data() + token * k, k);
                }
                oracle.push_back(sum);
            }
        }
        for (int rank = 0; rank < Width; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select verify rank");
            auto& r = *ranks[rank];
            const auto actual = from_device_bf16(residual[rank].data, N * tokens);
            failures += fixture::compare(label,
                fixture::gather_rows(actual, N, 0, N, tokens, 17), oracle,
                ReductionCriterion{2.0 / 256, 1e-5, 2.0 / 256});
            failures += verify_exact(label.c_str(),
                from_device<std::uint16_t>(residual[rank].data, N * tokens), expected);
            const auto norm_bits = from_device<std::uint16_t>(normalized[rank].data, N * tokens);
            std::vector<double> norm_values(norm_bits.size());
            for (std::size_t i = 0; i < norm_bits.size(); ++i) {
                norm_values[i] = fp16_image ? qw::detail::f16_to_f32(norm_bits[i])
                                           : bf16_to_f32(norm_bits[i]);
            }
            if (fp16_image) {
                auto expected_image = reference_norm[rank];
                for (auto& v : expected_image) v = qw::detail::f32_to_f16(bf16_to_f32(v));
                failures += verify_exact((label + " exact cast image").c_str(),
                                          norm_bits, expected_image);
            }
            std::vector<double> norm_oracle(N * tokens);
            for (int token = 0; token < tokens; ++token) {
                double ss = 0.0;
                for (int row = 0; row < N; ++row) {
                    const double v = actual[token * N + row]; ss += v * v;
                }
                const double inv = 1.0 / std::sqrt(ss / N + double(norm.eps));
                for (int row = 0; row < N; ++row) {
                    norm_oracle[token * N + row] = actual[token * N + row] * inv *
                        (1.0 + double(r.norm_weight[row]));
                }
            }
            // Use the collective RMSNorm contract: BF16 round-to-nearest alone
            // can incur up to 1/256 relative error. The old 0.0034 gross bound
            // rejected even the nearest representable output for wide fixtures
            // (for example 4.17179 -> 4.15625). Keep the existing L2 bound and
            // exact comparison with the separately composed BF16 sum/norm below.
            failures += verify_reduction((label + " norm FP64").c_str(), norm_values,
                                           norm_oracle, ReductionCriterion{1.85e-3, 1e-5, 1.0 / 256});
            if constexpr (Width == 2) {
                failures += verify_exact((label + " norm composition").c_str(),
                    from_device<std::uint16_t>(normalized[rank].data, N * tokens), reference_norm[rank]);
            }
            failures += r.normalized.verify_guards(label);
            failures += verify_exact((label + " norm weight immutable").c_str(),
                from_device_bf16(r.norm_weight_device, N),
                std::vector<double>(r.norm_weight.begin(), r.norm_weight.end()));
            failures += r.residual.verify_guards(label);
            failures += r.contribution.verify_guards(label);
            failures += r.weight.verify_preserved(label);
            failures += verify_exact((label + " input immutable").c_str(),
                from_device_bf16(r.input, r.activation.size()),
                std::vector<double>(r.activation.begin(), r.activation.end()));
            if (fp16_input) {
                auto bits = from_device<std::uint16_t>(r.input_image, r.activation.size());
                for (std::size_t i = 0; i < bits.size(); ++i) {
                    if (qw::detail::f16_to_f32(bits[i]) != r.activation[i]) {
                        std::cerr << label << ": FP16 input image was modified\n";
                        ++failures;
                        break;
                    }
                }
            }
        }
    };
    auto expected = reference_bits();
    reset(); invoke(); sync(); verify(expected);
    if (tokens == 5 || (Width == 2 && tokens >= 10) || (Width == 4 && tokens >= 8)) {
        cuda_check(cudaSetDevice(ec.dev[0]->device), "select capture origin");
        std::array<std::unique_ptr<DecodeGraphPeerBridge>, Width - 1> bridges;
        std::array<DecodeGraphPeerCapture, Width - 1> peers;
        for (int rank = 1; rank < Width; ++rank) {
            bridges[rank - 1] = std::make_unique<DecodeGraphPeerBridge>(
                ec.dev[0]->device, ec.dev[rank]->device);
            peers[rank - 1] = {bridges[rank - 1].get(), ec.dev[rank]->stream};
        }
        DecodeGraphDefinition definition;
        definition.capture(ec.dev[0]->stream, invoke, peers);
        DecodeGraphExecutable graph;
        graph.instantiate(definition);
        for (int replay = 0; replay < 2; ++replay) {
            for (int rank = 0; rank < Width; ++rank) {
                cuda_check(cudaSetDevice(ec.dev[rank]->device), "select changed input rank");
                auto& r = *ranks[rank];
                for (auto& value : r.activation) {
                    value = bf16_to_f32(f32_to_bf16(-value * .75F + .0005F));
                    if (fp16_input) {
                        value = qw::detail::f16_to_f32(qw::detail::f32_to_f16(value));
                    }
                }
                const auto staged = to_device_bf16(r.activation);
                cuda_check(cudaMemcpy(r.input.p, staged.p, staged.bytes, cudaMemcpyDeviceToDevice),
                           "replace captured input");
                if (fp16_input) {
                    std::vector<std::uint16_t> image(r.activation.size());
                    for (std::size_t i = 0; i < image.size(); ++i) {
                        image[i] = qw::detail::f32_to_f16(r.activation[i]);
                        if (qw::detail::f16_to_f32(image[i]) != r.activation[i]) {
                            throw std::logic_error("replay image must preserve its oracle input");
                        }
                    }
                    r.input_image.copy_from_host(image.data(), image.size() * 2);
                }
                cuda_synchronize();
            }
            for (auto& value : initial) { value = bf16_to_f32(f32_to_bf16(-value * .5F)); }
            expected = reference_bits(); reset();
            cuda_check(cudaSetDevice(ec.dev[0]->device), "select replay origin");
            graph.launch(ec.dev[0]->stream); sync(); verify(expected);
        }
    }
    for (int rank = 0; rank < Width; ++rank) {
        cuda_check(cudaSetDevice(ec.dev[rank]->device), "select release rank");
        ranks[rank].reset();
    }
    std::cout << label << (failures ? " FAIL\n" : " PASS\n");
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    const bool image_only = argc == 2 && std::string_view(argv[1]) == "--fp16-image-only";
    const bool tp4_batch_only = argc == 2 && std::string_view(argv[1]) == "--tp4-batch-only";
    if (argc != 1 && !image_only && !tp4_batch_only) return 2;
    if (cuda_unavailable()) { return 77; }
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count), "device count");
    if (count < 2) { return 77; }
    const ExecutionContext ec({0, 1});
    if (ec.dev[0]->sm() != 70) { return 77; }
    const ops::PeerEvents events(ec, ops::enable_peer_access(ec));
    int failures = 0;
    if (!image_only && !tp4_batch_only) for (bool selected : {false, true}) {
        fp32_scale = selected;
        for (int k : {3072, 8704}) {
            for (int tokens : {1, 5, 8, 9, 10, 20, 40, 64}) {
                failures += run<2>(ec, events, k, tokens);
            }
        }
    }
    if (tp4_batch_only && count < 4) return 77;
    if (count >= 4) {
        const ExecutionContext four({0, 1, 2, 3});
        const ops::PeerEvents peer(four, ops::enable_peer_access(four));
        for (bool selected : {false, true}) {
            fp32_scale = selected;
            for (int k : {1536, 4352}) {
                for (int tokens : {1, 5, 8, 9, 10, 16, 20, 32, 40, 64}) {
                    if (tp4_batch_only && tokens <= 8) continue;
                    if (!image_only) failures += run<4>(four, peer, k, tokens);
                    if (tokens <= 8) failures += run<4>(four, peer, k, tokens, true);
                    if (tokens <= 8) {
                        failures += run<4>(four, peer, k, tokens, true, true);
                    }
                }
            }
        }
    }
    return failures ? 1 : 0;
}
