#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"

#include "ops/input_projection_test_common.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/volta_tp4.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;
namespace qw = ninfer::test::quantized_weight;

#ifdef NINFER_VOLTA_BUILD
namespace {
bool fp32_scale = false;

Weight profile(Weight weight) {
    weight.volta_fp32_scale = fp32_scale;
    return weight;
}

constexpr ReductionCriterion kA16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

int check(std::string_view name, GuardedBf16Tensor& output, const qw::PackedWeight& weight,
          int row_begin, int rows, const std::vector<float>& x, int tokens) {
    return output.verify_guards(name) + output.verify_fully_written(name) +
        compare(name, gather_rows(output.values(), rows, 0, rows, tokens),
                projection_oracle(weight, row_begin, rows, x, 5120, tokens), kA16);
}

int exercise(bool attention, int tp) {
    const int n = (attention ? 14336 : 16384) / tp;
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor = 3.5F;
    const auto native = qw::make_patterned_weight(QType::NVFP4, n, 5120, attention ? 431U : 437U,
                                                  options);
    const auto packed = qw::nvfp4_qpn_payload(native);
    DeviceBuffer device_weight(packed.size());
    device_weight.copy_from_host(packed.data(), packed.size());
    auto weight = native.device_weight(device_weight.p);
    weight.layout = QuantLayout::VoltaQpnPrepacked;
    weight.volta_fp32_scale = fp32_scale;
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "projection test stream");
    int failures = 0;
    for (const int tokens : {1, 3, 4, 5, 6, 8, 9, 10, 16, 17, 20, 32, 33, 40, 41, 64,
                             127, 128, 129}) {
        auto x_host = make_bf16_activation(5120, tokens, 449U + tokens);
        auto device_x = to_device_bf16(x_host);
        Tensor x(device_x.p, DType::BF16, {5120, tokens});
        const auto prefix = std::string(attention ? "NVFP4 attention" : "NVFP4 GDN") +
            " TP" + std::to_string(tp) + " T=" + std::to_string(tokens);
        const auto capacity = tp > 1
            ? (attention ? ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                               QType::NVFP4, ops::LinearPolicy::A16Only, tokens, tokens, tp)
                         : ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
                               QType::NVFP4, ops::LinearPolicy::A16Only, tokens, tokens, tp))
            : attention
            ? ops::attn_input_proj_workspace_capacity_bytes(
                    QType::NVFP4, n, 5120, ops::LinearPolicy::A16Only, tokens, tokens)
            : ops::gdn_input_proj_workspace_capacity_bytes(
                    QType::NVFP4, n, 5120, ops::LinearPolicy::A16Only, tokens, tokens);
        DeviceArena workspace(std::max<std::size_t>(capacity, 1));
        if (attention) {
            const int q_rows = 6144 / tp, kv_rows = 1024 / tp;
            GuardedBf16Tensor query(q_rows, tokens), key(kv_rows, tokens),
                gate(q_rows, tokens), value(kv_rows, tokens);
            auto q = query.tensor(), k = key.tensor(), g = gate.tensor(), v = value.tensor();
            auto invoke = [&] {
                if (tp == 4) {
                    ops::detail::attn_input_volta_tp4_launch(
                        x, weight, q, g, k, v, &workspace, stream);
                } else if (tp == 2) {
                    // Qualify the real shard kernel directly against the same mathematical
                    // oracle; the split suite separately checks the public two-device wrapper.
                    ops::detail::nvfp4_attn_input_sm70_launch(
                        x, weight, q, g, k, v, &workspace, stream);
                } else {
                    ops::attn_input_proj(x, weight, q, g, k, v,
                                        ops::LinearPolicy::A16Only, workspace, stream);
                }
            };
            invoke();
            cuda_synchronize(stream);
            const auto verify = [&] {
                return check(prefix + " q", query, native, 0, q_rows, x_host, tokens) +
                    check(prefix + " k", key, native, q_rows, kv_rows, x_host, tokens) +
                    check(prefix + " gate", gate, native, q_rows + kv_rows, q_rows, x_host, tokens) +
                    check(prefix + " v", value, native, 2 * q_rows + kv_rows, kv_rows, x_host, tokens);
            };
            failures += verify();
            // Capture the same public route using exactly its declared scratch capacity.
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture input");
            invoke();
            cuda_check(cudaStreamEndCapture(stream, &graph), "finish input capture");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate input");
            x_host = make_bf16_activation(5120, tokens, 1741U + tokens);
            const auto changed = bf16_bits(x_host);
            device_x.copy_from_host(changed.data(), changed.size() * sizeof(std::uint16_t));
            cuda_check(cudaGraphLaunch(exec, stream), "replay input");
            cuda_synchronize(stream);
            failures += verify();
            cuda_check(cudaGraphExecDestroy(exec), "destroy input exec");
            cuda_check(cudaGraphDestroy(graph), "destroy input graph");
        } else {
            const int qkv_rows = 10240 / tp, z_rows = 6144 / tp;
            GuardedBf16Tensor projected(qkv_rows, tokens), z_out(z_rows, tokens);
            auto qkv = projected.tensor(), z = z_out.tensor();
            auto invoke = [&] {
                if (tp > 1) {
                    ops::detail::nvfp4_gdn_input_sm70_launch(
                        x, weight, qkv, z, &workspace, stream);
                } else {
                    ops::gdn_input_proj(x, weight, qkv, z,
                                       ops::LinearPolicy::A16Only, workspace, stream);
                }
            };
            invoke();
            cuda_synchronize(stream);
            const auto verify = [&] {
                return check(prefix + " qkv", projected, native, 0, qkv_rows, x_host, tokens) +
                    check(prefix + " z", z_out, native, qkv_rows, z_rows, x_host, tokens);
            };
            failures += verify();
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture GDN input");
            invoke();
            cuda_check(cudaStreamEndCapture(stream, &graph), "finish GDN input capture");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate GDN input");
            x_host = make_bf16_activation(5120, tokens, 1747U + tokens);
            const auto changed = bf16_bits(x_host);
            device_x.copy_from_host(changed.data(), changed.size() * sizeof(std::uint16_t));
            cuda_check(cudaGraphLaunch(exec, stream), "replay GDN input");
            cuda_synchronize(stream);
            failures += verify();
            cuda_check(cudaGraphExecDestroy(exec), "destroy GDN input exec");
            cuda_check(cudaGraphDestroy(graph), "destroy GDN input graph");
        }
        std::cout << prefix << '\n';
    }
    if (from_device<std::uint8_t>(device_weight.p, packed.size()) != packed) {
        std::cerr << "projection modified its packed weight\n";
        ++failures;
    }
    cuda_check(cudaStreamDestroy(stream), "destroy projection test stream");
    return failures;
}

int exercise_down_projection() {
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor = 3.5F;
    DevicePackedWeight weight(qw::make_patterned_weight(QType::NVFP4, 5120, 8704, 461U, options),
                               true);
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "down projection test stream");
    int failures = 0;
    for (const int tokens : {4, 5, 6}) {
        const auto host = make_bf16_activation(8704, tokens, 467U + tokens);
        auto device_input = to_device_bf16(host);
        Tensor input(device_input.p, DType::BF16, {8704, tokens});
        GuardedBf16Tensor result(5120, tokens);
        auto output = result.tensor();
        const auto view = profile(weight.view());
        const auto capacity = ops::linear_workspace_capacity_bytes(
            QType::NVFP4, 5120, 8704, ops::LinearPolicy::A16Only, tokens, tokens);
        DeviceArena workspace(capacity);
        ops::linear(input, view, output, ops::LinearPolicy::A16Only, workspace, stream);
        cuda_synchronize(stream);
        const auto expected = from_device<std::uint16_t>(output.data, 5120 * tokens);
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture down");
        ops::linear(input, view, output, ops::LinearPolicy::A16Only, workspace, stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "finish down capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate down");
        cuda_check(cudaGraphLaunch(executable, stream), "replay down");
        cuda_synchronize(stream);
        const auto name = "NVFP4 TP2 down T=" + std::to_string(tokens);
        failures += result.verify_guards(name) + result.verify_fully_written(name) +
            compare(name, gather_rows(result.values(), 5120, 0, 5120, tokens),
                    projection_oracle(weight.host, 0, 5120, host, 8704, tokens), kA16);
        if (from_device<std::uint16_t>(output.data, 5120 * tokens) != expected) { ++failures; }
        cuda_check(cudaGraphExecDestroy(executable), "destroy down executable");
        cuda_check(cudaGraphDestroy(graph), "destroy down graph");
        std::cout << name << '\n';
    }
    failures += weight.verify_preserved("down projection");
    cuda_check(cudaStreamDestroy(stream), "destroy down projection stream");
    return failures;
}

int exercise_swiglu() {
    constexpr int n = 8704, k = 5120;
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor = 3.5F;
    DevicePackedWeight weight(qw::make_patterned_weight(QType::NVFP4, n * 2, k, 479U, options), true);
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "SwiGLU test stream");
    int failures = 0;
    for (const int tokens : {4, 5, 6}) {
        const auto host = make_bf16_activation(k, tokens, 487U + tokens);
        auto device_input = to_device_bf16(host);
        Tensor input(device_input.p, DType::BF16, {k, tokens});
        GuardedBf16Tensor result(n, tokens);
        auto output = result.tensor();
        DeviceBuffer gate(sizeof(float) * n * tokens), activation(sizeof(std::uint16_t) * k * tokens);
        const auto view = profile(weight.view());
        auto invoke = [&] {
            ops::detail::nvfp4_linear_swiglu_qpn_split_launch(
                input, view, output, static_cast<float*>(gate.p), activation.p, stream);
        };
        invoke();
        cuda_synchronize(stream);
        const auto eager_bits = result.bits();
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture SwiGLU");
        invoke();
        cuda_check(cudaStreamEndCapture(stream, &graph), "finish SwiGLU capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate SwiGLU");
        cuda_check(cudaGraphLaunch(executable, stream), "replay SwiGLU");
        cuda_synchronize(stream);
        const auto name = "NVFP4 TP2 prepacked SwiGLU T=" + std::to_string(tokens);
        std::vector<double> expected;
        for (const int row : sampled_rows(n, 17)) {
            for (int token = 0; token < tokens; ++token) {
                const auto* values = host.data() + static_cast<std::size_t>(token) * k;
                const double gate_value = qw::dot_fp64(weight.host, row, values, k);
                const double up_value = qw::dot_fp64(weight.host, row + n, values, k);
                expected.push_back(gate_value / (1.0 + std::exp(-gate_value)) * up_value);
            }
        }
        failures += result.verify_guards(name) + result.verify_fully_written(name) +
            compare(name, gather_rows(result.values(), n, 0, n, tokens, 17), expected,
                    ReductionCriterion{3.3e-3, 5.0e-3, 6.3e-3});
        if (result.bits() != eager_bits) { ++failures; }
        cuda_check(cudaGraphExecDestroy(executable), "destroy SwiGLU executable");
        cuda_check(cudaGraphDestroy(graph), "destroy SwiGLU graph");
        std::cout << name << '\n';
    }
    failures += weight.verify_preserved("SwiGLU");
    cuda_check(cudaStreamDestroy(stream), "destroy SwiGLU test stream");
    return failures;
}

int exercise_conv_record() {
    constexpr int Hidden = 5120, Channels = 5120, QRows = 1024, VRows = 3072;
    int failures = 0;
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "record stream");
    for (float divisor : {0.125F, 1801.382080078125F}) {
        qw::PatternedWeightOptions options;
        options.decorrelate_coordinates = true;
        options.weight_scale_divisor = divisor;
        options.input_scale_divisor = 3.5F;
        DevicePackedWeight weight(qw::make_patterned_weight(QType::NVFP4, 8192, Hidden,
                                                               1871U, options), true);
        std::vector<float> conv_host(Channels * 4), state_host(Channels * 3 * 2);
        fill_uniform(conv_host, 1873U, -0.25F, 0.25F);
        fill_uniform(state_host, 1877U, -1.0F, 1.0F);
        round_to_bf16(conv_host);
        round_to_bf16(state_host);
        auto conv_device = to_device_bf16(conv_host), state_device = to_device_bf16(state_host);
        auto slots_device = to_device(std::vector<int>{1});
        Tensor conv(conv_device.p, DType::BF16, {Channels, 4});
        Tensor state(state_device.p, DType::BF16, {Channels, 3, 2});
        Tensor slots(slots_device.p, DType::I32, {1});
        for (int tokens : {1, 2, 4, 5, 7, 8}) {
            const auto host = make_bf16_activation(Hidden, tokens, 1879U + tokens);
            auto input_device = to_device_bf16(host);
            auto valid_device = to_device(std::vector<int>{tokens == 1 ? 1 : tokens - 1});
            Tensor x(input_device.p, DType::BF16, {Hidden, tokens});
            Tensor valid(valid_device.p, DType::I32, {1});
            GuardedBf16Tensor record(Channels, tokens), query(QRows, tokens), key(QRows, tokens),
                value(VRows, tokens), z(VRows, tokens), projected(Channels, tokens),
                reference_q(QRows, tokens), reference_k(QRows, tokens),
                reference_v(VRows, tokens), reference_z(VRows, tokens);
            auto r = record.tensor(), q = query.tensor(), k = key.tensor(), v = value.tensor(),
                zv = z.tensor(), p = projected.tensor(), rq = reference_q.tensor(),
                rk = reference_k.tensor(), rv = reference_v.tensor(), rz = reference_z.tensor();
            Tensor projected_3d(p.data, DType::BF16, {Channels, tokens, 1});
            const auto capacity = ops::detail::nvfp4_gdn_input_sm70_workspace_bytes(8192, tokens);
            DeviceArena arena(std::max<std::size_t>(capacity, 1));
            auto invoke = [&] {
                ops::detail::nvfp4_gdn_input_sm70_record_launch(x, profile(weight.view()), conv, state,
                    valid, slots, r, q, k, v, zv, arena, stream);
            };
            auto compare_composition = [&] {
                ops::detail::nvfp4_gdn_input_sm70_launch(x, profile(weight.view()), p, rz, &arena, stream);
                ops::detail::gdn_projected_conv_record_launch(projected_3d, conv, state, valid,
                                                                slots, rq, rk, rv, stream);
                cuda_synchronize(stream);
                if (record.bits() != projected.bits() || query.bits() != reference_q.bits() ||
                    key.bits() != reference_k.bits() || value.bits() != reference_v.bits() ||
                    z.bits() != reference_z.bits()) {
                    std::cerr << "GDN column record changed represented composition\n";
                    ++failures;
                }
            };
            invoke();
            compare_composition();
            std::vector<double> actual_record, ideal_record, actual_conv, ideal_conv;
            const auto record_values = record.values(), qv = query.values(), kv = key.values(),
                vv = value.values();
            for (int row : sampled_rows(Channels, 29)) {
                double s0 = state_host[Channels * 3 + row];
                double s1 = state_host[Channels * 4 + row];
                double s2 = state_host[Channels * 5 + row];
                for (int token = 0; token < tokens; ++token) {
                    const double ideal = qw::dot_fp64(weight.host, row,
                        host.data() + static_cast<std::size_t>(token) * Hidden, Hidden);
                    actual_record.push_back(record_values[token * Channels + row]);
                    ideal_record.push_back(ideal);
                    const int valid_tokens = tokens == 1 ? 1 : tokens - 1;
                    const double c = conv_host[row] * s0 + conv_host[Channels + row] * s1 +
                        conv_host[2 * Channels + row] * s2 + conv_host[3 * Channels + row] * ideal;
                    ideal_conv.push_back(token < valid_tokens ? c / (1.0 + std::exp(-c)) : 0.0);
                    actual_conv.push_back(row < QRows ? qv[token * QRows + row]
                        : row < 2 * QRows ? kv[token * QRows + row - QRows]
                        : vv[token * VRows + row - 2 * QRows]);
                    s0 = s1; s1 = s2; s2 = ideal;
                }
            }
            const auto label = "NVFP4 GDN column record T=" + std::to_string(tokens);
            failures += compare(label + " projection FP64", actual_record, ideal_record, kA16);
            failures += compare(label + " conv FP64", actual_conv, ideal_conv, kA16);
            failures += check(label + " z", z, weight.host, Channels, VRows, host, tokens);
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture record");
            invoke();
            cuda_check(cudaStreamEndCapture(stream, &graph), "finish record capture");
            cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate record");
            auto changed = make_bf16_activation(Hidden, tokens, 1889U + tokens);
            const auto changed_bits = bf16_bits(changed);
            input_device.copy_from_host(changed_bits.data(), changed_bits.size() * 2);
            cuda_check(cudaGraphLaunch(executable, stream), "replay changed record");
            compare_composition();
            cuda_check(cudaGraphExecDestroy(executable), "destroy record executable");
            cuda_check(cudaGraphDestroy(graph), "destroy record graph");
            for (auto* output : {&record, &query, &key, &value, &z}) {
                failures += output->verify_guards(label) + output->verify_fully_written(label);
            }
            if (from_device_bf16(state_device.p, state_host.size()) !=
                std::vector<double>(state_host.begin(), state_host.end())) { ++failures; }
            std::cout << label << " divisor=" << divisor << '\n';
        }
        failures += weight.verify_preserved("record weight");
    }
    cuda_check(cudaStreamDestroy(stream), "destroy record stream");
    return failures;
}

} // namespace
#endif

int main(int argc, char** argv) {
#ifndef NINFER_VOLTA_BUILD
    return 77;
#else
    if (cuda_unavailable()) { return 77; }
    cudaDeviceProp props{};
    cuda_check(cudaGetDeviceProperties(&props, 0), "query projection device");
    if (props.major != 7 || props.minor != 0) { return 77; }
    if (argc == 2 && std::string_view(argv[1]) == "--record") {
        return exercise_conv_record() ? 1 : 0;
    }
    const bool tp4 = argc == 2 && std::string_view(argv[1]) == "--tp4";
    if (argc != 1 && !tp4) { return 2; }
    int failures = 0;
    for (bool selected : {false, true}) {
        fp32_scale = selected;
        std::cout << "NVFP4 FP32 epilogue=" << selected << '\n';
        if (tp4) {
            failures += exercise(true, 4) + exercise(false, 4);
        } else {
            failures += exercise(true, 1) + exercise(false, 1) + exercise(true, 2) + exercise(false, 2)
                + exercise_down_projection() + exercise_swiglu() + exercise_conv_record();
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " NVFP4 SM70 input projections\n";
    return failures ? 1 : 0;
#endif
}
