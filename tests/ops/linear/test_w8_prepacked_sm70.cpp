#include "ninfer/ops/linear.h"
#include "ops/input_projection_test_common.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/w8/w8_prepack_sm70.h"
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
namespace fixture = ninfer::test::input_projection;
namespace qw = ninfer::test::quantized_weight;

#ifdef NINFER_VOLTA_BUILD
namespace {

int check_shape(int n, int k, std::span<const int> widths, bool exact_baseline = true,
                bool dflash = false) {
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    fixture::DevicePackedWeight weight(qw::make_patterned_weight(QType::W8G32_F16S, n, k, 149U, options));
    Weight view = weight.view();
    view.w8_dflash_projection = dflash;
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    // The fixture upload uses the legacy default stream; establish a device-wide
    // happens-before edge before launching the nonblocking prepack stream.
    cuda_synchronize();
    ops::detail::w8_prepack_qpn_sm70(view, stream);
    const auto codes = from_device<std::uint8_t>(view.qdata, static_cast<std::size_t>(n) * k);
    const auto scales = from_device<std::uint16_t>(view.scales, static_cast<std::size_t>(n) * k / 32);
    // Exact oracle reads the original represented artifact planes, not the GPU transform.
    const auto host_view = weight.host.device_weight(weight.host.payload.data());
    const auto* raw_codes = static_cast<const std::uint8_t*>(host_view.qdata);
    const auto* raw_scales = static_cast<const std::uint16_t*>(host_view.scales);
    for (int tile = 0; tile < n / 32; ++tile) {
        for (int row = 0; row < 32; ++row) {
            const int lane = ((row & 7) >> 2) * 16 + (row >> 3) * 4 + (row & 3);
            for (int group = 0; group < k / 32; ++group) {
                if (scales[(static_cast<std::size_t>(tile) * (k / 32) + group) * 32 + lane] !=
                    raw_scales[static_cast<std::size_t>(tile * 32 + row) * (k / 32) + group]) {
                    throw std::runtime_error("W8 scale permutation mismatch");
                }
                for (int sub = 0; sub < 4; ++sub) {
                    const auto* actual = codes.data() +
                        ((static_cast<std::size_t>(tile) * (k / 8) + group * 4 + sub) * 32 + lane) * 8;
                    const auto* expected = raw_codes +
                        static_cast<std::size_t>(tile * 32 + row) * k + group * 32 + sub * 8;
                    if (std::memcmp(actual, expected, 8) != 0) {
                        throw std::runtime_error("W8 code permutation mismatch");
                    }
                }
            }
        }
    }
    fixture::DevicePackedWeight baseline(weight.host);
    int failures = 0;
    for (int t : widths) {
        const auto host = fixture::make_bf16_activation(k, t, 157U + t);
        auto input = to_device_bf16(host);
        Tensor x(input.p, DType::BF16, {k, t});
        fixture::GuardedBf16Tensor expected(n, t), result(n, t);
        // Guard poisoning uses the default stream; explicitly order it before
        // either projection on the nonblocking test stream.
        cuda_synchronize();
        auto expected_tensor = expected.tensor();
        auto result_tensor = result.tensor();
        const auto baseline_view = baseline.view();
        const auto workspace_bytes = ops::linear_workspace_capacity_bytes(
            QType::W8G32_F16S, n, k, ops::LinearPolicy::A16Only, t, t);
        WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
        ops::linear(x, baseline_view, expected_tensor, stream);
        ops::linear(x, view, result_tensor, ops::LinearPolicy::A16Only, workspace, stream);
        cuda_synchronize(stream);
        const auto expected_bits = expected.bits();
        if (exact_baseline && result.bits() != expected_bits) {
            std::cerr << "W8 prepacked baseline mismatch N=" << n << " T=" << t << '\n';
            ++failures;
        }
        const auto oracle = fixture::projection_oracle(weight.host, 0, n, host, k, t, 17);
        failures += fixture::compare("W8 prepacked FP64",
            fixture::gather_rows(result.values(), n, 0, n, t, 17), oracle,
            ReductionCriterion{1.0 / 256, 1.0 / 256, 2.0 / 256});
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        ops::linear(x, view, result_tensor, ops::LinearPolicy::A16Only, workspace, stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, 0), "instantiate");
        for (int replay = 0; replay < 2; ++replay) {
            const auto changed = fixture::make_bf16_activation(k, t, 191U + t + replay);
            const auto changed_bits = fixture::bf16_bits(changed);
            cuda_check(cudaMemcpyAsync(input.p, changed_bits.data(), input.bytes,
                                        cudaMemcpyHostToDevice, stream), "changed input upload");
            cuda_check(cudaGraphLaunch(executable, stream), "graph replay");
            cuda_synchronize(stream);
            const auto changed_oracle = fixture::projection_oracle(weight.host, 0, n, changed, k, t, 17);
            failures += fixture::compare("W8 prepacked changed-input Graph FP64",
                fixture::gather_rows(result.values(), n, 0, n, t, 17), changed_oracle,
                ReductionCriterion{1.0 / 256, 1.0 / 256, 2.0 / 256});
            if (exact_baseline) {
                ops::linear(x, baseline_view, expected_tensor, stream);
                cuda_synchronize(stream);
                if (result.bits() != expected.bits()) { ++failures; }
            }
            failures += fixture::verify_preserved("W8 prepacked changed activation", input,
                                                  changed_bits);
        }
        failures += result.verify_guards("W8 prepacked output");
        cudaGraphExecDestroy(executable);
        cudaGraphDestroy(graph);
        std::cout << "W8 prepacked N=" << n << " K=" << k << " T=" << t
                  << " failures=" << failures << '\n';
    }
    if (from_device<std::uint8_t>(view.qdata, codes.size()) != codes ||
        from_device<std::uint16_t>(view.scales, scales.size()) != scales) {
        throw std::runtime_error("W8 prepacked persistent planes changed during inference");
    }
    cudaStreamDestroy(stream);
    return failures;
}

} // namespace
#endif

int main() {
#ifndef NINFER_VOLTA_BUILD
    std::cout << "SKIP: requires SM70 build\n";
    return 77;
#else
    if (cuda_unavailable()) { return 77; }
    try {
        const std::array tp2_widths{1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 33};
        const std::array other_widths{1, 2, 5, 9};
        int failures = check_shape(124160, 5120, tp2_widths);
        failures += check_shape(62080, 5120, tp2_widths);
        failures += check_shape(248320, 5120, other_widths);
        const std::array mtp_widths{1, 2, 3, 5, 8, 9, 33};
        for (const auto [n, k] : std::array<std::pair<int, int>, 15>{
                 {{14336, 5120}, {34816, 5120}, {5120, 10240}, {5120, 6144}, {5120, 17408},
                  {5120, 5120}, {7168, 5120}, {17408, 5120}, {5120, 3072}, {5120, 8704},
                  {3584, 5120}, {8704, 5120}, {5120, 1536}, {5120, 2560}, {5120, 4352}}}) {
            failures += check_shape(n, k, mtp_widths, false);
        }
        // DFlash's row permutation must preserve the pre-existing four-part
        // reduction as well as its original payload bits, including append T1/2
        // and the wide prefill fallback beyond the eight-column proposal block.
        const std::array dflash_widths{1, 2, 3, 7, 8, 9, 33};
        for (const auto [n, k] : std::array<std::pair<int, int>, 5>{
                 {{4096, 5120}, {34816, 5120}, {5120, 4096}, {5120, 17408}, {5120, 25600}}}) {
            failures += check_shape(n, k, dflash_widths, true, true);
        }
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
