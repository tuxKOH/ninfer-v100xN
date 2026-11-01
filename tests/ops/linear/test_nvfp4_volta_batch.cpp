#include "ops/linear/linear_test_common.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
#ifndef NINFER_VOLTA_BUILD
    return 77;
#else
    using namespace ninfer;
    using namespace ninfer::test::linear;
    if (!cuda_available()) { return 77; }
    try {
        // Real flattened MTP4 verifier widths, their tile boundaries, and the
        // maximum 8-request / 7-draft width. The independent oracle decodes the
        // original artifact codes/scales into naive FP64 dot products.
        constexpr std::array invocations{
            Invocation{9}, Invocation{10}, Invocation{16}, Invocation{17}, Invocation{20},
            Invocation{32}, Invocation{33}, Invocation{40}, Invocation{41}, Invocation{64}};
        int failures = 0;
        for (bool fast : {false, true}) {
            for (const auto shape : {std::array{7168, 5120}, std::array{8192, 5120},
                                     std::array{5120, 3072}, std::array{5120, 8704}}) {
                failures += run_shape("NVFP4 TP2 batch", ActivationCompute::A16,
                    make_nvfp4_weight, {shape[0], shape[1], 1841U,
                        Comparison::Sampled, true, invocations, true, fast});
            }
            constexpr std::array swiglu_widths{10, 20, 33, 40, 64};
            failures += test::linear_swiglu::run_column_parallel_profile(
                "NVFP4 TP2 batch SwiGLU",
                {QType::NVFP4, 17408, 5120, 8704, 1849U,
                 test::linear_swiglu::ActivationCompute::A16},
                swiglu_widths, true, fast);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4 TP2 batch oracles\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 TP2 batch test: " << error.what() << '\n';
        return 1;
    }
#endif
}
