// Two-device qualification suite for the cross-device collectives declared in
// include/ninfer/ops/allreduce.h.
//
// Every case REQUIRES two CUDA devices driven from ONE process; with fewer than two visible
// devices the suite reports the repository's skip code (77) instead of failing. This is also the
// first two-devices-in-one-process exercise of the per-device kernel-attribute memoization fix.
//
// Oracles:
//   allreduce_sum - independent FP64 elementwise sum of the two represented BF16 inputs. The
//     Op's observable output is BF16, so the comparison allows one BF16 ulp of output storage
//     rounding (relative 3.95e-3 >= 2^-8); the oracle itself performs no rounding.
//   allgather_rows/gather_columns_rank0 - exact: the Ops only relocate bytes, so every destination
//     byte is compared bit-for-bit against an independently constructed concatenation.
#include "ninfer/ops/allreduce.h"
#include "ops/op_tester.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// One BF16 ulp (2^-8 = 3.90625e-3) with the same small margin the residual_add suite uses. The
// only error the Op may introduce is the single round-to-nearest-even of a+b into BF16 storage,
// which is bounded by half an ulp; the criterion is deliberately not tighter than one ulp so a
// double-rounded tie cannot make the suite flaky.
constexpr PointwiseCriterion allreduce_sum_bf16_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// Independent oracle: the complete formula in FP64 from the represented BF16 inputs.
std::vector<double> allreduce_sum_oracle(const std::vector<float>& a,
                                         const std::vector<float>& b) {
    std::vector<double> expected(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        expected[i] = static_cast<double>(a[i]) + static_cast<double>(b[i]);
    }
    return expected;
}

void set_device(const ExecutionContext& ec, int rank) {
    cuda_check(cudaSetDevice(ec.dev[rank]->device), "cudaSetDevice");
}

void synchronize_both(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
    }
}

// Inputs are staged with cudaMemcpy/cudaMemset, which the runtime issues on each device's LEGACY
// default stream. DeviceContext::stream is created with cudaStreamNonBlocking and therefore does
// NOT implicitly synchronize with that default stream, so the staging writes must be retired
// before the collective's transfers read them. This is the caller obligation the Op contract
// documents; omitting the wait makes the suite intermittently read pre-staging bytes.
void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

template<class Body>
cudaGraph_t capture_both(const ExecutionContext& ec, Body&& body) {
    cudaEvent_t fork, join;
    set_device(ec, 0);
    cuda_check(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "capture fork");
    set_device(ec, 1);
    cuda_check(cudaEventCreateWithFlags(&join, cudaEventDisableTiming), "capture join");
    set_device(ec, 0);
    cuda_check(cudaStreamBeginCapture(ec.dev[0]->stream, cudaStreamCaptureModeGlobal), "capture begin");
    cuda_check(cudaEventRecord(fork, ec.dev[0]->stream), "capture fork record");
    set_device(ec, 1);
    cuda_check(cudaStreamWaitEvent(ec.dev[1]->stream, fork, 0), "capture fork wait");
    body();
    set_device(ec, 1);
    cuda_check(cudaEventRecord(join, ec.dev[1]->stream), "capture join record");
    set_device(ec, 0);
    cuda_check(cudaStreamWaitEvent(ec.dev[0]->stream, join, 0), "capture join wait");
    cudaGraph_t graph;
    cuda_check(cudaStreamEndCapture(ec.dev[0]->stream, &graph), "capture end");
    cuda_check(cudaEventDestroy(fork), "capture fork destroy");
    cuda_check(cudaEventDestroy(join), "capture join destroy");
    return graph;
}

// `ne0` is the contiguous dimension and `ne1` the outer one, so a 1-D buffer passes ne1 == 1 and
// the real row-parallel residual passes {5120, 48}.
int run_allreduce_case(const char* label, std::int32_t ne0, std::int32_t ne1, std::uint32_t seed,
                       const ExecutionContext& ec, const ops::PeerEvents& events, bool out_of_place = false, bool with_norm = false) {
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    std::vector<float> a(count), b(count);
    fill_uniform(a, seed, -8.0f, 8.0f);
    fill_uniform(b, seed + 1, -8.0f, 8.0f);
    round_to_bf16(a);
    round_to_bf16(b);

    const auto expected     = allreduce_sum_oracle(a, b);
    const auto a_bits       = encode_bf16(a);
    const auto b_bits       = encode_bf16(b);
    const std::size_t bytes = count * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes), normalized_0(with_norm ? bytes : 2);
    buffer_0.copy_from_host(a_bits.data(), bytes);
    staging_0.fill(0);

    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes), normalized_1(with_norm ? bytes : 2);
    buffer_1.copy_from_host(b_bits.data(), bytes);
    staging_1.fill(0);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {ne0, ne1}),
                                       Tensor(buffer_1.data(), DType::BF16, {ne0, ne1})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {ne0, ne1}),
                                        Tensor(staging_1.data(), DType::BF16, {ne0, ne1})};

    std::array<std::vector<float>, 2> host_weights;
    std::array<DeviceBuffer, 2> device_weights;
    std::array<Tensor, 2> norm_weights, normalized;
    if (with_norm) for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        host_weights[rank].resize(ne0);
        fill_uniform(host_weights[rank], seed + 5 + rank, -.75F, .75F);
        round_to_bf16(host_weights[rank]);
        device_weights[rank] = to_device_bf16(host_weights[rank]);
        norm_weights[rank] = Tensor(device_weights[rank].p, DType::BF16, {ne0});
        normalized[rank] = Tensor(rank ? normalized_1.data() : normalized_0.data(), DType::BF16, {ne0, ne1});
    }
    const ops::ReplicatedRmsNorm norm{norm_weights, normalized, 1e-6F, (seed & 1) != 0};
    const auto verify_norm = [&] {
        int errors = 0;
        if (!with_norm) return errors;
        for (int rank = 0; rank < 2; ++rank) {
            std::vector<double> oracle(count);
            for (int row = 0; row < ne1; ++row) {
                double ss = 0;
                for (int d = 0; d < ne0; ++d) {
                    const double v = bf16_to_f32(f32_to_bf16(expected[row * ne0 + d]));
                    ss += v * v;
                }
                const double inv = 1 / std::sqrt(ss / ne0 + double(norm.eps));
                for (int d = 0; d < ne0; ++d) oracle[row * ne0 + d] =
                    bf16_to_f32(f32_to_bf16(expected[row * ne0 + d])) * inv *
                    (double(host_weights[rank][d]) + (norm.unit_offset ? 1 : 0));
            }
            set_device(ec, rank);
            // A BF16 value rounded across a binade can exceed the old norm
            // fixture's 0.0034 * max bound even for correctly rounded FP64.
            // Keep its L2 bound; 1/256 covers BF16 round-to-nearest itself.
            errors += verify_reduction("collective RMSNorm FP64", from_device_bf16(normalized[rank].data, count),
                                         oracle, ReductionCriterion{1.85e-3, 1e-5, 1.0 / 256});
            errors += verify_exact("collective norm weight unchanged", from_device_bf16(device_weights[rank], ne0),
                                     std::vector<double>(host_weights[rank].begin(), host_weights[rank].end()));
        }
        errors += normalized_0.verify_guards("collective norm 0");
        errors += normalized_1.verify_guards("collective norm 1");
        return errors;
    };
    const auto invoke = [&] {
        if (with_norm) { ops::allreduce_sum_rmsnorm_to(buffer, staging, norm, ec, events); }
        else if (out_of_place) { ops::allreduce_sum_to(buffer, staging, ec, events); }
        else { ops::allreduce_sum(buffer, staging, ec, events); }
    };
    retire_staging(ec);
    invoke();
    synchronize_both(ec);

    int failures = verify_norm();
    set_device(ec, 0);
    failures += verify_pointwise((std::string(label) + " device 0").c_str(),
                                 from_device_bf16(out_of_place ? staging_0.data() : buffer_0.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_0.verify_guards("allreduce buffer device 0");
    failures += staging_0.verify_guards("allreduce staging device 0");
    set_device(ec, 1);
    failures += verify_pointwise((std::string(label) + " device 1").c_str(),
                                 from_device_bf16(out_of_place ? staging_1.data() : buffer_1.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_1.verify_guards("allreduce buffer device 1");
    failures += staging_1.verify_guards("allreduce staging device 1");
    // Replay the same public collective from original represented inputs, against the same
    // FP64 oracle. This exercises graph source-lifetime edges and both sum routes directly.
    auto graph = capture_both(ec, invoke);
    cudaGraphExec_t executable;
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "allreduce graph");
    for (int replay = 0; replay < 2; ++replay) {
        set_device(ec, 0); buffer_0.copy_from_host(a_bits.data(), bytes);
        set_device(ec, 1); buffer_1.copy_from_host(b_bits.data(), bytes);
        retire_staging(ec);
        set_device(ec, 0);
        cuda_check(cudaGraphLaunch(executable, ec.dev[0]->stream), "allreduce graph launch");
        synchronize_both(ec);
        failures += verify_norm();
        set_device(ec, 0);
        failures += verify_pointwise((std::string(label) + " graph device 0").c_str(),
                                     from_device_bf16(out_of_place ? staging_0.data() : buffer_0.data(), count), expected,
                                     allreduce_sum_bf16_criterion());
        failures += buffer_0.verify_guards("allreduce graph buffer device 0");
        failures += staging_0.verify_guards("allreduce graph staging device 0");
        set_device(ec, 1);
        failures += verify_pointwise((std::string(label) + " graph device 1").c_str(),
                                     from_device_bf16(out_of_place ? staging_1.data() : buffer_1.data(), count), expected,
                                     allreduce_sum_bf16_criterion());
        failures += buffer_1.verify_guards("allreduce graph buffer device 1");
        failures += staging_1.verify_guards("allreduce graph staging device 1");
    }
    if (out_of_place) {
        set_device(ec, 0);
        failures += verify_exact("out-of-place input 0 immutable",
                                  from_device<std::uint16_t>(buffer_0.data(), count), a_bits);
        set_device(ec, 1);
        failures += verify_exact("out-of-place input 1 immutable",
                                  from_device<std::uint16_t>(buffer_1.data(), count), b_bits);
    }
    cuda_check(cudaGraphExecDestroy(executable), "allreduce graph destroy");
    cuda_check(cudaGraphDestroy(graph), "allreduce definition destroy");
    return failures;
}

// Consecutive short collectives change active CTA counts and immediately reuse
// the same sources, without a gather or host wait between them. This exercises
// the packet epochs across changing sizes and transitions to scalar/DMA routes.
int run_varying_size_chain(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr int maximum = 5120 * 64 + 1, rounds = 128;
    const std::vector<std::uint16_t> initial(maximum, f32_to_bf16(std::ldexp(1.0f, -60)));
    std::vector<std::uint16_t> expected = initial;
    const std::array<int, 9> sizes{5120, 25600, 40960, 51200, 102400, 204800,
                                  327680, 4097, maximum};
    for (int round = 0; round < rounds; ++round) {
        for (int i = 0; i < sizes[round % sizes.size()]; ++i) {
            expected[i] = f32_to_bf16(2.0f * bf16_to_f32(expected[i]));
        }
    }
    set_device(ec, 0);
    GuardedDeviceBuffer a(2ULL * maximum), sa(2ULL * maximum), na(2ULL * maximum);
    auto wa = to_device_bf16(std::vector<float>(5120, 1.0F));
    set_device(ec, 1);
    GuardedDeviceBuffer b(2ULL * maximum), sb(2ULL * maximum), nb(2ULL * maximum);
    auto wb = to_device_bf16(std::vector<float>(5120, 1.0F));
    const auto reset = [&] {
        set_device(ec, 0); a.copy_from_host(initial.data(), 2ULL * maximum);
        set_device(ec, 1); b.copy_from_host(initial.data(), 2ULL * maximum);
        retire_staging(ec);
    };
    const auto invoke = [&] {
        for (int round = 0; round < rounds; ++round) {
            const int n = sizes[round % sizes.size()];
            const std::array<Tensor, 2> input{Tensor(a.data(), DType::BF16, {n}),
                                              Tensor(b.data(), DType::BF16, {n})};
            const std::array<Tensor, 2> staging{Tensor(sa.data(), DType::BF16, {n}),
                                                Tensor(sb.data(), DType::BF16, {n})};
            if (n % 5120 == 0 && (round & 1)) {
                const std::array<Tensor, 2> sources{input[0].view({5120, n/5120}), input[1].view({5120, n/5120})};
                const std::array<Tensor, 2> sums{staging[0].view({5120, n/5120}), staging[1].view({5120, n/5120})};
                const std::array<Tensor, 2> weights{Tensor(wa.p, DType::BF16, {5120}), Tensor(wb.p, DType::BF16, {5120})};
                const std::array<Tensor, 2> normalized{Tensor(na.data(), DType::BF16, {5120, n/5120}), Tensor(nb.data(), DType::BF16, {5120, n/5120})};
                ops::allreduce_sum_rmsnorm_to(sources, sums, {weights, normalized, 1e-6F, true}, ec, events);
                for (int rank=0; rank<2; ++rank) {
                    set_device(ec, rank);
                    cuda_check(cudaMemcpyAsync(input[rank].data, staging[rank].data, input[rank].bytes(),
                        cudaMemcpyDeviceToDevice, ec.dev[rank]->stream), "mixed packet source reuse");
                }
            } else ops::allreduce_sum(input, staging, ec, events);
        }
    };
    const auto check = [&] {
        int failures = 0;
        set_device(ec, 0);
        failures += verify_exact("varying-size chain rank 0", from_device<std::uint16_t>(a.data(), maximum), expected);
        failures += a.verify_guards("varying-size rank 0");
        failures += sa.verify_guards("varying-size staging 0");
        failures += na.verify_guards("varying-size norm 0");
        set_device(ec, 1);
        failures += verify_exact("varying-size chain rank 1", from_device<std::uint16_t>(b.data(), maximum), expected);
        failures += b.verify_guards("varying-size rank 1");
        failures += sb.verify_guards("varying-size staging 1");
        failures += nb.verify_guards("varying-size norm 1");
        return failures;
    };
    reset(); invoke(); synchronize_both(ec);
    int failures = check();
    const auto graph = capture_both(ec, invoke);
    cudaGraphExec_t executable;
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "varying-size graph");
    for (int repeat = 0; repeat < 2; ++repeat) {
        reset(); set_device(ec, 0);
        cuda_check(cudaGraphLaunch(executable, ec.dev[0]->stream), "varying-size graph launch");
        synchronize_both(ec); failures += check();
    }
    cuda_check(cudaGraphExecDestroy(executable), "varying-size graph destroy");
    cuda_check(cudaGraphDestroy(graph), "varying-size definition destroy");
    return failures;
}

int run_special_sum_inputs(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr int n = 5120;
    const std::array<std::uint16_t, 16> pattern{
        0x0000, 0x8000, 0x0001, 0x8001, 0x7f7f, 0xff7f, 0x7f80, 0xff80,
        0xffff, 0x7fff, 0x7fc0, 0xffc0, 0x3f80, 0xbf80, 0x0080, 0x8080};
    std::vector<std::uint16_t> x(n), y(n);
    for (int i = 0; i < n; ++i) { x[i] = pattern[i % 16]; y[i] = pattern[(i / 16) % 16]; }
    set_device(ec, 0); auto a = to_device(x); GuardedDeviceBuffer oa(2ULL * n);
    set_device(ec, 1); auto b = to_device(y); GuardedDeviceBuffer ob(2ULL * n);
    const std::array<Tensor, 2> input{Tensor(a.p, DType::BF16, {n}), Tensor(b.p, DType::BF16, {n})};
    const std::array<Tensor, 2> output{Tensor(oa.data(), DType::BF16, {n}), Tensor(ob.data(), DType::BF16, {n})};
    retire_staging(ec); ops::allreduce_sum_to(input, output, ec, events); synchronize_both(ec);
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        const auto actual = from_device<std::uint16_t>(output[rank].data, n);
        for (int i = 0; i < n; ++i) {
            const double ideal = double(bf16_to_f32(x[i])) + double(bf16_to_f32(y[i]));
            if (std::isnan(ideal)) {
                if (!std::isnan(bf16_to_f32(actual[i]))) { ++failures; }
            } else if (actual[i] != f32_to_bf16(static_cast<float>(ideal))) { ++failures; }
        }
    }
    set_device(ec, 0); failures += verify_exact("special input 0 unchanged", from_device<std::uint16_t>(a, n), x);
    failures += oa.verify_guards("special output 0");
    set_device(ec, 1); failures += verify_exact("special input 1 unchanged", from_device<std::uint16_t>(b, n), y);
    failures += ob.verify_guards("special output 1");
    return failures;
}

// `row_length` is ne[0] (the contiguous row) and the gathered axis is ne[1], so device 0 owns the
// leading `rows_0` rows and device 1 the trailing `rows_1`.
int run_allgather_case(const char* label, std::int32_t rows_0, std::int32_t rows_1,
                       std::int32_t row_length, std::uint32_t seed, const ExecutionContext& ec,
                       const ops::PeerEvents& events) {
    const std::int32_t rows  = rows_0 + rows_1;
    const std::size_t part_0 = static_cast<std::size_t>(rows_0) * row_length;
    const std::size_t part_1 = static_cast<std::size_t>(rows_1) * row_length;

    std::vector<float> source_0(part_0), source_1(part_1);
    fill_uniform(source_0, seed, -8.0f, 8.0f);
    fill_uniform(source_1, seed + 1, -8.0f, 8.0f);
    const auto bits_0 = encode_bf16(source_0);
    const auto bits_1 = encode_bf16(source_1);

    // Exact oracle: the gathered image is the concatenation of the two owned row ranges.
    std::vector<std::uint16_t> expected;
    expected.reserve(part_0 + part_1);
    expected.insert(expected.end(), bits_0.begin(), bits_0.end());
    expected.insert(expected.end(), bits_1.begin(), bits_1.end());

    const std::size_t full_bytes = expected.size() * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer source_device_0(part_0 * sizeof(std::uint16_t));
    GuardedDeviceBuffer destination_0(full_bytes);
    source_device_0.copy_from_host(bits_0.data(), source_device_0.bytes());
    destination_0.fill(0xcd);

    set_device(ec, 1);
    GuardedDeviceBuffer source_device_1(part_1 * sizeof(std::uint16_t));
    GuardedDeviceBuffer destination_1(full_bytes);
    source_device_1.copy_from_host(bits_1.data(), source_device_1.bytes());
    destination_1.fill(0xcd);

    const std::array<Tensor, 2> destination{
        Tensor(destination_0.data(), DType::BF16, {row_length, rows}),
        Tensor(destination_1.data(), DType::BF16, {row_length, rows})};
    const std::array<Tensor, 2> part{
        Tensor(source_device_0.data(), DType::BF16, {row_length, rows_0}),
        Tensor(source_device_1.data(), DType::BF16, {row_length, rows_1})};

    retire_staging(ec);
    ops::allgather_rows(destination, part, ec, events);
    synchronize_both(ec);

    int failures = 0;
    set_device(ec, 0);
    failures += verify_exact((std::string(label) + " device 0").c_str(),
                             from_device<std::uint16_t>(destination_0.data(), expected.size()),
                             expected);
    failures += verify_exact("allgather source device 0 unchanged",
                             from_device<std::uint16_t>(source_device_0.data(), part_0), bits_0);
    failures += destination_0.verify_guards("allgather destination device 0");
    failures += source_device_0.verify_guards("allgather source device 0");
    set_device(ec, 1);
    failures += verify_exact((std::string(label) + " device 1").c_str(),
                             from_device<std::uint16_t>(destination_1.data(), expected.size()),
                             expected);
    failures += verify_exact("allgather source device 1 unchanged",
                             from_device<std::uint16_t>(source_device_1.data(), part_1), bits_1);
    failures += destination_1.verify_guards("allgather destination device 1");
    failures += source_device_1.verify_guards("allgather source device 1");
    return failures;
}

int run_gather_columns_rank0_case(const char* label, std::int32_t width_0,
                                  std::int32_t width_1, std::int32_t columns,
                                  std::uint32_t seed, const ExecutionContext& ec,
                                  const ops::PeerEvents& events) {
    const std::int32_t width = width_0 + width_1;
    const std::size_t count_0 = static_cast<std::size_t>(width_0) * columns;
    const std::size_t count_1 = static_cast<std::size_t>(width_1) * columns;
    std::vector<float> values_0(count_0), values_1(count_1);
    fill_uniform(values_0, seed, -8.0f, 8.0f);
    fill_uniform(values_1, seed + 1, -8.0f, 8.0f);
    const auto bits_0 = encode_bf16(values_0);
    const auto bits_1 = encode_bf16(values_1);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(width) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        std::copy_n(bits_0.data() + static_cast<std::size_t>(column) * width_0, width_0,
                    expected.data() + static_cast<std::size_t>(column) * width);
        std::copy_n(bits_1.data() + static_cast<std::size_t>(column) * width_1, width_1,
                    expected.data() + static_cast<std::size_t>(column) * width + width_0);
    }
    const std::size_t full_bytes = expected.size() * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer source_0(count_0 * sizeof(std::uint16_t));
    GuardedDeviceBuffer destination_0(full_bytes);
    source_0.copy_from_host(bits_0.data(), source_0.bytes());
    destination_0.fill(0xcd);
    set_device(ec, 1);
    GuardedDeviceBuffer source_1(count_1 * sizeof(std::uint16_t));
    GuardedDeviceBuffer destination_1(full_bytes);
    source_1.copy_from_host(bits_1.data(), source_1.bytes());
    destination_1.fill(0xcd);

    const std::array<Tensor, 2> destination{
        Tensor(destination_0.data(), DType::BF16, {width, columns}),
        Tensor(destination_1.data(), DType::BF16, {width, columns})};
    const std::array<Tensor, 2> part{
        Tensor(source_0.data(), DType::BF16, {width_0, columns}),
        Tensor(source_1.data(), DType::BF16, {width_1, columns})};

    retire_staging(ec);
    ops::gather_columns_rank0(destination[0], part, ec, events);
    synchronize_both(ec);

    int failures = 0;
    set_device(ec, 0);
    failures += verify_exact((std::string(label) + " device 0").c_str(),
                             from_device<std::uint16_t>(destination_0.data(), expected.size()),
                             expected);
    failures += verify_exact("gather_columns_rank0 source device 0 unchanged",
                             from_device<std::uint16_t>(source_0.data(), count_0), bits_0);
    failures += destination_0.verify_guards("gather_columns_rank0 destination device 0");
    failures += source_0.verify_guards("gather_columns_rank0 source device 0");
    set_device(ec, 1);
    failures += verify_exact("gather_columns_rank0 unused destination remains untouched",
                             from_device<std::uint16_t>(destination_1.data(), expected.size()),
                             std::vector<std::uint16_t>(expected.size(), 0xcdcd));
    failures += verify_exact("gather_columns_rank0 source device 1 unchanged",
                             from_device<std::uint16_t>(source_1.data(), count_1), bits_1);
    failures += destination_1.verify_guards("gather_columns_rank0 unused destination device 1");
    failures += source_1.verify_guards("gather_columns_rank0 source device 1");
    return failures;
}

// Regression guard for the cross-call write-after-read hazard.
//
// kRounds alternating collectives are issued back-to-back on ONE set of buffers, staging, and
// events with NO host synchronization until the very end -- exactly what the tensor-parallel
// forward loop produces (128 all-reduces per decode token, plus one logit allgather per column)
// and what a captured graph replays. Under the earlier push-based
// design, call k+1's inbound write into the peer's staging was ordered only against the sender's
// own stream history, not against the receiver's call-k read of that staging; this loop is the
// pattern where that window opens.
//
// Two properties make a stale operand observable rather than benign:
//   * the chain value changes every round -- both devices start at 2^-40 and each all-reduce
//     doubles it, so after k rounds every element is exactly 2^(k-40). Every intermediate is a
//     power of two, hence exact in BF16 and in the FP32 accumulator, and an operand from the wrong
//     round yields a value that is not the expected power of two (3v instead of 2v, say);
//   * the interleaved all-gather reads the live all-reduce buffers as its parts, so it also
//     carries the round's value into its destination and is checked against the same expectation,
//     and its own cross-call ordering (the peer must finish reading part[r] before the next
//     all-reduce overwrites it) is exercised rather than merely occupying the streams.
//
// Deliberate skew: a large memset is queued on rank 0's stream first, with no host sync, so the
// two streams run genuinely out of step for the first rounds instead of in lockstep.
int run_chained_case(const ExecutionContext& ec, const ops::PeerEvents& events, bool out_of_place = false, bool with_norm = false) {
    const std::int32_t n = 5120 * (with_norm ? 5 : 1);
    constexpr int kRounds        = 64;
    constexpr int kStartExponent = -40;
    constexpr std::size_t kSkewBytes = 256u << 20;
    constexpr int kSkewRepeats       = 8;

    const float start           = std::ldexp(1.0f, kStartExponent);
    const double expected_value = std::ldexp(1.0, kStartExponent + kRounds);
    const std::size_t count     = static_cast<std::size_t>(n);
    const std::size_t bytes     = count * sizeof(std::uint16_t);

    const std::vector<std::uint16_t> start_bits(count, f32_to_bf16(start));
    const std::vector<double> expected(count, expected_value);
    // Both halves of the gathered image come from the two all-reduce buffers, which hold the same
    // value once the round's all-reduce has run.
    const std::vector<double> expected_gathered(2 * count, expected_value);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes), gathered_0(2 * bytes), norm_0(with_norm ? bytes : 2);
    auto weight_0 = to_device_bf16(std::vector<float>(5120, 1.0F));
    buffer_0.copy_from_host(start_bits.data(), bytes);
    staging_0.fill(0);
    gathered_0.fill(0xcd);
    DeviceBuffer skew(kSkewBytes);

    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes), gathered_1(2 * bytes), norm_1(with_norm ? bytes : 2);
    auto weight_1 = to_device_bf16(std::vector<float>(5120, 1.0F));
    buffer_1.copy_from_host(start_bits.data(), bytes);
    staging_1.fill(0);
    gathered_1.fill(0xcd);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {5120, with_norm ? 5 : 1}),
                                       Tensor(buffer_1.data(), DType::BF16, {5120, with_norm ? 5 : 1})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {5120, with_norm ? 5 : 1}),
                                        Tensor(staging_1.data(), DType::BF16, {5120, with_norm ? 5 : 1})};
    const std::array<Tensor, 2> gathered{Tensor(gathered_0.data(), DType::BF16, {n, 2}),
                                         Tensor(gathered_1.data(), DType::BF16, {n, 2})};
    const std::array<Tensor, 2> part{Tensor(buffer_0.data(), DType::BF16, {n, 1}),
                                     Tensor(buffer_1.data(), DType::BF16, {n, 1})};

    const std::array<Tensor, 2> norm_weights{Tensor(weight_0.p, DType::BF16, {5120}),
                                            Tensor(weight_1.p, DType::BF16, {5120})};
    const std::array<Tensor, 2> norm_outputs{Tensor(norm_0.data(), DType::BF16, {5120, 5}),
                                            Tensor(norm_1.data(), DType::BF16, {5120, 5})};
    const ops::ReplicatedRmsNorm norm{norm_weights, norm_outputs, 1e-6F, false};
    const auto invoke = [&] {
        if (out_of_place) {
            if (with_norm) ops::allreduce_sum_rmsnorm_to(buffer, staging, norm, ec, events);
            else ops::allreduce_sum_to(buffer, staging, ec, events);
            // Immediately overwrite the sources on their own streams; a missing
            // peer-read retirement edge becomes a real cross-rank race here.
            for (int rank = 0; rank < 2; ++rank) {
                set_device(ec, rank);
                cuda_check(cudaMemcpyAsync(buffer[rank].data, staging[rank].data, bytes,
                    cudaMemcpyDeviceToDevice, ec.dev[rank]->stream), "reuse sum sources");
            }
        } else {
            ops::allreduce_sum(buffer, staging, ec, events);
        }
    };
    retire_staging(ec);

    set_device(ec, 0);
    for (int i = 0; i < kSkewRepeats; ++i) {
        cuda_check(cudaMemsetAsync(skew.p, i & 0xff, skew.bytes, ec.dev[0]->stream),
                   "cudaMemsetAsync skew");
    }

    for (int round = 0; round < kRounds; ++round) {
        invoke();
        ops::allgather_rows(gathered, part, ec, events);
    }
    synchronize_both(ec);

    const auto check_norm = [&] {
        int errors = 0;
        if (with_norm) {
            const double expected_norm = expected_value / std::sqrt(expected_value * expected_value + double(norm.eps));
            for (int rank = 0; rank < 2; ++rank) {
                set_device(ec, rank);
                errors += verify_pointwise("chained norm FP64", from_device_bf16(norm_outputs[rank].data, count),
                                            std::vector<double>(count, expected_norm), PointwiseCriterion{1e-6, 1.0 / 256});
            }
            errors += norm_0.verify_guards("chained norm 0");
            errors += norm_1.verify_guards("chained norm 1");
        }
        return errors;
    };
    int failures = check_norm();
    set_device(ec, 0);
    failures += verify_pointwise("chained 64x device 0 buffer",
                                 from_device_bf16(buffer_0.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += verify_pointwise("chained 64x device 0 gathered",
                                 from_device_bf16(gathered_0.data(), 2 * count), expected_gathered,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_0.verify_guards("chained buffer device 0");
    failures += staging_0.verify_guards("chained staging device 0");
    failures += gathered_0.verify_guards("chained gathered device 0");
    set_device(ec, 1);
    failures += verify_pointwise("chained 64x device 1 buffer",
                                 from_device_bf16(buffer_1.data(), count), expected,
                                 allreduce_sum_bf16_criterion());
    failures += verify_pointwise("chained 64x device 1 gathered",
                                 from_device_bf16(gathered_1.data(), 2 * count), expected_gathered,
                                 allreduce_sum_bf16_criterion());
    failures += buffer_1.verify_guards("chained buffer device 1");
    failures += staging_1.verify_guards("chained staging device 1");
    failures += gathered_1.verify_guards("chained gathered device 1");
    // Repeat the skewed state transition in a single cross-device graph, then replay from
    // original states. No host wait exists between the 64 dependent collectives.
    const auto graph = capture_both(ec, [&] {
        set_device(ec, 0);
        for (int i = 0; i < kSkewRepeats; ++i) {
            cuda_check(cudaMemsetAsync(skew.p, i, skew.bytes, ec.dev[0]->stream), "captured skew");
        }
        for (int round = 0; round < kRounds; ++round) {
            invoke();
            ops::allgather_rows(gathered, part, ec, events);
        }
    });
    cudaGraphExec_t executable;
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "chained graph");
    for (int replay = 0; replay < 2; ++replay) {
        set_device(ec, 0); buffer_0.copy_from_host(start_bits.data(), bytes);
        set_device(ec, 1); buffer_1.copy_from_host(start_bits.data(), bytes);
        retire_staging(ec);
        set_device(ec, 0);
        cuda_check(cudaGraphLaunch(executable, ec.dev[0]->stream), "chained graph launch");
        synchronize_both(ec);
        failures += check_norm();
        set_device(ec, 0);
        failures += verify_pointwise("chained graph buffer 0", from_device_bf16(buffer_0.data(), count),
                                     expected, allreduce_sum_bf16_criterion());
        failures += verify_pointwise("chained graph gathered 0", from_device_bf16(gathered_0.data(), 2 * count),
                                     expected_gathered, allreduce_sum_bf16_criterion());
        set_device(ec, 1);
        failures += verify_pointwise("chained graph buffer 1", from_device_bf16(buffer_1.data(), count),
                                     expected, allreduce_sum_bf16_criterion());
        failures += verify_pointwise("chained graph gathered 1", from_device_bf16(gathered_1.data(), 2 * count),
                                     expected_gathered, allreduce_sum_bf16_criterion());
    }
    cuda_check(cudaGraphExecDestroy(executable), "chained graph destroy");
    cuda_check(cudaGraphDestroy(graph), "chained definition destroy");
    return failures;
}

// The per-decode-token all-reduce payload is one 5120-element BF16 hidden row = 10 KiB, issued
// 128 times per token. Measure the full call: both peer transfers plus both local combines, with
// the host waiting for both devices to retire the work.
//
// The figure is SYNC-DOMINATED and is an upper bound, not the op's device cost: each iteration
// pays two host-side cudaStreamSynchronize round trips that the production path does not, because
// the forward loop issues 128 of these back-to-back and the decode path replays them inside a
// captured CUDA graph with no host sync at all. Read it as "one isolated, fully drained
// all-reduce", which is what the < 100 us budget is stated against.
int run_microbenchmark(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr std::int32_t n        = 5120;
    constexpr double kLimitMicros   = 100.0;
    constexpr int kWarmupIterations = 50;
    constexpr int kTimedIterations  = 500;
    const std::size_t bytes         = static_cast<std::size_t>(n) * sizeof(std::uint16_t);

    set_device(ec, 0);
    GuardedDeviceBuffer buffer_0(bytes), staging_0(bytes);
    buffer_0.fill(0);
    staging_0.fill(0);
    set_device(ec, 1);
    GuardedDeviceBuffer buffer_1(bytes), staging_1(bytes);
    buffer_1.fill(0);
    staging_1.fill(0);

    const std::array<Tensor, 2> buffer{Tensor(buffer_0.data(), DType::BF16, {n}),
                                       Tensor(buffer_1.data(), DType::BF16, {n})};
    const std::array<Tensor, 2> staging{Tensor(staging_0.data(), DType::BF16, {n}),
                                        Tensor(staging_1.data(), DType::BF16, {n})};

    retire_staging(ec);
    for (int i = 0; i < kWarmupIterations; ++i) {
        ops::allreduce_sum(buffer, staging, ec, events);
        synchronize_both(ec);
    }

    std::vector<double> samples;
    samples.reserve(kTimedIterations);
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < kTimedIterations; ++i) {
        const auto call_started = std::chrono::steady_clock::now();
        ops::allreduce_sum(buffer, staging, ec, events);
        synchronize_both(ec);
        const std::chrono::duration<double, std::micro> call_elapsed =
            std::chrono::steady_clock::now() - call_started;
        samples.push_back(call_elapsed.count());
    }
    const std::chrono::duration<double, std::micro> elapsed =
        std::chrono::steady_clock::now() - started;
    const double mean_micros = elapsed.count() / kTimedIterations;
    std::sort(samples.begin(), samples.end());

    std::cout << "allreduce microbench: " << bytes << " B bf16 mean " << mean_micros << " us, p50 "
              << samples[samples.size() / 2] << " us, p99 " << samples[(samples.size() * 99) / 100]
              << " us, max " << samples.back() << " us over " << kTimedIterations
              << " iterations (limit " << kLimitMicros << " us, host-sync dominated)\n";

    int failures = 0;
    set_device(ec, 0);
    failures += buffer_0.verify_guards("microbench buffer device 0");
    failures += staging_0.verify_guards("microbench staging device 0");
    set_device(ec, 1);
    failures += buffer_1.verify_guards("microbench buffer device 1");
    failures += staging_1.verify_guards("microbench staging device 1");
    if (mean_micros >= kLimitMicros) {
        std::cerr << "allreduce microbench: mean " << mean_micros << " us exceeds the "
                  << kLimitMicros << " us budget for a 10 KiB two-device all-reduce\n";
        ++failures;
    }
    return failures;
}

int run_argmax_collective_case(int columns, const ExecutionContext& ec,
                               const ops::PeerEvents& events, bool mapped = false) {
    constexpr int Rounds = 16, Shard = 65536;
    const std::size_t bytes = sizeof(std::int32_t) * columns;
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> value, index, out, history, seeds;
    std::array<Tensor, 2> values, indices, outputs;
    std::array<std::vector<float>, 2> host_values;
    std::array<std::vector<int>, 2> host_indices;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        value[rank] = std::make_unique<GuardedDeviceBuffer>(bytes);
        index[rank] = std::make_unique<GuardedDeviceBuffer>(bytes);
        out[rank] = std::make_unique<GuardedDeviceBuffer>(bytes);
        history[rank] = std::make_unique<GuardedDeviceBuffer>(Rounds * bytes);
        seeds[rank] = std::make_unique<GuardedDeviceBuffer>(4 * bytes);
        values[rank] = Tensor(value[rank]->data(), DType::FP32, {1, columns});
        indices[rank] = Tensor(index[rank]->data(), DType::I32, {1, columns});
        outputs[rank] = Tensor(out[rank]->data(), DType::I32, {columns});
        host_values[rank].resize(2 * columns);
        host_indices[rank].resize(2 * columns);
    }
    set_device(ec, 0);
    GuardedDeviceBuffer sv(2 * bytes), si(2 * bytes);
    Tensor gathered_values(sv.data(), DType::FP32, {2, columns});
    Tensor gathered_indices(si.data(), DType::I32, {2, columns});
    GuardedDeviceBuffer map_buffer(2 * Shard * sizeof(int));
    std::vector<int> map(2 * Shard);
    for (int i = 0; i < 2 * Shard; ++i) { map[i] = (i * 97 + 13) % (2 * Shard); }
    cuda_check(cudaMemcpy(map_buffer.data(), map.data(), 2 * Shard * sizeof(int),
                          cudaMemcpyHostToDevice), "argmax map");
    const auto invoke = [&] {
        for (int round = 0; round < Rounds; ++round) {
            for (int rank = 0; rank < 2; ++rank) {
                set_device(ec, rank);
                auto* seed = static_cast<std::uint8_t*>(seeds[rank]->data());
                cuda_check(cudaMemcpyAsync(values[rank].data, seed + (round % 2) * bytes,
                    bytes, cudaMemcpyDeviceToDevice, ec.dev[rank]->stream), "reuse argmax scores");
                cuda_check(cudaMemcpyAsync(indices[rank].data, seed + (2 + round % 2) * bytes,
                    bytes, cudaMemcpyDeviceToDevice, ec.dev[rank]->stream), "reuse argmax indices");
            }
            ops::allreduce_argmax(values, indices, outputs, gathered_values, gathered_indices,
                                   Shard, ec, events, mapped ? static_cast<const int*>(map_buffer.data()) : nullptr);
            for (int rank = 0; rank < 2; ++rank) {
                set_device(ec, rank);
                cuda_check(cudaMemcpyAsync(static_cast<std::uint8_t*>(history[rank]->data()) +
                    round * bytes, outputs[rank].data, bytes, cudaMemcpyDeviceToDevice,
                    ec.dev[rank]->stream), "retire argmax output");
            }
        }
    };
    cudaGraph_t graph = capture_both(ec, invoke);
    cudaGraphExec_t executable;
    set_device(ec, 0);
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "argmax graph");
    int failures = 0;
    for (int probe = 0; probe < 3; ++probe) {
        for (int rank = 0; rank < 2; ++rank) {
            for (int mode = 0; mode < 2; ++mode) {
                for (int t = 0; t < columns; ++t) {
                    const int i = mode * columns + t;
                    float v = float((t + mode + probe + rank) % 7 - 3);
                    if (t % 4 == 0) { v = mode ? -INFINITY : INFINITY; }
                    if (t % 4 == 1) { v = rank ? -0.0F : 0.0F; }
                    host_values[rank][i] = v;
                    host_indices[rank][i] = (t * 11 + probe * 13 + mode * 17) % Shard;
                }
            }
            set_device(ec, rank);
            cuda_check(cudaMemcpy(seeds[rank]->data(), host_values[rank].data(), 2 * bytes,
                cudaMemcpyHostToDevice), "argmax score seeds");
            cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(seeds[rank]->data()) + 2 * bytes,
                host_indices[rank].data(), 2 * bytes, cudaMemcpyHostToDevice), "argmax index seeds");
        }
        retire_staging(ec);
        if (probe == 0) { invoke(); }
        else {
            set_device(ec, 0);
            cuda_check(cudaGraphLaunch(executable, ec.dev[0]->stream), "argmax replay");
        }
        synchronize_both(ec);
        std::vector<int> expected(Rounds * columns);
        for (int round = 0; round < Rounds; ++round) {
            for (int t = 0; t < columns; ++t) {
                const int i = (round % 2) * columns + t;
                const double a = host_values[0][i], b = host_values[1][i];
                const int ia = host_indices[0][i], ib = Shard + host_indices[1][i];
                const int winner = b > a || (b == a && ib < ia) ? ib : ia;
                expected[round * columns + t] = mapped ? map[winner] : winner;
            }
        }
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            failures += verify_exact("replicated argmax and source retirement",
                from_device<int>(history[rank]->data(), expected.size()), expected);
            failures += value[rank]->verify_guards("argmax values");
            failures += index[rank]->verify_guards("argmax indices");
            failures += out[rank]->verify_guards("argmax output");
            failures += history[rank]->verify_guards("argmax history");
            failures += seeds[rank]->verify_guards("argmax seeds");
            failures += verify_exact("argmax seed values unchanged",
                from_device<float>(seeds[rank]->data(), 2 * columns), host_values[rank]);
        }
    }
    set_device(ec, 0);
    failures += sv.verify_guards("argmax gather values");
    failures += si.verify_guards("argmax gather indices");
    failures += map_buffer.verify_guards("argmax ID map");
    failures += verify_exact("argmax ID map unchanged",
        from_device<int>(map_buffer.data(), map.size()), map);
    cuda_check(cudaGraphExecDestroy(executable), "destroy argmax executable");
    cuda_check(cudaGraphDestroy(graph), "destroy argmax graph");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: cross-device collectives require two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }

    const ExecutionContext ec({0, 1});
    bool peer_access = ops::enable_peer_access(ec);
    const bool require_host_packets = std::getenv("NINFER_TEST_HOST_PACKETS") != nullptr;
    if (peer_access && (std::getenv("NINFER_TEST_PEER_OFF") != nullptr || require_host_packets)) {
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            cuda_check(cudaDeviceDisablePeerAccess(ec.dev[1 - rank]->device), "disable test peer access");
        }
        peer_access = false;
    }
    std::cout << "peer access: "
              << (peer_access ? "enabled (direct P2P)"
                              : "unavailable (verified CUDA UVA D2D staging)")
              << '\n';
    const ops::PeerEvents events(ec, peer_access, require_host_packets
        ? ops::PeerEvents::HostPackets::Required : ops::PeerEvents::HostPackets::Auto);
    if (events.host_packets()) std::cout << "small reductions: mapped-host packets\n";

    int failures = 0;
    for (int columns : {1, 5, 8, 33}) {
        failures += run_argmax_collective_case(columns, ec, events);
        failures += run_argmax_collective_case(columns, ec, events, true);
        const ops::PeerEvents dma_events(ec, false, ops::PeerEvents::HostPackets::Disabled);
        failures += run_argmax_collective_case(columns, ec, dma_events);
        failures += run_argmax_collective_case(columns, ec, dma_events, true);
    }
    // Real decode shape first: 5120 is the hidden dimension all-reduced 128 times per token.
    // Direct-read threshold, scalar tail, wide DMA fallback, graph replay and immutable inputs.
    for (int count : {1, 4097, 5120, 25600, 40960, 40961, 51200, 102400, 204800,
                     327680, 327681}) {
        failures += run_allreduce_case("allreduce_sum_to", count, 1, 121u, ec, events, true);
        const ops::PeerEvents dma_events(ec, false, ops::PeerEvents::HostPackets::Disabled);
        failures += run_allreduce_case("allreduce_sum_to forced DMA", count, 1, 122u,
                                       ec, dma_events, true);
    }
    for (int tokens : {1, 4, 5, 8, 9, 10, 20, 40, 64, 65}) {
        for (unsigned seed : {141u, 142u}) {
            failures += run_allreduce_case("sum + RMSNorm", 5120, tokens, seed, ec, events, true, true);
            const ops::PeerEvents dma_events(ec, false, ops::PeerEvents::HostPackets::Disabled);
            failures += run_allreduce_case("sum + RMSNorm DMA", 5120, tokens, seed, ec, dma_events, true, true);
        }
    }
    failures += run_allreduce_case("allreduce_sum [5120]", 5120, 1, 101u, ec, events);
    failures += run_allreduce_case("allreduce_sum [5120,4]", 5120, 4, 111u, ec, events);
    failures += run_allreduce_case("allreduce_sum [40959]", 40959, 1, 112u, ec, events);
    failures += run_allreduce_case("allreduce_sum [40960]", 40960, 1, 113u, ec, events);
    failures += run_allreduce_case("allreduce_sum [40961]", 40961, 1, 114u, ec, events);
    // The real row-parallel residual: a full 48-token prefill chunk, 2-D.
    failures += run_allreduce_case("allreduce_sum [5120,48]", 5120, 48, 102u, ec, events);
    failures += run_allreduce_case("allreduce_sum [4097]", 4097, 1, 103u, ec, events);
    failures += run_allreduce_case("allreduce_sum [1]", 1, 1, 104u, ec, events);
    failures += run_allreduce_case("allreduce_sum [3]", 3, 1, 105u, ec, events);
    failures += run_allreduce_case("allreduce_sum [17,3]", 17, 3, 106u, ec, events);

    failures += run_allgather_case("allgather_rows [5120,1024]", 512, 512, 5120, 201u, ec, events);
    // Gathered logits: 248320 vocabulary rows for one token, split by vocabulary half.
    failures += run_allgather_case("allgather_rows [1,248320]", 124160, 124160, 1, 202u, ec, events);
    failures += run_allgather_case("allgather_rows [5120,3] uneven", 2, 1, 5120, 203u, ec, events);
    failures += run_allgather_case("allgather_rows [7,2] minimal", 1, 1, 7, 204u, ec, events);
    failures += run_gather_columns_rank0_case("gather_columns_rank0 [248320,3]", 124160,
                                              124160, 3, 205u, ec, events);
    failures += run_gather_columns_rank0_case("gather_columns_rank0 [11,7] uneven", 4, 7, 7,
                                              206u, ec, events);

    failures += run_chained_case(ec, events);
    failures += run_chained_case(ec, events, true);
    failures += run_chained_case(ec, events, true, true);
    failures += run_varying_size_chain(ec, events);
    failures += run_special_sum_inputs(ec, events);
    failures += run_microbenchmark(ec, events);

    std::cout << (failures ? "FAIL" : "OK") << " allreduce\n";
    return failures ? 1 : 0;
}
