// SM70 tensor-core route of the plain Q4/Q5 Linear at the registered Vision geometries and wide T
// (ops/linear/q{4,5}/*_cutlass_sm70), reached through public linear() with a caller workspace.
//
// Each case runs one random problem through both public call forms and checks, under the A16
// Linear criterion: the workspace form against an independent FP64 oracle, the workspace-free form
// against the same oracle, and the two forms against each other over the whole output. It also
// checks the reported workspace capacity, output guards, unchanged activation and weight bytes,
// and CUDA Graph replay.

#include "ninfer/ops/linear.h"
#include "ops/input_projection_test_common.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/q4/q4_cutlass_sm70.h"
#include "ops/linear/q5/q5_cutlass_sm70.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#ifdef NINFER_VOLTA_BUILD
namespace {

using namespace ninfer;
using namespace ninfer::test;
namespace fixture = ninfer::test::input_projection;
namespace qw      = ninfer::test::quantized_weight;

// The A16 Linear criterion (tests/ops/linear/linear_test_common.cpp, tolerance_for): one BF16
// unit roundoff of relative L2, and a gross bound of one roundoff plus two of the largest
// reference magnitude.
constexpr ReductionCriterion kA16Criterion{1.0 / 256, 1.0 / 256, 2.0 / 256};

struct Case {
    QType qtype;
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
};

std::int32_t chunk_columns(QType qtype) {
    return qtype == QType::Q4G64_F16S ? ops::detail::kQ4CutlassSm70ChunkCols
                                      : ops::detail::kQ5CutlassSm70ChunkCols;
}

std::string label_of(const Case& c, const char* what) {
    return std::string(c.qtype == QType::Q4G64_F16S ? "Q4" : "Q5") + " CUTLASS Sm70 [" +
           std::to_string(c.n) + "," + std::to_string(c.k) + "] T=" + std::to_string(c.t) + " " +
           what;
}

fixture::DevicePackedWeight make_weight(const Case& c, std::uint32_t seed) {
    std::vector<float> source(static_cast<std::size_t>(c.n) * c.k);
    fill_uniform(source, seed, -0.05F, 0.05F);
    return fixture::DevicePackedWeight(qw::pack_row_split_lowbit(source, c.n, c.k, c.qtype));
}

std::vector<float> make_activation(const Case& c, std::uint32_t seed) {
    std::vector<float> values(static_cast<std::size_t>(c.k) * c.t);
    fill_uniform(values, seed, -2.0F, 2.0F);
    round_to_bf16(values);
    return values;
}

// Evenly spread columns plus both sides of every column-chunk seam the route introduces.
std::vector<std::int32_t> oracle_columns(const Case& c) {
    std::vector<std::int32_t> columns = fixture::sampled_rows(c.t, 33);
    const std::int32_t chunk          = chunk_columns(c.qtype);
    for (std::int32_t seam = chunk; seam < c.t; seam += chunk) {
        columns.push_back(seam - 1);
        columns.push_back(seam);
    }
    std::sort(columns.begin(), columns.end());
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
    return columns;
}

struct Samples {
    std::vector<std::int32_t> rows;
    std::vector<std::int32_t> columns;
    std::vector<double> oracle;
};

Samples make_oracle(const Case& c, const qw::PackedWeight& weight,
                    const std::vector<float>& activation) {
    Samples out;
    out.rows    = fixture::sampled_rows(c.n, 17);
    out.columns = oracle_columns(c);
    const std::vector<float> rows_fp32 = qw::materialize_rows_fp32(weight, out.rows);
    out.oracle.reserve(out.rows.size() * out.columns.size());
    for (const std::int32_t column : out.columns) {
        const float* x = activation.data() + static_cast<std::size_t>(column) * c.k;
        for (std::size_t row = 0; row < out.rows.size(); ++row) {
            const float* w = rows_fp32.data() + row * static_cast<std::size_t>(c.k);
            double sum     = 0.0;
            for (std::int32_t inner = 0; inner < c.k; ++inner) {
                sum += static_cast<double>(w[inner]) * static_cast<double>(x[inner]);
            }
            out.oracle.push_back(sum);
        }
    }
    return out;
}

std::vector<double> gather(const std::vector<double>& full, std::int32_t n,
                           const Samples& samples) {
    std::vector<double> out;
    out.reserve(samples.oracle.size());
    for (const std::int32_t column : samples.columns) {
        for (const std::int32_t row : samples.rows) {
            out.push_back(full[static_cast<std::size_t>(column) * n + row]);
        }
    }
    return out;
}

// The guarded outputs are poisoned with cudaMemset, which is issued on the default stream and is
// asynchronous with respect to the host. The launches below go to a non-blocking stream (graph
// capture needs one), which does not order itself after the default stream, so without this the
// poison fill can land on top of the first kernel's output -- at these sizes, on a GPU that is
// busy with another process, it does.
void retire_fixture_fills() { cuda_synchronize(); }

int run_case(const Case& c, std::uint32_t seed, cudaStream_t stream) {
    int failures = 0;
    const fixture::DevicePackedWeight weight = make_weight(c, seed);
    const Weight w                           = weight.view();
    const std::vector<float> activation      = make_activation(c, seed + 1U);
    DeviceBuffer input                       = to_device_bf16(activation);
    const Tensor x(input.p, DType::BF16, {c.k, c.t});

    const std::size_t capacity = ops::linear_workspace_capacity_bytes(
        c.qtype, c.n, c.k, ops::LinearPolicy::A16Only, c.t, c.t);
    if (capacity == 0) {
        std::cerr << label_of(c, "workspace") << ": Linear reports no workspace requirement\n";
        return 1;
    }
    DeviceArena workspace(capacity);

    fixture::GuardedBf16Tensor workspace_free(c.n, c.t), actual(c.n, c.t);
    Tensor reference = workspace_free.tensor(), output = actual.tensor();
    retire_fixture_fills();
    ops::linear(x, w, reference, stream);
    ops::linear(x, w, output, ops::LinearPolicy::A16Only, workspace, stream);
    cuda_synchronize(stream);

    // The workspace is call-scoped and bounded by the reported capacity; a zero peak would mean
    // the workspace form never reached the route under test.
    if (workspace.used() != 0 || workspace.peak_used() == 0 || workspace.peak_used() > capacity) {
        std::cerr << label_of(c, "workspace") << ": used=" << workspace.used()
                  << " peak=" << workspace.peak_used() << " capacity=" << capacity << '\n';
        ++failures;
    }

    failures += actual.verify_guards(label_of(c, "guards"));
    failures += actual.verify_fully_written(label_of(c, "coverage"));
    failures += fixture::verify_preserved(label_of(c, "activation"), input,
                                          fixture::bf16_bits(activation));
    failures += weight.verify_preserved(label_of(c, "weight"));

    const Samples samples             = make_oracle(c, weight.host, activation);
    const std::vector<double> values  = actual.values();
    const std::vector<double> control = workspace_free.values();
    failures += fixture::compare(label_of(c, "vs FP64"), gather(values, c.n, samples),
                                 samples.oracle, kA16Criterion);
    failures += fixture::compare(label_of(c, "workspace-free vs FP64"),
                                 gather(control, c.n, samples), samples.oracle, kA16Criterion);
    failures += fixture::compare(label_of(c, "vs workspace-free"), values, control, kA16Criterion);

    // Replaying a captured graph must reproduce the eager result exactly: the route keeps no
    // host-side state between its dequant, cast and GEMM launches.
    const std::vector<std::uint16_t> eager_bits = actual.bits();
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    ops::linear(x, w, output, ops::LinearPolicy::A16Only, workspace, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "capture end");
    cuda_check(cudaGraphInstantiate(&executable, graph, 0), "instantiate");
    for (int repeat = 0; repeat < 2; ++repeat) {
        cuda_check(cudaGraphLaunch(executable, stream), "replay");
        cuda_synchronize(stream);
        if (actual.bits() != eager_bits) {
            std::cerr << label_of(c, "graph replay") << ": output differs from eager launch\n";
            ++failures;
        }
    }
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);

    std::cout << label_of(c, "failures=") << failures << '\n';
    return failures;
}

int run_all() {
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");

    // Vision qkv, mlp/fc1, attention/output and mlp/fc2, each at the routed band's lower edge,
    // one whole column chunk, four columns past the chunk seam, and two whole chunks.
    struct Shape {
        QType qtype;
        std::int32_t n;
        std::int32_t k;
        std::int32_t min_t;
    };
    constexpr Shape kShapes[] = {
        {QType::Q4G64_F16S, 3456, 1152, ops::detail::kQ4CutlassSm70MinT},
        {QType::Q4G64_F16S, 4304, 1152, ops::detail::kQ4CutlassSm70MinT},
        {QType::Q5G64_F16S, 1152, 1152, ops::detail::kQ5CutlassSm70MinT},
        {QType::Q5G64_F16S, 1152, 4304, ops::detail::kQ5CutlassSm70MinT},
    };

    int failures       = 0;
    std::uint32_t seed = 7001U;
    for (const Shape& shape : kShapes) {
        const std::int32_t chunk = chunk_columns(shape.qtype);
        for (const std::int32_t t : {shape.min_t, chunk, chunk + 4, 2 * chunk}) {
            failures += run_case({shape.qtype, shape.n, shape.k, t}, seed, stream);
            seed += 2U;
        }
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
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_all();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4/Q5 CUTLASS Sm70 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q4/Q5 CUTLASS Sm70 Linear: " << error.what() << '\n';
        return 1;
    }
#endif
}
