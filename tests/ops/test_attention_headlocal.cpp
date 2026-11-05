// Two/four-device head-local parity suite for the full-attention core under TP2/TP4
// tensor parallelism. Sibling of test_linear_split.cpp / test_attn_input_proj_split.cpp;
// read test_linear_split.cpp's header comment first, this one documents what differs.
//
// Attention is separable over KV heads. The 24-query/4-KV parent is partitioned
// into 12-query/2-KV TP2 shards or 6-query/1-KV TP4 shards, retaining group size six.
// Each rank owns contiguous global head ranges; every kernel/cache index is local.
// A per-head comparison against the unsplit Op catches an incorrect shard mapping.
// Short fixtures also reject head permutations explicitly.
//
// Single-request cases additionally evaluate the complete decoded-input FP64
// attention oracle at every head/key, including the 85K occupied-context cases.
// Both geometry parity and the oracle retain the registered BF16/INT8 criteria
// from gqa_attention_fixture.h. Reduction partitions differ by geometry, so
// geometry parity is numerical rather than bit-exact. Exact cache checks protect
// appends, masked columns, fragmentation and graph replay state transitions.
//
// Each target requires its configured two or four devices in one process;
// fewer visible devices return skip code 77.
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/gqa_attention_fixture.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

using namespace ninfer::test::gqa;

#ifndef NINFER_ATTENTION_TP
#define NINFER_ATTENTION_TP 2
#endif
constexpr int kRanks = NINFER_ATTENTION_TP;
static_assert(kRanks == 2 || kRanks == 4);

// Registered global geometry and this executable's head-local shard.
constexpr Geometry kGlobalGeometry{"qwen3_6_27b", 24, 4};
constexpr Geometry kLocalGeometry{kRanks == 4 ? "qwen3_6_27b_tp4" : "qwen3_6_27b_tp2",
                                   24 / kRanks, 4 / kRanks};

static_assert(kGlobalGeometry.q_heads == kRanks * kLocalGeometry.q_heads);
static_assert(kGlobalGeometry.kv_heads == kRanks * kLocalGeometry.kv_heads);

std::int32_t local_q_head_begin(int rank) { return rank * kLocalGeometry.q_heads; }
std::int32_t local_kv_head_begin(int rank) { return rank * kLocalGeometry.kv_heads; }

void set_device(int device) { cuda_check(cudaSetDevice(device), "cudaSetDevice"); }

struct CurrentDeviceScope {
    int previous = 0;
    explicit CurrentDeviceScope(int device) {
        cuda_check(cudaGetDevice(&previous), "cudaGetDevice");
        set_device(device);
    }
    CurrentDeviceScope(const CurrentDeviceScope&)            = delete;
    CurrentDeviceScope& operator=(const CurrentDeviceScope&) = delete;
    ~CurrentDeviceScope() { (void)cudaSetDevice(previous); }
};

// --- global -> local slicing -------------------------------------------------------------
//
// Both slices re-index the same represented logical data. The caller applies
// global-to-local numbering once, as the model's sharded input projection does.

// q/k/v are token-major: index = d + 256*(head + heads*token). One device's block is a per-token
// gather of a contiguous head window.
std::vector<float> slice_heads_token_major(const std::vector<float>& source,
                                           std::int32_t source_heads, std::int32_t head_begin,
                                           std::int32_t head_count, std::int32_t tokens) {
    std::vector<float> result(static_cast<std::size_t>(kHeadDim) * head_count * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t head = 0; head < head_count; ++head) {
            const std::size_t src = static_cast<std::size_t>(kHeadDim) *
                                    (static_cast<std::size_t>(head_begin + head) +
                                     static_cast<std::size_t>(source_heads) * token);
            const std::size_t dst =
                static_cast<std::size_t>(kHeadDim) *
                (static_cast<std::size_t>(head) + static_cast<std::size_t>(head_count) * token);
            std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(src), kHeadDim,
                        result.begin() + static_cast<std::ptrdiff_t>(dst));
        }
    }
    return result;
}

// Cache planes are head-major: index = leading + leading_extent*(position + capacity*head). One
// device's pool is therefore a CONTIGUOUS slab of the global pool's logical image -- no
// re-indexing at all, which is what makes "each rank holds only its own contiguous head range" an exact
// statement rather than an approximation.
template <typename T>
std::vector<T> slice_head_major(const std::vector<T>& source, std::int32_t leading_extent,
                                std::int32_t capacity, std::int32_t head_begin,
                                std::int32_t head_count) {
    const std::size_t slab = static_cast<std::size_t>(leading_extent) * capacity;
    std::vector<T> result(slab * head_count);
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(slab * head_begin), result.size(),
                result.begin());
    return result;
}

HostCache slice_cache_heads(const HostCache& source, int rank) {
    const std::int32_t begin = local_kv_head_begin(rank);
    const std::int32_t count = kLocalGeometry.kv_heads;
    HostCache result{kLocalGeometry, source.dtype, source.max_context, source.logical_capacity};
    if (source.dtype == DType::BF16) {
        result.k_bf16 =
            slice_head_major(source.k_bf16, kHeadDim, source.logical_capacity, begin, count);
        result.v_bf16 =
            slice_head_major(source.v_bf16, kHeadDim, source.logical_capacity, begin, count);
        return result;
    }
    result.k_i8  = slice_head_major(source.k_i8, kHeadDim, source.logical_capacity, begin, count);
    result.v_i8  = slice_head_major(source.v_i8, kHeadDim, source.logical_capacity, begin, count);
    result.k_scale =
        slice_head_major(source.k_scale, kQuantGroups, source.logical_capacity, begin, count);
    result.v_scale =
        slice_head_major(source.v_scale, kQuantGroups, source.logical_capacity, begin, count);
    return result;
}

// --- per-head comparison ------------------------------------------------------------------

std::vector<double> head_block(const std::vector<double>& values, const Geometry& geometry,
                               std::int32_t head, std::int32_t tokens, std::int32_t columns) {
    std::vector<double> block(static_cast<std::size_t>(kHeadDim) * tokens * columns);
    std::size_t cursor = 0;
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t token = 0; token < tokens; ++token) {
            const std::size_t base =
                q_index(geometry, head, 0, token + column * tokens);
            for (std::int32_t d = 0; d < kHeadDim; ++d) { block[cursor++] = values[base + d]; }
        }
    }
    return block;
}

double maximum_absolute_difference(const std::vector<double>& a, const std::vector<double>& b) {
    double maximum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { maximum = std::max(maximum, std::abs(a[i] - b[i])); }
    return maximum;
}

double maximum_absolute(const std::vector<double>& values) {
    double maximum = 0.0;
    for (const double value : values) { maximum = std::max(maximum, std::abs(value)); }
    return maximum;
}

// A head permutation must not be able to hide inside the tolerance.
//
// Two global heads are only DISTINGUISHABLE on a given input if the reference itself separates
// them by much more than the criterion allows; that separation is a property of the data, not of
// the kernel. It fails legitimately in exactly one place: when a single key is visible, softmax
// over one key is 1 and every query head in a KV group returns that head's V verbatim, so the six
// heads of a group are mathematically identical. The check therefore gates on the reference's own
// separation, and asserts SEPARATELY that heads belonging to DIFFERENT KV heads are always
// separated -- that leg is the KV-head-mapping check and no input may make it vacuous.
constexpr double kPermutationMargin = 20.0;

int verify_head_mapping(const std::string& label, const std::vector<double>& observed,
                        const std::vector<double>& reference, const Geometry& reference_geometry,
                        std::int32_t correct_head, const ReductionCriterion& criterion,
                        std::int32_t tokens, std::int32_t columns) {
    int failures = 0;
    const std::vector<double> correct =
        head_block(reference, reference_geometry, correct_head, tokens, columns);
    const std::int32_t group = reference_geometry.query_group();
    for (std::int32_t head = 0; head < reference_geometry.q_heads; ++head) {
        if (head == correct_head) { continue; }
        const std::vector<double> other =
            head_block(reference, reference_geometry, head, tokens, columns);
        const double limit =
            std::max(criterion.gross_absolute,
                     criterion.gross_relative_to_max_reference * maximum_absolute(other));
        const double reference_separation = maximum_absolute_difference(correct, other);
        const bool different_kv_head      = (head / group) != (correct_head / group);
        if (reference_separation <= kPermutationMargin * limit) {
            if (different_kv_head) {
                std::cerr << label << ": reference heads " << correct_head << " and " << head
                          << " read DIFFERENT KV heads yet are indistinguishable (separation="
                          << reference_separation << "), so KV-head mapping is untested here\n";
                ++failures;
            }
            continue; // same KV head, too few visible keys to separate query heads: not a defect
        }
        // The observed block must stay far closer to its own reference head than to this one.
        const double distance = maximum_absolute_difference(observed, other);
        if (!(distance > 0.5 * reference_separation)) {
            std::cerr << label << ": head block is closer to global head " << head
                      << " than a correct mapping allows (max_abs_diff=" << distance
                      << " needs > " << 0.5 * reference_separation << ")\n";
            ++failures;
        }
    }
    return failures;
}

// --- device-side run ----------------------------------------------------------------------

struct RunResult {
    std::vector<std::uint16_t> output;
    HostCache cache_after;
    int failures = 0;
};

// One A1 (append + attend) invocation on `device` with `geometry`-shaped inputs.
RunResult run_a1(int device, const Geometry& geometry, DType dtype, const HostCache& initial,
                 MappingPattern mapping, const std::vector<float>& q, const std::vector<float>& k,
                 const std::vector<float>& v, const std::vector<std::int32_t>& positions,
                 ops::GqaExecutionEnvelope envelope, const std::string& label,
                 std::int32_t valid_tokens = -1) {
    const CurrentDeviceScope scope(device);
    const auto tokens = static_cast<std::int32_t>(positions.size());
    DeviceCache cache(initial, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    const std::vector<std::uint16_t> canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dp.data(), DType::I32, {tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});

    // MTP's single-row proposal carries an explicit valid-column mask even at B=1.
    GuardedDeviceBuffer dvalid(sizeof(std::int32_t));
    if (valid_tokens < 0) valid_tokens = tokens;
    dvalid.copy_from_host(&valid_tokens, sizeof(valid_tokens));
    Tensor tvalid(dvalid.data(), DType::I32, {1});

    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, tokens, tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp,
                       envelope.volta_long_attention || valid_tokens != tokens ? tvalid : Tensor{},
                       ttable_row, kAttentionScale, cache.batch_view(),
                       envelope, workspace, tout, nullptr);
    cuda_synchronize();

    RunResult result;
    result.output      = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    result.cache_after = cache.snapshot();
    result.failures += verify_input(label + " q unchanged", dq, q_bits);
    result.failures += verify_input(label + " k unchanged", dk, k_bits);
    result.failures += verify_input(label + " v unchanged", dv, v_bits);
    result.failures += verify_positions(label + " positions unchanged", dp, positions);
    result.failures += dout.verify_guards((label + " output").c_str());
    result.failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++result.failures;
    }
    result.failures += cache.verify_guards(label);
    return result;
}

// One A3 (attend over an already populated pool, no cache write) invocation.
RunResult run_a3(int device, const Geometry& geometry, DType dtype, const HostCache& initial,
                 MappingPattern mapping, const std::vector<float>& q,
                 const std::vector<std::int32_t>& positions, ops::GqaExecutionEnvelope envelope,
                 const std::string& label) {
    const CurrentDeviceScope scope(device);
    const auto tokens = static_cast<std::int32_t>(positions.size());
    DeviceCache cache(initial, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::vector<std::uint16_t> canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});
    Tensor tp(dp.data(), DType::I32, {tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, tokens});

    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, tokens, tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                              nullptr);
    cuda_synchronize();

    RunResult result;
    result.output      = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    result.cache_after = cache.snapshot();
    result.failures += verify_input(label + " q unchanged", dq, q_bits);
    result.failures += verify_positions(label + " positions unchanged", dp, positions);
    result.failures += dout.verify_guards((label + " output").c_str());
    result.failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    result.failures += cache.verify_guards(label);
    return result;
}

// --- cases ---------------------------------------------------------------------------------

struct HeadLocalCase {
    const char* entry; // "gqa_attention" (A1) or "gqa_attention_cached" (A3)
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    MappingPattern mapping;
    std::uint32_t seed;
    bool check_head_mapping = true;
    std::int32_t valid_tokens = -1;
};

std::string case_label(const HeadLocalCase& test_case, DType dtype) {
    return std::string(test_case.entry) + " " + cache_name(dtype) +
           " mapping=" + mapping_name(test_case.mapping) +
           " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max);
}

// Re-indexes a global 24-head tensor into the 12-head layout of one device: the [D,H,W,B]
// element for global head 12r+h becomes the element for local head h.
std::vector<double> slice_output_heads(const std::vector<double>& global_value, int rank,
                                       std::int32_t tokens, std::int32_t columns) {
    std::vector<double> result(static_cast<std::size_t>(kHeadDim) * kLocalGeometry.q_heads * tokens *
                               columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t token = 0; token < tokens; ++token) {
            const std::int32_t flat = token + column * tokens;
            for (std::int32_t head = 0; head < kLocalGeometry.q_heads; ++head) {
                const std::size_t src =
                    q_index(kGlobalGeometry, local_q_head_begin(rank) + head, 0, flat);
                const std::size_t dst = q_index(kLocalGeometry, head, 0, flat);
                std::copy_n(global_value.begin() + static_cast<std::ptrdiff_t>(src), kHeadDim,
                            result.begin() + static_cast<std::ptrdiff_t>(dst));
            }
        }
    }
    return result;
}

// Two independently BF16-rounded evaluations of the same quantity are each within the registered
// criterion of the FP64 oracle (leg A below for the split side; the GQA conformance suite for the
// tp1 side), so by the triangle inequality their difference is bounded by TWICE that criterion.
// This factor is that bound, not a fitted tolerance -- and it is needed, because the split does
// not reproduce tp1's arithmetic bit for bit: the split-KV decode route rounds its per-split
// partial accumulators to BF16 and the tp2 geometry's DecodeSplitScale=2 changes the number of
// splits, hence the partition of the reduction.
constexpr double kParityBound = 2.0;

ReductionCriterion scaled(const ReductionCriterion& criterion, double factor) {
    return ReductionCriterion{criterion.relative_l2 * factor, criterion.gross_absolute * factor,
                              criterion.gross_relative_to_max_reference * factor};
}

// Compares one device's 12-head output against (a) the FP64 ideal oracle for its heads at the
// Op's own registered criterion and the granularity that criterion was measured at, (b) the tp1
// 24-head output sliced to the same global heads at the triangle bound, (c) the same, PER HEAD,
// so a head-mapping error is localized rather than averaged away. Fixtures with distinguishable
// reference heads additionally prove that a head permutation could not have passed (c).
int compare_device(const std::string& label, int rank, const std::vector<std::uint16_t>& local_bits,
                   const std::vector<double>& global_reference,
                   const std::vector<double>* oracle, const ReductionCriterion& criterion,
                   std::int32_t tokens, std::int32_t columns, bool check_head_mapping = true) {
    int failures                          = 0;
    const std::vector<double> local_value = bf16_bits_to_double(local_bits);
    const std::string rank_label          = label + " rank=" + std::to_string(rank);
    const ReductionCriterion parity       = scaled(criterion, kParityBound);

    if (oracle != nullptr) {
        // Leg A: the registered numerical contract, applied unmodified to the new geometry.
        failures += verify_attention(rank_label + " vs oracle", local_value,
                                     slice_output_heads(*oracle, rank, tokens, columns), criterion);
    }
    const std::vector<double> reference_slice =
        slice_output_heads(global_reference, rank, tokens, columns);
    failures += verify_attention(rank_label + " vs tp1", local_value, reference_slice, parity);

    for (std::int32_t head = 0; head < kLocalGeometry.q_heads; ++head) {
        const std::int32_t global_head = local_q_head_begin(rank) + head;
        const std::string head_label    = rank_label + " local_head=" + std::to_string(head) +
                                       " global_head=" + std::to_string(global_head);
        const std::vector<double> observed =
            head_block(local_value, kLocalGeometry, head, tokens, columns);
        failures += verify_attention(
            head_label + " vs tp1",
            observed, head_block(global_reference, kGlobalGeometry, global_head, tokens, columns),
            parity);
        if (check_head_mapping) {
            failures += verify_head_mapping(head_label, observed, global_reference, kGlobalGeometry,
                                            global_head, criterion, tokens, columns);
        }
    }
    return failures;
}

// Full-width prefill runs use the same independent FP64 formula on a bounded set of query
// columns. Every selected query still attends over ALL of its visible keys, all 256 dimensions,
// and every head. GPU execution, output parity, and exact cache validation remain full-width.
// Include both sides of tile/page boundaries and the final tiles, where stream-K fixups differ
// from the existing T=66 fixture. This keeps the CPU oracle linear in the number of keys rather
// than evaluating an entire 1024-by-1024 attention matrix.
std::vector<std::int32_t> prefill_oracle_columns(std::int32_t tokens) {
    std::vector<std::int32_t> columns{0, 1, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256,
                                      tokens / 2 - 1, tokens / 2, tokens - 17, tokens - 16,
                                      tokens - 2, tokens - 1};
    std::sort(columns.begin(), columns.end());
    std::erase_if(columns, [tokens](auto column) { return column < 0 || column >= tokens; });
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
    return columns;
}

template <typename Value>
std::vector<Value> select_query_columns(const std::vector<Value>& source, std::int32_t heads,
                                         const std::vector<std::int32_t>& columns) {
    const std::size_t column_elements = static_cast<std::size_t>(kHeadDim) * heads;
    std::vector<Value> selected;
    selected.reserve(columns.size() * column_elements);
    for (const auto column : columns) {
        const auto begin = source.begin() + static_cast<std::ptrdiff_t>(column * column_elements);
        selected.insert(selected.end(), begin, begin + static_cast<std::ptrdiff_t>(column_elements));
    }
    return selected;
}

enum class ReferenceRoute { GlobalGpuAndOracle, OracleOnly };

int run_case(DType dtype, const HeadLocalCase& test_case, bool sampled_prefill_oracle = false,
             bool volta_long_attention = false,
             ReferenceRoute reference_route = ReferenceRoute::GlobalGpuAndOracle) {
    const bool append              = std::string(test_case.entry) == "gqa_attention";
    const std::int32_t tokens      = test_case.tokens;
    const std::int32_t valid_tokens = test_case.valid_tokens < 0 ? tokens : test_case.valid_tokens;
    const std::int32_t total       = test_case.base + tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::string label = case_label(test_case, dtype);

    const std::size_t q_elements =
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.q_heads * tokens;
    const std::size_t kv_elements =
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.kv_heads * tokens;
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(kGlobalGeometry, tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] =
            test_case.base + (token < valid_tokens ? token : std::max(0, valid_tokens - 1));
    }

    const HostCache initial = make_cache(kGlobalGeometry, dtype, max_context, test_case.seed + 10u);
    HostCache expected      = initial;
    if (append) {
        append_cache(expected, k, v,
                     std::vector<std::int32_t>(positions.begin(), positions.begin() + valid_tokens));
    }
    const auto oracle_columns = sampled_prefill_oracle ? prefill_oracle_columns(tokens)
                                                        : std::vector<std::int32_t>{};
    std::vector<std::int32_t> oracle_positions;
    for (const auto column : oracle_columns) { oracle_positions.push_back(positions[column]); }
    std::vector<double> oracle =
        sampled_prefill_oracle
            ? ideal_attention(select_query_columns(q, kGlobalGeometry.q_heads, oracle_columns),
                              expected, oracle_positions)
            : ideal_attention(q, expected, positions);
    if (!sampled_prefill_oracle) {
        const auto column_size = static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.q_heads;
        std::fill(oracle.begin() + valid_tokens * column_size, oracle.end(), 0.0);
    }
    if (sampled_prefill_oracle) {
        std::cout << label << ": full-width GPU/cache checks, FP64 oracle on "
                  << oracle_columns.size() << " query columns x 24 heads\n" << std::flush;
    }

    const ops::GqaExecutionEnvelope envelope{
                                             static_cast<std::uint32_t>(test_case.base +
                                                                       std::max(1, valid_tokens)),
                                             test_case.envelope_max, volta_long_attention};
    const ReductionCriterion criterion = attention_criterion(dtype);

    int failures = 0;

    // (1) The optional single-device GPU comparison. It is supplementary to the
    // mathematical oracle: the 85K prefill fixture qualifies TP4 directly, since
    // the unrelated TP1 flash reduction fails this fixture's existing criterion.
    std::vector<double> reference_value;
    std::vector<double> oracle_reference;
    if (reference_route == ReferenceRoute::GlobalGpuAndOracle) {
        const RunResult reference =
            append ? run_a1(0, kGlobalGeometry, dtype, initial, test_case.mapping, q, k, v, positions,
                            envelope, label + " tp1", valid_tokens)
                   : run_a3(0, kGlobalGeometry, dtype, initial, test_case.mapping, q, positions,
                            envelope, label + " tp1");
        failures += reference.failures;
        failures += verify_cache(label + " tp1 cache", reference.cache_after, expected);
        reference_value = bf16_bits_to_double(reference.output);
        oracle_reference =
            sampled_prefill_oracle
                ? select_query_columns(reference_value, kGlobalGeometry.q_heads, oracle_columns)
                : reference_value;
        failures += verify_attention(label + " tp1 vs oracle", oracle_reference, oracle, criterion);
    }

    // (2) Head-local runs, each on its own device and disjoint KV-head pool.
    std::array<HostCache, kRanks> local_initial;
    for (int rank = 0; rank < kRanks; ++rank) {
        local_initial[rank] = slice_cache_heads(initial, rank);
    }
    // Distinctness: if the two devices' pools carried the same bytes, every head-mapping check
    // below would be vacuous. make_cache generates per-head-distinct data; assert it.
    if (dtype == DType::BF16) {
        if (local_initial[0].k_bf16 == local_initial[1].k_bf16 ||
            local_initial[0].v_bf16 == local_initial[1].v_bf16) {
            std::cerr << label << ": the two devices' KV pools are byte-identical, so head "
                                 "mapping is untested\n";
            ++failures;
        }
    } else if (local_initial[0].k_i8 == local_initial[1].k_i8 ||
               local_initial[0].v_i8 == local_initial[1].v_i8) {
        std::cerr << label << ": the two devices' KV pools are byte-identical, so head mapping "
                             "is untested\n";
        ++failures;
    }

    for (int rank = 0; rank < kRanks; ++rank) {
        const std::string rank_label = label + " tp" + std::to_string(kRanks) + " rank=" + std::to_string(rank);
        const std::vector<float> local_q = slice_heads_token_major(
            q, kGlobalGeometry.q_heads, local_q_head_begin(rank), kLocalGeometry.q_heads, tokens);
        const std::vector<float> local_k =
            slice_heads_token_major(k, kGlobalGeometry.kv_heads, local_kv_head_begin(rank),
                                    kLocalGeometry.kv_heads, tokens);
        const std::vector<float> local_v =
            slice_heads_token_major(v, kGlobalGeometry.kv_heads, local_kv_head_begin(rank),
                                    kLocalGeometry.kv_heads, tokens);

        const RunResult observed =
            append ? run_a1(rank, kLocalGeometry, dtype, local_initial[rank], test_case.mapping,
                            local_q, local_k, local_v, positions, envelope, rank_label, valid_tokens)
                   : run_a3(rank, kLocalGeometry, dtype, local_initial[rank], test_case.mapping,
                            local_q, positions, envelope, rank_label);
        failures += observed.failures;
        // The device wrote its new K/V into the LOCAL head rows of its own pool: the expected
        // pool is the same head slice of the tp1 expected pool.
        failures += verify_cache(rank_label + " cache", observed.cache_after,
                                 slice_cache_heads(expected, rank));
        if (reference_route == ReferenceRoute::OracleOnly) {
            const auto local_value = bf16_bits_to_double(observed.output);
            const auto selected = sampled_prefill_oracle
                                      ? select_query_columns(local_value, kLocalGeometry.q_heads,
                                                             oracle_columns)
                                      : local_value;
            const auto selected_tokens = sampled_prefill_oracle
                                             ? static_cast<std::int32_t>(oracle_columns.size())
                                             : tokens;
            const auto local_oracle = slice_output_heads(oracle, rank, selected_tokens, 1);
            failures += verify_attention(rank_label + " vs FP64 oracle", selected, local_oracle,
                                         criterion);
            for (std::int32_t head = 0; head < kLocalGeometry.q_heads; ++head) {
                failures += verify_attention(
                    rank_label + " head=" + std::to_string(head) + " vs FP64 oracle",
                    head_block(selected, kLocalGeometry, head, selected_tokens, 1),
                    head_block(local_oracle, kLocalGeometry, head, selected_tokens, 1), criterion);
            }
        } else if (sampled_prefill_oracle) {
            failures += verify_attention(
                rank_label + " full output vs tp1", bf16_bits_to_double(observed.output),
                slice_output_heads(reference_value, rank, tokens, 1),
                scaled(criterion, kParityBound));
            failures += compare_device(
                label + " sampled query columns", rank,
                select_query_columns(observed.output, kLocalGeometry.q_heads, oracle_columns),
                oracle_reference, &oracle, criterion,
                static_cast<std::int32_t>(oracle_columns.size()), 1, test_case.check_head_mapping);
        } else {
            failures += compare_device(label, rank, observed.output, reference_value, &oracle,
                                       criterion, tokens, 1, test_case.check_head_mapping);
        }
    }
    return failures;
}

int run_long_split_graph_boundary(int device, int tokens, bool long_attention = true) {
    const CurrentDeviceScope device_scope(device);
    constexpr int first_window = 65536;
    constexpr int last_window = first_window + 8;
    const auto host = make_cache(kLocalGeometry, DType::I8, last_window + 8, 1701u + tokens);
    DeviceCache cache(host, MappingPattern::Fragmented);
    const auto query = make_bf16_values(kHeadDim * kLocalGeometry.q_heads * tokens,
                                       1711u + tokens, -0.25f, 0.25f);
    const auto query_bits = to_bf16_bits(query);
    GuardedDeviceBuffer dq(query_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(tokens * sizeof(std::int32_t));
    GuardedDeviceBuffer eager(query_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer replay(query_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(query_bits.data(), query_bits.size() * sizeof(std::uint16_t));
    Tensor q(dq.data(), DType::BF16, {kHeadDim, kLocalGeometry.q_heads, tokens});
    Tensor pos(dp.data(), DType::I32, {tokens});
    Tensor eager_out(eager.data(), DType::BF16,
                     {kHeadDim, kLocalGeometry.q_heads, tokens});
    Tensor replay_out(replay.data(), DType::BF16,
                      {kHeadDim, kLocalGeometry.q_heads, tokens});
    const ops::GqaExecutionEnvelope envelope{first_window, last_window, long_attention};
    const auto bytes = ops::gqa_attention_workspace_capacity_bytes(
        kLocalGeometry.q_heads, DType::I8, envelope, 1, tokens, tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(bytes, 256));
    WorkspaceArena work(DeviceSpan{scratch.data(), scratch.bytes()});
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "split graph stream");
    auto invoke = [&](Tensor& output, ops::GqaExecutionEnvelope range) {
        ops::gqa_attention_cached(q, pos, kAttentionScale, cache.view(), range, work, output, stream);
    };
    std::vector<std::int32_t> positions(tokens);
    for (int token = 0; token < tokens; ++token) positions[token] = first_window - tokens + token;
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    invoke(replay_out, envelope);
    cuda_synchronize(stream);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "split capture");
    invoke(replay_out, envelope);
    cuda_check(cudaStreamEndCapture(stream, &graph), "split capture end");
    cuda_check(cudaGraphInstantiate(&executable, graph, 0), "split graph instantiate");
    int failures = 0;
    for (int window : {first_window, last_window}) {
        for (int token = 0; token < tokens; ++token) positions[token] = window - tokens + token;
        dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
        invoke(eager_out, {static_cast<std::uint32_t>(window), static_cast<std::uint32_t>(window), long_attention});
        cuda_check(cudaGraphLaunch(executable, stream), "split graph replay");
        cuda_synchronize(stream);
        const auto observed = copy_from_guarded<std::uint16_t>(replay, query_bits.size());
        const auto expected = copy_from_guarded<std::uint16_t>(eager, query_bits.size());
        const std::string label = "TP" + std::to_string(kRanks) + " INT8 split graph T=" + std::to_string(tokens) +
                                  " keys=" + std::to_string(window);
        if (observed != expected) {
            std::cerr << label << ": eager/replay outputs differ\n";
            ++failures;
        }
        failures += verify_attention(label + " FP64", bf16_bits_to_double(observed),
                                     ideal_attention(query, host, positions), kAttentionInt8Criterion);
        failures += verify_positions(label + " positions unchanged", dp, positions);
    }
    cuda_check(cudaGraphExecDestroy(executable), "split graph destroy");
    cuda_check(cudaGraphDestroy(graph), "split capture destroy");
    cuda_check(cudaStreamDestroy(stream), "split stream destroy");
    failures += verify_input("split graph query unchanged", dq, query_bits);
    failures += verify_cache("split graph cache unchanged", cache.snapshot(), host);
    failures += cache.verify_guards("split graph cache");
    failures += eager.verify_guards("split graph eager output");
    failures += replay.verify_guards("split graph replay output");
    failures += scratch.verify_guards("split graph workspace");
    return failures;
}

// --- batched (speculative-verify) route ------------------------------------------------------
//
// B>1 is the only way a 12-head geometry reaches GqaAttentionRoute::ChunkedSmallT, and it is the
// only route that instantiates the decode kernels' MultiBatch=true / Masked=true specializations.
// Each row is also extracted for an independent full-key FP64 oracle. Small-T
// cases check changed-query Graph replay and guarded output/workspace writes.
struct BatchHeadLocalCase {
    std::int32_t width;
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    MappingPattern mapping;
    std::uint32_t seed;
    bool persistent_v_head_bias = false;
    bool long_attention = false;
};

struct BatchRunResult {
    std::vector<std::uint16_t> output;
    int failures = 0;
};

BatchRunResult run_batch(int device, const Geometry& geometry, DType dtype,
                         std::span<const HostCache> rows, const BatchHeadLocalCase& test_case,
                         const std::vector<float>& q, const std::vector<float>& k,
                         const std::vector<float>& v,
                         const std::vector<std::int32_t>& positions,
                         ops::GqaExecutionEnvelope envelope, const std::string& label) {
    const CurrentDeviceScope scope(device);
    const auto batch = static_cast<std::int32_t>(rows.size());
    BatchDeviceCache cache(rows, test_case.mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvalid(test_case.valid_columns.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer drows(test_case.table_rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dvalid.copy_from_host(test_case.valid_columns.data(),
                          test_case.valid_columns.size() * sizeof(std::int32_t));
    drows.copy_from_host(test_case.table_rows.data(),
                         test_case.table_rows.size() * sizeof(std::int32_t));
    const std::vector<std::uint16_t> canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tp(dp.data(), DType::I32, {test_case.width, batch});
    Tensor tvalid(dvalid.data(), DType::I32, {batch});
    Tensor trows(drows.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});

    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, batch, test_case.width, test_case.width);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    const bool masked = std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](std::int32_t valid) { return valid != test_case.width; });
    ops::gqa_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, trows, kAttentionScale,
                       cache.view(), envelope, workspace, tout, nullptr);
    cuda_synchronize();

    BatchRunResult result;
    result.output = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    result.failures += verify_input(label + " q unchanged", dq, q_bits);
    result.failures += verify_positions(label + " positions unchanged", dp, positions);
    result.failures += dout.verify_guards((label + " output").c_str());
    result.failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (test_case.width <= 6) {
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "batch graph stream");
        const auto invoke = [&](ops::GqaExecutionEnvelope current) {
            workspace.reset();
            ops::gqa_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, trows, kAttentionScale,
                               cache.view(), current, workspace, tout, stream);
        };
        auto replay_envelope = envelope;
        replay_envelope.min_visible_keys = 1;
        invoke(replay_envelope);
        cuda_synchronize(stream);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "batch capture");
        invoke(replay_envelope);
        cuda_check(cudaStreamEndCapture(stream, &graph), "batch capture end");
        cuda_check(cudaGraphInstantiate(&executable, graph, 0), "batch graph instantiate");
        auto changed_query = q_bits;
        for (std::size_t i = 0; i < changed_query.size(); ++i) {
            if (i % 3 != 0) changed_query[i] ^= 0x8000u;
        }
        for (const auto* query : std::array<const std::vector<std::uint16_t>*, 2>{&q_bits, &changed_query}) {
            dq.copy_from_host(query->data(), query->size() * sizeof(std::uint16_t));
            invoke(envelope);
            cuda_synchronize(stream);
            const auto eager = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
            dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));
            cuda_check(cudaGraphLaunch(executable, stream), "batch graph replay");
            cuda_synchronize(stream);
            const auto replay = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
            result.failures += verify_exact((label + " changed-query Graph/eager").c_str(),
                                             replay, eager);
        }
        cuda_check(cudaGraphExecDestroy(executable), "batch graph destroy");
        cuda_check(cudaGraphDestroy(graph), "batch capture destroy");
        cuda_check(cudaStreamDestroy(stream), "batch stream destroy");
        result.failures += verify_input(label + " changed query unchanged", dq, changed_query);
        result.failures += dout.verify_guards((label + " replay output").c_str());
        result.failures += workspace_buffer.verify_guards((label + " replay workspace").c_str());
    }
    return result;
}

int run_batch_case(DType dtype, const BatchHeadLocalCase& test_case,
                   bool compare_global_execution = true) {
    const auto batch = static_cast<std::int32_t>(test_case.contexts.size());
    std::int32_t maximum_visible = 1;
    for (std::int32_t row = 0; row < batch; ++row) {
        maximum_visible =
            std::max(maximum_visible, test_case.contexts[static_cast<std::size_t>(row)] +
                                          test_case.valid_columns[static_cast<std::size_t>(row)]);
    }
    const std::int32_t max_context = maximum_visible + 3;
    const std::size_t columns      = static_cast<std::size_t>(test_case.width) * batch;
    std::vector<float> q = make_bf16_values(
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.q_heads * columns, test_case.seed,
        -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.kv_heads * columns,
        test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.kv_heads * columns,
        test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(kGlobalGeometry, static_cast<std::int32_t>(columns), k, v);

    std::vector<std::int32_t> positions(columns, 0);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < valid; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] =
                test_case.contexts[static_cast<std::size_t>(row)] + token;
        }
        const std::int32_t padding =
            valid == 0 ? 0 : test_case.contexts[static_cast<std::size_t>(row)] + valid - 1;
        for (std::int32_t token = valid; token < test_case.width; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] = padding;
        }
    }

    std::vector<HostCache> global_rows;
    global_rows.reserve(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        global_rows.push_back(make_cache(kGlobalGeometry, dtype, max_context,
                                         test_case.seed + 20u + 3u * row));
        if (test_case.persistent_v_head_bias) {
            auto& cache = global_rows.back();
            const auto scale = f32_to_f16_bits(1.0f / 128.0f);
            std::fill(cache.v_scale.begin(), cache.v_scale.end(), scale);
            // Keep distinct KV heads distinguishable after long-history averaging.
            for (int head = 0; head < kGlobalGeometry.kv_heads; ++head) {
                const std::size_t slab = static_cast<std::size_t>(kHeadDim) * cache.logical_capacity;
                for (std::size_t index = 0; index < slab; ++index) {
                    cache.v_i8[head * slab + index] = static_cast<std::int8_t>(
                        16 + 32 * head + static_cast<int>(index % 9) - 4);
                }
            }
        }
    }

    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(maximum_visible),
                                             static_cast<std::uint32_t>(maximum_visible), test_case.long_attention};
    const ReductionCriterion criterion = attention_criterion(dtype);
    const std::string label = std::string("gqa_attention batch ") + cache_name(dtype) +
                              " mapping=" + mapping_name(test_case.mapping) +
                              " B=" + std::to_string(batch) +
                              " W=" + std::to_string(test_case.width);

    int failures = 0;
    std::vector<double> reference_value;
    if (compare_global_execution) {
        const auto reference = run_batch(0, kGlobalGeometry, dtype, global_rows, test_case, q,
                                          k, v, positions, envelope, label + " tp1");
        failures += reference.failures;
        reference_value = bf16_bits_to_double(reference.output);
    }
    std::vector<double> oracle(q.size(), 0.0);
    for (int row = 0; row < batch; ++row) {
        const int valid = test_case.valid_columns[row];
        if (valid == 0) { continue; }
        auto expected_cache = global_rows[row];
        const auto first_column = row * test_case.width;
        const auto q_stride = kHeadDim * kGlobalGeometry.q_heads;
        const auto kv_stride = kHeadDim * kGlobalGeometry.kv_heads;
        const auto take = [](const auto& values, int offset, int count) {
            return std::vector<float>(values.begin() + offset, values.begin() + offset + count);
        };
        const std::vector<std::int32_t> row_positions(positions.begin() + first_column,
                                                    positions.begin() + first_column + valid);
        append_cache(expected_cache, take(k, first_column * kv_stride, valid * kv_stride),
                     take(v, first_column * kv_stride, valid * kv_stride), row_positions);
        const auto values = ideal_attention(take(q, first_column * q_stride, valid * q_stride),
                                             expected_cache, row_positions);
        std::copy(values.begin(), values.end(), oracle.begin() + first_column * q_stride);
    }
    if (compare_global_execution) {
        failures += verify_attention(label + " batched vs FP64 oracle", reference_value, oracle, criterion);
    }

    for (int rank = 0; rank < kRanks; ++rank) {
        std::vector<HostCache> local_rows;
        local_rows.reserve(static_cast<std::size_t>(batch));
        for (std::int32_t row = 0; row < batch; ++row) {
            local_rows.push_back(slice_cache_heads(global_rows[static_cast<std::size_t>(row)],
                                                   rank));
        }
        const std::vector<float> local_q = slice_heads_token_major(
            q, kGlobalGeometry.q_heads, local_q_head_begin(rank), kLocalGeometry.q_heads,
            static_cast<std::int32_t>(columns));
        const std::vector<float> local_k = slice_heads_token_major(
            k, kGlobalGeometry.kv_heads, local_kv_head_begin(rank), kLocalGeometry.kv_heads,
            static_cast<std::int32_t>(columns));
        const std::vector<float> local_v = slice_heads_token_major(
            v, kGlobalGeometry.kv_heads, local_kv_head_begin(rank), kLocalGeometry.kv_heads,
            static_cast<std::int32_t>(columns));

        const BatchRunResult observed =
            run_batch(rank, kLocalGeometry, dtype, local_rows, test_case, local_q, local_k, local_v,
                      positions, envelope, label + " tp" + std::to_string(kRanks) + " rank=" + std::to_string(rank));
        failures += observed.failures;
        if (compare_global_execution) {
            failures += compare_device(label, rank, observed.output, reference_value, &oracle,
                                       criterion, test_case.width, batch);
        } else {
            // The focused TP4 route is judged directly against the mathematical
            // oracle. The unrelated TP1 FP16-PV implementation does not pass
            // these persistent-bias fixtures and is not a reference authority.
            failures += verify_attention(label + " rank=" + std::to_string(rank) + " vs oracle",
                bf16_bits_to_double(observed.output),
                slice_output_heads(oracle, rank, test_case.width, batch), criterion);
        }
    }
    return failures;
}

// --- standalone append (A2) into a head-local pool -------------------------------------------
//
// A2 has no Q heads at all, so it selects its geometry by KV-head count. A head-local pool has 2
// KV heads, which is also the 16|2 geometry's count -- the append kernels read Geometry::KVHeads
// and nothing else, so the two are the same operation. This case proves the LOCAL head rows are
// the ones written, and (with T>=128) reaches the KVHeads==2 paged INT8 fill kernel.
int run_append_case(DType dtype, MappingPattern mapping, std::int32_t tokens, std::int32_t base,
                    std::uint32_t seed) {
    const std::int32_t max_context = base + tokens + 4;
    const std::size_t elements =
        static_cast<std::size_t>(kHeadDim) * kGlobalGeometry.kv_heads * tokens;
    std::vector<float> k = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);
    inject_codec_edges(kGlobalGeometry, tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base + token;
    }

    const HostCache initial = make_cache(kGlobalGeometry, dtype, max_context, seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);

    const std::string label = std::string("gqa_kv_append tp") + std::to_string(kRanks) + " " + cache_name(dtype) +
                              " mapping=" + mapping_name(mapping) +
                              " T=" + std::to_string(tokens);
    int failures = 0;
    for (int rank = 0; rank < kRanks; ++rank) {
        const CurrentDeviceScope scope(rank);
        const std::string rank_label = label + " rank=" + std::to_string(rank);
        const HostCache local        = slice_cache_heads(initial, rank);
        DeviceCache cache(local, mapping);

        const std::vector<float> local_k =
            slice_heads_token_major(k, kGlobalGeometry.kv_heads, local_kv_head_begin(rank),
                                    kLocalGeometry.kv_heads, tokens);
        const std::vector<float> local_v =
            slice_heads_token_major(v, kGlobalGeometry.kv_heads, local_kv_head_begin(rank),
                                    kLocalGeometry.kv_heads, tokens);
        const std::vector<std::uint16_t> k_bits = to_bf16_bits(local_k);
        const std::vector<std::uint16_t> v_bits = to_bf16_bits(local_v);
        GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
        dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
        dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
        dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
        Tensor tk(dk.data(), DType::BF16, {kHeadDim, kLocalGeometry.kv_heads, tokens});
        Tensor tv(dv.data(), DType::BF16, {kHeadDim, kLocalGeometry.kv_heads, tokens});
        Tensor tp(dp.data(), DType::I32, {tokens});

        ops::gqa_kv_append(tk, tv, tp, cache.view(), nullptr);
        cuda_synchronize();

        failures += verify_cache(rank_label, cache.snapshot(), slice_cache_heads(expected, rank));
        failures += verify_input(rank_label + " k unchanged", dk, k_bits);
        failures += verify_input(rank_label + " v unchanged", dv, v_bits);
        failures += verify_positions(rank_label + " positions unchanged", dp, positions);
        failures += cache.verify_guards(rank_label);
    }
    return failures;
}

// --- rejections -------------------------------------------------------------------------------

int verify_rejections() {
    int failures = 0;
    // A 12-head geometry is registered and must be admitted by the public capacity query.
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(kLocalGeometry.q_heads, DType::I8,
                                                          {1, 4096}, 1, 1, 6);
    } catch (const std::invalid_argument&) {
        std::cerr << "gqa_attention rejected the registered head-local geometry\n";
        ++failures;
    }
    // Head counts that are not a registered geometry stay rejected: a shard extent is only legal
    // where it was registered; six Q heads are registered for TP4.
    for (const std::int32_t q_heads : {5, 8, 12 + 1, 20}) {
        try {
            (void)ops::gqa_attention_workspace_capacity_bytes(q_heads, DType::I8, {1, 4096}, 1, 1,
                                                              6);
            std::cerr << "gqa_attention accepted unregistered Q-head count " << q_heads << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const bool batch_decode_only = argc == 2 && std::string(argv[1]) == "--batch-decode-only";
    const bool prefill_only = argc == 2 && std::string(argv[1]) == "--prefill-only";
    const bool long_context_from = argc == 3 && std::string(argv[1]) == "--long-context-from";
    int first_long_width = 1;
    if (long_context_from) {
        const std::string width(argv[2]);
        if (width != "1" && width != "4" && width != "5" && width != "6") {
            std::cerr << "--long-context-from requires 1, 4, 5 or 6\n";
            return 2;
        }
        first_long_width = width[0] - '0';
    }
    const bool long_context_only = long_context_from ||
        (argc == 2 && std::string(argv[1]) == "--long-context-only");
    const bool default_single_only = argc == 2 && std::string(argv[1]) == "--default-single-only";
    const bool long_verify5_only = argc == 2 && std::string(argv[1]) == "--long-verify5-only";
    if (argc != 1 && !prefill_only && !long_context_only && !default_single_only &&
        !long_verify5_only && !batch_decode_only) {
        std::cerr << "usage: " << argv[0]
                  << " [--prefill-only|--long-context-only|--long-context-from 1|4|5|6"
                     "|--default-single-only|--long-verify5-only|--batch-decode-only]\n";
        return 2;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < kRanks) {
        std::cout << "SKIP: head-local attention parity requires " << kRanks
                  << " CUDA devices, found " << device_count << '\n';
        return 77;
    }

    if (batch_decode_only) {
        int failures = 0;
        for (bool long_attention : {false, true}) {
            failures += run_batch_case(DType::I8,
                {5, {2048, 2039, 2300, 2548, 2111, 127, 513, 2560},
                 {5, 3, 5, 1, 4, 5, 0, 5}, {0, 1, 2, 3, 4, 5, 6, 7},
                 MappingPattern::Fragmented, 1934u, true, long_attention}, false);
            for (int width : {3, 4, 6}) {
                failures += run_batch_case(DType::I8,
                    {width, {2048, 4091, 5000, 128}, {width, width, width - 1, 0},
                     {0, 1, 2, 3}, MappingPattern::Fragmented,
                     static_cast<std::uint32_t>(1940 + width), true, long_attention}, false);
            }
        }
        std::cout << (failures == 0 ? "PASS" : "FAIL")
                  << " head-local INT8 concurrent decode (FP64 and changed-query Graph/eager)\n";
        return failures == 0 ? 0 : 1;
    }

    if (long_verify5_only) {
        int failures = run_case(DType::I8,
            {"gqa_attention", 5, 84995, 180000, MappingPattern::Identity, 1606u, false},
            false, true);
        failures += run_long_split_graph_boundary(1, 5);
        failures += run_case(DType::I8,
            {"gqa_attention", 5, 16386, 16391, MappingPattern::Fragmented, 1751u, false, 3},
            false, true);
        failures += run_case(DType::I8,
            {"gqa_attention_cached", 5, 32763, 32768, MappingPattern::Fragmented, 1753u, false},
            false, true);
        failures += run_batch_case(DType::I8,
            {5, {2048, 16384, 85000, 32767}, {5, 3, 5, 0}, {3, 1, 0, 2},
             MappingPattern::Fragmented, 1834u, true, true});
        std::cout << (failures == 0 ? "PASS" : "FAIL")
                  << " head-local INT8 five-query long decode (independent FP64 oracle)\n";
        return failures == 0 ? 0 : 1;
    }

    // Exact envelopes and no valid-column mask reach the VoltaFlash path on sm_70. These are
    // real prefill widths, not small-T decode calls against a larger cache capacity. The second
    // chunk fixture additionally verifies that all history survives while 512 new rows append.
    int failures = 0;
    const HeadLocalCase prefill_cases[] = {
        {"gqa_attention", 65, 0, 65, MappingPattern::Identity, 1500u},
        {"gqa_attention", 520, 512, 1032, MappingPattern::Fragmented, 1506u},
        {"gqa_attention", 512, 0, 512, MappingPattern::Identity, 1501u},
        {"gqa_attention", 1024, 0, 1024, MappingPattern::Identity, 1502u},
        {"gqa_attention", 512, 512, 1024, MappingPattern::Fragmented, 1503u},
        {"gqa_attention", 2560, 1024, 3584, MappingPattern::Fragmented, 1504u},
        {"gqa_attention", 1281, 1024, 2305, MappingPattern::Offset, 1505u},
    };
    if (!long_context_only && !default_single_only) {
        for (const auto& test_case : prefill_cases) {
            failures += run_case(DType::I8, test_case, true);
        }
        if constexpr (kRanks == 4) {
            // Real TP4 prefill at 85K occupied keys. Each sampled query evaluates
            // every visible key and dimension against FP64 at the unchanged
            // registered criterion, both per rank and per head. Full-width
            // guards/cache checks remain exact; short cases above additionally
            // qualify full-output TP1 parity and distinguishable head mapping.
            failures += run_case(
                DType::I8,
                {"gqa_attention", 2560, 82440, 85000, MappingPattern::Fragmented, 1507u, false},
                true, false, ReferenceRoute::OracleOnly);
        }
    }
    if (prefill_only) {
        std::cout << (failures == 0 ? "PASS" : "FAIL")
                  << " head-local INT8 prefill (512..2560 columns, independent FP64 oracle)\n";
        return failures == 0 ? 0 : 1;
    }

    // One/four/five/six decode queries at the acceptance workload's occupied context, with the full
    // 180K capacity envelope. Evaluate every head/key against FP64 without subsampling.
    // Averaging 85K zero-mean V rows need not separate different heads by the fixture's
    // anti-permutation margin; the short cases retain that distinct-input requirement.
    // Oracle, per-head parity, and exact cache checks keep their original criteria here.
    for (const int width : {1, 4, 5, 6}) {
        if (width < first_long_width) { continue; }
        if (default_single_only && width != 1) { continue; }
        const HeadLocalCase long_context_case{
            .entry = "gqa_attention",
            .tokens = width,
            .base = 85000 - width,
            .envelope_max = 180000,
            .mapping = MappingPattern::Identity,
            .seed = static_cast<std::uint32_t>(1601 + width),
            .check_head_mapping = false,
        };
        std::cout << "head-local INT8 85K decode: full FP64 oracle, " << width
                  << " queries x 24 heads; 180K capacity\n" << std::flush;
        if (!default_single_only) {
            failures += run_case(DType::I8, long_context_case, false, true);
        }
        if (kRanks == 4 || width == 1 || width == 5) {
            failures += run_case(DType::I8, long_context_case, false, false);
        }
    }
    if (!default_single_only) {
        failures += run_long_split_graph_boundary(1, 5, false);
        failures += run_case(DType::I8,
            {"gqa_attention", 5, 16386, 180000, MappingPattern::Fragmented, 1741u, false, 3}, false, false);
        failures += run_case(DType::I8,
            {"gqa_attention", 5, 16386, 180000, MappingPattern::Offset, 1742u, false, 0}, false, false);
        failures += run_case(DType::I8,
            {"gqa_attention_cached", 5, 32763, 32768, MappingPattern::Offset, 1743u, false}, false, false);
        failures += run_batch_case(DType::I8,
            {5, {16386, 32763}, {3, 0}, {1, 0}, MappingPattern::Fragmented, 1744u, true});
    }
    if (!default_single_only) {
        failures += run_long_split_graph_boundary(0, 1);
        failures += run_long_split_graph_boundary(1, 5);
        failures += run_long_split_graph_boundary(1, 6);
        failures += run_case(DType::I8,
            {"gqa_attention", 1, 16390, 16391, MappingPattern::Fragmented, 1721u, false}, false, true);
        failures += run_case(DType::I8,
            {"gqa_attention_cached", 1, 32767, 32768, MappingPattern::Offset, 1722u, false}, false, true);
        failures += run_case(DType::I8,
            {"gqa_attention", 5, 16386, 16391, MappingPattern::Fragmented, 1751u, false, 3}, false, true);
        failures += run_case(DType::I8,
            {"gqa_attention", 5, 16386, 16391, MappingPattern::Offset, 1752u, false, 0}, false, true);
        failures += run_case(DType::I8,
            {"gqa_attention_cached", 5, 32763, 32768, MappingPattern::Fragmented, 1753u, false}, false, true);
        failures += run_case(DType::I8,
            {"gqa_attention", 6, 16385, 16391, MappingPattern::Fragmented, 1723u, false}, false, true);
        failures += run_case(DType::I8,
            {"gqa_attention", 6, 16385, 16391, MappingPattern::Fragmented, 1724u, false, 4}, false, true);
        failures += run_case(DType::I8,
            {"gqa_attention", 6, 16385, 16391, MappingPattern::Offset, 1725u, false, 0}, false, true);
    }
    failures += run_long_split_graph_boundary(1, 1, false);
    failures += run_case(DType::I8,
        {"gqa_attention", 1, 16390, 16391, MappingPattern::Fragmented, 1731u, false}, false, false);
    failures += run_case(DType::I8,
        {"gqa_attention_cached", 1, 32767, 32768, MappingPattern::Offset, 1732u, false}, false, false);
    failures += run_case(DType::I8,
        {"gqa_attention", 1, 16390, 180000, MappingPattern::Offset, 1733u, false, 0}, false, false);
    failures += run_batch_case(DType::I8,
        {1, {16390, 32767}, {1, 0}, {1, 0}, MappingPattern::Fragmented, 1734u, true});
    failures += run_case(DType::I8,
        {"gqa_attention", 1, 0, 180000, MappingPattern::Fragmented, 1735u, false}, false, false);
    failures += run_case(DType::I8,
        {"gqa_attention_cached", 1, 64, 180000, MappingPattern::Offset, 1736u, false}, false, false);
    if (!default_single_only) {
        failures += run_batch_case(DType::I8,
            {5, {2048, 16384, 85000, 32767}, {5, 3, 5, 0}, {3, 1, 0, 2},
             MappingPattern::Fragmented, 1834u, true, true});
        failures += run_batch_case(DType::I8,
            {1, {85000, 2048}, {1, 1}, {1, 0}, MappingPattern::Offset, 1835u, true, true});
    }
    if (long_context_only || default_single_only) {
        std::cout << (failures == 0 ? "PASS" : "FAIL")
                  << " head-local INT8 85K decode (independent FP64 oracle)\n";
        return failures == 0 ? 0 : 1;
    }

    failures += verify_rejections();

    // T=1 is the decode edge; T=6 is the split-KV small-T maximum; T=17 and T=66 take the prompt
    // route. Every case's key range crosses at least one 64-key page boundary, and the base=1000
    // and base=190 cases span many pages inside one split.
    const HeadLocalCase cases[] = {
        {"gqa_attention", 1, 0, 1, MappingPattern::Identity, 1101u},
        {"gqa_attention", 1, 64, 70, MappingPattern::Offset, 1102u},
        {"gqa_attention", 6, 61, 67, MappingPattern::Fragmented, 1103u},
        {"gqa_attention", 6, 190, 512, MappingPattern::Identity, 1104u},
        {"gqa_attention", 17, 31, 48, MappingPattern::Fragmented, 1105u},
        {"gqa_attention", 66, 63, 129, MappingPattern::Identity, 1106u},
        {"gqa_attention_cached", 1, 128, 129, MappingPattern::Fragmented, 1201u},
        {"gqa_attention_cached", 1, 1000, 1024, MappingPattern::Offset, 1202u},
        {"gqa_attention_cached", 6, 61, 512, MappingPattern::Identity, 1203u},
        {"gqa_attention_cached", 17, 31, 48, MappingPattern::Identity, 1204u},
    };
    for (const DType dtype : {DType::BF16, DType::I8}) {
        for (const HeadLocalCase& test_case : cases) { failures += run_case(dtype, test_case); }
        // W in (6,16] with B>1 is the only route into GqaAttentionRoute::ChunkedSmallT for a
        // 12-head geometry; W<=6 with B>1 is SmallT with MultiBatch=true.
        failures += run_batch_case(dtype, {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Identity,
                                           1301u + (dtype == DType::I8 ? 10u : 0u)});
        failures += run_batch_case(dtype, {1, {63, 511}, {1, 1}, {1, 0}, MappingPattern::Fragmented,
                                           1311u + (dtype == DType::I8 ? 10u : 0u)});
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            failures += run_append_case(dtype, mapping, 3, 63, 1401u);
        }
        // T>=128 reaches the paged INT8 fill kernel, which is selected on KVHeads==2.
        failures += run_append_case(dtype, MappingPattern::Fragmented, 129, 61, 1451u);
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " head-local attention parity (" << kLocalGeometry.q_heads << '|'
              << kLocalGeometry.kv_heads << " per device vs 24|4 single device)\n";
    return failures == 0 ? 0 : 1;
}
