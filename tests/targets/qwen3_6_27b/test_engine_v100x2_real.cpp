// Opt-in Qwen3.8 Q4_K_M/NVFP4 integration gate for the selected V100 TP2/TP4 domain.
// Engines are destroyed before the next one is constructed: this artifact
// cannot fit on one card. NINFER_V100X2_PROPOSAL_HEAD selects full (default) or
// optimized; both run the same graph/eager and non-speculative target checks.
//
// This is a behavioral/state gate, not an independent mathematical model oracle.
// The GGML_K and attention Op tests supply the independent FP64/FP32 oracles.
// Here exact graph/eager output and acceptance checks protect graph replay and
// cross-device commits. Fresh, non-speculative teacher-forced logits check every
// speculative output position, recording the chosen token's actual logit deficit (not
// merely the reference's top-two gap). Q4_K_M/NVFP4 retain the strict zero-disagreement
// criterion. Native block-FP8 also measures ordinary decode's re-prefill discrepancy:
// each probe's speculative worst deficit must not exceed its ordinary control and
// both stay within the established 0.5-logit BF16 grouping bound. This is not a
// claim of bit-identical trajectories or an independent whole-model quality score.
//
// NINFER_V100X2_ARTIFACT=/path/to/qwen3_8_27b_q4_k_m.ninfer ctest -R v100x2_real
// Add NINFER_V100X2_PROPOSAL_HEAD=optimized to exercise the shortlist proposal head.
// NINFER_V100X2_SPEC=dflash selects DFlash7 with the full vocabulary head.

#include "ninfer/engine.h"
#include "v100x2_test_profile.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kContext = 4096;
constexpr std::uint32_t kChunk = 256;
constexpr std::uint32_t kOutputs = 32;
// Registered Qwen3.8 tokenizer domain; the output matrix includes padded rows.
constexpr std::size_t kTokenDomain = 248077;
constexpr std::size_t kProbes = 2;
constexpr std::array<std::size_t, 3> kLogitPositions{0, 15, 31};

using Tokens = std::vector<ninfer::TokenId>;
using Logits = std::vector<std::uint16_t>;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact, bool speculation, bool graphs,
                                     ninfer::SpeculativeOptions profile) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.tp = v100x2_test::tp();
    options.pp = v100x2_test::pp();
    options.devices = v100x2_test::devices();
    options.max_context = kContext;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kContext);
    options.prefill_chunk = kChunk;
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph = graphs;
    if (speculation) { options.speculative = profile; }
    return options;
}

ninfer::RequestOptions request_options(std::uint32_t count) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = count;
    options.execution.allow_prefix_reuse = false;
    options.execution.sampling.temperature = 0.0F;
    options.execution.sampling.presence_penalty = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.stop.include_model_defaults = false;
    return options;
}

ninfer::GenerationResult generate(ninfer::Engine& engine, const Tokens& prompt,
                                  std::uint32_t count = kOutputs) {
    auto result = engine.generate(engine.prepare_tokens(prompt, false), request_options(count));
    require(result.generated_token_ids.size() == count, "generation ended before its token budget");
    require(result.reused_prompt_tokens == 0, "the full-reset probe unexpectedly reused a prefix");
    return result;
}

ninfer::RequestOptions sampled_options() {
    auto options = request_options(kOutputs);
    options.execution.sampling.temperature = 0.7F;
    options.execution.sampling.top_k = 20;
    options.execution.sampling.top_p = 0.8F;
    options.execution.sampling.presence_penalty = 0.5F;
    options.execution.sampling.frequency_penalty = 0.3F;
    options.execution.sampling.seed = 20260826ULL;
    return options;
}

ninfer::GenerationResult generate_sampled(ninfer::Engine& engine, const Tokens& prompt) {
    return engine.generate(engine.prepare_tokens(prompt, false), sampled_options());
}

Tokens chat_tokens(ninfer::Engine& engine, const std::string& text) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    ninfer::ChatMessage message;
    message.parts.push_back({ninfer::MessagePartKind::Text, text, {}});
    input.messages.push_back(std::move(message));
    return engine.prepare(std::move(input)).debug_token_ids();
}

std::array<Tokens, kProbes> prompts(ninfer::Engine& engine) {
    std::string long_text = "Read the following observations and summarize their pattern.\n";
    for (int i = 0; i < 32; ++i) {
        long_text += "Observation " + std::to_string(i) +
                     ": the morning temperature rises, the ice melts, and the river flows faster.\n";
    }
    long_text += "Explain the causal relationship in several sentences.";
    std::array<Tokens, kProbes> result{
        chat_tokens(engine, "Write a Python function that returns the first n Fibonacci numbers. "
                            "Then explain its time and space complexity."),
        chat_tokens(engine, long_text)};
    require(result[0].size() < kChunk, "the short probe no longer fits in one prefill chunk");
    require(result[1].size() > 2 * kChunk && result[1].size() + kOutputs < kContext,
            "the long probe must cross multiple prefill chunks and fit its context");
    return result;
}

float value(std::uint16_t bits) {
    const std::uint32_t raw = std::uint32_t(bits) << 16U;
    float result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

ninfer::TokenId argmax(const Logits& logits) {
    require(logits.size() >= kTokenDomain, "captured logits omit valid vocabulary rows");
    float best = -std::numeric_limits<float>::infinity();
    ninfer::TokenId token = -1;
    for (std::size_t i = 0; i < kTokenDomain; ++i) {
        const float x = value(logits[i]);
        require(std::isfinite(x), "captured logits contain a non-finite vocabulary value");
        if (x > best) {
            best = x;
            token = static_cast<ninfer::TokenId>(i);
        }
    }
    return token;
}

Logits probe(ninfer::Engine& engine, const Tokens& context) {
    const auto result = generate(engine, context, 1);
    auto logits = engine.debug_last_round_logits_bf16();
    const auto best = argmax(logits);
    require(result.generated_token_ids.front() == best,
            "the captured logit argmax is not the token sampled by the same request");
    logits.resize(kTokenDomain);
    return logits;
}

void require_speculation(const ninfer::GenerationResult& result,
                         ninfer::SpeculativeOptions profile) {
    const auto& stats = result.speculative;
    require(stats.enabled && stats.backend == profile.backend &&
                stats.draft_window == profile.draft_tokens && stats.rounds > 0 &&
                stats.drafted_tokens > 0,
            "the selected probe did not execute speculative rounds");
    require(stats.accepted_tokens <= stats.drafted_tokens, "invalid accepted draft count");
}

void compare_rounds(const ninfer::GenerationResult& captured,
                    const ninfer::GenerationResult& eager) {
    const auto& a_ids = captured.generated_token_ids;
    const auto& b_ids = eager.generated_token_ids;
    const auto mismatch = std::mismatch(a_ids.begin(), a_ids.end(), b_ids.begin(), b_ids.end());
    if (mismatch.first != a_ids.end() || mismatch.second != b_ids.end()) {
        const auto position = static_cast<std::size_t>(mismatch.first - a_ids.begin());
        throw std::runtime_error(
            "graph/eager speculative committed token sequences differ: position=" +
            std::to_string(position) + " graph=" +
            (mismatch.first == a_ids.end() ? "end" : std::to_string(*mismatch.first)) +
            " eager=" +
            (mismatch.second == b_ids.end() ? "end" : std::to_string(*mismatch.second)));
    }
    const auto& a = captured.speculative;
    const auto& b = eager.speculative;
    require(a.rounds == b.rounds && a.drafted_tokens == b.drafted_tokens &&
                a.accepted_tokens == b.accepted_tokens && a.fallback_steps == b.fallback_steps &&
                a.accepted_per_position == b.accepted_per_position,
            "graph/eager speculative acceptance or fallback patterns differ");
}

void exercise_mtp_batches(const char* artifact, ninfer::SpeculativeOptions profile,
                           const std::array<Tokens, kProbes>& inputs,
                           unsigned concurrency = 2, bool long_fast = false) {
    std::array<std::array<ninfer::GenerationResult, 8>, 2> captured;
    const unsigned outputs = concurrency > 2 ? 128 : kOutputs;
    for (bool graphs : {true, false}) {
        auto options = engine_options(artifact, true, graphs, profile);
        options.max_concurrency = concurrency;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(concurrency * kContext);
        options.volta_long_attention = long_fast;
        options.volta_fast_decode = long_fast;
        ninfer::Engine engine(options);
        engine.debug_enable_peer_egress_check(true);
        for (std::size_t mixed = 0; mixed < 2; ++mixed) {
            std::vector<ninfer::PreparedPrompt> prepared;
            for (unsigned lane = 0; lane < concurrency; ++lane) {
                prepared.push_back(engine.prepare_tokens(inputs[lane % inputs.size()], false));
            }
            const auto before = engine.runtime_stats();
            std::vector<ninfer::GenerationHandle> handles;
            for (unsigned lane = 0; lane < concurrency; ++lane) {
                auto request = mixed && lane % 2 ? sampled_options() : request_options(outputs);
                request.execution.requested_output_tokens = outputs;
                handles.push_back(engine.submit(std::move(prepared[lane]),
                                                request));
            }
            unsigned peak_ready = 0;
            std::jthread observer([&](std::stop_token stop) {
                while (!stop.stop_requested()) {
                    peak_ready = std::max(peak_ready, engine.runtime_stats().decode_ready_requests);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
            std::vector<ninfer::GenerationResult> results;
            for (auto& handle : handles) { results.push_back(handle.wait()); }
            observer.request_stop();
            observer.join();
            const auto after = engine.runtime_stats();
            require(peak_ready == concurrency, "MTP batch probe did not reach its full batch");
            require(after.decode_row_rounds - before.decode_row_rounds >
                        after.decode_rounds - before.decode_rounds,
                    "greedy/mixed MTP probe did not execute a multi-row decode round");
            for (unsigned lane = 0; lane < concurrency; ++lane) {
                require(results[lane].generated_token_ids.size() == outputs,
                        "MTP batch probe ended before its output budget");
                require_speculation(results[lane], profile);
                if (graphs) {
                    captured[mixed][lane] = std::move(results[lane]);
                } else {
                    compare_rounds(captured[mixed][lane], results[lane]);
                }
            }
        }
        v100x2_test::check_peer_egress(engine, profile.backend);
    }
    std::cout << "MTP B=" << concurrency << (long_fast ? " long/fast" : " normal")
              << " greedy/mixed graph/eager outputs and acceptance PASS" << std::endl;
}

int exercise(const char* artifact, ninfer::SpeculativeOptions profile) {
    std::cout << "spec="
              << (profile.backend == ninfer::SpeculativeBackend::Mtp ? "mtp" : "dflash")
              << " drafts=" << profile.draft_tokens << " proposal_head="
              << (profile.proposal_head == ninfer::ProposalHead::Full ? "full" : "optimized")
              << std::endl;
    std::array<Tokens, kProbes> inputs;
    std::array<ninfer::GenerationResult, kProbes> captured;
    ninfer::GenerationResult captured_sampled;
    std::array<std::array<Logits, kLogitPositions.size()>, kProbes> speculative_logits;
    std::uint64_t accepted_total = 0;
    bool native_fp8 = false;
    {
        ninfer::Engine engine(engine_options(artifact, true, true, profile));
        v100x2_test::check_identity(engine);
        native_fp8 = engine.load_summary().weights_id == "fp8";
        inputs = prompts(engine);
        engine.debug_enable_peer_egress_check(true);
        for (std::size_t p = 0; p < kProbes; ++p) {
            captured[p] = generate(engine, inputs[p]);
            require_speculation(captured[p], profile);
            accepted_total += captured[p].speculative.accepted_tokens;
            std::cout << "captured probe=" << p << " prompt_tokens=" << inputs[p].size()
                      << " output_tokens=" << captured[p].generated_token_ids.size()
                      << " accepted=" << captured[p].speculative.accepted_tokens
                      << "/" << captured[p].speculative.drafted_tokens << std::endl;
        }
        require(accepted_total > 0, "no proposal was accepted; the commit path was not exercised");
        // Exercise both graph topologies in one Engine, then switch back to the greedy route.
        // Nonzero penalties also protect rank-local counter ownership on the full-logit path.
        captured_sampled = generate_sampled(engine, inputs[0]);
        compare_rounds(captured[0], generate(engine, inputs[0]));
        v100x2_test::check_peer_egress(engine, profile.backend);
        require(engine.memory_summary().cuda_graph_node_count > 0,
                "graphs were requested but no decode graph was captured");
        engine.debug_enable_logit_capture(true);
        for (std::size_t p = 0; p < kProbes; ++p) {
            for (std::size_t i = 0; i < kLogitPositions.size(); ++i) {
                Tokens context = inputs[p];
                const auto& emitted = captured[p].generated_token_ids;
                context.insert(context.end(), emitted.begin(), emitted.begin() + kLogitPositions[i]);
                speculative_logits[p][i] = probe(engine, context);
            }
        }
    }
    {
        ninfer::Engine engine(engine_options(artifact, true, false, profile));
        engine.debug_enable_peer_egress_check(true);
        for (std::size_t p = 0; p < kProbes; ++p) {
            const auto eager = generate(engine, inputs[p]);
            require_speculation(eager, profile);
            std::cout << "eager probe=" << p << " prompt_tokens=" << inputs[p].size()
                      << " output_tokens=" << eager.generated_token_ids.size()
                      << " accepted=" << eager.speculative.accepted_tokens
                      << "/" << eager.speculative.drafted_tokens << std::endl;
            compare_rounds(captured[p], eager);
        }
        compare_rounds(captured_sampled, generate_sampled(engine, inputs[0]));
        compare_rounds(captured[0], generate(engine, inputs[0]));
        std::cout << "greedy/sampling route switch and sampled graph/eager parity PASS" << std::endl;
        v100x2_test::check_peer_egress(engine, profile.backend);
        require(engine.memory_summary().cuda_graph_node_count == 0,
                "the eager control unexpectedly captured a graph");
    }

    std::size_t disagreements = 0;
    float worst_deficit = 0.0F;
    std::size_t ordinary_disagreements = 0;
    float ordinary_worst_deficit = 0.0F;
    std::array<float, kProbes> speculative_probe_deficits{};
    std::array<float, kProbes> ordinary_probe_deficits{};
    {
        ninfer::Engine engine(engine_options(artifact, false, false, profile));
        engine.debug_enable_logit_capture(true);
        for (std::size_t p = 0; p < kProbes; ++p) {
            const auto plain = generate(engine, inputs[p]);
            require(!plain.speculative.enabled, "the target control unexpectedly enabled speculation");
            std::cout << "probe=" << p << " speculative_vs_plain_sequence_equal="
                      << (plain.generated_token_ids == captured[p].generated_token_ids) << std::endl;
            Tokens context = inputs[p];
            Tokens ordinary_context = inputs[p];
            for (std::size_t i = 0; i < kOutputs; ++i) {
                const auto logits = probe(engine, context);
                const auto best = argmax(logits);
                const auto emitted = captured[p].generated_token_ids[i];
                require(emitted >= 0 && std::size_t(emitted) < kTokenDomain,
                        "speculation emitted a token outside the registered tokenizer domain");
                if (best != emitted) {
                    const float deficit = value(logits[best]) - value(logits[emitted]);
                    worst_deficit = std::max(worst_deficit, deficit);
                    speculative_probe_deficits[p] = std::max(speculative_probe_deficits[p], deficit);
                    ++disagreements;
                    std::cerr << "teacher_force probe=" << p << " position=" << i
                              << " target=" << best << " speculative=" << emitted
                              << " emitted_logit_deficit=" << deficit << '\n';
                }
                for (std::size_t j = 0; j < kLogitPositions.size(); ++j) {
                    if (i == kLogitPositions[j]) {
                        require(logits == speculative_logits[p][j],
                                "enabling speculation changed target prefill logit bits");
                    }
                }
                context.push_back(emitted);
                const auto ordinary_logits = probe(engine, ordinary_context);
                const auto ordinary_best = argmax(ordinary_logits);
                const auto ordinary_emitted = plain.generated_token_ids[i];
                if (ordinary_best != ordinary_emitted) {
                    const float deficit = value(ordinary_logits[ordinary_best]) -
                                          value(ordinary_logits[ordinary_emitted]);
                    ordinary_worst_deficit = std::max(ordinary_worst_deficit, deficit);
                    ordinary_probe_deficits[p] = std::max(ordinary_probe_deficits[p], deficit);
                    ++ordinary_disagreements;
                    std::cerr << "ordinary_teacher_force probe=" << p << " position=" << i
                              << " target=" << ordinary_best << " emitted=" << ordinary_emitted
                              << " emitted_logit_deficit=" << deficit << '\n';
                }
                ordinary_context.push_back(ordinary_emitted);
            }
        }
    }
    std::cout << "teacher_force_positions=" << kProbes * kOutputs
              << " disagreements=" << disagreements << " worst_emitted_logit_deficit="
              << worst_deficit << std::endl;
    std::cout << "ordinary_teacher_force_positions=" << kProbes * kOutputs
              << " disagreements=" << ordinary_disagreements << " worst_emitted_logit_deficit="
              << ordinary_worst_deficit << std::endl;
    if (native_fp8) {
        // Existing model-level BF16 grouping bound, not fitted to this FP8 fixture.
        // Independent FP64 Op oracles still qualify the represented block codes/scales.
        constexpr float kNearTieBound = 0.5F;
        for (std::size_t p = 0; p < kProbes; ++p) {
            require(ordinary_probe_deficits[p] <= kNearTieBound &&
                        speculative_probe_deficits[p] <= ordinary_probe_deficits[p],
                    "native FP8 speculation exceeds its ordinary re-prefill error envelope");
        }
        std::cout << "native FP8 per-probe ordinary-controlled BF16 grouping check PASS" << std::endl;
    } else {
        require(disagreements == 0, "outputs failed strict non-speculative teacher-forced argmax");
    }
    if (profile.backend == ninfer::SpeculativeBackend::Mtp && v100x2_test::pp() == 1) {
        exercise_mtp_batches(artifact, profile, inputs);
    }
    return 0;
}


// Exercise the actual long-kernel window, rather than increasing capacity while
// retaining a short prompt. The attention Op suite supplies the FP64 oracle;
// this gate protects graph replay, MTP commits and exact saved frontiers.
int exercise_long_attention(const char* artifact, ninfer::SpeculativeOptions profile,
                            bool fast_decode) {
    require(profile.backend == ninfer::SpeculativeBackend::Mtp && v100x2_test::pp() == 1,
            "long attention state gate requires Text/MTP without pipeline parallelism");
    std::array<ninfer::GenerationResult, 3> captured;
    std::array<Logits, 3> captured_logits;
    Tokens input;
    for (const bool graphs : {true, false}) {
        auto options = engine_options(artifact, true, graphs, profile);
        options.max_context = 90000;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(90000);
        options.prefill_chunk = 2560;
        options.prefix_cache_tokens = 0;
        options.volta_long_attention = true;
        options.volta_fast_decode = fast_decode;
        ninfer::Engine engine(options);
        v100x2_test::check_identity(engine);
        engine.debug_enable_peer_egress_check(true);
        engine.debug_enable_logit_capture(true);
        if (input.empty()) {
            const auto seed = prompts(engine)[1];
            constexpr std::size_t suffix = 64;
            for (std::size_t i = 0; i < 85000 - suffix; ++i) {
                input.push_back(seed[i % (seed.size() - suffix)]);
            }
            input.insert(input.end(), seed.end() - suffix, seed.end());
        }
        for (std::size_t replay = 0; replay < captured.size(); ++replay) {
            auto request = request_options(kOutputs);
            // The empty Engine is cold; enabling reuse also captures its frontiers
            // for subsequent requests. Disabling reuse would suppress those images.
            request.execution.allow_prefix_reuse = true;
            const auto before = engine.runtime_stats();
            std::cout << "85K state graphs=" << graphs << " replay=" << replay
                      << " begin" << std::endl;
            auto result = engine.generate(engine.prepare_tokens(input), request);
            const auto logits = engine.debug_last_round_logits_bf16();
            const auto computed = engine.runtime_stats().computed_prefill_tokens -
                                  before.computed_prefill_tokens;
            require(result.generated_token_ids.size() == kOutputs,
                    "long attention state gate ended before its token budget");
            require_speculation(result, profile);
            require(result.reused_prompt_tokens == (replay ? input.size() : 0) &&
                    computed == (replay ? 0 : input.size()),
                    "long attention cold/replay prefill accounting differs: graphs=" +
                        std::to_string(graphs) + " replay=" + std::to_string(replay) +
                        " reused=" + std::to_string(result.reused_prompt_tokens) +
                        " computed=" + std::to_string(computed));
            if (graphs && replay) {
                require(captured[0].generated_token_ids == result.generated_token_ids,
                        "long attention exact replay changed committed output");
            }
            std::cout << "85K state graphs=" << graphs << " replay=" << replay
                      << " reused=" << result.reused_prompt_tokens << " computed=" << computed
                      << " PASS" << std::endl;
            if (graphs) {
                captured[replay] = std::move(result);
                captured_logits[replay] = logits;
                if (replay == 2) {
                    compare_rounds(captured[1], captured[2]);
                    require(captured_logits[1] == captured_logits[2],
                            "long attention repeated replay changed target logits");
                }
            } else {
                compare_rounds(captured[replay], result);
                require(captured_logits[replay] == logits,
                        "long attention graph/eager target logits differ");
            }
        }
        Tokens frontier = input;
        frontier.insert(frontier.end(), captured.back().generated_token_ids.begin(),
                        captured.back().generated_token_ids.end() - 1);
        auto request = request_options(1);
        request.execution.allow_prefix_reuse = true;
        const auto resumed = engine.generate(engine.prepare_tokens(frontier), request);
        require(resumed.reused_prompt_tokens == frontier.size() &&
                resumed.generated_token_ids.size() == 1 &&
                resumed.generated_token_ids.front() == captured.back().generated_token_ids.back(),
                "long attention committed frontier did not resume exactly");
        require((engine.memory_summary().cuda_graph_node_count > 0) == graphs,
                "long attention graph admission differs from the selected route");
        v100x2_test::check_peer_egress(engine, profile.backend);
    }
    std::cout << "85K long attention" << (fast_decode ? " + fast decode" : "")
              << " MTP graph/eager, exact replay logits/counters and committed frontier PASS"
              << std::endl;
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const bool long_attention = argc == 2 && std::string(argv[1]) == "--long-attention";
    const bool long_fast = argc == 2 && std::string(argv[1]) == "--long-fast-decode";
    const bool batches = argc == 2 && std::string(argv[1]) == "--mtp-batches";
    if (argc != 1 && !long_attention && !long_fast && !batches) { return 2; }
    const char* artifact = std::getenv("NINFER_V100X2_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_V100X2_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess ||
        devices < v100x2_test::tp() * v100x2_test::pp()) {
        std::cout << "skip: insufficient devices for the selected tensor-parallel width\n";
        return 77;
    }
    try {
        const auto profile = v100x2_test::profile(ninfer::ProposalHead::Full);
        if (batches) {
            require((v100x2_test::tp() == 2 || v100x2_test::tp() == 4) && v100x2_test::pp() == 1 &&
                        profile.backend == ninfer::SpeculativeBackend::Mtp,
                    "batch gate requires TP2/TP4 PP1 MTP");
            std::array<Tokens, kProbes> inputs;
            {
                ninfer::Engine engine(engine_options(artifact, true, false, profile));
                v100x2_test::check_identity(engine);
                inputs = prompts(engine);
            }
            for (bool fast : {false, true}) {
                for (unsigned concurrency : {2U, 4U, 8U}) {
                    exercise_mtp_batches(artifact, profile, inputs, concurrency, fast);
                }
            }
            return 0;
        }
        return long_attention || long_fast
            ? exercise_long_attention(artifact, profile, long_fast) : exercise(artifact, profile);
    } catch (const std::exception& error) {
        std::cerr << "V100X2 integration: " << error.what() << '\n';
        return 1;
    }
}
