// 1Cat-style workload rows through NInfer's public Engine; no forced draft acceptance.
#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
struct Sample { double wall; ninfer::RuntimeStats stats; };

static Json measure(const std::vector<Sample>& samples, unsigned width) {
    struct Interval { double seconds, wall; std::uint64_t tokens, drafted, accepted; };
    std::vector<Interval> full;
    unsigned merged = 0;
    for (std::size_t i = 1; i < samples.size(); ++i) {
        const auto& a = samples[i - 1]; const auto& b = samples[i];
        const auto rounds = b.stats.decode_rounds - a.stats.decode_rounds;
        if (rounds == 0) { continue; }
        if (a.stats.running_requests != width || b.stats.running_requests != width ||
            a.stats.decode_ready_requests != width || b.stats.decode_ready_requests != width ||
            a.stats.prefilling_requests || b.stats.prefilling_requests ||
            a.stats.waiting_requests || b.stats.waiting_requests ||
            a.stats.computed_prefill_tokens != b.stats.computed_prefill_tokens ||
            b.stats.decode_row_rounds - a.stats.decode_row_rounds != rounds * width) { continue; }
        if (rounds != 1) { ++merged; continue; }
        full.push_back({b.stats.decode_wall_seconds - a.stats.decode_wall_seconds,
            b.wall - a.wall,
            b.stats.committed_decode_tokens - a.stats.committed_decode_tokens,
            b.stats.speculative_drafted_tokens - a.stats.speculative_drafted_tokens,
            b.stats.speculative_accepted_tokens - a.stats.speculative_accepted_tokens});
    }
    Json out{{"eligible_full_width_rounds", full.size()}, {"merged_poll_intervals", merged},
        {"trim_rounds_per_end", 8}, {"aggregate_decode_tok_s", nullptr},
        {"step_ms_mean", nullptr}, {"normalized_70pct_tok_s", nullptr}};
    if (full.size() < 32) { out["status"] = "insufficient simultaneous decode rounds"; return out; }
    double seconds = 0, wall = 0; std::uint64_t tokens = 0, drafted = 0, accepted = 0;
    std::vector<double> steps;
    for (std::size_t i = 8; i + 8 < full.size(); ++i) {
        seconds += full[i].seconds; wall += full[i].wall; tokens += full[i].tokens;
        drafted += full[i].drafted; accepted += full[i].accepted;
        steps.push_back(full[i].seconds * 1000);
    }
    std::sort(steps.begin(), steps.end());
    out.update({{"status", "measured"}, {"measured_rounds", steps.size()},
        {"committed_tokens", tokens}, {"executor_seconds", seconds},
        {"observation_wall_seconds", wall}, {"aggregate_decode_tok_s", tokens / seconds},
        {"observed_wall_tok_s", tokens / wall}, {"step_ms_mean", seconds * 1000 / steps.size()},
        {"step_ms_p50", steps[steps.size() / 2]},
        {"step_ms_p90", steps[static_cast<std::size_t>((steps.size() - 1) * .9)]},
        {"observed_wall_step_ms_mean", wall * 1000 / steps.size()},
        {"emitted_tokens_per_request_round", double(tokens) / (width * steps.size())},
        {"drafted", drafted}, {"accepted", accepted},
        {"acceptance", drafted ? double(accepted) / drafted : 0}});
    return out;
}

int main(int argc, char** argv) {
    Json report;
    std::string destination;
    const auto save = [&] {
        if (destination.empty()) { return; }
        std::ofstream out(destination + ".tmp"); out << report.dump(2) << '\n'; out.close();
        if (!out) { throw std::runtime_error("cannot write report"); }
        std::filesystem::rename(destination + ".tmp", destination);
    };
    try {
        if (argc < 5) { throw std::invalid_argument(
            "bench ARTIFACT mtp|dflash 8192|65536 REPORT [--experiments 0|1] "
            "[--capacity N|0(auto)] [--prefill-chunk N] [--output N] [--repeats N] "
            "[--probe-only 0|1] [--fixture REPORT]"); }
        const std::string backend = argv[2]; const unsigned length = std::stoul(argv[3]);
        destination = argv[4]; bool experimental = false, probe = false;
        unsigned output = 2048, repeats = 2, capacity = 0, chunk = 1024; std::string fixture;
        for (int i = 5; i < argc; i += 2) {
            if (i + 1 == argc) { throw std::invalid_argument("missing option value"); }
            const std::string key = argv[i];
            if (key == "--fixture") { fixture = argv[i + 1]; continue; }
            const auto n = std::stoul(argv[i + 1]);
            if (key == "--experiments" && n <= 1) { experimental = n; }
            else if (key == "--probe-only" && n <= 1) { probe = n; }
            else if (key == "--capacity") { capacity = n; }
            else if (key == "--prefill-chunk") { chunk = n; }
            else if (key == "--output") { output = n; }
            else if (key == "--repeats") { repeats = n; }
            else { throw std::invalid_argument("invalid option " + key); }
        }
        if ((backend != "mtp" && backend != "dflash") || (length != 8192 && length != 65536) ||
            output < 256 || output > 16384 || repeats == 0 || chunk < 128 ||
            chunk > 16384 || chunk % 128 != 0) { throw std::invalid_argument("invalid workload"); }
        const unsigned width = length == 8192 ? 8 : 4;
        const unsigned per_row = (length + output + 8 + 63) / 64 * 64;
        const unsigned required = per_row * width;
        ninfer::EngineOptions options;
        options.artifact_path = argv[1]; options.tp = 2; options.devices = {0, 1};
        options.max_context = (length + output + 4095) / 4096 * 4096;
        options.kv_capacity = capacity ? ninfer::KvCapacityPolicy::explicit_capacity(capacity)
                                       : ninfer::KvCapacityPolicy::automatic();
        options.prefill_chunk = chunk; options.prefill_decode_rounds = 1;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.max_concurrency = width; options.pending_timeout_ms = 1800000;
        options.use_cuda_graph = true; options.prefix_cache_tokens = 0;
        options.speculative.backend = backend == "mtp" ? ninfer::SpeculativeBackend::Mtp
                                                       : ninfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens = backend == "mtp" ? 4 : 7;
        options.speculative.proposal_head = experimental && backend == "mtp"
            ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
        options.volta_long_attention = experimental; options.volta_fast_decode = experimental;
        report = {{"artifact", argv[1]}, {"backend", backend}, {"tp", 2},
            {"input_tokens", length}, {"concurrency", width}, {"output_tokens", output},
            {"required_kv_capacity", required}, {"requested_kv_capacity", capacity},
            {"max_context", options.max_context}, {"prefill_chunk", options.prefill_chunk},
            {"experiments", experimental}, {"optimized_head", experimental && backend == "mtp"},
            {"drafts", options.speculative.draft_tokens}, {"kv", "int8-group64"},
            {"graphs", true}, {"prefix_reuse", false}, {"thinking", false}, {"temperature", 0},
            {"runs", Json::array()}};
        save();
        ninfer::Engine engine(options);
        const auto memory = engine.memory_summary();
        report["resolved_kv_capacity"] = memory.kv_capacity;
        report["all_rows_can_reside"] = memory.kv_capacity >= required;
        report["graph_bytes"] = memory.cuda_graph_observed_bytes;
        report["graph_allowance_bytes"] = memory.cuda_graph_allowance_bytes;
        report["device_memory"] = Json::array();
        for (unsigned rank = 0; rank < 2; ++rank) {
            const auto m = engine.load_summary().devices[rank];
            report["device_memory"].push_back({{"device", m.device}, {"weights", m.weights_bytes},
                {"sequence", m.sequence_bytes}, {"workspace", m.workspace_bytes},
                {"reserved", m.reserved_bytes}, {"free", m.free_after_startup_bytes}});
        }
        if (probe) { report["status"] = "capacity probe"; save(); std::cout << report.dump() << std::endl; return 0; }
        Json cases = Json::array();
        if (!fixture.empty()) {
            std::ifstream in(fixture); Json saved; in >> saved; cases = saved.at("cases");
        } else {
            const std::string filler = "SM70 V100 release benchmark prompt. Continue with stable, coherent "
                "technical prose about CUDA kernels, attention backends, KV cache, speculative decoding, "
                "quantization, and long-context serving. ";
            const auto chat_ids = [&](const std::string& text) {
                ninfer::PromptInput input; input.options.enable_thinking = false;
                ninfer::ChatMessage user;
                user.parts.push_back({.text = text});
                input.messages.push_back(std::move(user));
                return engine.prepare(std::move(input)).debug_token_ids();
            };
            const auto a = chat_ids("a"), b = chat_ids("b");
            std::size_t prefix = 0, suffix = 0;
            while (prefix < std::min(a.size(), b.size()) && a[prefix] == b[prefix]) { ++prefix; }
            while (suffix < std::min(a.size(), b.size()) - prefix &&
                   a[a.size() - suffix - 1] == b[b.size() - suffix - 1]) { ++suffix; }
            const auto content_ids = [&](const std::string& text) {
                auto ids = chat_ids(text);
                if (!std::equal(a.begin(), a.begin() + prefix, ids.begin()) ||
                    !std::equal(a.end() - suffix, a.end(), ids.end() - suffix)) {
                    throw std::runtime_error("chat template changed while building fixture");
                }
                return std::vector<ninfer::TokenId>(ids.begin() + prefix, ids.end() - suffix);
            };
            const auto fill = content_ids(filler);
            if (fill.empty()) { throw std::runtime_error("empty filler"); }
            for (unsigned row = 0; row < width; ++row) {
                const auto start = content_ids("Independent benchmark stream " + std::to_string(row) + ".\n");
                const auto question = content_ids("\nQuestion for stream " + std::to_string(row) +
                    ": Explain how KV caching and speculative decoding improve inference. "
                    "Give a detailed technical answer.");
                std::vector<ninfer::TokenId> exact(a.begin(), a.begin() + prefix);
                exact.insert(exact.end(), start.begin(), start.end());
                const auto body_end = length - question.size() - suffix;
                while (exact.size() < body_end) { exact.insert(exact.end(), fill.begin(), fill.end()); }
                exact.resize(body_end);
                exact.insert(exact.end(), question.begin(), question.end());
                exact.insert(exact.end(), a.end() - suffix, a.end());
                cases.push_back({{"stream", row}, {"prompt_ids", exact}});
            }
        }
        if (cases.size() != width) { throw std::runtime_error("wrong fixture width"); }
        report["cases"] = cases; save();
        ninfer::RequestOptions request;
        request.execution.allow_prefix_reuse = false; request.execution.requested_output_tokens = output;
        request.execution.sampling.temperature = 0; request.execution.sampling.presence_penalty = 0;
        request.execution.sampling.frequency_penalty = 0; request.stop.include_model_defaults = false;
        auto warm = request; warm.execution.requested_output_tokens = 33;
        (void)engine.generate(engine.prepare_tokens({198, 220, 198, 220}, false), warm);
        for (unsigned rep = 0; rep < repeats; ++rep) {
            const auto origin = Clock::now(); const auto before = engine.runtime_stats();
            std::vector<Sample> samples{{0, before}}; samples.reserve(10000);
            std::vector<ninfer::GenerationHandle> handles;
            for (const auto& item : cases) {
                auto ids = item.at("prompt_ids").get<std::vector<ninfer::TokenId>>();
                if (ids.size() != length) { throw std::runtime_error("fixture length changed"); }
                handles.push_back(engine.submit(engine.prepare_tokens(std::move(ids), false), request));
            }
            std::jthread observer([&](std::stop_token stop) {
                while (!stop.stop_requested()) {
                    const auto stats = engine.runtime_stats();
                    const double wall = std::chrono::duration<double>(Clock::now() - origin).count();
                    const auto& last = samples.back().stats;
                    if (stats.decode_rounds != last.decode_rounds ||
                        stats.computed_prefill_tokens != last.computed_prefill_tokens ||
                        stats.decode_ready_requests != last.decode_ready_requests ||
                        stats.running_requests != last.running_requests) { samples.push_back({wall, stats}); }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
            Json requests = Json::array();
            for (auto& handle : handles) {
                const auto r = handle.wait();
                if (r.generated_token_ids.size() != output || r.reused_prompt_tokens != 0) {
                    throw std::runtime_error("incomplete output or unexpected prefix reuse");
                }
                requests.push_back({{"ids", r.generated_token_ids}, {"rounds", r.speculative.rounds},
                    {"drafted", r.speculative.drafted_tokens}, {"accepted", r.speculative.accepted_tokens},
                    {"accepted_per_position", r.speculative.accepted_per_position},
                    {"fallback_steps", r.speculative.fallback_steps}, {"finish_reason", int(r.finish_reason)},
                    {"ttft_seconds", r.timings.first_token_seconds}, {"total_seconds", r.timings.total_seconds}});
            }
            observer.request_stop(); observer.join();
            const auto after = engine.runtime_stats();
            samples.push_back({std::chrono::duration<double>(Clock::now() - origin).count(), after});
            auto measured = measure(samples, width);
            if (measured.at("status") == "measured") {
                const double per_round = backend == "mtp" ? 3.8 : 5.9;
                measured["normalized_70pct_tok_s"] = per_round * width * 1000 /
                    measured.at("step_ms_mean").get<double>();
            }
            Json trace = Json::array();
            for (const auto& s : samples) {
                trace.push_back({s.wall, s.stats.decode_rounds, s.stats.decode_row_rounds,
                    s.stats.committed_decode_tokens, s.stats.decode_wall_seconds,
                    s.stats.computed_prefill_tokens, s.stats.running_requests,
                    s.stats.decode_ready_requests, s.stats.prefilling_requests, s.stats.waiting_requests});
            }
            measured["repeat"] = rep; measured["requests"] = requests; measured["samples"] = trace;
            measured["whole_wave_seconds"] = samples.back().wall;
            measured["all_decode_tok_s"] = double(after.committed_decode_tokens - before.committed_decode_tokens) /
                (after.decode_wall_seconds - before.decode_wall_seconds);
            measured["mean_all_decode_batch"] = double(after.decode_row_rounds - before.decode_row_rounds) /
                (after.decode_rounds - before.decode_rounds);
            report["runs"].push_back(measured); save();
            measured.erase("requests"); measured.erase("samples"); std::cout << measured.dump() << std::endl;
        }
        report["status"] = "complete"; save(); return 0;
    } catch (const std::exception& error) {
        report["status"] = "failed"; report["error"] = error.what();
        try { save(); } catch (...) {}
        std::cerr << error.what() << std::endl; return 1;
    }
}
