// Fixed-token QUASAR/MTP4 request groups through the public Engine.
#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
int main(int argc, char** argv) {
    try {
        if (argc != 7 && argc != 8) { throw std::invalid_argument("usage: bench ARTIFACT CORPUS INPUT_TOKENS CONCURRENCY OUTPUT_TOKENS REPORT.json [PREFILL_DECODE_ROUNDS]"); }
        const unsigned length = std::stoul(argv[3]), concurrency = std::stoul(argv[4]), output = std::stoul(argv[5]);
        const unsigned decode_rounds = argc == 8 ? std::stoul(argv[7]) : 1;
        if (length < 1 || length > 85000 || concurrency < 1 || concurrency > 8 || output < 2) {
            throw std::invalid_argument("invalid workload");
        }
        if (decode_rounds < 1 || decode_rounds > 64) { throw std::invalid_argument("invalid decode burst"); }
        std::vector<ninfer::TokenId> ids;
        std::ifstream corpus(argv[2]);
        ninfer::TokenId token;
        while (corpus >> token && ids.size() < length) { ids.push_back(token); }
        if (ids.size() != length) { throw std::runtime_error("corpus too short"); }
        ninfer::EngineOptions options;
        options.artifact_path = argv[1]; options.tp = 2; options.devices = {0, 1};
        options.max_context = 180000;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(180000);
        options.prefill_chunk = 1024;
        options.prefill_decode_rounds = decode_rounds;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.max_concurrency = concurrency;
        options.use_cuda_graph = std::getenv("NINFER_BENCH_EAGER") == nullptr;
        options.pending_timeout_ms = 1800000; // Eight cold 85K prefills can queue for minutes.
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 4;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        options.volta_long_attention = true; options.volta_fast_decode = true;
        options.prefix_cache_tokens = 0; // Cold independent requests; no historical RAM transfers.
        ninfer::Engine engine(options);
        const auto settled_stats = [&](std::uint64_t expected) {
            const auto deadline = Clock::now() + std::chrono::seconds(5);
            for (;;) {
                const auto stats = engine.runtime_stats();
                if (stats.running_requests == 0 && stats.committed_decode_tokens >= expected) {
                    return stats;
                }
                if (Clock::now() >= deadline) { throw std::runtime_error("final decode counters did not settle"); }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        };
        ninfer::RequestOptions request;
        request.execution.allow_prefix_reuse = false;
        request.execution.requested_output_tokens = output;
        request.execution.sampling.temperature = 0.0F;
        request.execution.sampling.presence_penalty = 0.0F;
        request.execution.sampling.frequency_penalty = 0.0F;
        request.stop.include_model_defaults = false;
        // Warm the graph execution path outside the group measurement.
        auto warm = request; warm.execution.requested_output_tokens = 32;
        (void)engine.generate(engine.prepare_tokens({198, 220, 198, 220}), warm);
        std::vector<ninfer::PreparedPrompt> prompts;
        for (unsigned i = 0; i < concurrency; ++i) { prompts.push_back(engine.prepare_tokens(ids)); }
        Json rows = Json::array(), samples = Json::array();
        const auto before = settled_stats(31);
        const auto started = Clock::now();
        const bool profile = std::getenv("NINFER_BENCH_PROFILE") != nullptr;
        if (profile && cudaProfilerStart() != cudaSuccess) {
            throw std::runtime_error("cannot start CUDA profiler");
        }
        std::vector<ninfer::GenerationHandle> handles;
        std::vector<double> submitted;
        for (auto& prompt : prompts) {
            submitted.push_back(std::chrono::duration<double>(Clock::now() - started).count());
            handles.push_back(engine.submit(std::move(prompt), request));
        }
        std::jthread sampler([&](std::stop_token stop) {
            while (!stop.stop_requested()) {
                const auto stats = engine.runtime_stats();
                samples.push_back({{"seconds", std::chrono::duration<double>(Clock::now() - started).count()},
                    {"running", stats.running_requests}, {"prefilling", stats.prefilling_requests},
                    {"decode_ready", stats.decode_ready_requests}, {"waiting", stats.waiting_requests},
                    {"computed_prefill_tokens", stats.computed_prefill_tokens - before.computed_prefill_tokens},
                    {"committed_decode_tokens", stats.committed_decode_tokens - before.committed_decode_tokens}});
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
        double all_live_start = 0, all_live_end = std::numeric_limits<double>::infinity();
        unsigned request_index = 0;
        for (auto& handle : handles) {
            const auto result = handle.wait();
            if (result.generated_token_ids.size() != output) { throw std::runtime_error("incomplete output"); }
            const double origin = submitted[request_index++] - result.timings.prepare_seconds;
            all_live_start = std::max(all_live_start, origin + result.timings.first_token_seconds);
            all_live_end = std::min(all_live_end, origin + result.timings.total_seconds);
            rows.push_back({{"prompt_tokens", result.prompt.prompt_tokens},
                {"completion_tokens", result.generated_token_ids.size()}, {"cached_tokens", result.reused_prompt_tokens},
                {"prefill_seconds", result.timings.prefill_seconds}, {"decode_seconds", result.timings.decode_seconds},
                {"first_token_seconds", result.timings.first_token_seconds}, {"total_seconds", result.timings.total_seconds},
                {"stream_wall_tok_s", double(output - 1) / (result.timings.total_seconds - result.timings.first_token_seconds)},
                {"generated_token_ids", result.generated_token_ids},
                {"drafted", result.speculative.drafted_tokens}, {"accepted", result.speculative.accepted_tokens}});
        }
        const double makespan = std::chrono::duration<double>(Clock::now() - started).count();
        if (profile && cudaProfilerStop() != cudaSuccess) {
            throw std::runtime_error("cannot stop CUDA profiler");
        }
        sampler.request_stop(); sampler.join();
        const auto after = settled_stats(before.committed_decode_tokens + std::uint64_t(concurrency) * (output - 1));
        const auto committed = after.committed_decode_tokens - before.committed_decode_tokens;
        const double decode_wall = after.decode_wall_seconds - before.decode_wall_seconds;
        const auto rounds = after.decode_rounds - before.decode_rounds;
        const auto row_rounds = after.decode_row_rounds - before.decode_row_rounds;
        if (committed != std::uint64_t(concurrency) * (output - 1) || decode_wall <= 0 || rounds == 0) {
            throw std::runtime_error("invalid committed-token timing counters");
        }
        unsigned peak_running = 0, peak_ready = 0;
        for (const auto& sample : samples) {
            peak_running = std::max(peak_running, sample.at("running").get<unsigned>());
            peak_ready = std::max(peak_ready, sample.at("decode_ready").get<unsigned>());
        }
        // Only intervals whose starting sample has both phases runnable. Counts are committed
        // decode tokens; wall time includes intervening prefill stalls. Boundaries have nominal
        // 10ms sampling uncertainty. This is separate from decode-kernel/round throughput below.
        double mixed_wall = 0;
        std::uint64_t mixed_committed = 0;
        for (std::size_t i = 1; i < samples.size(); ++i) {
            const auto& previous = samples[i - 1];
            const auto& current = samples[i];
            if (previous.at("prefilling").get<unsigned>() > 0 &&
                previous.at("decode_ready").get<unsigned>() > 0) {
                mixed_wall += current.at("seconds").get<double>() - previous.at("seconds").get<double>();
                mixed_committed += current.at("committed_decode_tokens").get<std::uint64_t>() -
                    previous.at("committed_decode_tokens").get<std::uint64_t>();
            }
        }
        // Complete 10ms sample intervals inside the common decode lifetime. Wall time includes
        // intervening prefill; no window is reported if requests never all reach decode together.
        const Json* live_first = nullptr;
        const Json* live_last = nullptr;
        for (const auto& sample : samples) {
            const double seconds = sample.at("seconds").get<double>();
            if (seconds >= all_live_start && seconds < all_live_end) {
                if (!live_first) { live_first = &sample; }
                live_last = &sample;
            }
        }
        const double live_wall = live_first && live_last
            ? live_last->at("seconds").get<double>() - live_first->at("seconds").get<double>() : 0;
        const std::uint64_t live_tokens = live_wall > 0
            ? live_last->at("committed_decode_tokens").get<std::uint64_t>() -
              live_first->at("committed_decode_tokens").get<std::uint64_t>() : 0;
        Json report{{"artifact_type", "ninfer_v100_long_concurrency"}, {"input_tokens", length},
            {"concurrency", concurrency}, {"output_tokens_per_request", output},
            {"tp", 2}, {"mtp_drafts", 4}, {"graphs", options.use_cuda_graph}, {"kv_capacity", 180000},
            {"prefill_chunk", 1024}, {"prefill_decode_rounds", decode_rounds},
            {"long_attention", true}, {"fast_decode", true},
            {"prefix_reuse", false}, {"committed_decode_tokens", committed},
            {"decode_wall_seconds", decode_wall}, {"total_decode_tok_s", committed / decode_wall},
            {"group_makespan_seconds", makespan}, {"group_output_tok_s", double(concurrency * output) / makespan},
            {"decode_rounds", rounds}, {"mean_decode_batch", double(row_rounds) / rounds},
            {"all_live_sampled_wall_seconds", live_wall}, {"all_live_sampled_tokens", live_tokens},
            {"all_live_sampled_tok_s", live_wall > 0 ? Json(live_tokens / live_wall) : Json(nullptr)},
            {"mixed_wall_seconds", mixed_wall}, {"mixed_committed_decode_tokens", mixed_committed},
            {"mixed_wall_decode_tok_s", mixed_wall > 0 ? Json(mixed_committed / mixed_wall) : Json(nullptr)},
            {"peak_running", peak_running}, {"peak_decode_ready", peak_ready},
            {"requests", rows}, {"scheduler_samples", samples}};
        const std::string temporary = std::string(argv[6]) + ".tmp";
        { std::ofstream stream(temporary); stream << report.dump(2) << '\n'; if (!stream) { throw std::runtime_error("cannot write report"); } }
        std::filesystem::rename(temporary, argv[6]);
        report.erase("scheduler_samples"); report.erase("requests");
        std::cout << report.dump() << std::endl;
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
