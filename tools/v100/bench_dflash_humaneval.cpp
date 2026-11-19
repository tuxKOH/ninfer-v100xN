// HumanEval code-generation throughput through the public Engine. Does not execute model code.
#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <cuda_profiler_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using Json = nlohmann::json;

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            throw std::invalid_argument("usage: bench ARTIFACT HumanEval.jsonl REPORT.json "
                "[--count N] [--offset N] [--repeats N] [--max-output N] [--eager 0|1] "
                "[--fixture REPORT.json] [--spec dflash|mtp] [--proposal-head full|optimized] "
                "[--long-attention 0|1] [--fast-decode 0|1]");
        }
        unsigned count = 32, offset = 0, repeats = 2, max_output = 2048;
        bool eager = false, long_attention = false, fast_decode = false;
        std::string backend = "dflash", proposal_head = "full";
        std::string fixture;
        for (int i = 4; i < argc; i += 2) {
            if (i + 1 == argc) { throw std::invalid_argument("missing option value"); }
            const std::string key = argv[i];
            if (key == "--fixture") { fixture = argv[i + 1]; continue; }
            if (key == "--spec") {
                backend = argv[i + 1];
                if (backend != "dflash" && backend != "mtp") { throw std::invalid_argument("invalid spec backend"); }
                continue;
            }
            if (key == "--proposal-head") {
                proposal_head = argv[i + 1];
                if (proposal_head != "full" && proposal_head != "optimized") { throw std::invalid_argument("invalid proposal head"); }
                continue;
            }
            std::size_t used = 0;
            const auto value = std::stoul(argv[i + 1], &used);
            if (used != std::string(argv[i + 1]).size()) { throw std::invalid_argument("invalid integer"); }
            if (key == "--count") { count = value; }
            else if (key == "--offset") { offset = value; }
            else if (key == "--repeats") { repeats = value; }
            else if (key == "--max-output") { max_output = value; }
            else if (key == "--eager" && value <= 1) { eager = value; }
            else if (key == "--long-attention" && value <= 1) { long_attention = value; }
            else if (key == "--fast-decode" && value <= 1) { fast_decode = value; }
            else { throw std::invalid_argument("unknown option: " + key); }
        }
        if (!count || !repeats || max_output < 2) { throw std::invalid_argument("empty workload"); }
        Json cases = Json::array();
        std::ifstream data(fixture.empty() ? argv[2] : fixture);
        if (!data) { throw std::runtime_error("cannot open dataset/fixture"); }
        if (!fixture.empty()) {
            Json saved; data >> saved;
            cases = saved.at("cases");
        } else {
            std::string line;
            unsigned index = 0;
            while (std::getline(data, line)) {
                if (index++ < offset) { continue; }
                const auto row = Json::parse(line);
                cases.push_back({{"task_id", row.at("task_id")}, {"prompt", row.at("prompt")}});
                if (cases.size() == count) { break; }
            }
            if (cases.size() != count) { throw std::runtime_error("dataset too short"); }
        }
        ninfer::EngineOptions options;
        options.artifact_path = argv[1]; options.tp = 2; options.devices = {0, 1};
        options.max_context = 98304;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
        options.prefill_chunk = 1024;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.use_cuda_graph = !eager;
        options.prefix_cache_tokens = 0;
        options.speculative.backend = backend == "dflash"
            ? ninfer::SpeculativeBackend::DFlash : ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = backend == "dflash" ? 7 : 4;
        options.speculative.proposal_head = proposal_head == "full"
            ? ninfer::ProposalHead::Full : ninfer::ProposalHead::Optimized;
        options.volta_long_attention = long_attention;
        options.volta_fast_decode = fast_decode;
        ninfer::Engine engine(options);
        ninfer::RequestOptions request;
        request.execution.allow_prefix_reuse = false;
        request.execution.sampling.temperature = 0;
        request.execution.sampling.presence_penalty = 0;
        request.execution.sampling.frequency_penalty = 0;
        request.execution.requested_output_tokens = max_output;
        request.stop.include_model_defaults = true;
        for (auto& item : cases) {
            if (item.contains("prompt_ids")) { continue; }
            ninfer::PromptInput input;
            input.options.enable_thinking = false;
            ninfer::ChatMessage user;
            user.parts.push_back({.text = "Complete the following Python function. Return runnable Python code only.\n\n" +
                item.at("prompt").get<std::string>()});
            input.messages.push_back(std::move(user));
            item["prompt_ids"] = engine.prepare(std::move(input)).debug_token_ids();
        }
        auto warm = request;
        warm.execution.requested_output_tokens = 33;
        (void)engine.generate(engine.prepare_tokens(cases[0].at("prompt_ids").get<std::vector<ninfer::TokenId>>(), false), warm);
        Json report{{"artifact", argv[1]}, {"backend", backend}, {"drafts", options.speculative.draft_tokens}, {"tp", 2},
            {"capacity", 98304}, {"chunk", 1024}, {"kv", "int8-group64"}, {"graphs", !eager},
            {"proposal_head", proposal_head}, {"volta_long_attention", long_attention},
            {"volta_fast_decode", fast_decode},
            {"greedy", true}, {"thinking", false}, {"prefix_reuse", false},
            {"max_output", max_output}, {"repeats", repeats}, {"cases", cases}, {"runs", Json::array()}};
        const auto save = [&] { std::ofstream out(argv[3]); out << report.dump(2) << '\n'; };
        save();
        const bool profile = std::getenv("NINFER_BENCH_PROFILE") != nullptr;
        if (profile && cudaProfilerStart() != cudaSuccess) { throw std::runtime_error("profiler start failed"); }
        for (unsigned rep = 0; rep < repeats; ++rep) {
            for (const auto& item : cases) {
                auto result = engine.generate(engine.prepare_tokens(item.at("prompt_ids").get<std::vector<ninfer::TokenId>>(), false), request);
                const double seconds = result.timings.total_seconds - result.timings.first_token_seconds;
                const auto committed = result.generated_token_ids.empty() ? 0 : result.generated_token_ids.size() - 1;
                Json row{{"task_id", item.at("task_id")}, {"repeat", rep},
                    {"prompt_tokens", result.prompt.prompt_tokens}, {"completion_tokens", result.generated_token_ids.size()},
                    {"committed_decode_tokens", committed}, {"decode_wall_seconds", seconds},
                    {"tok_s", seconds > 0 ? committed / seconds : 0},
                    {"prefill_seconds", result.timings.prefill_seconds},
                    {"ttft_seconds", result.timings.first_token_seconds}, {"finish_reason", static_cast<int>(result.finish_reason)},
                    {"ids", result.generated_token_ids}, {"content", result.content},
                    {"rounds", result.speculative.rounds}, {"drafted", result.speculative.drafted_tokens},
                    {"accepted", result.speculative.accepted_tokens}, {"accepted_per_position", result.speculative.accepted_per_position},
                    {"fallback_steps", result.speculative.fallback_steps}};
                std::cout << item.at("task_id") << " rep=" << rep << " tokens=" << result.generated_token_ids.size()
                          << " tok/s=" << row["tok_s"] << std::endl;
                report["runs"].push_back(std::move(row)); save();
            }
        }
        if (profile && cudaProfilerStop() != cudaSuccess) { throw std::runtime_error("profiler stop failed"); }
        double decode = 0, mean = 0;
        std::uint64_t tokens = 0, drafted = 0, accepted = 0;
        unsigned truncated = 0;
        for (const auto& row : report["runs"]) {
            decode += row.at("decode_wall_seconds").get<double>(); mean += row.at("tok_s").get<double>();
            tokens += row.at("committed_decode_tokens").get<std::uint64_t>();
            drafted += row.at("drafted").get<std::uint64_t>(); accepted += row.at("accepted").get<std::uint64_t>();
            truncated += row.at("completion_tokens").get<unsigned>() == max_output;
        }
        report["summary"] = {{"pooled_tok_s", tokens / decode}, {"request_mean_tok_s", mean / report["runs"].size()},
            {"acceptance", drafted ? double(accepted) / drafted : 0}, {"truncated_requests", truncated}};
        save(); std::cout << report["summary"].dump() << std::endl;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
