#include "ninfer/engine.h"

#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

namespace {
ninfer::PromptInput input_for(std::size_t repetitions, bool image) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    if (image) {
        ninfer::MessagePart media;
        media.kind = ninfer::MessagePartKind::Media;
        media.media.kind = ninfer::MediaKind::Image;
        media.media.media_type = "image/x-portable-pixmap";
        media.media.source_name = "red-square.ppm";
        const std::string header = "P6\n128 128\n255\n";
        media.media.bytes.assign(header.begin(), header.end());
        for (int y = 0; y < 128; ++y) {
            for (int x = 0; x < 128; ++x) {
                const bool red = x >= 32 && x < 96 && y >= 32 && y < 96;
                media.media.bytes.insert(media.media.bytes.end(), {255, std::uint8_t(red ? 0 : 255),
                                                                       std::uint8_t(red ? 0 : 255)});
            }
        }
        message.parts.push_back(std::move(media));
    }
    std::string text;
    text.reserve(repetitions * 2 + 200);
    for (std::size_t i = 0; i < repetitions; ++i) { text += " a"; }
    text += image ? "\nDescribe the image's shape, foreground color and background color."
                  : "\nWrite a detailed explanation of how a compiler works.";
    message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
    input.messages.push_back(std::move(message));
    return input;
}
} // namespace

int main(int argc, char** argv) {
    const char* path = std::getenv("NINFER_QUASAR_ARTIFACT");
    if (!path || !*path) { std::cout << "SKIP: set NINFER_QUASAR_ARTIFACT\n"; return 77; }
    try {
        bool cpu = false, image = false, eager = false, vision = true;
        std::size_t wanted = 85000;
        std::uint32_t output_tokens = 129;
        std::optional<int> vision_gpu;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--cpu") { cpu = true; }
            else if (flag == "--image") { image = true; }
            else if (flag == "--eager") { eager = true; }
            else if (flag == "--no-vision") { vision = false; }
            else if (flag == "--vision-gpu" && i + 1 < argc) {
                const std::string value = argv[++i];
                int ordinal = 0;
                const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), ordinal);
                if (error != std::errc{} || end != value.data() + value.size() || ordinal < 0) {
                    throw std::invalid_argument("--vision-gpu requires a non-negative CUDA ordinal");
                }
                vision_gpu = ordinal;
            }
            else if (flag == "--prompt-tokens" && i + 1 < argc) { wanted = std::stoul(argv[++i]); }
            else if (flag == "--output-tokens" && i + 1 < argc) { output_tokens = std::stoul(argv[++i]); }
            else { throw std::invalid_argument("unknown Vision qualification option: " + flag); }
        }
        if ((!vision && (cpu || image || vision_gpu)) || (cpu && vision_gpu) ||
            (vision_gpu && (*vision_gpu == 0 || *vision_gpu == 1)) ||
            wanted < 256 || wanted > 85000 || output_tokens < 2) {
            throw std::invalid_argument("invalid Vision qualification dimensions/options");
        }
        ninfer::EngineOptions options;
        options.artifact_path = path;
        options.tp = 2;
        options.devices = {0, 1};
        options.max_context = static_cast<std::uint32_t>(wanted + 1024);
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.prefix_cache_tokens = 0;
        options.prefill_chunk = 1024;
        options.enable_vision = vision;
        options.vision_device = cpu ? ninfer::VisionDevice::Cpu : ninfer::VisionDevice::Gpu;
        options.vision_gpu = vision_gpu;
        options.vision_max_tokens = 4096;
        options.volta_long_attention = true;
        options.use_cuda_graph = !eager;
        options.speculative = {ninfer::SpeculativeBackend::Mtp, 4, ninfer::ProposalHead::Optimized};
        ninfer::Engine engine(options);
        auto repetitions = wanted - 256;
        for (int attempt = 0; attempt < 4; ++attempt) {
            const auto count = engine.count_tokens(input_for(repetitions, image));
            if (count == wanted) { break; }
            const auto adjusted = static_cast<std::int64_t>(repetitions) +
                                  static_cast<std::int64_t>(wanted) - count;
            if (adjusted < 0) { throw std::runtime_error("image is larger than requested token count"); }
            repetitions = static_cast<std::size_t>(adjusted);
        }
        auto prompt = engine.prepare(input_for(repetitions, image));
        if (prompt.summary().prompt_tokens != wanted || prompt.summary().has_media != image) {
            throw std::runtime_error("qualified prompt occupancy/modality mismatch");
        }
        ninfer::RequestOptions request;
        request.execution.sampling.temperature = 0;
        request.execution.sampling.presence_penalty = 0;
        request.execution.requested_output_tokens = output_tokens;
        request.execution.allow_prefix_reuse = false;
        request.stop.include_model_defaults = false;
        const auto result = engine.generate(std::move(prompt), request);
        if (result.generated_token_ids.size() != output_tokens || result.reused_prompt_tokens != 0 ||
            result.speculative.draft_window != 4 || (image && result.timings.vision_seconds <= 0)) {
            throw std::runtime_error("Vision qualification did not complete cold MTP4 inference");
        }
        const auto& t = result.timings;
        const auto memory = engine.memory_summary();
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        std::cout << "{\"cpu\":" << (cpu ? "true" : "false") << ",\"vision\":" << (vision ? "true" : "false")
                  << ",\"image\":" << (image ? "true" : "false") << ",\"graphs\":" << (!eager ? "true" : "false")
                  << ",\"prompt_tokens\":" << wanted << ",\"output_tokens\":" << output_tokens
                  << ",\"vision_seconds\":" << t.vision_seconds << ",\"prefill_seconds\":" << t.prefill_seconds
                  << ",\"first_token_seconds\":" << t.first_token_seconds << ",\"total_seconds\":" << t.total_seconds
                  << ",\"committed_wall_tps\":" << (output_tokens - 1) / (t.total_seconds - t.first_token_seconds)
                  << ",\"weights_gpu_bytes\":" << memory.weights.used_bytes << ",\"max_rss_kib\":" << usage.ru_maxrss
                  << ",\"drafted\":" << result.speculative.drafted_tokens << ",\"accepted\":" << result.speculative.accepted_tokens
                  << ",\"token_ids\":[";
        for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
            if (i) { std::cout << ','; }
            std::cout << result.generated_token_ids[i];
        }
        std::cout << "]}\n";
        std::cerr << "Vision long-context public Engine PASS\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}
