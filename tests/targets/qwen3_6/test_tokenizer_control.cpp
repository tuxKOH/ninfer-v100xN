// Regression for #6: interrupt actual in-progress BPE work, including the
// retrieval query, count-tokens and the shared multimodal text encoder.
#include <ninfer/targets/qwen3_6/frontend.h>

#include "targets/qwen3_6/impl/frontend/processor.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
namespace fi = ninfer::targets::qwen3_6::frontend_internal;

ninfer::targets::qwen3_6::FrontendResources resources() {
    ninfer::targets::qwen3_6::FrontendResources out;
    Json vocab = {{"x", 0}, {"xx", 1}, {"Ċ", 10}, {"Ġ", 32}};
    for (int c = 33; c <= 126; ++c) {
        if (c != 'x') { vocab[std::string(1, static_cast<char>(c))] = 1000 + c; }
    }
    out.tokenizer_json = Json{
        {"model", {{"type", "BPE"}, {"vocab", std::move(vocab)},
                   {"merges", Json::array({"x x"})}}},
        {"added_tokens", Json::array()}}.dump();
    out.generation_config_json = R"({"eos_token_id":0})";
    std::ifstream source(NINFER_SOURCE_DIR "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja");
    out.chat_template_jinja = std::string(std::istreambuf_iterator<char>(source), {});
    if (!out.chat_template_jinja.empty() && out.chat_template_jinja.back() == '\n') {
        out.chat_template_jinja.pop_back();
    }
    out.tokenizer_config_json = Json{{"added_tokens_decoder", Json::object()},
        {"add_bos_token", false}, {"add_prefix_space", false},
        {"pad_token", "<|endoftext|>"}, {"chat_template", out.chat_template_jinja}}.dump();
    out.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    out.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return out;
}

ninfer::PromptInput prompt(std::size_t length) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    ninfer::MessagePart text;
    text.kind = ninfer::MessagePartKind::Text;
    text.text = std::string(length, 'x');
    message.parts.push_back(std::move(text));
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.add_generation_prompt = false;
    input.options.enable_thinking = false;
    return input;
}

int interrupted(const char* label, const std::function<void(const ninfer::PreparationControl&)>& run,
                bool cancellation) {
    std::atomic<bool> cancelled = false;
    ninfer::PreparationControl control;
    const auto start = Clock::now();
    std::jthread cancel_thread;
    if (cancellation) {
        control.cancellation = ninfer::CancellationView([&] { return cancelled.load(); });
        cancel_thread = std::jthread([&] {
            std::this_thread::sleep_for(50ms);
            cancelled = true;
        });
    } else {
        control.deadline = start + 50ms;
    }
    try {
        run(control);
        std::cerr << label << " completed instead of being interrupted\n";
        return 1;
    } catch (const ninfer::RequestError& error) {
        const auto expected = cancellation ? ninfer::RequestErrorKind::Cancelled
                                           : ninfer::RequestErrorKind::QueueTimeout;
        if (error.kind() != expected || Clock::now() - start > 2s ||
            std::string(error.what()).find("tokenization") == std::string::npos) {
            std::cerr << label << ": wrong or late interruption: " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
} // namespace

int main() {
    const auto owned = resources();
    const fi::Tokenizer tokenizer({owned.tokenizer_json, owned.tokenizer_config_json,
                                    owned.generation_config_json});
    const auto frontend = ninfer::targets::qwen3_6::FrontendTestAccess::create_component(owned, false);
    int failures = 0;
    // Equal-rank ties still merge left to right, with/without special-token parsing.
    for (const bool added : {false, true}) {
        ninfer::PreparationControl live{.deadline = Clock::now() + 2s};
        const auto expected = std::vector<int>{1, 1, 0};
        if (tokenizer.encode("xxxxx", {.parse_added_tokens = added}) != expected ||
            tokenizer.encode("xxxxx", {.parse_added_tokens = added, .control = live}) != expected) {
            std::cerr << "control changed BPE output\n";
            ++failures;
        }
        for (bool cancel : {false, true}) {
            failures += interrupted("tokenizer BPE", [&](const auto& control) {
                (void)tokenizer.encode(std::string(20000, 'x'),
                                      {.parse_added_tokens = added, .control = control});
            }, cancel);
        }
    }
    for (const auto length : {16000U, 20000U}) {
        failures += interrupted("prepare (retrieval/rendered chat)", [&](const auto& control) {
            (void)frontend.prepare(prompt(length), control);
        }, false);
    }
    failures += interrupted("count_tokens", [&](const auto& control) {
        (void)frontend.count_tokens(prompt(20000), control);
    }, false);
    fi::RenderedChat rendered;
    rendered.text = std::string(20000, 'x');
    failures += interrupted("processor rendered text", [&](const auto& control) {
        (void)fi::encode_rendered_chat(tokenizer, rendered, control);
    }, true);
    return failures == 0 ? 0 : 1;
}
