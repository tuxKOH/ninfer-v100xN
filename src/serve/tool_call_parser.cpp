#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

std::string trim_ascii(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(begin, end - begin));
}

std::string rtrim_ascii(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(0, end));
}

void skip_ws(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}

std::string new_tool_call_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "call_%016llx",
                  static_cast<unsigned long long>(dist(rng)));
    return std::string(buf.data());
}

constexpr std::string_view kToolOpen      = "<tool_call>";
constexpr std::string_view kToolClose     = "</tool_call>";
constexpr std::string_view kFunctionOpen  = "<function=";
constexpr std::string_view kFunctionClose = "</function>";
constexpr std::string_view kParamOpen     = "<parameter=";
constexpr std::string_view kParamClose    = "</parameter>";

bool is_call_open(std::string_view text, std::size_t pos) {
    if (!starts_with_at(text, pos, kToolOpen)) { return false; }
    pos += kToolOpen.size();
    skip_ws(text, pos);
    return starts_with_at(text, pos, kFunctionOpen);
}

bool consume_call_close(std::string_view text, std::size_t& pos) {
    std::size_t after = pos;
    if (!starts_with_at(text, after, kFunctionClose)) { return false; }
    after += kFunctionClose.size();
    skip_ws(text, after);
    if (!starts_with_at(text, after, kToolClose)) { return false; }
    pos = after + kToolClose.size();
    return true;
}

struct ParseContext {
    // A malformed nested example can be encountered from several enclosing values.
    // Remember its result instead of repeatedly parsing the same suffix.
    std::unordered_map<std::size_t, std::size_t> nested_ends;
    bool nesting_limit = false;
};

bool parse_call_at(std::string_view text, std::size_t& pos, std::size_t max_name_length,
                   ToolCall& out, ParseContext& context, unsigned nesting = 0) {
    if (nesting == 64) {
        context.nesting_limit = true;
        return false;
    }
    if (!is_call_open(text, pos)) { return false; }
    pos += kToolOpen.size();
    skip_ws(text, pos);
    const std::size_t name_begin = pos + kFunctionOpen.size();
    const std::size_t name_end   = text.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string name = std::string(text.substr(name_begin, name_end - name_begin));
    if (!valid_function_name(name, max_name_length)) { return false; }
    pos = name_end + 1;

    Json args = Json::object();
    for (;;) {
        skip_ws(text, pos);
        if (consume_call_close(text, pos)) { break; }
        if (!starts_with_at(text, pos, kParamOpen)) { return false; }
        const std::size_t key_begin = pos + kParamOpen.size();
        const std::size_t key_end   = text.find('>', key_begin);
        if (key_end == std::string_view::npos || key_end == key_begin) { return false; }
        const std::string key(text.substr(key_begin, key_end - key_begin));
        const std::size_t value_begin = key_end + 1;
        pos = value_begin;
        for (;;) {
            pos = text.find('<', pos);
            if (pos == std::string_view::npos) { return false; }
            // Only a complete nested call is structural. A lone quoted opener
            // must not hide the actual parameter/function terminators.
            if (is_call_open(text, pos)) {
                auto cached = context.nested_ends.find(pos);
                if (cached == context.nested_ends.end()) {
                    std::size_t end = pos;
                    ToolCall nested;
                    if (!parse_call_at(text, end, max_name_length, nested, context, nesting + 1)) {
                        if (context.nesting_limit) { return false; }
                        end = std::string_view::npos;
                    }
                    cached = context.nested_ends.emplace(pos, end).first;
                }
                const std::size_t nested_end = cached->second;
                if (nested_end != std::string_view::npos) {
                    pos = nested_end;
                    continue;
                }
            }
            if (starts_with_at(text, pos, kParamClose)) {
                std::size_t after = pos + kParamClose.size();
                skip_ws(text, after);
                std::size_t call_end = after;
                // Tag spellings inside code/regex strings are data unless the
                // surrounding grammar can continue with a sibling or call end.
                if (starts_with_at(text, after, kParamOpen) ||
                    consume_call_close(text, call_end)) {
                    const std::string raw = trim_ascii(text.substr(value_begin, pos - value_begin));
                    Json parsed = Json::parse(raw, nullptr, false);
                    args[key] = parsed.is_discarded() ? Json(raw) : std::move(parsed);
                    pos = after;
                    break;
                }
            }
            ++pos;
        }
    }

    out.name           = name;
    out.arguments_json = args.dump();
    return true;
}

ParsedToolCallOutput fallback(const std::string& text) {
    ParsedToolCallOutput out;
    out.content = text;
    return out;
}

} // namespace

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length) {
    std::size_t first = text.find(kToolOpen);
    while (first != std::string::npos && !is_call_open(text, first)) {
        first = text.find(kToolOpen, first + kToolOpen.size());
    }
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));
    ParseContext context;

    std::size_t pos = first;
    while (pos < text.size()) {
        skip_ws(text, pos);
        if (pos >= text.size()) { break; }
        ToolCall call;
        if (!parse_call_at(text, pos, max_tool_name_length, call, context)) {
            return fallback(text);
        }
        call.id = new_tool_call_id();
        out.tool_calls.push_back(std::move(call));
    }

    if (out.tool_calls.empty()) { return fallback(text); }
    out.is_tool_call_response = true;
    return out;
}

std::string ToolCallStreamFilter::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    if (text.empty()) { return {}; }
    if (saw_tool_marker_) { return {}; }

    pending_.append(text);
    const std::size_t marker = pending_.find(kToolOpen);
    if (marker != std::string::npos) {
        std::size_t safe_end = marker;
        while (safe_end != 0 &&
               std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
            --safe_end;
        }
        std::string visible = pending_.substr(0, safe_end);
        pending_.clear();
        saw_tool_marker_ = true;
        emitted_bytes_ += visible.size();
        return visible;
    }

    const std::size_t prefix = longest_suffix_prefix(pending_, kToolOpen);
    std::size_t safe_end     = pending_.size() - prefix;
    while (safe_end != 0 && std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
        --safe_end;
    }
    std::string visible = pending_.substr(0, safe_end);
    pending_.erase(0, safe_end);
    emitted_bytes_ += visible.size();
    return visible;
}

std::string ToolCallStreamFilter::finish(std::string_view final_content) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    finished_ = true;
    if (emitted_bytes_ > final_content.size()) {
        throw std::logic_error("streamed tool-call prefix exceeds terminal content");
    }
    // The first literal marker may precede the real call (e.g. a regex in prose).
    // Publish all remaining parsed content, or all raw content on parse failure.
    std::string tail(final_content.substr(emitted_bytes_));
    pending_.clear();
    emitted_bytes_ += tail.size();
    return tail;
}

} // namespace ninfer::serve
