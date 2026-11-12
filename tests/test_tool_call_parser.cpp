#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

int test_single_call() {
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output("Calling weather.\n"
                                                   "<tool_call>\n"
                                                   "<function=get_weather>\n"
                                                   "<parameter=city>\nParis\n</parameter>\n"
                                                   "<parameter=days>\n2\n</parameter>\n"
                                                   "</function>\n"
                                                   "</tool_call>",
                                                   64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "single call parsed as tool response");
    failures += check(parsed.content == "Calling weather.", "content prefix trimmed");
    failures += check(parsed.tool_calls.size() == 1, "one parsed call");
    failures += check(parsed.tool_calls[0].id.rfind("call_", 0) == 0, "generated call id prefix");
    failures += check(parsed.tool_calls[0].name == "get_weather", "function name parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "string parameter parsed");
    failures += check(args.at("days") == 2, "number parameter parsed");
    return failures;
}

int test_multiple_calls_and_json_values() {
    const ninfer::serve::ParsedToolCallOutput parsed = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=first>\n"
        "<parameter=payload>\n{\"ok\":true,\"items\":[1,2]}\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=second>\n"
        "<parameter=value>\nplain text\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "multiple calls parsed as tool response");
    failures += check(parsed.tool_calls.size() == 2, "two parsed calls");
    failures += check(parsed.tool_calls[0].name == "first", "first call name");
    failures += check(parsed.tool_calls[1].name == "second", "second call name");
    const Json first = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(first.at("payload").at("ok") == true, "object parameter bool");
    failures += check(first.at("payload").at("items").at(1) == 2, "object parameter array");
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures += check(second.at("value") == "plain text", "plain text parameter string");
    return failures;
}

int test_malformed_falls_back_to_text() {
    const std::string text = "<tool_call>\n<function=get_weather>\n";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "malformed xml is not tool response");
    failures += check(parsed.content == text, "malformed xml preserved as text");
    failures += check(parsed.tool_calls.empty(), "malformed xml has no calls");
    return failures;
}

int test_suffix_after_tool_falls_back_to_text() {
    const std::string text = "<tool_call>\n"
                             "<function=get_weather>\n"
                             "<parameter=city>\nParis\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "extra answer";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "non-whitespace suffix falls back to text");
    failures += check(parsed.content == text, "suffix fallback preserves text");
    return failures;
}

int test_configured_name_limit() {
    const std::string name(128, 'a');
    const std::string text = "<tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";

    const ninfer::serve::ParsedToolCallOutput anthropic =
        ninfer::serve::parse_qwen_tool_call_output(text, 128);
    const ninfer::serve::ParsedToolCallOutput openai =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    const std::string too_long_text =
        "<tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";
    const ninfer::serve::ParsedToolCallOutput too_long =
        ninfer::serve::parse_qwen_tool_call_output(too_long_text, 128);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1 &&
                          anthropic.tool_calls[0].name == name,
                      "128-character name accepted with Anthropic limit");
    failures +=
        check(!openai.is_tool_call_response, "128-character name rejected with OpenAI limit");
    failures +=
        check(!too_long.is_tool_call_response, "129-character name rejected with Anthropic limit");
    return failures;
}

int test_incremental_filter_valid_tool() {
    ninfer::serve::ToolCallStreamFilter filter;
    std::string visible;
    visible += filter.feed("Calling weather.  \n<tool_");
    visible += filter.feed("call>\n<function=get_weather>");
    visible += filter.feed("\n</function>\n</tool_call>");
    visible += filter.finish("Calling weather.");
    int failures = 0;
    failures += check(visible == "Calling weather.",
                      "valid tool filter did not stream the trimmed content prefix");
    failures +=
        check(filter.emitted_bytes() == visible.size(), "valid tool filter byte count mismatch");
    return failures;
}

int test_incremental_filter_fallback() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    ninfer::serve::ToolCallStreamFilter malformed;
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    restored += malformed.finish(original);

    ninfer::serve::ToolCallStreamFilter normal;
    std::string ordinary;
    ordinary += normal.feed("ordinary text  ");
    ordinary += normal.finish("ordinary text  ");

    int failures = 0;
    failures += check(restored == original, "malformed tool filter fallback lost raw bytes");
    failures +=
        check(ordinary == "ordinary text  ", "ordinary filtered output lost trailing whitespace");
    return failures;
}

int check_stream(const std::string& raw, const std::string& content) {
    int failures = 0;
    // Include every byte boundary (also inside UTF-8 and each markup delimiter).
    for (std::size_t split = 0; split <= raw.size(); ++split) {
        ninfer::serve::ToolCallStreamFilter filter;
        std::string streamed = filter.feed(std::string_view(raw).substr(0, split));
        streamed += filter.feed(std::string_view(raw).substr(split));
        streamed += filter.finish(content);
        if (streamed != content || filter.emitted_bytes() != content.size()) {
            failures += fail("stream/terminal content differs at byte " + std::to_string(split));
            break;
        }
    }
    ninfer::serve::ToolCallStreamFilter filter;
    std::string streamed;
    for (const char byte : raw) { streamed += filter.feed(std::string_view(&byte, 1)); }
    streamed += filter.finish(content);
    failures += check(streamed == content, "one-byte stream differs from terminal content");
    return failures;
}

int test_literal_tags() {
    const std::vector<std::string> values{
        "b = \"start <tool_call> end </tool_call> here\"",
        "TXT_PARAM = re.compile(r\"<parameter=([a-zA-Z_]+)>(.*?)</parameter>\", re.S)",
        "unmatched tags: <tool_call> and <function=write> and </function> and </tool_call>",
        "a literal </parameter> followed by text, not a sibling",
        "<tool_call><function=example><parameter=value>nested</parameter></function></tool_call>",
        "before\n<tool_call><function=outer><parameter=x>"
        "<tool_call><function=inner></function></tool_call>"
        "</parameter></function></tool_call>\nafter",
        "{\"tags\":[\"</parameter>\",\"</function>\",\"</tool_call>\"],\"ok\":true}",
        "\"quoted JSON value with </parameter> inside\"",
        "多字节内容 <tool_call>保留</tool_call> ✅",
    };
    int failures = 0;
    for (const std::string& value : values) {
        const std::string raw = "<tool_call>\n<function=write>\n<parameter=content>\n" + value +
                                "\n</parameter>\n<parameter=path>/tmp/t.py</parameter>\n"
                                "</function>\n</tool_call>";
        const auto parsed = ninfer::serve::parse_qwen_tool_call_output(raw, 64);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                          "literal tags caused fallback: " + value);
        if (parsed.tool_calls.size() != 1) { continue; }
        const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
        const Json expected = Json::parse(value, nullptr, false);
        failures += check(args.at("content") == (expected.is_discarded() ? Json(value) : expected),
                          "literal-tag parameter changed: " + value);
        failures += check(args.at("path") == "/tmp/t.py" && args.size() == 2,
                          "literal-tag call lost sibling parameter");
        failures += check_stream(raw, parsed.content);
    }
    return failures;
}

int test_quoted_opener_before_call() {
    const std::string prefix =
        "格式说明：TXT_CALL = re.compile(r\"<tool_call>\\s*<function=([a-z_]+)>\")";
    const std::string raw = prefix + "\n<tool_call>\n<function=bash>\n"
                                    "<parameter=command>ls</parameter>\n"
                                    "</function>\n</tool_call>";
    const auto parsed = ninfer::serve::parse_qwen_tool_call_output(raw, 64);
    int failures = check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                         "prose quoting opener hid actual call");
    failures += check(parsed.content == prefix, "prose with quoted opener changed");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls[0].name == "bash" &&
                              Json::parse(parsed.tool_calls[0].arguments_json).at("command") == "ls",
                          "actual call after quoted opener changed");
    }
    failures += check_stream(raw, parsed.content);
    return failures;
}

int test_malformed_and_ordinary_streams() {
    const std::vector<std::string> values{
        "ordinary text  \n",
        "a literal <tool_call> in prose  \n",
        "<tool_call><function=broken><parameter=x>truncated",
        "<tool_call><function=broken>stray text</function></tool_call>",
        "<tool_call><function=broken><parameter=x>missing close</function></tool_call>",
        "<tool_call><function=broken><parameter=>value</parameter></function></tool_call>",
        "<tool_call><function=ok></function></tool_call>\ntrailing text",
        "<tool_call><function=ok></function></tool_call>\n<tool_call><function=broken>",
    };
    int failures = 0;
    for (const auto& raw : values) {
        const auto parsed = ninfer::serve::parse_qwen_tool_call_output(raw, 64);
        failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty() &&
                              parsed.content == raw,
                          "malformed/ordinary content was not preserved: " + raw);
        failures += check_stream(raw, parsed.content);
    }
    return failures;
}

int test_unclosed_nested_calls() {
    for (const int depth : {32, 80}) {
        std::string raw;
        for (int i = 0; i < depth; ++i) {
            raw += "<tool_call><function=write><parameter=content>";
        }
        raw += "unclosed examples</parameter></function></tool_call>";
        const auto parsed = ninfer::serve::parse_qwen_tool_call_output(raw, 64);
        if (parsed.is_tool_call_response || parsed.content != raw || !parsed.tool_calls.empty()) {
            return fail("unclosed nested examples did not fall back to raw text");
        }
    }
    return 0;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_single_call();
    failures += test_multiple_calls_and_json_values();
    failures += test_malformed_falls_back_to_text();
    failures += test_suffix_after_tool_falls_back_to_text();
    failures += test_configured_name_limit();
    failures += test_incremental_filter_valid_tool();
    failures += test_incremental_filter_fallback();
    failures += test_literal_tags();
    failures += test_quoted_opener_before_call();
    failures += test_malformed_and_ordinary_streams();
    failures += test_unclosed_nested_calls();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
