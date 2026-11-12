#pragma once

#include "serve/request.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
};

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length);

// Incrementally publishes text that is provably outside a possible Qwen
// <tool_call> suffix. Finish against the parser's terminal content so quoted
// markers in prose and malformed markup are published without losing bytes.
class ToolCallStreamFilter {
public:
    std::string feed(std::string_view text);
    std::string finish(std::string_view final_content);

    [[nodiscard]] std::size_t emitted_bytes() const noexcept { return emitted_bytes_; }

private:
    std::string pending_;
    std::size_t emitted_bytes_ = 0;
    bool saw_tool_marker_      = false;
    bool finished_             = false;
};

} // namespace ninfer::serve
