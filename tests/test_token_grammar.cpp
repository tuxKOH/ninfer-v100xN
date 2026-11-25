// Structured-output matcher: draft-window planning and committed-token transitions over a
// miniature vocabulary.
#include "runtime/generation/token_grammar.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::runtime;

namespace {

enum Token : TokenId { Open, Close, Quote, Letter, Colon, Digit, Comma, Space, Stop, Special, Think };

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

TokenGrammarCompiler make_compiler() {
    TokenGrammarVocabulary vocabulary;
    vocabulary.token_bytes    = {"{", "}", "\"", "a", ":", "1", ",", " ", "", "", ""};
    vocabulary.stop_token_ids = {Stop};
    vocabulary.reasoning_end  = Think;
    return TokenGrammarCompiler(vocabulary);
}

struct Window {
    std::vector<std::uint32_t> rows;
    std::vector<std::int32_t> constrained;
    std::uint32_t licensed = 0;

    [[nodiscard]] bool allows(std::size_t column, TokenId token) const {
        return ((rows[column] >> token) & 1u) != 0u; // one word per row
    }
};

Window plan(TokenGrammarMatcher& matcher, std::vector<TokenId> drafts) {
    Window window;
    window.rows.assign((drafts.size() + 1) * static_cast<std::size_t>(matcher.words()), 0u);
    window.constrained.assign(drafts.size() + 1, -1);
    window.licensed = matcher.plan(drafts, window.rows, window.constrained);
    return window;
}

void test_json_object(const TokenGrammarCompiler& compiler) {
    TokenGrammarMatcher matcher(compiler.compile({.kind = StructuredOutputKind::JsonObject}),
                                false);
    check(matcher.words() == 1, "eleven tokens need one mask word");

    const Window first = plan(matcher, {});
    check(first.constrained == std::vector<std::int32_t>{1}, "first column is constrained");
    check(first.allows(0, Open) && !first.allows(0, Letter) && !first.allows(0, Stop) &&
              !first.allows(0, Special),
          "an object starts with '{'");

    // "{" "}" form a complete object; the third draft is invalid and nothing after it verifies.
    const Window window = plan(matcher, {Open, Close, Letter, Digit});
    check(window.licensed == 2, "drafts after the first rejected draft are not licensed");
    check(window.constrained == std::vector<std::int32_t>{1, 1, 1, 0, 0},
          "constrained columns end after the first rejected draft");
    check(window.allows(1, Close) && window.allows(1, Quote) && !window.allows(1, Open),
          "after '{' the object closes or opens a key");
    check(window.allows(2, Stop) && !window.allows(2, Letter), "a complete object only stops");

    // Planning never moves committed state.
    const Window again = plan(matcher, {});
    check(again.rows == first.rows, "plan rolls back speculative drafts");

    matcher.commit(std::array<TokenId, 2>{Open, Close});
    const Window complete = plan(matcher, {Stop, Letter});
    check(complete.allows(0, Stop) && !complete.allows(0, Open), "committed object only stops");
    check(complete.licensed == 2 && complete.constrained == std::vector<std::int32_t>{1, 0, 0},
          "no column after an accepted stop token is constrained");

    bool rejected = false;
    try {
        matcher.commit(std::array<TokenId, 1>{Letter});
    } catch (const std::logic_error&) { rejected = true; }
    check(rejected, "a committed grammar violation is an invariant failure");

    matcher.commit(std::array<TokenId, 1>{Stop});
    check(plan(matcher, {}).constrained == std::vector<std::int32_t>{0},
          "a terminated grammar constrains nothing");
}

void test_reasoning(const TokenGrammarCompiler& compiler) {
    TokenGrammarMatcher matcher(compiler.compile({.kind = StructuredOutputKind::JsonObject}),
                                true);
    const Window window = plan(matcher, {Letter, Think, Open});
    check(window.licensed == 3, "reasoning drafts are unconstrained");
    check(window.constrained == std::vector<std::int32_t>{0, 0, 1, 1},
          "constraints start after the reasoning-end draft");
    check(window.allows(2, Open) && !window.allows(2, Letter), "content starts as an object");

    matcher.commit(std::array<TokenId, 1>{Letter});
    check(plan(matcher, {}).constrained == std::vector<std::int32_t>{0},
          "reasoning content stays unconstrained");
    matcher.commit(std::array<TokenId, 1>{Think});
    check(plan(matcher, {}).constrained == std::vector<std::int32_t>{1},
          "committing the reasoning end arms the grammar");
}

void test_schema(const TokenGrammarCompiler& compiler) {
    TokenGrammarMatcher matcher(
        compiler.compile({.kind        = StructuredOutputKind::JsonSchema,
                          .json_schema = R"({"type":"object","properties":{"a":{"type":"integer"}},)"
                                         R"("required":["a"]})"}),
        false);
    // {"a": 1} is the only shape; "}" right after "{" would omit the required property.
    const Window window =
        plan(matcher, {Open, Quote, Letter, Quote, Colon, Space, Digit, Close});
    check(window.licensed == 8, "the required object is licensed token by token");
    check(!window.allows(1, Close), "a required property cannot be skipped");
    check(window.allows(8, Stop), "the complete instance may stop");

    bool rejected = false;
    try {
        (void)compiler.compile({.kind = StructuredOutputKind::JsonSchema, .json_schema = "{"});
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "an unparsable schema is a request error");
}

} // namespace

int main() {
    const TokenGrammarCompiler compiler = make_compiler();
    test_json_object(compiler);
    test_reasoning(compiler);
    test_schema(compiler);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " token_grammar\n";
    return failures == 0 ? 0 : 1;
}
