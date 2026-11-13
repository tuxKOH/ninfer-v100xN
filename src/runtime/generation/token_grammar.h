#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::runtime {

// One constraint row licenses token g through bit (g & 31) of word g / 32.
[[nodiscard]] constexpr std::int32_t token_bitmask_words(std::int32_t token_domain) noexcept {
    return (token_domain + 31) / 32;
}

// The generated-token vocabulary as raw bytes. An empty entry is never licensed by a grammar
// (added and special tokens); stop tokens are licensed exactly where the grammar may end.
struct TokenGrammarVocabulary {
    std::vector<std::string> token_bytes; // [token domain]
    std::vector<TokenId> stop_token_ids;
    TokenId reasoning_end = -1; // the token closing the reasoning channel
};

class CompiledTokenGrammar;

// Owns the vocabulary-specific compiler. compile() may run concurrently on submitting threads.
class TokenGrammarCompiler {
public:
    explicit TokenGrammarCompiler(const TokenGrammarVocabulary& vocabulary);
    ~TokenGrammarCompiler();

    TokenGrammarCompiler(const TokenGrammarCompiler&)            = delete;
    TokenGrammarCompiler& operator=(const TokenGrammarCompiler&) = delete;

    // Throws RequestError(InvalidStructuredOutput) when the schema is not a usable JSON Schema.
    [[nodiscard]] std::shared_ptr<const CompiledTokenGrammar>
    compile(const StructuredOutput& output) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Per-request grammar state over the committed token ledger. A request that starts in the
// reasoning channel is unconstrained until the reasoning-end token commits.
class TokenGrammarMatcher {
public:
    TokenGrammarMatcher(std::shared_ptr<const CompiledTokenGrammar> grammar,
                        bool starts_in_reasoning);
    ~TokenGrammarMatcher();
    TokenGrammarMatcher(TokenGrammarMatcher&&) noexcept;
    TokenGrammarMatcher& operator=(TokenGrammarMatcher&&) noexcept;

    [[nodiscard]] std::int32_t words() const noexcept;

    // Plans one verification window after the committed ledger. Column j follows drafts[0:j).
    // Writes rows[j * words(), +words()) and constrained[j] = 1 for each constrained column and
    // constrained[j] = 0 otherwise. Returns the longest draft prefix the grammar licenses; columns
    // after that prefix are unconstrained and must not be verified. Committed state is unchanged.
    [[nodiscard]] std::uint32_t plan(std::span<const TokenId> drafts, std::span<std::uint32_t> rows,
                                     std::span<std::int32_t> constrained);

    // Appends committed tokens. Throws std::logic_error when a token violates the grammar.
    void commit(std::span<const TokenId> tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::runtime
