#include "runtime/generation/token_grammar.h"

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::runtime {
namespace {

// Compilation shares host cores with the serving process; one schema compiles at a time.
constexpr int kCompileThreads           = 8;
constexpr std::int64_t kCacheLimitBytes = 256LL << 20;

xgrammar::TokenizerInfo tokenizer_info(const TokenGrammarVocabulary& vocabulary) {
    if (vocabulary.token_bytes.empty()) {
        throw std::invalid_argument("token grammar vocabulary is empty");
    }
    return xgrammar::TokenizerInfo(
        vocabulary.token_bytes, xgrammar::VocabType::RAW,
        static_cast<int>(vocabulary.token_bytes.size()),
        std::vector<std::int32_t>(vocabulary.stop_token_ids.begin(),
                                  vocabulary.stop_token_ids.end()),
        false);
}

void fill_row(xgrammar::GrammarMatcher& matcher, std::span<std::uint32_t> row) {
    std::int64_t shape[1] = {static_cast<std::int64_t>(row.size())};
    DLTensor tensor{};
    tensor.data   = row.data();
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim   = 1;
    tensor.dtype  = xgrammar::GetBitmaskDLType();
    tensor.shape  = shape;
    matcher.FillNextTokenBitmask(&tensor, 0);
}

} // namespace

class CompiledTokenGrammar {
public:
    CompiledTokenGrammar(xgrammar::CompiledGrammar compiled, TokenId reasoning_end,
                         std::int32_t words)
        : grammar(std::move(compiled)), reasoning_end(reasoning_end), words(words) {}

    const xgrammar::CompiledGrammar grammar;
    const TokenId reasoning_end;
    const std::int32_t words;
};

struct TokenGrammarCompiler::Impl {
    explicit Impl(const TokenGrammarVocabulary& vocabulary)
        : compiler(tokenizer_info(vocabulary), kCompileThreads, true, kCacheLimitBytes),
          reasoning_end(vocabulary.reasoning_end),
          words(token_bitmask_words(static_cast<std::int32_t>(vocabulary.token_bytes.size()))) {}

    xgrammar::GrammarCompiler compiler;
    const TokenId reasoning_end;
    const std::int32_t words;
    std::mutex mutex;
};

TokenGrammarCompiler::TokenGrammarCompiler(const TokenGrammarVocabulary& vocabulary)
    : impl_(std::make_unique<Impl>(vocabulary)) {}

TokenGrammarCompiler::~TokenGrammarCompiler() = default;

std::shared_ptr<const CompiledTokenGrammar>
TokenGrammarCompiler::compile(const StructuredOutput& output) const {
    // Free whitespace keeps the model's own JSON layout; the per-gap bound stops blank runs.
    constexpr bool kAnyWhitespace    = true;
    constexpr int kMaximumWhitespace = 64;
    std::optional<xgrammar::CompiledGrammar> compiled;
    try {
        const std::lock_guard lock(impl_->mutex);
        switch (output.kind) {
        case StructuredOutputKind::JsonObject:
            compiled = impl_->compiler.CompileJSONSchema(R"({"type":"object"})", kAnyWhitespace,
                                                         std::nullopt, std::nullopt, false,
                                                         kMaximumWhitespace);
            break;
        case StructuredOutputKind::JsonSchema:
            compiled = impl_->compiler.CompileJSONSchema(output.json_schema, kAnyWhitespace,
                                                         std::nullopt, std::nullopt, true,
                                                         kMaximumWhitespace);
            break;
        }
    } catch (const std::exception& error) {
        // xgrammar prefixes "[time] source:line: " and appends a newline.
        std::string_view reason = error.what();
        if (const auto source = reason.find(".cc:"); source != std::string_view::npos) {
            if (const auto text = reason.find(": ", source); text != std::string_view::npos) {
                reason.remove_prefix(text + 2);
            }
        }
        while (!reason.empty() && reason.back() == '\n') { reason.remove_suffix(1); }
        throw RequestError(RequestErrorKind::InvalidStructuredOutput,
                           "invalid JSON schema: " + std::string(reason));
    }
    if (!compiled) {
        throw RequestError(RequestErrorKind::InvalidStructuredOutput,
                           "unknown structured output kind");
    }
    return std::make_shared<const CompiledTokenGrammar>(std::move(*compiled), impl_->reasoning_end,
                                                        impl_->words);
}

struct TokenGrammarMatcher::Impl {
    Impl(std::shared_ptr<const CompiledTokenGrammar> compiled, bool starts_in_reasoning)
        : grammar(std::move(compiled)), matcher(grammar->grammar), armed(!starts_in_reasoning) {}

    std::shared_ptr<const CompiledTokenGrammar> grammar;
    xgrammar::GrammarMatcher matcher;
    bool armed;
};

TokenGrammarMatcher::TokenGrammarMatcher(std::shared_ptr<const CompiledTokenGrammar> grammar,
                                         bool starts_in_reasoning) {
    if (grammar == nullptr) { throw std::invalid_argument("token grammar is empty"); }
    impl_ = std::make_unique<Impl>(std::move(grammar), starts_in_reasoning);
}

TokenGrammarMatcher::~TokenGrammarMatcher()                                 = default;
TokenGrammarMatcher::TokenGrammarMatcher(TokenGrammarMatcher&&) noexcept    = default;
TokenGrammarMatcher& TokenGrammarMatcher::operator=(TokenGrammarMatcher&&) noexcept = default;

std::int32_t TokenGrammarMatcher::words() const noexcept { return impl_->grammar->words; }

std::uint32_t TokenGrammarMatcher::plan(std::span<const TokenId> drafts,
                                        std::span<std::uint32_t> rows,
                                        std::span<std::int32_t> constrained) {
    const std::size_t columns = drafts.size() + 1;
    const auto words          = static_cast<std::size_t>(impl_->grammar->words);
    if (constrained.size() < columns || rows.size() < columns * words) {
        throw std::invalid_argument("token grammar plan has too few constraint rows");
    }
    xgrammar::GrammarMatcher& matcher = impl_->matcher;
    bool armed                        = impl_->armed;
    int advanced                      = 0;
    std::uint32_t licensed            = 0;
    std::size_t column                = 0;
    for (; column < columns; ++column) {
        const bool active   = armed && !matcher.IsTerminated();
        constrained[column] = active ? 1 : 0;
        if (active) { fill_row(matcher, rows.subspan(column * words, words)); }
        if (column == drafts.size()) { break; }
        const TokenId draft = drafts[column];
        if (active) {
            if (!matcher.AcceptToken(draft)) { break; }
            ++advanced;
        } else if (!armed && draft == impl_->grammar->reasoning_end) {
            armed = true;
        }
        ++licensed;
    }
    for (std::size_t later = column + 1; later < columns; ++later) { constrained[later] = 0; }
    if (advanced != 0) { matcher.Rollback(advanced); }
    return licensed;
}

void TokenGrammarMatcher::commit(std::span<const TokenId> tokens) {
    for (const TokenId token : tokens) {
        if (!impl_->armed) {
            impl_->armed = token == impl_->grammar->reasoning_end;
            continue;
        }
        if (impl_->matcher.IsTerminated() || !impl_->matcher.AcceptToken(token)) {
            throw std::logic_error("committed token " + std::to_string(token) +
                                   " violates the structured output grammar");
        }
    }
}

} // namespace ninfer::runtime
