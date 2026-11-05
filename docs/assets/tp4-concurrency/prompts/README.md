# Complete coding prompts

`code-chat-2048.ids` and `code-chat-85000.ids` are complete, non-thinking chat
prompts for the same C++20 bounded blocking move-only task queue implementation.
Both retain the full system message, final task, 512-token response instruction,
and assistant prefix. They differ only in the amount of repository reference code.
The final reference excerpt may end mid-file; the implementation task does not.

The artifact's embedded tokenizer and chat template produce 72 introductory tokens
and 214 task/template suffix tokens. Reference code occupies 1762 or 84714 tokens.
Sources are consumed once in the order below, without repeated-text padding.
Literal special-token spellings in source files are encoded as ordinary text.
Pass these saved IDs directly to both engines; do not decode, retokenize or apply
another chat template.

From the repository root, after building `ninfer_v100_corpus`:

```bash
MODEL=/path/to/quasar-v3/qwen3_8_27b_nvfp4.ninfer
CORPUS=build/bench/ninfer_v100_corpus
for tokens in 2048 85000; do
  "$CORPUS" "$MODEL" "docs/assets/tp4-concurrency/prompts/code-chat-$tokens.ids" \
    --code-chat "$tokens" --output-tokens 512 \
    src/core/host_worker_pool.h \
    src/core/host_worker_pool.cpp \
    src/runtime/engine/concurrent_executor.h \
    src/targets/qwen3_6/impl/runtime/program_impl.h \
    src/targets/qwen3_6/impl/runtime/text_context_impl.h \
    src/targets/qwen3_6/impl/runtime/layouts_impl.h \
    src/targets/qwen3_6/impl/runtime/mtp_impl.h \
    src/ops/kernel/gqa_attention_decode_i8.cuh \
    src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh \
    src/ops/linear/nvfp4/nvfp4_volta_batch.cuh
done
```

The checked-in IDs are the benchmark authority. Regenerating after changing source
excerpts can produce a different workload. Earlier 2K/16K measurements instead
truncated a single 85K corpus, losing its final task and assistant prefix. Those
raw-prefix diagnostics remain separate and are not software before/after controls
for these complete coding prompts.
