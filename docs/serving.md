# HTTP serving

Optional native RAM-KV uses the same public Engine and HTTP protocols, not a llama.cpp backend.
Add `--ram-kv-window 98304 --ram-kv-budget-bytes 32000000000` to a 27B SM70 TP2 Text/MTP
launch with `--max-context 180000` or `262144`. It supports one active request and CUDA Graphs.
Omit `--kv-capacity`, or set it equal to the resident window. This is **approximate Q/K mean-vector
retrieval**, including long prefill, and can miss history. RAM archive bytes are lossless, but that
does not establish quality equivalence. No Vision/DFlash/TP4 in this mode. The archive budget is
not an entire-server RAM cap; apply cgroup MemoryMax and disable swap separately when required.
Server-start JSON records identify the resident window and maximum archive bytes. Default full
history behavior and the advertised OpenAI/Anthropic request schemas are unchanged.

`build/apps/ninfer-serve` loads one registered artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident NInfer Engine.

## Model residency

The GUI/TUI can release GPU residency while retaining one immutable copy of the original `.ninfer`
container in host RAM. These NInfer control endpoints use the server's configured bearer or
`x-api-key` authentication; they do not change the OpenAI/Anthropic request formats:

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/ninfer/model` | Return model state, retained bytes, progress and active request count |
| POST | `/ninfer/model/offload` | Copy original container to RAM once, then destroy the GPU Engine |
| POST | `/ninfer/model/load` | Reconstruct the same configured Engine directly from RAM |

Example status: `{"state":"ram","retained_host_bytes":19782432752,"phase":"ready",`
`"done":0,"total":0,"active_requests":0,"cache_preserved":false}`. The four states are `gpu`,
`offloading`, `ram`, `loading`; `done`/`total` are phase-local bytes or work units (zero total means
indeterminate progress). POST waits until the transition completes; launchers issue it in a worker
and poll GET separately. Repeating offload in `ram` or load in `gpu` succeeds without doing work.

An active request owns its reservation through response consumption and generation-handle
destruction. Offload returns **409 `model_busy`** while any such reservation exists; competing
transitions also return 409. New inference/token-count requests return **503 `model_not_loaded`**
while offloaded or transitioning. `/health` remains process liveness (`{"status":"ok"}`), including
in RAM state. Model listing and stored Responses remain accessible. No inference is auto-reloaded.

First offload requires the artifact still on disk and available host RAM covering the complete file
plus 1 GiB headroom (otherwise **507 `insufficient_host_memory`**). A failed copy leaves the active
GPU Engine intact. Successful offload releases its model allocations, KV/GDN, workspace and graphs;
CUDA driver context/module overhead can remain. Recreate uses original codes/scales/divisors and
the same artifact materialization route, with no disk fallback or requantization. RAM is retained
after load for subsequent unloads and freed when the server stops. It is ordinary pageable RAM,
not pinned/locked memory or an RSS limit. Reload failure returns an error and keeps the RAM state
retryable. Runtime throughput counters remain cumulative across residency cycles.

This is model residency control, **not request pause/resume**: KV, archived prefixes and processed
media caches are discarded, and the next request starts cold. TP/PP, capacity, Vision placement,
speculation and all other startup settings remain fixed. Reload rebuilds runtime allocations and
CUDA Graphs; it is faster storage access, not an instant GPU context restoration.

## Start the server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 16384 \
  --kv-capacity 32768 \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For the 35B-A3B artifact, select its artifact path; the public model ID follows the container
identity automatically:

```bash
./build/apps/ninfer-serve models/qwen3_6_35b_a3b.ninfer \
  --max-context 16384 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

When `--model-id` is omitted, the server advertises and accepts the loaded container's exact
`identity.model_id`. An explicit `--model-id` remains a public HTTP alias override and does not
select or alter the artifact.

Vision is disabled by default: its weights, Vision scratch phase, and frozen request-transient
buffer are not allocated, and media
requests and token-count requests fail with HTTP 400 `vision_disabled`. Add `--vision` when the
server must accept image or video input. Speculative residency is likewise frozen by
`--spec mtp|dflash` and `--draft-tokens`; omitting `--spec` loads neither backend.
`--lm-head-draft` additionally loads the optimized proposal head. DFlash is text-only: 35B-A3B
supports up to fifteen drafts and Qwen3.8-27B artifacts with the optional DFlash2 package support
up to seven. DFlash cannot be combined with `--vision`. A later request cannot enable a capability
omitted at startup.

`--tp 2` splits one model across two GPUs and requires an explicit `--devices A,B` naming one
distinct device per rank. The 27B package supports `--spec mtp` and optional Qwen3.8 DFlash2.
Experimental `--vision` is available at TP2 for artifacts with execution-qualified Vision weights;
QUASAR NVFP4 admits TP2 None/MTP; GGUF-derived Q4_K_M rejects it. Vision weights and encoding reside only on the primary
rank; composed embeddings are copied to the peer before split text prefill, and both ranks use
the same three-axis positions and per-request RoPE delta. Vision requests participate in the
same compact concurrent decode batches as text requests. Vision requires additional startup
memory; lower the text-only context profile rather than simply appending `--vision` to a
near-capacity command. On the two 16 GB V100s, the qualified NVFP4 v3 DFlash2 profile uses
`--max-context 98304 --prefill-chunk 1024 --kv-dtype int8 --spec dflash --draft-tokens 7`, without
`--lm-head-draft`: its selector uses the full vocabulary. It does not fit the 180000-capacity MTP
profile. The 35B-A3B package has no TP2 path, but it can use two GPUs as a single-device compute
path plus an expert-only storage device. Keep `--tp 1`, select the compute GPU with `--device`,
and pass the other GPU with `--storage-device`; this requires peer access and leaves attention,
KV, GDN, and scheduling on the primary GPU.

`--tp 1 --pp 2 --devices 0,1 --max-concurrency 1` selects the experimental SM70
Qwen3.8 NVFP4/QUASAR layer pipeline through the same Engine and HTTP protocols.
`--pipeline-chunks 1..8` controls the prefill wave (default 8; 1 is serial).
It supports Text, None/MTP, prefix reuse and optional sparse RAM-KV; Vision, DFlash, expert storage
and concurrency above one are rejected. Both prefill and decode use PP2 throughout
the resident Engine's lifetime; there is no request-time TP/PP switch.

Compatible-prefix reuse is enabled by default at both TP widths, including `--tp 2 --spec mtp`.
The visual-prefix bridge test fails on this two-card SM70 host; the difference has not been
established as either a state bug or floating-point path variation. For experimental TP2 Vision,
the recommended server profile uses `--no-prefix-reuse` to avoid that unqualified cache path.
This flag disables Engine prefix reuse for text requests on the same server too, but does not
disable the immutable-media preprocessing cache. The text-only recommendation is unchanged.
See [test status](../tests/README.md#experimental-tp2-vision-status).
The new SM70 `--tp 4 --devices 0,1,2,3` route is limited to Qwen3.8-27B NVFP4/native FP8
Text/MTP and has completed its four-card state, prefix and performance qualification. It uses
the same Engine and HTTP protocols; it is not a separate serving backend. Native FP8 requires
TP4, and Vision and DFlash are unavailable at that width.
It applies transparently to the HTTP APIs: submit the normal conversation history. A matching
resident frontier or saved turn/response checkpoint reuses its complete model state, prefilling only
the suffix; an exact hit requires no prompt-token prefill. See the cache behavior below for matching
and retention rules. `--no-prefix-reuse` forces cold requests for comparison.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 262144 --kv-capacity auto --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

### Extended context: `--rope yarn`

`--rope native` (the default) runs the checkpoint's own rotary embedding and caps `--max-context`
at the artifact's registered native position capacity, `262144`. `--rope yarn` applies YaRN
frequency correction to the text rotary subspace and raises that ceiling to
`--yarn-origin` x `--yarn-factor`, up to a product maximum of `1048576` tokens:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
  --max-context 1048576 --kv-capacity auto --kv-dtype int8
```

`--yarn-origin` must equal the artifact's registered native capacity (`262144`); a different origin
is rejected at startup rather than silently reinterpreted, because the correction table is derived
from that exact window. `--yarn-factor` accepts a finite value in `[1.0, 64.0]` and defaults to `4.0`;
`origin x factor` must land on a whole number of tokens at or below `1048576`.

YaRN is available at `--tp 1` as well as `--tp 2` -- the flag is independent of tensor parallelism,
though the KV pool for a very long context is what usually decides which width fits. It is rejected
together with `--vision` (the vision encoder ropes 2-D image-grid positions through its own table)
and with `--spec dflash` (a different rotary geometry), and it is only offered by targets whose
text attention carries a YaRN rope domain -- `qwen3.6-27b`/`qwen3.8-27b`; `qwen3.6-35b-a3b` rejects
it at startup. Under `--rope native` nothing about the rope path changes, at either `--tp` width.

The server's startup record and the CLI load summary both report the resolved regime: mode, factor,
origin, the effective max-context ceiling, and the YaRN `mscale` multiplier.

At the full `1048576` ceiling the configuration is single-slot: one sequence costs 16.66 GiB per
device without a speculative backend and 17.69 GiB with `--spec mtp --draft-tokens 3`, so
`--max-concurrency` must be `1` and a second slot cannot be allocated on a 32 GiB card. Concurrency
at extended context requires a smaller `--max-context` (roughly 500,000 tokens for two slots at
INT8 KV). `--kv-dtype int8` is mandatory at that window.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | process health |
| `GET /v1/models` | configured OpenAI model alias |
| `GET /v1/models/{id}` | lookup of the configured alias |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

The HTTP server enables `TCP_NODELAY` so small SSE frames following the headers are sent
without Nagle/delayed-ACK stalls on persistent connections. SSE responses also set
`Cache-Control: no-cache` and `X-Accel-Buffering: no`. Clients should read incrementally;
proxy buffering and remote network latency can still delay visible output. The local
[fully cached HTTP TTFT measurement](performance.md#http-fully-cached-first-token-latency-2026-10-19)
times the first nonempty text delta, not the initial role or lifecycle event.

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history;
- string content and ordered text, `image_url`, and `video_url` parts;
- `max_completion_tokens` and the legacy `max_tokens` spelling;
- `temperature`, `top_p`, `top_k`, presence/frequency penalties, and a nonnegative `seed`;
- one stop string or an array of stop strings;
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- function tools, tool choices, assistant tool-call history, and tool-result messages;
- `response_format` of type `text`, `json_object`, or `json_schema` (see
  [Structured output](#structured-output));
- the top-level `reasoning_effort` field;
- the `enable_thinking` extension;
- `chat_template_kwargs.preserve_thinking` and the top-level `preserve_thinking` alias.

The request `model` must equal the public model ID: the artifact `identity.model_id` by default, or
the explicit `--model-id` override. Reasoning is returned separately as `reasoning_content`; answer
text remains in `content`.

Message roles retain their input order through schema translation. The Qwen family frontend maps
both `system` and `developer` to system-class ChatML blocks at their original positions; it does not
move later instructions to the beginning of the conversation. A leading instruction keeps the
artifact template's existing tool/reasoning-instruction composition.

At startup, NInfer resolves prompt capabilities from the exact `frontend/chat_template.jinja`
resource embedded in the loaded artifact. It does not infer them from the request's `model` field,
the artifact identity, or a target profile. A recognized effort-capable template exposes `low`,
`medium`, and `xhigh`; omitting effort uses the server's `--reasoning-effort`, or else that
template's declared default. After any
protocol-specific mapping described below, an effort not exposed by the loaded template
returns HTTP 400 with code `reasoning_effort_not_supported` before prompt preparation.

For Chat Completions, `reasoning_effort: "none"` disables thinking. `low`, `medium`, and `xhigh`
select the corresponding template effort when available. The other OpenAI protocol values
`minimal`, `high`, and `max` are parsed but rejected when the loaded template does not expose them.
`enable_thinking` controls the same new-turn thinking switch; a contradictory combination with
`reasoning_effort` returns `conflicting_template_option`.

`preserve_thinking` controls whether reasoning from closed assistant turns remains in later
prompts. It defaults to the server setting, which is off unless `--preserve-thinking` is used. If
both OpenAI spellings are present they must carry the same boolean value. Unknown non-null
`chat_template_kwargs` are rejected.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage.

Chat Completions usage includes `prompt_tokens_details.cached_tokens`: the actual input tokens
restored from the Engine's prefix cache, including resident-frontier and saved-checkpoint reuse.
`prompt_tokens` still includes these tokens; neither completion nor speculative draft tokens count
as cache hits. Cold requests and `--no-prefix-reuse` return zero. Non-streaming text and tool-call
responses return this field in `usage`; streaming clients must request
`"stream_options":{"include_usage":true}` and read it from the final usage chunk before `[DONE]`.

```json
{
  "usage": {
    "prompt_tokens": 1000,
    "completion_tokens": 128,
    "total_tokens": 1128,
    "prompt_tokens_details": {"cached_tokens": 900}
  }
}
```

The token cache-hit rate for this request is `900 / 1000 = 90%`. It is not the fraction of requests
with a hit. The Responses API reports the same reuse as `usage.input_tokens_details.cached_tokens`.

### Structured output

`response_format` restricts the generated content with grammar-constrained decoding:

```json
{"type": "json_object"}
```

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "answer",
    "schema": {
      "type": "object",
      "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
      "required": ["city", "population"]
    },
    "strict": true
  }
}
```

Before every token selection the server masks the vocabulary to tokens that continue a valid
document, so content is a JSON object (`json_object`) or an instance of the schema (`json_schema`)
unless the output-token budget ends it first (`finish_reason: "length"`). Every schema is enforced
regardless of `strict`, properties absent from `properties` are rejected, and properties are
generated in the schema's order. Whitespace between JSON tokens follows the model, at most 64
characters per gap. An unusable schema returns HTTP 400 with code `invalid_response_format` before
prompt preparation. xgrammar compiles masks on the submitting thread; the first structured request
also builds the tokenizer index once.

Reasoning is not constrained: with thinking enabled, the grammar starts after the model closes its
reasoning channel, and reasoning still appears in `reasoning_content`. With MTP, drafts the grammar
rejects are dropped before verification and every verified column is masked, so speculative output
remains exactly the constrained result. Structured output is unavailable with `--spec dflash`
(`invalid_response_format`): DFlash drafts are produced and verified inside one device round.
Decode CUDA Graphs are captured with and without the mask; rounds without a constrained request
replay the unmasked graphs, so unconstrained throughput is unchanged.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

Text and media requests use one complete-prompt context contract. After chat-template rendering and
media-token expansion, the result must fit Engine `--max-context`. The separately configurable
`--vision-max-tokens` limits merged visual tokens across the prompt (four raw patches per merged
token); it defaults to 32,768, and the effective Vision limit is
`min(--max-context, --vision-max-tokens)`. Both its media-processing allowance and the startup
Vision scratch/output reservations use this bound. A smaller visual-token budget leaves room for
long text context on memory-constrained hardware; exceeding the media budget returns an error,
not a shorter image. There is no fixed image/video item-count limit: item count is admitted
through aggregate source-byte, decoded-pixel, raw-patch, Vision-token, and live-memory budgets.

Media cache misses run as independent decode → resize → BF16-pack tasks on a bounded host worker
pool. Prepared payloads are keyed by SHA-256 of the acquired bytes plus modality, so repeated media
in later requests reuses the exact immutable BF16 patch input; concurrent identical misses use one
single-flight build. `--media-cache-mib` bounds LRU-retained payloads, while
`--media-live-mib` bounds every cache-, request-, or runtime-referenced payload. Cache eviction does
not invalidate a request reference, and live bytes are returned only when the final reference is
released. A request-level preparation gate derived from the live limit prevents concurrent partial
builds from deadlocking the memory account.

An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded`, including
the prepared token count and configured context ceiling. A media preprocessing resource rejection
returns HTTP 400 `media_budget_exceeded`. HTTP 413 `request_too_large` is reserved for a raw request
body that exceeds `--max-request-mib` before JSON parsing; it is not used for model-context or media
resource errors.

## OpenAI Responses Core

NInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
registered artifact identities use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.6-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override |
| `input` | required string or non-empty typed Item array |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | integer at least `16`; default is `--default-max-tokens` |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `reasoning.effort` | `none` disables thinking. On graded Qwen templates, `minimal`/`low` map to `low`, `medium` to `medium`, and `high`/`xhigh`/`max` to `xhigh`. On toggle-only templates, every non-`none` value enables thinking without a graded budget. |
| `chat_template_kwargs.preserve_thinking` | optional boolean controlling whether closed-turn reasoning remains in reconstructed prompts |
| `preserve_thinking` | top-level alias for the same option; conflicting values are rejected |
| `text.format` | `text`, `json_object`, or flat `json_schema` with `name`, `schema`, `strict`, and `description`; see [Structured output](#structured-output). The response echoes the requested format. |
| `tools` | flat Responses function definitions; see below |
| `tool_choice` | `auto` or `none` |
| `parallel_tool_calls` | omitted or `true` |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted or an empty array |
| `stream_options` | omitted or `{"include_obfuscation":false}` |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored.

For example, use `"reasoning":{"effort":"high"}` in a Responses request. This is a local
mapping to the checkpoint's template controls, not an OpenAI-equivalent reasoning token budget.
The response echoes the requested effort. `reasoning.summary` remains unsupported.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `input_image` | user-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires server `--vision` |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with an empty `summary` and `reasoning_text` content parts |
| `function_call` | completed assistant call with optional `id`, and required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed tool result with required `call_id` and string `output` |

Adjacent function-call Items are grouped into one assistant history turn. A reasoning Item attaches
to the following assistant message or function call. Input Item IDs are preserved when supplied and
generated otherwise; duplicate IDs fail.

System and developer message Items retain their positions in the input array. Top-level
`instructions` is represented as a leading developer turn for the current request; target-specific
role lowering occurs only in the Qwen family frontend.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, reasoning summaries or
encrypted reasoning, message `phase`, and other Item/content types are not supported. HTTP media
URLs stored in a response chain are fetched again when that chain is continued; use data URIs when
the historical media bytes must be immutable.

### Function tools

Responses function definitions are flat rather than Chat Completions' nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

NInfer renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. NInfer does not execute functions or enforce JSON Schema through constrained
decoding, so `strict:true`, `tool_choice:required`, named tool choice, hosted tools, MCP tools, and
custom free-form tools are rejected.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. NInfer may emit:

- a `reasoning` Item containing raw `reasoning_text` and an empty summary;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion
produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after
an SSE response has started produce `response.failed`; validation and preparation errors remain
normal HTTP error responses.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
resident prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token
IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen
output decoder while accepted tokens are still in the reasoning channel; it is not estimated by
re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

Function arguments use `response.function_call_arguments.delta` and `.done`. IDs, output indices,
and content indices remain stable, and concatenated deltas equal the terminal Item. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only an ambiguous `<tool_call>` suffix or the structured tool region is held.
Malformed tool markup is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so resident
prefix reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
rendering and identity still determine reuse. Changing the boolean alone never invalidates an exact
current frontier or a complete matching rewrite checkpoint.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found` |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`) |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored.

### Responses input token count

`POST /v1/responses/input_tokens` accepts exactly `model` and `input`, performs the same typed Item,
template, and media expansion, and does not run generation:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, prompt-cache controls, safety/user identifiers, non-empty `include`, background
execution, compaction, files/audio, and OpenAI-hosted/MCP/custom tools. These are compatibility
boundaries, not silently accepted placeholders.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint supports top-level system text, ordered mid-conversation system messages,
user/assistant history, text and image blocks, thinking blocks, tool-use history, tool results,
client-defined tools, non-streaming responses, and Anthropic SSE events.
Mid-conversation system messages remain at their `messages` array position and are not merged into
the top-level system instruction. A system section must follow a user/tool-result message and be
final or immediately precede an assistant message; it cannot interrupt a tool-use/tool-result pair.
Consecutive system messages remain separate ordered turns.

`thinking.type: "disabled"` disables thinking; other supported values enable it.
The independent top-level `preserve_thinking` boolean controls closed-turn history and otherwise
uses the server default.

Anthropic `output_config.effort` accepts the protocol values `low`, `medium`, `high`, `xhigh`, and
`max`. The value is then checked against the loaded chat template in the same way as the OpenAI
endpoints; the registered effort-capable template exposes `low`, `medium`, and `xhigh`. Combining
an effort with `thinking.type: "disabled"` is rejected as contradictory.

Anthropic's `model` field is treated as a response label and does not select the loaded artifact.

Messages usage reports `cache_read_input_tokens` for actual prefix-cache reuse and `input_tokens`
for the remaining uncached input. Their sum is the complete templated input-token count.
`cache_creation_input_tokens` is zero: NInfer does not expose a separate cache-creation accounting
category. For SSE, `message_start.usage.input_tokens` is provisional (the prepared full input);
the terminal `message_delta.usage` contains authoritative cumulative input, cache-read and output
counts. Replace the initial counts with these final values rather than adding them.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without running GPU generation:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

## Server options

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `identity.model_id` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `8192` |
| `--rope native\|yarn` | rotary regime; `yarn` applies YaRN frequency correction and raises the `--max-context` ceiling to `--yarn-origin` x `--yarn-factor` | `native` |
| `--yarn-factor F` | YaRN scaling factor, in `[1.0, 64.0]`; only read under `--rope yarn`; `origin x factor` must be a whole token count not exceeding `1048576` | `4.0` |
| `--yarn-origin O` | YaRN origin window; must equal the artifact's registered native context capacity (`262144`) | `262144` |
| `--max-concurrency N` | maximum admitted requests; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `30000` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--prefill-decode-rounds N` | experimental decode rounds between prefill steps while both phases are runnable; `1..64`; larger values favor existing streams and delay the prefilling request's first token | `1` |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--device N` | CUDA device index | `0` |
| `--tp 1\|2` | tensor-parallel width; `2` splits the model across two GPUs | `1` |
| `--pp 1\|2` | experimental SM70 NVFP4/QUASAR layer pipeline; PP2 requires TP1, concurrency 1, Text/None/MTP | `1` |
| `--pipeline-chunks N` | PP prefill wave of `1..8` chunks; `1` is serial | `8` |
| `--devices A,B` | exactly `tp * pp` distinct CUDA device indices; required for parallel execution | `--device` |
| `--storage-device N` | 35B-A3B expert-only VMM backing GPU; keeps `--tp 1` | unset |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--kv-dtype bf16\|int8` | KV-cache storage | `bf16` |
| `--prefix-cache-tokens N` | historical host KV budget in unique page-rounded Main Text tokens; `0` disables historical branches, preserving current GPU-prefix reuse | `384000` |
| `--volta-long-attention` | startup-fixed experimental complete-history attention; SM70 27B TP2/TP4, PP1, experimental concurrency 1–8, INT8 KV, Text/None/MTP (Vision on TP2 only); excludes RAM-KV; floating-point association and output IDs may change | off |
| `--volta-fast-decode` | experimental QUASAR SM70 TP2/TP4 Text/None/MTP numeric profile: packed NVFP4 FP32 epilogue and MTP W8 reductions; may change logits, acceptance and output; excludes Vision/DFlash/PP2 | off |
| `--spec mtp\|dflash` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash `1..7` on 27B, `1..15` on 35B-A3B | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--default-max-tokens N` | output limit when omitted by a request | `65536` |
| `--max-output-tokens N` | server-wide hard output limit for Chat Completions, Responses and Messages; `0` disables it | `0` |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--vision-device gpu\|cpu` | visual encoder placement; requires `--vision` for CPU; CPU weights remain in host RAM, final BF16 embeddings upload; may change numerical outputs | `gpu` |
| `--vision-max-tokens N` | merged visual-token limit per prompt and reserved Vision workspace bound, independent of the text context ceiling; `1..32768` | `32768` |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--no-thinking` | disable thinking by default | thinking on |
| `--reasoning-effort low\|medium\|xhigh` | effort for thinking requests that omit one; must be exposed by the loaded template; excludes `--no-thinking` | template default |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--cors` | permissive browser CORS headers | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |

Requests above `--max-output-tokens` return HTTP 400 with code `max_output_tokens_exceeded`
before model preparation. An omitted default output budget is lowered to this ceiling when needed;
an explicit `--default-max-tokens` above it is rejected at startup. The Engine additionally limits
generation to the remaining total context capacity.

Engine selects sampling defaults from the loaded model and the request's resolved thinking mode.
Qwen3.6-27B and Qwen3.8-27B use `1.0/0.95/20/0/0` for
temperature/top-p/top-k/min-p/presence penalty in thinking mode and `0.7/0.80/20/0/1.5` in
non-thinking mode. Qwen3.6-35B-A3B differs only in its thinking presence penalty, which is `1.5`.
Frequency penalty is `0` for all registered presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --request-log-jsonl profiles/bench/run/server.requests.jsonl
```

Every line is one `ninfer_serve_request_log` schema-v10 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | target/weights identity and artifact, resolved Engine including tensor-parallel width `tp` and the resolved per-rank `devices` list, registered thinking/non-thinking sampler defaults plus process overrides, thinking-history defaults, weights/sequence/workspace/request-transient arenas, KV sizing ledger, CUDA Graph observed/allowance bytes, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, thinking modes, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, unrounded phase seconds, and complete speculative-decoding counters |
| `request_error` | the resolved request configuration and generation error message |
| `throughput` | interval token deltas and rates, drafted/accepted draft deltas, scheduler occupancy, and decode-round batch statistics |

At `--tp 2` every other `engine` and `memory` field in `server_start` is the primary rank's; `tp`
and `devices` are what identify the record as a dual-device run and name the second GPU. Per-device
memory rows remain a console load-summary table rather than structured-log fields.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position`. Rates can be
derived downstream from raw token counts and seconds instead of rounded stderr strings.
`throughput.speculative` contains interval `drafted_tokens` and `accepted_tokens` deltas for
live draft acceptance (`accepted / drafted`, undefined when drafted is zero). These count
target-accepted drafts, excluding bonus tokens, not all committed output tokens.

The JSONL file contains no generated response text and never records an API-key value; `argv`
replaces that value with `<redacted>`. The existing stderr summaries remain available for operators
but are rounded and are not the aggregation source. Console lines use local
`[YYYY-MM-DD HH:MM:SS.mmm] [level]` timestamps. OpenAI Responses, OpenAI Chat, and Anthropic
generation requests receive a request ID when they enter synchronous preparation. Successful
preparation produces `request_start`; a preparation failure produces `request_rejected` without a
matching start. Later generation failures produce `request_error`. Schema/model validation
rejections before preparation and token-count-only calls are not measurement requests and do not
receive request IDs.

By default the server also reports aggregate activity every five seconds. `prefill` counts prompt
suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode` counts
tokens finally committed by decode rounds, excluding the first token produced by prefill. For MTP
and DFlash this is the accepted committed output, not draft or rejected tokens.
`avg_decode_batch` is decode row-rounds divided by decode rounds during the same interval. The
`running`, `prefilling`, `decode_ready`, and `waiting` fields are the Engine scheduler snapshot at
the end of the interval. Fully idle zero intervals are omitted. The JSONL `throughput` event keeps
the raw token and round deltas as well as derived rates; downstream measurement should prefer those
raw values.

## Execution behavior

The server owns one resident Engine with a startup-fixed capacity of `1..8` active generation
requests. At each decode boundary, every decode-ready request is compacted into one batch and
processed by one model traversal and, when graphs are enabled, one exact-batch CUDA Graph replay. A
request joins that batch only after its single-request prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row.

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
There is no admission ETA or unbounded overflow queue.

When prefill and decode are both runnable, `--prefill-decode-rounds N` runs up to `N`
compact decode rounds between bounded prefill steps. The default `1` retains alternation;
larger values favor existing streams and increase the prefilling request's TTFT. Each round
still includes every decode-ready lane and processes completion/cancellation at its boundary.
With one active request, or when only one phase is runnable, this option has no effect.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media requests additionally share one preparation permit, so a waiting
media request retains the same cancellation and timeout deadline. Model output is bounded by the
same finite request count and each request's effective output-token limit; output callbacks and
network serialization run outside the GPU executor and do not delay formation of the next batch.

`--max-context` and the resolved `--kv-capacity` are independent limits. The former is each
sequence's logical ceiling; the latter sizes the shared Main Text KV pool used by all active
requests and retained prefixes. Both are represented with 64-token pages internally, while a
sequence can never cross the exact `--max-context` frontier. `--kv-capacity N` requests an explicit
capacity; `--kv-capacity auto` chooses the largest legal capacity that fits the memory remaining
after weights are loaded while keeping 1 GiB of sizing headroom. When omitted it follows
`--max-context`, preserving one full-length request's capacity. The shared pool is fixed at startup
and is not divided evenly among request lanes.

Automatic sizing evaluates the complete target runtime layout for the chosen concurrency, KV
dtype, speculative backend, draft window, Vision setting, workspace, and CUDA Graph allowance. It
uses a direct page-capacity calculation rather than allocation probing. Startup reports the policy,
resolved capacity, runtime reservation, free memory after weights, automatic headroom, planned
slack, actual free memory after complete startup, and observed Graph memory. An explicit capacity
is never silently reduced, and neither policy permits request-time pool growth.

Admission reserves the full prompt-plus-effective-output page entitlement, so an admitted request
can finish within its declared bound. A later request waits in FIFO order when the remaining shared
pages cannot satisfy its complete entitlement; the Engine never admits it and later truncates an
older request to recover capacity. Startup rejects a KV pool smaller than one sequence, too small to
provide one page per configured lane, or larger than all configured lanes could use.

Compatible resident prefixes are reused for both text and multimodal histories unless the server is
started with `--no-prefix-reuse`. A multimodal hit requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans; changing an earlier image or video
therefore resets the prefix instead of reusing placeholder-token KV. Media wholly inside a matched
prefix skips Vision execution, while new suffix media is encoded normally. The completion log
reports the reused token count as `cache=`.

The prefix cache is process-local Engine state; restarting the server clears it. Text/None/MTP
with complete GPU KV also retains historical branches and multiple conversation trees in host RAM.
Before an old resident branch is overwritten or its lane evicted, its original Main/MTP KV and
complete continuation checkpoints can be archived. `--prefix-cache-tokens` defaults to **384000**
unique page-rounded Main Text tokens, shared across all trees; overflow evicts the oldest archived
branch first. A historical hit restores private writable GPU pages and exact state, including
completed-sibling fan-out. `ABD → ABC → ABDE` can therefore recover ABD. Setting the budget
to zero keeps current resident reuse but disables historical retention. This is not disk persistence
or a process RSS cap: QUASAR INT8 Main KV plus MTP can use about **12.8 GiB RAM** at the 384K default
(about **33.4 GiB RAM** at 1M tokens),
plus state images and transfer headroom. In addition to the current resident frontier and a saved complete
turn/response checkpoint, reusable text requests with None/MTP speculation and complete GPU KV
retain bounded RAM checkpoints of every rank's exact GDN state and the target tail hidden.
Shorter prompts or edits to a later suffix restore the longest saved frontier within their common
token/position prefix and recompute only the remaining suffix. Cache usage counts only tokens
whose computation was actually skipped. There is no arbitrary substring reuse or guarantee that
every common-prefix token has a saved recurrent state.

Active-lane state images have a 2 GiB per-Program budget divided among configured lanes;
archived state has a separate 2 GiB cap, in addition to the historical KV token budget. Capture uses
one early boundary (half the prompt, capped at 64 tokens), then
budget-spaced actual prefill chunk ends and the final prompt boundary. The early split rejoins
the ordinary chunk grid; not every small GPU chunk is copied to RAM. Under the budget, older
redundant images are evicted, preserving a first and recent checkpoint when two images fit.
Capture/restore transfer time is included
in prefill timing. `--no-prefix-reuse` also skips these captures. Vision, DFlash and sparse RAM-KV
keep the existing frontier/turn/response rules and do not archive historical branches.
See [prefix state ownership](maintainer/paged-kv-cache.md#10-prefix-reuse).

The shared family runtime distinguishes `full_reset`, `append_frontier`,
`restore_turn_checkpoint`, `restore_response_checkpoint`, and `restore_prefix_checkpoint`.
Every restored checkpoint includes the
recurrent, hidden, and selected speculative-backend continuation state required to recompute a
rewritten suffix; matching KV tokens alone never authorize a partial hit. With stable
`preserve_thinking=true`, the auxiliary checkpoint rolls to the prompt frontier after the current
response's complete deterministic generation prologue. For thinking generation this includes
`<think>\n`; for non-thinking generation it includes the complete empty thinking block. Capturing
that frontier does not split a tiny trailing prologue into a separate prefill unit, and a normalized
response which no longer matches the raw generated tokens replays only that response and its
suffix. Stable `false` keeps the first assistant opener in the open turn so a newly closed turn can
be recomputed without its reasoning.

`preserve_thinking` selects where the next checkpoint should live; it is not a cache-compatibility
bit. An exact current frontier or matching complete checkpoint remains reusable across a mode
change. If the newly desired boundary is already behind the selected reuse frontier and no snapshot
exists there, the Engine keeps the valid hit and defers installing that new checkpoint rather than
forcing an eager full reset. A later request that diverges before every retained checkpoint then
resets normally. The JSONL completion record exposes the checkpoint actually restored as
`prefix_reuse_path`. Changing reasoning effort changes rendered tokens and therefore does not reuse
a prefix whose effort instruction differs.

An appended mid-conversation system message is an ordinary prompt suffix, so an unchanged prior
history remains eligible for `append_frontier`. If the client modifies, removes, or moves a
historical system message, the token prefix genuinely differs and a miss/reset is correct.

Speculative decoding is an engine option and does not change protocol output shapes, stop behavior,
or usage accounting. If a stop truncates a multi-token MTP or DFlash round, the Engine commits the
exact accepted target prefix so a following compatible turn can still reuse it. Output-limit and
context-capacity finishes map to `length`/ `max_tokens`; ordinary model or string stops map to
`stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools and does not enforce client JSON Schema through constrained
decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.

Generated text is always valid UTF-8. The checkpoint vocabulary is byte-level, so a single code
point can span several generated tokens: the Engine holds an incomplete trailing sequence and
publishes it in the delta that completes it, and no streamed chunk ever ends part-way through a
code point. A generated byte sequence that no continuation can complete -- a byte-level token that
can neither begin nor continue a code point, or a code point cut short by the output-token limit --
is published as one U+FFFD replacement character per ill-formed sequence. This is a per-request
text outcome: it never fails the request and never takes the server out of service. Token
accounting is unaffected, since the generated token IDs remain exactly what the model produced.

## CPU visual encoding (0.1.2)

Use `--vision --vision-device cpu` to trade visual encoding latency and host RAM for lower GPU
residency. Source codes and FP16 group scales are decoded exactly to one FP32 host copy on load;
there is no new weight quantization. CPU Ops retain BF16 activation/result boundaries, 2-D RoPE,
all causally permitted language keys and all keys within each temporal vision segment. CPU and
CUDA reduction/GEMM arithmetic are not bit-identical, so embeddings and output can differ.
No new GPQA or task-accuracy percentage is available.

CPU visual encoding runs in the submitting thread after the bounded ingress slot is reserved,
before the request joins the GPU FIFO. It reads immutable weights and does not hold the GPU
execution lock; other requests may continue decoding. Explicit submission deadlines and cancellation also bound
CPU work; the ordinary Engine default queue timeout begins after encoding. Request references are
freed after their GPU consumers finish. A Program-local immutable embedding cache retains at most 128 MiB in FIFO order, keyed by media digest, modality and grid;
repeated images do not rerun the CPU tower. This cache is separate from the text KV history budget. Large media still consumes host scratch
space in addition to resident tower weights. CPU placement does not enable Vision on unsupported
artifacts, TP4, PP2, DFlash or sparse RAM-KV.
For serving large CPU media, increase `--pending-timeout-ms` (for example `600000`); the server
default of 30 seconds still covers preparation and can expire during CPU encoding. GUI/TUI select
600000 ms when CPU Vision is enabled. The deadline is finite and can still reject oversized work.

`--volta-long-attention` can be combined with GPU or CPU Vision on SM70 27B TP2/PP1, complete
INT8 group-64 KV, None/MTP and experimental concurrency 1–8. It remains default-off and changes
floating-point association. `--volta-fast-decode` remains incompatible with Vision.
