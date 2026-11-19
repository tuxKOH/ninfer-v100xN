# NInfer CLI

Experimental native RAM-KV: add `--ram-kv-window 98304` to a 27B SM70 TP1/TP2/PP2 Text/MTP command.
This is approximate, training-free Q/K mean-vector retrieval, disabled by default. It keeps packed history
in RAM but omits unselected pages from attention (including long prefill). `--max-context` remains
the logical limit; omitted `--kv-capacity` follows the RAM window instead. If explicitly provided,
KV capacity must equal that window. `--ram-kv-budget-bytes` defaults to 32000000000 and bounds the
maximum archive across all ranks, not process RSS. Window must be a multiple of 64, at least
prefill-chunk+8192 (PP2: prefill-chunk*pipeline-chunks+8192), and no larger than max-context.
Only one active request; no Vision/DFlash/TP4.

`build/apps/ninfer` runs one request against one registered `.ninfer` artifact. Build NInfer and
download an artifact using the [project README](../README.md) before following this guide.

## Text input

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 16384 \
  --max-new 256
```

Exactly one of `--prompt` and `--messages` is required.

Answer content is streamed to stdout. Reasoning, model loading (including the registered target and
canonical `weights_id`), timings, throughput, GPU memory, and speculative-decoding statistics are
written to stderr, so stdout can be redirected independently:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Return one sentence." --max-new 64 \
  > answer.txt 2> run.log
```

Thinking is enabled by default. If the chat template embedded in the loaded artifact exposes
reasoning effort, `--reasoning-effort low|medium|xhigh` selects it; omitting the option uses the
template's default. An artifact whose template does not expose effort rejects the option. Add
`--no-thinking` for direct-response prompt rendering; it cannot be combined with
`--reasoning-effort`. `--greedy` selects exact argmax decoding independently.

## Startup memory profile

`--tp 4 --devices 0,1,2,3` selects the Qwen3.8-27B official NVFP4, experimental QUASAR NVFP4,
or native FP8 Text/MTP route on SM70. Four-card model/state and occupied-context results are
recorded in [`tp4.md`](tp4.md); native FP8 TP4 qualification and the measured occupancy
and capacity controls are complete.
The native `qwen3.8-27b/fp8` identity requires TP4. TP4 excludes Vision and DFlash.
SM70 TP4 admits the default-off long-attention profile and, for QUASAR, fast decode.
Existing TP2 qualification and results remain separate.

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash weights and state and the optimized proposal head;
- `--spec mtp` loads only MTP, while `--spec dflash` loads the artifact's optional text-only
  DFlash package and omits MTP;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights, Vision scratch phase, and frozen
  request-transient allocation;
- `--vision` enables image/video input. `--vision-device gpu` (default) loads those GPU allocations;
  `--vision-device cpu` instead keeps the tower in host RAM and reserves only the GPU embedding transient.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: a
text-only Engine rejects media and cannot enable Vision later. DFlash and Vision are mutually
exclusive. The default speculative and Vision settings produce the smallest resident profile.
`--vision-max-tokens N` independently bounds merged visual tokens per request (`1..32768`,
default `32768`) and startup Vision reservations; the prepared text-plus-media prompt still must
fit `--max-context`. On memory-constrained GPUs a smaller visual budget allows a larger text
context without promising to process images or videos beyond that separate budget.
QUASAR TP2/MTP4 passes the existing Vision/prefix integration test at 4096 capacity, including
the visual-prefix bridge. The older official-NVFP4 cached/cold discrepancy remains separate;
see [test status](../tests/README.md#experimental-tp2-vision-status). CLI processes do not share
Engine prefix state. The supplied Vision server preset starts with `--no-prefix-reuse`; QUASAR
users may enable prefix reuse manually.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
System and developer messages retain their array positions; the Qwen family frontend renders both
as system-class ChatML turns rather than moving later instructions to the beginning.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP with one to five draft positions.
Text-only DFlash supports one to fifteen on 35B-A3B, or one to seven on Qwen3.8-27B artifacts
containing the optional DFlash2 package. The official NVFP4 v3 container includes that package.
`--lm-head-draft` selects the optimized proposal head and requires a selected backend:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

For the two-V100 NVFP4 v3 DFlash2 profile:

```bash
./build-v100/apps/ninfer /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 --max-context 98304 --prefill-chunk 1024 --kv-dtype int8 \
  --prompt "Write a complete C++20 program." --max-new 2048 --greedy --no-thinking \
  --spec dflash --draft-tokens 7
```

MTP and DFlash cannot be enabled together, and DFlash cannot enable Vision. Qwen3.8 DFlash2's
lattice selector always consumes the full target vocabulary; the optimized MTP shortlist is not
its proposal path. The NVFP4 v3 drafter adds about 2.074 GiB of resident weights per TP rank.
The demonstrated V100X2 configuration fits at 98304 capacity/chunk1024; 180000 capacity does not.
On SM70, its W8 feature, query, attention-output and MLP projections automatically use an exact
load-time QPN layout permutation. Original codes/scales and one resident copy are retained;
narrow projections reuse a caller-owned image of the same FP16 input operands. There is no
additional option or artifact conversion.
QUASAR TP2 DFlash additionally admits `--volta-long-attention` and
`--volta-fast-decode`, independently or together. Long attention reuses all target
KV with a seven/eight-query verifier; fast decode changes the target NVFP4 arithmetic,
while the native W8/BF16 drafter is retained. Both remain default-off and may change
logits, acceptance and outputs; see the measured profile in the performance reference.
See [performance results](performance.md) for measured workloads and limitations. Seven drafts
are a measured choice, not the 35B-A3B semantic limit.

## Common options

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--rope native\|yarn` | rotary regime; `yarn` applies YaRN frequency correction and raises the `--max-context` ceiling to `--yarn-origin` x `--yarn-factor` | `native` |
| `--yarn-factor F` | YaRN scaling factor, in `[1.0, 64.0]`; only read under `--rope yarn`; `origin x factor` must be a whole token count not exceeding `1048576` | `4.0` |
| `--yarn-origin O` | YaRN origin window; must equal the artifact's registered native context capacity (`262144`) | `262144` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--tp 1\|2` | tensor-parallel width; `2` splits the model across two GPUs | `1` |
| `--pp 1\|2` | layer-pipeline degree; `2` requires SM70 Qwen3.8 NVFP4/QUASAR, TP1 and Text/None/MTP | `1` |
| `--pipeline-chunks N` | PP prefill wave of `1..8` chunks; `1` is the serial control | `8` |
| `--devices A,B` | exactly `tp * pp` distinct CUDA device indices; required for parallel execution | `--device` |
| `--kv-dtype bf16\|int8` | KV-cache storage | `bf16` |
| `--prefix-cache-tokens N` | historical host KV budget in unique page-rounded Main Text tokens; `0` disables historical branches, preserving current GPU-prefix reuse | `384000` |
| `--volta-long-attention` | experimental complete-history attention arithmetic; SM70 27B TP2/TP4, PP1, 1–8 active requests, INT8 KV; Vision and DFlash on TP2 only; excludes RAM-KV; floating-point association and output IDs may change | off |
| `--volta-fast-decode` | experimental QUASAR SM70 TP2/TP4 Text/None/MTP or TP2 Text/DFlash numeric profile: packed NVFP4 FP32 epilogue and MTP W8 reductions; DFlash retains native W8 drafter; may change logits, acceptance and output; excludes Vision/PP2 | off |
| `--spec mtp\|dflash` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash `1..7` on 27B, `1..15` on 35B-A3B | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--vision-device gpu\|cpu` | visual encoder placement; requires `--vision` for CPU; CPU weights remain in host RAM, final BF16 embeddings upload; may change numerical outputs | `gpu` |
| `--vision-max-tokens N` | merged visual tokens per prompt and startup Vision reservation, independent of text context (`1..32768`) | `32768` |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-thinking` | disable thinking in prompt rendering | thinking on |
| `--reasoning-effort low\|medium\|xhigh` | select an effort exposed by the loaded chat template | template default |
| `--greedy` | exact argmax decoding | off |
| `--ignore-eos` | drop the checkpoint's own end-of-turn token ids from the request stop policy | off |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |

When a sampling flag is omitted, Engine selects the official general-task preset registered for
the loaded model and the rendered prompt mode. The current presets are:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Qwen's separate precise-coding recommendation
is task-specific and is therefore an explicit override rather than an inferred Engine default.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

`--ignore-eos` removes the checkpoint's registered `eos_token_id` set from the request stop policy,
so decode continues until `--max-new` is exhausted or the remaining context capacity is reached
(`finish reason` `output-limit` or `context-capacity`) instead of ending on the model's end-of-turn
token. Stop conditions added with `--stop-token-id`, `--stop`, and `--reasoning-stop` are
unaffected. This exists for fixed-length decode work such as sustained long-context soak and
throughput measurement; normal use should leave it off, because generation past the end-of-turn
token is off-distribution and its text is not a product output.

Run `./build/apps/ninfer --help` for the exact option contract.

## Dual-GPU execution

`--tp 2` splits one resident model across two CUDA devices and requires an explicit `--devices A,B`
naming two distinct devices of the same compute capability. `--device` alone selects the single
device used at the default `--tp 1`; when both are given they must agree on the primary device.

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 262144 --kv-dtype int8 --kv-capacity auto \
  --messages long_prompt.json --max-new 256 --no-thinking
```

Tensor-parallel execution is implemented for the 27B execution package (`qwen3.6-27b` and
`qwen3.8-27b`, either weight profile). `qwen3.6-35b-a3b` has no tensor-parallel path and rejects
`--tp 2` at startup. `--spec mtp` and optional Qwen3.8 DFlash2 support `--tp 2`, including
compatible-prefix reuse in a resident Engine. Experimental `--vision` is also available at TP2
for artifacts with execution-qualified Vision weights; QUASAR NVFP4 admits TP2 None/MTP; GGUF-derived Q4_K_M rejects it.
Both suffix prefill and exact-frontier
sampling restore the complete state on both devices. The HTTP server enables reuse by default;
separate CLI processes do not share a cache.

`--tp 1 --pp 2 --devices 0,1` instead selects the experimental SM70 Qwen3.8-27B
NVFP4/QUASAR layer pipeline. Device 1 owns layers 0..31; device 0 owns layers 32..63,
the output head and MTP. `--pipeline-chunks 8` submits a bounded prefill wave so the
next first-half chunk overlaps the previous second-half chunk; `1` is the non-overlapped
control. Decode traverses the two stages serially. PP2 supports single-request Text,
None/MTP speculation, prefix reuse and optional sparse RAM-KV, not Vision, DFlash or expert-storage mode.
The choice is fixed at startup: it does not switch to TP2 after prefill or by prompt length.

```bash
./build-v100-tp4/apps/ninfer /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --tp 1 --pp 2 --devices 0,1 --pipeline-chunks 8 \
  --max-context 180000 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --prompt "Write a bounded queue in C++."
```

The 35B-A3B package also supports expert-storage mode on two GPUs without tensor parallelism. Keep
`--tp 1`, select the compute GPU with `--device`, and pass the other GPU as `--storage-device`:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --device 0 --storage-device 1 --max-context 65536 \
  --kv-dtype int8 --kv-capacity auto --prompt "Hello"
```

This maps the routed expert banks into one primary-device virtual address range; attention, KV,
GDN, sampling, and scheduling remain single-device. The two GPUs must support peer access and
share `sm_70`/`sm_86`/`sm_89` (the option is rejected with `--tp 2`).

The load summary reports weights, KV pool, GDN state, sequence, workspace, CUDA Graph and reserved
bytes per device, plus a free/total row for each. `--no-cuda-graph` runs decode eagerly; at `--tp 2`
that is the same two-stream forward pass without a captured graph, and the tokens are identical.
`--tp 1` is bit-identical to single-device execution.

Speculative decoding at `--tp 2` is output-equivalent to non-speculative decoding rather than
bit-identical: a verify round evaluates the target model over `draft_tokens + 1` columns at once
while an ordinary round evaluates one, which selects different GEMM shapes, so greedy streams can
diverge on a near-tie token. Every committed token is still one the target model's own argmax
selected.

## Context and memory

The registered model IDs have a native context limit of 262,144 tokens, which is the ceiling
`--max-context` is checked against under the default `--rope native`. `--rope yarn` applies YaRN
frequency correction and raises that ceiling to `--yarn-origin` x `--yarn-factor`, up to 1,048,576
tokens; `--yarn-origin` must equal the registered native capacity (262,144); `--yarn-factor`
accepts a finite value in [1.0, 64.0] and defaults to 4.0. YaRN works at either `--tp` width and is
rejected together with `--vision`, `--spec dflash`, or a target with no YaRN rope domain
(`qwen3.6-35b-a3b`). The load summary's
`rope` row reports the resolved mode, factor, origin, effective ceiling and `mscale`. The practical
allocation on one RTX 5090 depends on the selected artifact, media workload, output budget, and
KV-cache type.
Use `--kv-dtype int8` for large context allocations; at `--max-context 1048576` it is mandatory,
because a BF16 pool at that window does not fit two RTX 5090s. The 1,048,576-token configuration is
`--tp 2 --devices A,B --rope yarn --yarn-factor 4.0 --yarn-origin 262144 --kv-dtype int8
--kv-capacity auto`; it reserves 26.93 GiB per device without a speculative backend and 28.42 GiB
with `--spec mtp --draft-tokens 3`. A full-ceiling prefill takes about 19 minutes on the measured
host. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, workspace, Vision
request transient, and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program scratch arena, the maximum Vision request-transient buffer when Vision is enabled, and a
separate CUDA Graph driver allowance. Scratch is the maximum of the enabled Text, MTP, DFlash, and
Vision phases, not their sum. Its prefill bound uses
`min(--prefill-chunk,--max-context)`. The request-transient buffer is also frozen at startup; a
media request activates only the needed prefix and performs no project-owned device allocation or
growth.

All weight, sequence, workspace, request-transient, and graph allocations are released when the
Engine is destroyed.

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
