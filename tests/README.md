# Tests

Native RAM-KV checks: `ninfer_ram_kv_test` covers exact packed page eviction/reload and sparse
GQA against the independent FP64 oracle (TP2 geometry, BF16/INT8, decode/cached/Volta prefill).
It also checks Q aggregation, inverse-RoPE cached-K means, masked partial-page replacement and
cosine scoring against independent FP64 formulas at H=2/4, including original positions near 85K.
`NINFER_RAM_KV_REAL_WEIGHTS=/absolute/model.ninfer ctest -R '^ninfer_ram_kv_real_test$'`
is opt-in: default/full-window token parity, real 29.8K eviction, MTP partial commit, complete
checkpoint replay, and appended-query reuse on two V100s. It does not qualify general quality
equivalence of approximate retrieval. `tools/v100/bench_ram_kv.py` evaluates existing long-input
fixtures through HTTP/public Engine with a whole-server 32 GB cgroup cap.

The retained tests protect current `.ninfer`, numerical operator, target, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); Op contract and CUDA implementation guidance is in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Organization

`ninfer_tokenizer_control_test` is a self-contained CPU regression for #6. It interrupts live
repetitive BPE work with both cancellation and deadlines, exercises `prepare`, retrieval-query,
`count_tokens` and the processor's text encoder, and checks unchanged successful token IDs.
`ninfer_responses_schema_test` includes DSH's cache/reasoning hints, raw reasoning replay and SSE
Item lifecycle alongside the existing tools, continuation and capability-rejection contracts.

`ninfer_nvfp4_volta_batch_test` checks TP2 dense batch projections and paired SwiGLU against
independent FP64 oracles, covering T=9..64 tile boundaries and normal/fast scale profiles.
`ninfer_nvfp4_row_add_sm70_test` checks the contraction/packet/sum/norm path through T=64,
including changing-input Graph replay, guards and exact composition at its BF16 boundaries.
The opt-in `ninfer_qwen3_8_27b_v100x2_real_test --mtp-batches` mode checks real QUASAR MTP4
C2/C4/C8, normal and long/fast, greedy and mixed sampling, with complete Graph/eager output IDs
and speculative counters. It also requires the requested full batch to have executed.
`ninfer_gdn_tp4_decode_test` additionally covers prepacked TP4 normal/fast projection at
T=1/2/4/5/6/8 and masked record/convolution at T>=2. Its FP64 oracle applies the record
interface's observable BF16 projection boundary; exact composed-route, read-only state,
record/history and masked-zero checks supplement that oracle. It requires four CUDA devices.

SM70 NVFP4 input checks use independent host QPN encoding and FP64 logical projection oracles
at TP1/TP2 shapes, including QPN/wide-CUTLASS boundaries and graph replay. Run
`ctest -R '^ninfer_nvfp4_(input_sm70|gdn_.*)_test$'` for projection, snapshot-state, masked
record and public TP2 split checks.
`ninfer_gdn_tp4_decode_test` additionally covers prepacked TP4 normal/fast projection at
T=1/2/4/5/6/8 and masked record/convolution at T>=2. Its FP64 oracle applies the record
interface's observable BF16 projection boundary; exact composed-route, read-only state,
record/history and masked-zero checks supplement that oracle. It requires four CUDA devices.

- `artifact/` — Python container, registered layout, quantization, and resource behavior;
- `ops/` — one identifiable qualification suite per semantic Op or closely related overload group,
  using independent numerical/state-transition oracles at real supported shapes;
- `ops/linear/` — weight/activation-profile-specific public Linear conformance tests plus their
  one shared input generator, FP64 GEMM oracle, tolerance registry, and output/effects mechanics;
- `ops/linear_add/`, `ops/linear_pair/`, `ops/linear_swiglu/` — fused-Op suites split by registered
  weight/activation profile, each evaluating its complete formula rather than composing production
  Ops;
- `ops/test_allreduce.cpp` — the two-device `allreduce_sum`/`allgather_rows` collectives; it needs
  two CUDA devices visible to one process and reports the shared skip code when fewer are present.
  It checks eager/captured sums, small-message route boundaries and skewed 64-round chains.
  `NINFER_TEST_PEER_OFF=1` disables peer access only inside the test process to qualify the DMA
  fallback; it does not change the system's IOMMU or driver configuration;
- `ops/test_allreduce_tp4.cpp` — four-rank FP64 summation/byte-relocation oracles, uneven
  gathers, repeated broadcasts and source reuse, and production multi-stream graph capture/replay
  gates. Both staged DMA and qualified direct-P2P paths are exercised; needs four CUDA devices;
- `ops/test_linear_tp4.cpp` — four-rank column/row projection and exactly-once residual addition
  at 27B quarter-shard geometries,
  eager and graph replay, against complete-dot FP64 oracles with represented BF16 operands.
  The oracle does not round intermediate partials; row routes use the existing two-BF16-ulp
  tensor-scale criterion. This is not qualification
  of a new FP8/NVFP4 codec or a four-rank Engine route. Needs four CUDA devices;
- `targets/qwen3_6/` — shared tokenizer/template, multimodal preprocessing, MRoPE, prepared-prompt,
  stop/output decoding, hybrid topology, decoder/GDN and round-state layouts/views, shifted-MTP
  alignment, Vision control, and family runtime mechanisms;
- `targets/qwen3_6_27b/` — registered inventory, converter recipe, source verifier, artifact
  bindings, reference diagnostics, family Program/multimodal/MTP behavior, and the opt-in real-Engine
  prefix test;
- `targets/qwen3_6_35b_a3b/` — registered inventory/converter contracts, artifact-native diagnostic
  reference, MoE oracle, typed binding, selected-expert row access, 256K INT8 memory calculation,
  and the opt-in real public-Engine route;
- `test_ninfer_artifact_reader.cpp` — C++ framing, directory, encoded-size, payload-span, and
  geometry behavior against a self-contained C++ fixture;
- `test_request_memory.cpp` — startup-frozen request-transient capacity, stable address,
  activation alignment, rejection, and peak semantics;
- `test_openai_schema.cpp`, `test_responses_schema.cpp`, `test_response_store.cpp`,
  `test_anthropic_schema.cpp`, and `test_tool_call_parser.cpp` — current protocol translation,
  Responses Item/state/SSE behavior, and incremental tool-call behavior;
- `test_request_log.cpp` and `test_http_error_handler.cpp` — generation lifecycle records,
  preparation rejections, protocol-shaped payload-limit errors, and application-error preservation;
- `test_ninfer_bench_support.cpp` — product benchmark CLI, timing boundary, and schema-v15 reports;
- `test_bench_matrix.py` — schema-v15 report consumption by the Python matrix summarizer;
- `test_serve_corpus.py` — serving request-log schema compatibility at the measurement consumer;
- device/tensor/arena tests — reusable lower-component behavior; KV tests cover the core physical
  container, family runtime tests cover dimension-driven GDN storage/view mechanics, and Op tests
  cover mathematical state transitions at their own boundary.

Tests are grouped by observable risk, not by mirroring every source file or class.
`ops/op_tester.h` and `ops/op_check.h` own only reusable device/guard and comparison mechanics.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

`ops/quantized_weight.h` is the common packed-weight fixture for Q4/Q5/Q6/W8 and NVFP4 Op tests. It
owns deterministic payload generation, device `Weight` views, row views, and independent logical
weight decoding.

## Build and run

Model hot unload/load: `test_launcher_model.py` checks asynchronous control, authenticated HTTP,
progress, busy errors and recovery without CUDA. The artifact reader fixture verifies exact packed
bytes and aligned reads after deleting the source file. For a real QUASAR TP2/MTP4 CUDA Graph
lifecycle (busy admission, 503 while in RAM, disk-independent reload, deterministic output,
discarded KV/prefix cache and cumulative stats), run:

```bash
NINFER_RESIDENCY_MODEL=/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build-v100-tp4 -R '^ninfer_model_residency_real_test$' --output-on-failure
.venv/bin/python -m unittest tests.test_launcher_model tests.test_gui_cache tests.test_ninfer_tui
```

The real-model test uses a temporary symlink, removes only that link, and never modifies the model
artifact. Without the explicit environment variable it is skipped. It requires two free V100 GPUs
and enough host RAM to retain the complete container.

The standard-library TUI launcher checks cover admission, saved preferences, metrics and owned
child lifetime, including an actual curses session in a pseudo-terminal with a fixture server.
The GUI checks use Qt when installed (otherwise only the log consumer checks run). Neither
suite builds CUDA tests or loads a model:

```bash
python3 -m unittest discover -s tests -p test_ninfer_tui.py
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -p test_gui_cache.py
python3 -m unittest discover -s tests -p test_launcher_resources.py
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

On a host with more GPUs, select the physical cards for CTest at configuration time:

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DNINFER_TEST_CUDA_VISIBLE_DEVICES=4,6
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure -j1
```

This sets `CUDA_VISIBLE_DEVICES=4,6` for every CTest process, including child processes.
CUDA renumbers that pair as logical devices `0,1`, in the selected order. Four-device tests
skip with only two visible devices; select four physical cards to run them. An empty value
(`-DNINFER_TEST_CUDA_VISIBLE_DEVICES=`) leaves the calling environment unchanged, so
`CUDA_VISIBLE_DEVICES=4,6 ctest --test-dir build ...` also works without a configured mask.
Compilation targets the selected architecture and needs no GPU or fixed GPU count. Directly
invoked executables use their environment and the CLI's `--devices`/`--device` options;
the CTest mask does not select GPUs for an independently launched server or CLI.

Run a focused target for a localized change:

```bash
cmake --build build --parallel --target ninfer_sampling_test
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninfer_(rmsnorm|gqa_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

Linear tests are independently runnable by weight and activation-compute profile:

`ninfer_linear_w8_prepacked_sm70_test` additionally checks the exact load-time W8 vocabulary
permutation at full/half/quarter-head geometries, the independent FP64 projection oracle,
original-layout output parity, graph replay and persistent-plane/input guards. It covers
one/two-column SIMT, narrow QPN and wide MMA consumers through public `linear()`.

`ninfer_linear_w8_volta_decode_test` checks the five SM70 one-token MTP shard
projections against the independent FP64 oracle and the original SIMT output,
including CUDA Graph replay, output guards and unchanged activation/weight bytes.

`ninfer_linear_q4_volta_proposal_test` checks the SM70 TP2 one-token Q4 proposal
head against the independent FP64 oracle and original staged GEMV output, with
CUDA Graph replay and unchanged activation, weight and output-guard bytes.

`ninfer_long_attention_real_test` is an opt-in SM70 TP2 QUASAR 85K full-history
logit/state check, not a routine CTest. It uses one explicit artifact, corpus,
captured continuation report and output prefix. Set
`NINFER_LONG_ATTENTION_TEST_DRAFTS=4` or `5`. Run `reference` before `candidate`,
and `graph` before `eager`:

```bash
NINFER_LONG_ATTENTION_TEST_DRAFTS=4 \
  build-v100-tp4/tests/ninfer_long_attention_real_test reference \
  /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  profiles/bench/v100-code-85000-iommu-pt.ids \
  profiles/bench/quasar-150-tp2/mtp4-single-qpn.json \
  profiles/bench/quasar-150-tp2/long-qpn-mtp4-quality
```

The reference and candidate share teacher-forced history across single-token and
verification-width suffixes. The diagnostic `control` changes only the default
prefill chunk from 2560 to 1024; its arithmetic variation does not waive the
candidate's bounds. State modes check graph/eager commit IDs, acceptance counters,
peer egress and output-prefix equality with the supplied benchmark report. These
checks do not establish general model quality equivalence.
The `default` mode instead requires every captured BF16 vocabulary logit to equal
the saved default reference exactly. `default-graph` and `default-eager` check
the default arithmetic's MTP state using its own measured continuation report.

`tools/v100/bench_gpqa.py` and `ninfer_v100_gpqa` run a frozen paired GPQA-Diamond
**85K unrelated-code occupancy variant**, with resumable per-question results.
`python3 -m unittest tests.test_gpqa` checks paired scoring, truncation accounting
and shared-setting validation. The delivered evaluation stopped at 104 common
questions; this does not qualify a full 198-question score or universal numerical
equivalence. See the [local results and limits](../docs/performance.md#local-gpqa-diamond-comparison-104-paired-questions).

```bash
cmake --build build --parallel --target \
  ninfer_linear_q4_a16_test ninfer_linear_q5_a16_test \
  ninfer_linear_q6_a16_test ninfer_linear_w8_a16_test
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|w8)_a16_test$' --output-on-failure
```

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/targets/qwen3_6_27b tests/targets/qwen3_6_35b_a3b \
  tests/test_bench_matrix.py tests/test_serve_corpus.py
```

The Python binding tests use `NINFER_QWEN3_6_27B_ARTIFACT` when set, otherwise they look for
`out/qwen3_6_27b.ninfer`. They report a pytest skip when neither path provides the real
artifact. The 35B-A3B reference binding test follows the same rule with
`NINFER_QWEN3_6_35B_A3B_ARTIFACT` and `out/qwen3_6_35b_a3b.ninfer`. The remaining Python
target tests still run without either artifact.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine:

```bash
NINFER_QWEN3_6_27B_WEIGHTS=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_prefix_real_test --output-on-failure
```

For TP2 Vision on the official Qwen3.8 NVFP4 artifact, build
`ninfer_qwen3_6_27b_prefix_real_test` and run the retained two-device variant:

```bash
NINFER_QWEN3_6_27B_NVFP4_WEIGHTS=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build-v100-duo -R '^ninfer_qwen3_6_27b_vision_tp2_real_test$' --output-on-failure
```

It checks real multimodal prefill, cross-chunk image lifetime, shifted MTP alignment,
same-media reuse, changed/appended media, and visual prefix bridges. To verify actual concurrent
HTTP generation, use the README's experimental Vision server profile with port `18081`,
`--max-concurrency 2`, `--vision-max-tokens 2048`, `--greedy`,
`--request-log-jsonl /tmp/ninfer-vision.jsonl` and `--log-stats-interval-ms 500` (retain
`--no-prefix-reuse`). These larger reservations require additional free GPU memory. Then run
with Python 3.11:

```bash
/path/to/python3.11 tools/v100/check_vision_http.py --request-log /tmp/ninfer-vision.jsonl
```

The HTTP check generates exact red-circle/blue-square PPM scenes on white backgrounds, verifies
their reported colors and geometry (not exact wording), tests a 1,630-token cross-chunk prompt
and two images, and sends two simultaneous streaming requests. It requires overlapping content
streams and structured-log evidence of an actual multi-row decode batch, not just two HTTP 200s.
Use a dedicated idle server with a 1,024-token prefill chunk and at least two request slots.

Run the peer 35B-A3B route independently:

```bash
NINFER_QWEN3_6_35B_A3B_WEIGHTS=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_35b_a3b_real_test --output-on-failure
```

Without the corresponding variable CTest marks each C++ integration test as skipped. Neither test
uses another numerical/execution path's generated tokens as a golden.

The V100X2-named gates accept registered Qwen3.8 Q4_K_M/NVFP4 at TP2 (default), or
Qwen3.8 NVFP4/native block-FP8 on four SM70 cards with `NINFER_TEST_TP=4`.
`NINFER_V100X2_SPEC=mtp` (default) selects MTP3; `NINFER_TEST_MTP_DRAFTS=4` selects MTP4.
`dflash` selects DFlash7 and requires an
artifact with the optional drafter. `NINFER_V100X2_PROPOSAL_HEAD=optimized` is MTP-only.
`NINFER_TEST_PP=2` selects the experimental SM70 Qwen3.8 NVFP4/QUASAR TP1/PP2
single-request gates. Its real gate retains graph/eager, sampling and all 64
teacher-forced checks but does not run the unsupported B=2 TP test. The prefix gate
keeps checkpoint/append/partial-stop/exact-frontier replay across both layer-local
states. `ninfer_gdn_replay_fold_test` qualifies the 24-layer, 48-value-head PP2
geometry against the independent recurrent-state oracle.
The real gate checks exact graph/eager commits, sampled routing and B=2 state. Q4_K_M/NVFP4
require strict non-speculative teacher-forced argmax at all 64 output positions. Native FP8
additionally measures ordinary decode's re-prefill discrepancy: each prompt's speculative
worst emitted-logit deficit must not exceed its ordinary control, within the established
0.5-logit BF16 grouping bound. This is not a bit-identical-trajectory or quality-score claim.
The prefix gate checks checkpoint, append, partial-stop and exact-frontier replay, including
an exact stop-vs-budget truncation control. It also checks exact-hit single-output termination,
subsequent replay/suffix restoration and B=2 mixed-sampling MTP checkpoint isolation.
Cold re-prefill can change BF16 grouping; its
first differing choice is checked separately against fresh target logits. DFlash does not
use the MTP-only peer-egress diagnostic and is unavailable at TP4.

```bash
NINFER_V100X2_ARTIFACT=/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
NINFER_TEST_MTP_DRAFTS=4 \
  ctest --test-dir build-v100-tp4 -R 'v100x2_(real|prefix_real)' --output-on-failure
```

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares restored and cold greedy output, compares stripped and preserved closed-turn
prompt lengths, and verifies turn/response rewrite-checkpoint reuse paths plus Responses
inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

### Experimental TP2 Vision status

The TP2 Vision path is opt-in. On 2026-10-19, QUASAR v3 with default numerical profiles,
INT8 group-64 KV, 4096 capacity, 1024 chunks, Graphs and optimized-head MTP4 passes
`ninfer_qwen3_6_27b_prefix_real_test`, including image lifetime, same/changed/appended media,
stop/resume, exact-frontier reuse and the visual-prefix bridge. Select the artifact with
`NINFER_QWEN3_6_27B_NVFP4_WEIGHTS`, TP2 with `NINFER_TEST_TP2=1`, and four drafts with
`NINFER_TEST_MTP_DRAFTS=4`. Real image/video CLI and Vision ON/OFF HTTP checks also pass.
These are integration checks, not a scored visual-quality evaluation.

The earlier official-NVFP4 qualification is a separate unresolved result:
With the official Qwen3.8-27B NVFP4 artifact, INT8 group-64 KV, 4096 capacity, 1024 prefill chunks
and optimized-head MTP3, `ninfer_qwen3_6_27b_vision_tp2_real_test` repeatedly fails its visual-prefix
bridge greedy-output comparison at the first output token. Earlier cross-chunk image lifetime,
same/changed/appended-media reuse and stop/resume checks complete before this failure.

The differing candidates are not an exact BF16 tie, but this does not by itself establish a state
bug or a model-quality regression. The cause remains unresolved; no numerical criterion has been
relaxed and the failing test is retained. The supplied Vision presets start with `--no-prefix-reuse`. The older official result does not
negate the QUASAR integration result above, and the QUASAR result does not qualify the official
artifact or prove all chunkings and numerical profiles equivalent.

## What belongs here

A permanent test should protect one current risk, such as:

- exact registered artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- family Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.

## CPU Vision and visual long attention (0.1.2)

`ninfer_host_vision_test` runs without a GPU. It checks exact signed-code/FP16-scale decoding for
all real tower/merger matrix shapes, FP64 linear and full segmented attention oracles, independent
LayerNorm/GELU/RoPE/position interpolation, and BF16 representation. The CPU path retains original
mathematical weights and BF16 Op outputs; pairwise GPU equality is not its numerical oracle.

`ninfer_quasar_load_plan_test` checks that CPU Vision places every tower object on host and none
on either GPU, while GPU Vision remains primary-rank-only. `ninfer_qwen3_6_27b_prefix_real_test`
accepts `NINFER_TEST_CPU_VISION=1` and `NINFER_TEST_LONG_ATTENTION=1` alongside the existing TP2
and MTP4 settings to exercise real cross-chunk media, continuation, edited media and MTP state.
`ninfer_vision_long_real_test` accepts `--cpu`, `--image`, `--eager`, `--prompt-tokens N` and
`--output-tokens N`; set the exact `NINFER_QUASAR_ARTIFACT` path. Its default is 85,000 occupied
text tokens with GPU Vision enabled, long attention and MTP4; it prints JSON wall timing,
acceptance, memory and committed IDs. These checks are not GPQA or proof of checkpoint quality.
