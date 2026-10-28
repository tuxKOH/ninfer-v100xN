# AGENTS.md

These rules apply to the whole repository.

## Governing objective

Complete the user's explicit deliverable within the applicable product contract. For the declared
product target and requested outcome, choose the technically strongest coherent solution. Optimize
for architectural integrity, clear ownership, functional and numerical correctness, and maximum
relevant performance. Never optimize a solution for a small diff, few changed files, low
implementation effort, short-term simplicity, backward compatibility, or preservation of a
superseded internal path. Make every affected implementation, test, tool, and active authority
consistent with the selected design.

Correctness, performance, tests, profiling, documentation, provenance, cleanup, and tooling are
means to the requested outcome, not independent objectives. Do not let supporting work replace,
delay, or materially enlarge the requested deliverable.

## Responding to user corrections

When the user points out an error in the agent's execution or reasoning, never reply with the
formulaic opening “你说得对，……” ("you're right, ..."). This phrase and cosmetic variants of it
are strictly prohibited in correction responses because they sound reflexive and insincere. Do not
replace it with another generic agreement such as “确实如此”, “完全正确”, or “好问题”. Instead,
state the specific mistake directly, explain its concrete effect when relevant, and say what has
been or will be changed. Keep the response proportionate; do not add performative apology or praise.

When choosing between possible work, use this order:

1. respect applicable product and external-contract constraints;
2. satisfy the user's explicit deliverable and acceptance criteria;
3. preserve functional and numerical correctness of supported behavior;
4. choose the strongest architecture and clearest ownership for the declared product model;
5. maximize performance at the scope relevant to the task;
6. gather only the evidence and provenance needed to support the result.

The product and architecture described here are the current contract for ordinary work. A task may
explicitly change that contract; when it does, update the affected implementation, tests, and active
authorities consistently rather than treating the current description as an immutable prohibition.

## Scope control

Before substantial work, determine the requested output, the behavior or decision it must support,
and the conditions under which it is complete. This is an execution discipline, not a requirement
to create a separate planning artifact.

Work is in scope only when it:

- directly contributes to the requested deliverable;
- is necessary to preserve an applicable product, semantic, or external contract;
- resolves uncertainty that could materially change the result; or
- checks a realistic regression introduced by the change.

An architectural redesign, cross-cutting refactor, or replacement of an existing path is in scope
when it is necessary to deliver the strongest solution for the requested outcome. Do not use scope
control as a reason to ship an inferior patch. Do not expand into unrelated audits, cleanup,
hardening, compatibility work, benchmark campaigns, or documentation projects. General engineering
preferences, possible future scenarios, and concerns outside the declared product model do not
create requirements by themselves.

Handle incidental findings proportionally:

- address them when they block the requested outcome or make it materially incorrect;
- include them when they are inseparable from a coherent implementation;
- otherwise leave them unchanged and mention them only when they are useful to the user.

For analysis, review, or design work, the requested explanation or design artifact is the
deliverable; experiments and code inspection serve only to resolve material questions. For
implementation work, implement the selected design completely across its affected boundaries,
remove the superseded project-owned path, and validate its supported observable behavior. For
diagnosis, establish the cause and supporting evidence without turning the task into an unrequested
fix or redesign.

## Evidence, provenance, and completion

Select evidence from the claim or decision it supports. The availability of a tool, test suite,
artifact, or profiler does not make its use necessary. Prefer representative evidence over
exhaustive evidence, and do not repeat an experiment unless the previous result is invalid or
inconclusive, or the new result could change a live decision.

Verification must match the semantic contract: use exact comparison for exact formats and
transformations, and numerical or behavioral criteria for floating-point and probabilistic work.
Do not substitute final-output plausibility for verification of an operator or state transition.

Record only the provenance needed to interpret a material result. By default, this is the relevant
target, hardware/toolchain, workload or command, and summarized outcome. Fixed hashes, clean
worktrees, full command transcripts, raw profiler inventories, byte-identical regeneration, and
exact probabilistic outputs are not validity requirements unless a concrete contract or the user
requires them.

Stop when:

- the requested deliverable exists;
- applicable contracts are satisfied;
- material claims have sufficient evidence;
- relevant checks pass, or their limitations are stated clearly; and
- no known in-scope issue prevents the result from being used.

Do not continue merely to eliminate all uncertainty, collect more metrics, complete a process loop,
improve descriptive provenance, investigate unrelated observations, or make working notes
exhaustive. The final result should lead with the deliverable, key decisions, relevant verification,
and material limitations. Raw logs, experiment diaries, exhaustive command histories, hashes, and
intermediate artifacts are excluded unless requested or themselves the deliverable.

## Current product contract

NInfer is a from-scratch C++/CUDA inference engine for a small set of explicitly registered
checkpoint artifacts. This checkout's active target is maximum single-request performance on
two Tesla V100-SXM2 16 GB cards using CUDA 12.8 and `sm_70`. It combines the RTX 3060 TP2 path
with the Volta implementation from `geoffwatts/ninfer-v100`. Ampere `sm_86` and early Ada `sm_89`
builds remain available; the RTX 5090 `sm_120a` results are inherited upstream evidence, not
measurements of this target. The supported identities are
`qwen3.6-27b/groupwise-int`, `qwen3.6-27b/nvfp4`, `qwen3.8-27b/groupwise-int`,
`qwen3.8-27b/nvfp4`, `qwen3.8-27b/fp8`, `qwen3.8-27b/gguf-q4-k-m`, and
`qwen3.6-35b-a3b/groupwise-int`.
The explicitly requested QUASAR trial adds `qwen3.8-27b/quasar-nvfp4`, currently restricted to
SM70 TP2 Text/None/MTP/DFlash, TP2 image/video Vision with None/MTP, experimental TP4
Text/None/MTP, and experimental PP2 Text/None/MTP. Vision excludes DFlash, PP2 and the fast-decode numerical profile. Vision admits the default-off
long-attention profile on TP2 with complete INT8 group-64 KV. It registers the published QUASAR v3 container identity, preserves its
256 NVFP4 parents and FP32 activation divisors, and keeps its W8 vocabulary/MTP and BF16 GDN
controls and optional W8/BF16 DFlash2 package. The 27B draft window remains at most seven.
This artifact's QAT/conversion quality is not established by inference smoke tests.
Text/None/MTP with complete GPU KV additionally retains exact per-rank GDN and tail-hidden
checkpoints in at most 2 GiB host RAM per Program, divided among configured lanes. Shorter or
edited shared prefixes restore the longest matching saved frontier and recompute its remaining
suffix; reported cached tokens exclude that recomputation. Historical branches additionally retain
immutable exact host KV chunks and complete checkpoints under a default 384000-token unique
page-rounded Main Text budget, with FIFO branch eviction and an independent 2 GiB archived state
cap. Active lane state images retain their separate 2 GiB cap. Restored lineage can share immutable
host chunks; every active request receives private writable GPU pages and GDN state. This admits
historical branch and completed-sibling restoration, not arbitrary substring reuse or partial-state
reuse for Vision/DFlash/RAM-KV. The token budget is not a process RSS limit.
The explicitly requested remote extension targets four NVLink-connected V100-SXM2 16 GB cards,
single-request TP4 Text/MTP on Qwen3.8-27B official NVFP4, QUASAR NVFP4 and native block-128
FP8. QUASAR TP4 admits the default-off long-attention and fast-decode profiles; its four-card
qualification and measurements are recorded separately from official NVFP4/native FP8. Native FP8 preserves
source E4M3 codes and BF16 block multipliers without requantization and requires TP4 on SM70.
TP4 Vision/DFlash and other TP4 identities are not admitted. Operator checks on the local two
cards do not substitute for four-card whole-model/state or end-to-end performance evidence.
The explicitly requested WSL2 extension targets QUASAR SM70 TP2 Text/MTP4 on two V100-SXM2
16 GB cards without direct P2P. Small aligned BF16 reductions may use startup-owned mapped
host packets with system fences and device-local epochs; wider reductions and gathers retain
verified CUDA UVA staging. This transport preserves the represented sum formula and introduces
no model quantization change. Its whole-model measurements are separate from native Linux.
The remote WSL deployment disables historical branches and prefix reuse at the user's request
to avoid host cache growth. See `docs/wsl/README.md` for qualification and deployment settings.
The V100X2 workload uses the GGUF-derived Qwen3.8-27B Q4_K_M artifact from the local LM Studio
directory, 180000-token context capacity, INT8 group-64 KV, CUDA Graphs, and MTP with at most
three drafts; accepting zero drafts is valid. Capacity must not be confused with prompt occupancy.
Its Q4_K/Q6_K codes and scales must remain unchanged. Numerical transformations and inference
optimizations require evidence against their applicable exact, mathematical, or behavioral oracle;
preserving packed bytes alone does not prove absence of end-to-end quality loss.

The five published identities retain Text, image/video Vision, MTP, prefix reuse, CLI,
OpenAI/Anthropic serving, and measurement through the same public `.ninfer` Engine route.
OpenAI `response_format` and Responses `text.format` JSON object/schema output is enforced by
xgrammar grammar masks applied before every None/MTP target selection; DFlash rejects it.
The GGUF-derived identity uses this same route for Text/MTP and rejects Vision; its retained
Vision objects are validation-only. The 35B-A3B target additionally supports text-only DFlash.
Qwen3.8-27B artifacts containing the optional DFlash2 package also support text-only DFlash,
including TP2; the official NVFP4 v3 package preserves W8 projections and BF16 auxiliary tensors.
The 27B draft window is at most seven. On this V100X2, the verified DFlash profile is 98304-token
capacity with a 1024-token prefill chunk, not the 180000-capacity MTP acceptance profile.

The optional `volta_long_attention` profile admits SM70 27B TP2/TP4, PP1 experimental 1–8-request
Text/None/MTP and TP2 Vision/None/MTP with complete INT8 group-64 KV, without RAM-KV. It retains every
causally visible key but changes split partitioning, softmax association and PV accumulation;
it is default-off and not bit-equivalent to ordinary attention. The 104-question local GPQA
comparison is behavioral evidence, not a full 198-question evaluation or proof of losslessness.

Vision optionally uses `VisionDevice::Cpu` / `--vision-device cpu` with `--vision`. Its tower
weights decode once from original signed groupwise codes and exact FP16 scales into one FP32
host copy, without requantization or GPU tower residency. This explicit CPU representation is
not a GPU repacking path. Family-owned host scheduling keeps BF16 activation boundaries, 2-D
RoPE, complete temporal-segment attention and the merger. Only final BF16 embeddings upload.
CPU encoding runs on the submitting host thread after ingress-capacity reservation and before
joining the GPU FIFO; it neither locks the GPU executor nor changes text/decode leaves. A caller's
explicit submission deadline and cancellation cover CPU work; the default queue timeout starts after staging.
Each request references immutable embeddings in a Program-local 128 MiB FIFO cache, keyed by media
digest, modality and grid, never the frontend's shared media cache. The CPU/GPU
arithmetic and long attention need not produce identical embeddings, logits or output. Real
qualification for this release uses the QUASAR v3 TP2/MTP4 artifact; no new GPQA is claimed.
Release 0.1.2 contains these Vision changes and an optional qualified Linux SM70 prebuilt;
QUASAR TP4 qualification is recorded separately and ships in release 0.2.0.

The optional `volta_fast_decode` profile is default-off and admits QUASAR SM70 TP2/TP4
Text/None/MTP. It selects packed NVFP4 parents with FP32 scale-after-K-reduction and
shape-specific packed MTP W8 reductions. The normal profile retains per-group FP16 scaling
with the same exact weight layout, fusion and communication optimizations. Both paths keep
original codes/scales/divisors and one resident copy. The fast profile can change logits and
draft acceptance. Together with long attention the final 85K/MTP4 measurements are
198.105/198.097 committed wall tok/s in the 2026-10-19 campaign; 200 was not achieved.
After device recovery the user reopened QUASAR TP2 optimization. On 2026-10-22 the user
removed the fixed speed goal and requested six decode trials, retaining MTP4: attention load
overlap, MTP tail fusion, TP2 communication/norm, NVFP4 QPN scheduling, GDN value tiles,
and verified history lookup. Neither 200 nor 250 tok/s is a completion condition or measured limit.
No current-combination GPQA result is available.
The earlier remote TP4 campaign retains MTP4 and that host's 220 W / 1200 MHz limits. After the
216.934 tok/s result, the user requested a measured same-host 1Cat comparison and reopened
optimization toward a 300 committed tok/s stretch target at 85000 occupied input tokens.
Match the saved prompt IDs, output budget, sampling, draft window and request concurrency;
record any unavoidable artifact, KV or arithmetic differences rather than claiming exact
equivalence. Short-context rates and clock extrapolations do not establish the 85K objective.
The retained 2026-10-23–24 TP4 optimization measures 283.118/283.367 committed wall tok/s
(mean 283.242) with both experimental profiles, versus 191.997/192.210 (mean 192.104)
with both flags off. Prefill averages 2334.73/2327.73 tok/s, respectively. The 1Cat-inspired
large collective and TP4 Split-D prefill improve the preceding approximately 1743 tok/s
prefill; Split-D changes floating-point association and the saved continuation first differs
from the preceding 231 stage at output 505. Publication, norm scheduling, verify5 prefetch,
proposal reduction, exact BF16→FP16 norm-image and single-launch split-10 GDN controls
additionally improve decode. TP4 GDN projection/record/conv fusion then removes intermediate
projection copies and launches, retaining the observable BF16 record boundary. These later
steps preserve all 513 fixture IDs and MTP counters of the qualified 244.655
stage, not universal output equality. The subsequent TP4 N64 prefill tile improves prefill
by 3.46%/3.45% and passes the independent FP64 oracle at the real final 85K chunk;
its saved continuations first differ from N32 at zero-based output 504/505. It changes
long/fast acceptance, so its 0.81% decode-rate increase is not an isolated decode-kernel gain.
Ordered long-attention split reduction and a BF16-rounded FP16 SwiGLU output image then
improve N64's 260.133/182.133 by 1.87%/0.80%, preserving both fixtures' 513 IDs and every
speculative counter. A GDN gated-norm FP16 image then adds 0.94%/0.67%, preserving
these complete fixtures and the explicit FP32→BF16→FP16 rounding boundary. The output
images are explicitly typed. Last-CTA GDN record/norm fusion adds 0.55%/0.30%, retaining
the Dv2/W4 recurrence, BF16 readout/records and resettable per-Program/rank completion counters.
A caller-owned FP16 W8 verifier input image then adds 0.65%/0.45%, retaining represented
operands, Split4 reduction, BF16 output and one resident weight image. Both preserve every
fixture ID/counter; independent recurrence/norm/W8 FP64 oracles, changed-input Graphs and
normal/longfast/prefix state gates pass without relaxed criteria. Direct-peer TP4 argmax
then adds 4.05%/2.78%; one origin kernel reads completed four-rank shards and publishes the
exact global tie-broken/mapped winner. Direct-peer I32 broadcasts through 4 KiB add another
0.56%/0.38%. Ready/completion edges protect source and output-consumer lifetimes. Exact CPU
argmax/relocation oracles, changed-input Graphs, reuse/guard checks and all three model-state
gates pass; all 513 fixture IDs and speculative counters remain unchanged. An MTP stem
sum/norm trial shows no benefit and is withdrawn. A Q/K-norm/RoPE prototype is also withdrawn
without whole-model qualification; neither direction is considered exhausted.
The same-host 1Cat 1.5.1 MTP4 control measures
98.249/102.509 (mean 100.379), with the same prompt IDs/output budget; FP16/E4M3-KV differs
from NInfer BF16/INT8-G64. Its captured comparison diverges after 325 shared output IDs.
Mean TTFT is 35.9378 s versus NInfer long/fast's 36.4077 s. These are not equal-quality
results. 300 tok/s was not reached and this is not a measured hardware limit.
After recovery, initcheck identified an eager-only residual/RMSNorm trial reading
uninitialized scratch. That branch was removed. Current four-rank numerical oracles and
normal, prefix and actual-85K long/fast Graph/eager, replay-logit/counter and committed-frontier
gates pass; the residual/norm regression passes initcheck with zero errors. These checks
do not establish model quality; no current-combination GPQA was run. TP4 work ships in release 0.2.0, separately from the earlier 0.1.2 Vision qualification. The subsequent 2026-10-24 host campaign explicitly
authorizes 300 W on its four test cards, to remain enabled for the user's video. It measures
322.178 committed wall tok/s at 85K/MTP4 with both experimental profiles. Its no-NVLink-traffic
control forces the qualified no-P2P staging route and verifies zero link-counter deltas; it is
not a direct PCIe P2P measurement.
See `docs/tp4.md`.

The explicit RAM-KV experiment is optional SM70 27B TP1/TP2/PP2 single-request Text/None/MTP.
It stores original packed KV in a bounded RAM archive and uses KVMem-inspired Q/K cosine
mean-128 retrieval with chronological resident page tables. Keys are decoded from the original
cache and inverse-rotated for the index; attention still consumes unchanged packed KV.
This is approximate attention (also during long prefill), not full-history semantic equivalence.
The default route is unchanged.
RAM archive budget and whole-process RSS/cgroup limits are distinct; external llama.cpp results
do not establish native NInfer performance. See docs/maintainer/paged-kv-cache.md section 15.

The requested pipeline experiment additionally admits SM70 Qwen3.8-27B NVFP4/QUASAR
startup-fixed TP1/PP2 single-request Text/None/MTP with prefix reuse. Device 1 owns
layers 0..31, device 0 layers 32..63 and output/MTP. Bounded chronological prefill waves
overlap stages; decode visits both stages serially. Optional sparse RAM-KV is admitted.
Vision, DFlash, storage mode
and concurrency above one are not admitted. No request-time TP/PP migration is provided.

One Engine owns one resident model, with one device, TP2, experimental PP2, or the selected TP4 domain on the 27B
package and a startup-fixed
one to eight active requests. The V100X2 acceptance workload is one active request on two cards.
The Engine forms one compact decode batch at every round boundary and uses bounded
FIFO ingress with no request preemption. Large-scale or preemptive continuous batching, priority/QoS
scheduling, additional checkpoint targets, and additional execution platforms beyond those named
above are outside the current product. This is a local, single-owner project. Registered models,
generated artifacts, and the local workflow are trusted.
Requirements derived from a different workload, trust model, or deployment model are out of scope
until that product contract is explicitly changed.

The explicitly requested GUI/TUI model hot unload/load retains one immutable original `.ninfer`
container in host RAM and destroys/reconstructs the GPU Engine through its public interface.
Serving admits transitions only with zero request lifetimes, rejects inference while RAM-only or
transitioning, and preserves the startup configuration and cumulative execution counters.
KV/GDN/prefix/processed-media caches do not survive; this is not request pause/resume. RAM survives
reload for subsequent cycles and is released on server shutdown; failed reload remains retryable.

The 27B and 35B-A3B execution packages are peer compile-time Variants of one identity-free Qwen3.6
family runtime. The family owns the shared `SequencePlan<Variant>`, `RequestPlan<Variant>`, and
`Program<Variant>` algorithms; frontend and output semantics; Text/Vision/speculative schedules;
state transactions; workspace composition; and CUDA Graph capture/replay mechanics. Each package
separately owns its registered artifact identities and bindings, immutable model view,
dimensions/storage facts, three closed execution-leaf families, graph frontier data, and Program
instance bytes. No mutable state or device allocation is shared between Programs, neither package
is defined as a delta from the other, and there is no runtime family selection or target-dependent
branch inside family scheduling. All artifacts embed the same six frontend resources, and a
prepared prompt carries no exact-target tag.

## Engineering priorities

Prioritize functional correctness, architectural quality, clear ownership, direct code, and maximum
requested inference performance. Change size, implementation difficulty, short-term simplicity,
and backward compatibility for project-owned contracts are not quality criteria. Low maintenance
cost may distinguish otherwise equivalent designs, but it never justifies worse architecture or
performance. Generality, defensive hardening, formal completeness, broad compatibility, and test
coverage are not goals by themselves.

Prefer explicit target-specific implementation over framework-like abstraction. Do not add generic
model graphs, family base classes, plugin discovery, string-driven execution, hidden device
allocation, runtime weight repacking, or placeholders for hypothetical models or hardware unless an
explicitly changed product contract requires them.

## Sources of truth

Read only current authorities relevant to a live decision in the task. The following list is a
routing map, not a mandatory reading list:

- `README.md` and executable `--help`: delivered capabilities and exact commands;
- `docs/README.md`: public documentation map;
- `docs/cli.md`: CLI input, output, sampling, MTP, and runtime options;
- `docs/serving.md`: OpenAI/Anthropic HTTP behavior;
- `docs/performance.md`: published performance methodology and results;
- `docs/maintainer/concurrent-inference-architecture.md`: bounded ingress, request/slot lifecycle,
  scheduling, batched execution, CUDA Graph, and speculative-concurrency semantics;
- `docs/maintainer/paged-kv-cache.md`: shared KV capacity, page ownership, retention, physical
  layouts, and paged consumer contracts;
- `docs/maintainer/artifact-container.md`, `storage-layouts.md`, and `tensor-formats.md`:
  generic `.ninfer` contracts;
- `docs/maintainer/qwen3.6-27b-artifact.md`, `qwen3.8-27b-artifact.md`, and
  `qwen3.6-35b-a3b-artifact.md`: exact target inventories, conversion, and binding;
- `docs/maintainer/qwen3.6-27b-model.md` and `qwen3.6-35b-a3b-model.md`: exact model mathematics,
  dimensions, and state semantics;
- `docs/maintainer/op-development.md`: Op admission, contracts, implementation ownership,
  qualification, and performance evidence rules;
- `include/ninfer/engine.h` and `include/ninfer/types.h`: in-tree C++ product interface.

Do not survey unrelated references for completeness. Read additional documents only when they
govern a live decision in the current task.

## Product and ownership boundaries

These boundaries govern ordinary implementation work. An explicit architecture task may revise
them, but must update the corresponding active authorities and affected implementation together.

- `.ninfer` is the only C++ product artifact. Do not add `.qus` fallback, extension detection,
  compatibility shims, or a second product lane.
- `include/ninfer/engine.h` and `include/ninfer/types.h` are the opaque Engine interface used by
  in-tree applications and owning host values. NInfer does not currently install or export a C++
  SDK. `include/ninfer/ops/` contains repository-internal semantic Op contracts.
- `src/core` owns device primitives, tensors/views, checked layouts, arenas, graph RAII, physical
  KV-cache containers, and raw transfer mechanisms.
- `src/artifact` owns generic `.ninfer` framing, descriptors, binding primitives, and
  materialization. It has no checkpoint execution semantics.
- `src/ops` owns every semantically closed Op implementation, including fused, fixed-shape, and
  device-specialized paths. Op ownership follows the mathematical or state-transition contract,
  not its first model caller or demonstrated cross-target reuse.
- `src/targets/qwen3_6` owns only the Qwen3.6-family invariants shared by the 27B and 35B-A3B
  targets: tokenizer/template and output semantics, media preprocessing and MRoPE prompt
  construction, owning prepared-prompt/output-session types, semantic weight-view schemas, passive
  Vision definitions, and the fixed
  planning/Program/Text/Vision/speculative/state/workspace/CUDA-Graph algorithms. It has no target
  identity, registry entry, artifact binder, target leaf
  implementation, or storage for a live Program instance.
- `src/targets/<package>` owns its registered checkpoint identities, storage profiles, binder,
  `LoadedModel`, configuration, populated family model-view values and private leaf payloads,
  diagnostics, graph frontier values, and exactly three execution-leaf families: attention
  projection, GDN projection/control, and post-mixer. It aliases and instantiates the family
  runtime types; it does not own a copied Program, Text/Vision/speculative schedule, workspace
  composition, state transaction, or graph-capture algorithm. Leaf Ops remain implemented under
  `src/ops`.
- `src/runtime` owns common contracts, generated-token transaction/publication policy, and the
  public Engine PIMPL. It does not own model mathematics or target state.
- `src/media/decode` consumes already-owned bytes. URL/path/data acquisition belongs to
  `src/product/media_acquire`, CLI, or serving and is not linked into a target.
- `src/product/prompt_input` owns the shared product-side JSON/message-to-owning-input adapter.
- `src/serve` owns protocol translation and transport. CLI, server, and benchmark call only the
  public Engine for inference.
- `tools/convert/<target>`, `tools/reference/<target>`, and `tools/parity/<target>` remain
  target-private conversion, correctness, and diagnostic implementations.

## Compatibility and document lifecycle

Project-owned C++ APIs, CLIs, Python tools, fixtures, reports, formats, and active documentation do
not preserve backward compatibility. When a task replaces project-owned behavior, remove the
obsolete aliases, fallbacks, transition branches, and tests in the affected contract instead of
maintaining two paths. Do not turn that rule into unrelated repository-wide cleanup.

The advertised OpenAI and Anthropic protocol surfaces are real external contracts. A change to
their behavior must update the affected schema tests and serving documentation together.

Integrate stable requirements into the existing active reference. Use a temporary dated plan only
when active work genuinely needs one; a plan is not a substitute for the requested deliverable.
Remove completed or abandoned plans instead of retaining a historical documentation tree. Do not
create parallel `final`, `v2`, or `new-design` references.

## Numerical correctness

When a task changes numerical behavior or makes a numerical claim, identify the mathematical
oracle, represented public inputs, explicit semantic cast/quantization/state boundaries, output
criterion, and real model shapes relevant to that claim. If a route's private precision or
reduction profile matters to the evidence, describe it as an implementation profile rather than a
semantic requirement. Apply exact, tolerance-based, or behavioral comparison according to the
actual semantic contract.

Every floating-point Op has one independent naive FP32/FP64 mathematical oracle; exact transforms
and codecs have one independent exact oracle. The oracle evaluates the complete logical formula
from the represented public inputs and, for packed weights, decodes the signed code with the exact
stored scale. It does not copy a production kernel's staging casts, reduction tree, workspace dtype,
or another implementation's output.

The oracle does not prescribe a production arithmetic path. Unless an intermediate value is an
observable Op output, explicit Cast/quantize/dequantize result, registered codec value, or specified
persistent state, kernels may choose the natural intermediate precision, instruction operands,
reduction association, workspace representation, and kernel decomposition for their route. A fused
kernel is neither required to reproduce an unfused BF16 materialization nor forbidden from using a
lower-precision intermediate when that is the natural qualified implementation. Every production
route is checked directly against the same oracle with a criterion appropriate to its output and
implementation profile; pairwise implementation parity is supplementary evidence only.

Where relevant to the changed behavior, account for numeric-format decode, BF16 fusion order, FP32
GDN state, BF16/INT8 KV, MTP accept/commit state, arena lifetime, and CUDA Graph address stability.
This is a risk map, not a checklist for every numerical task.

## Performance work

The requested V100X2 acceptance is above the user's LM Studio baseline without further quality
loss: about 45 committed decode tok/s at roughly 85000 occupied context tokens, with a reported
57 tok/s peak whose occupied context was not specified. The estimate of about 40 tok/s at longer context is not a
measurement. Both sides use 180000 capacity, Q4_K_M weights, Q8/INT8 KV, and a maximum draft
window of three. Record actual token counts, sampling, acceptance, and committed throughput.
Short prompts, draft throughput, or inherited RTX 5090 tables cannot establish this acceptance.

Define a performance claim at the level where it matters: operator, schedule, request phase, or
end-to-end inference. Measure that level directly when practical. An isolated microbenchmark can
support an operator-level claim but does not establish an end-to-end improvement.

Use whole-inference profiling when end-to-end attribution remains unresolved. Use kernel profiling
only after a relevant kernel has been identified and a kernel-level answer could materially change
the current design or implementation decision. Do not collect additional profiling data once the
relevant alternatives can be distinguished and the requested claim has adequate support.

Retain concise context sufficient to interpret a reported result: relevant hardware/toolchain,
artifact identity at the descriptive level, workload or command, and summarized measurements. Raw
reports and fixed repository or artifact hashes are not required by default.

## Tests and verification

Add or retain a test only when it protects supported observable behavior or a realistic regression:
numerical kernel/model correctness, `.ninfer` framing/binding, external schema/report behavior, a
small real integration route, GPU lifetime, or a reproduced bug. Do not add tests for coverage,
private file/class shape, getters/constructors, deleted compatibility, source-string scans,
hypothetical failures, or test ceremony.

Run a focused set of checks sufficient to support the changed behavior and its material claims.
The following are typical choices, not a cumulative checklist:

| Change | Relevant evidence |
|---|---|
| documentation | affected active-link/stale-reference review and `git diff --check` |
| C++ runtime/API | affected explicit targets and meaningful tests |
| Python tooling | `py_compile` and affected Python tests |
| `.ninfer` reader/converter/binder | affected contract tests and a real artifact when semantics require it |
| CUDA math | independent numerical oracle at relevant shapes |
| memory/lifetime | the affected execution; sanitizer only for a concrete lifetime risk |
| performance | measurement at the claimed scope; attribution tools only when needed |
| serving | affected OpenAI/Anthropic schema tests and observable request/stream behavior |

Do not replace weak verification with low-value tests. State clearly when a relevant check could not
run and why.

## Local environment

Keep aggregate host CPU usage below the user's approximately 85% ceiling while leaving sufficient
headroom for the session. Choose build concurrency or an external process quota accordingly; this
user constraint supersedes the inherited unrestricted-build rule. Low CPU use during CUDA waits
does not by itself indicate throttling, and busy waiting is not a performance objective.
The user has removed the GPU benchmark cooldown requirement. Keep detailed health logs and
resume only missing saved cases after interruption. Do not change power limits or clocks.

These are conventional project resources, not a checklist of resources every task must use:

| Purpose | Path |
|---|---|
| repository | current checkout |
| Python 3.11 | `python3` in the selected maintainer environment |
| Q4_K_M source | `/Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf` |
| companion Vision source | same directory, `mmproj-Qwen3.8-27B-BF16.gguf` |
| product artifact | `/Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer` |
| conversion report | product artifact path plus `.conversion.json` |
| normal build | `build-v100/` |
| private dependency prefix | `build/_deps/install/` |
| profiler output | `profiles/ncu/`, `profiles/nsys/`, `profiles/bench/` |
| hardware/toolchain | 2 x Tesla V100-SXM2 16 GB, `sm_70`, CUDA 12.8 |

Use the selected Python 3.11 interpreter explicitly. Do not install or upgrade dependencies unless
the task requires it. Never select an artifact by glob, modification time, or an unqualified
“latest” name. Large artifacts, source checkpoints, and profiler outputs are local prerequisites;
do not download or regenerate them unless that work is in scope.

```bash
PYTHON=python3
MODEL=/Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf
NINFER_WEIGHTS=/Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer
```

## Commits

Create a commit only when the user requests one. Use Conventional Commit-style subjects, for
example:

```text
feat(engine): cut over the registered target to native artifacts
```

Use concise lowercase types consistent with repository history (`feat`, `fix`, `perf`, `bench`,
`test`, `build`, `refactor`, `docs`, `chore`).
