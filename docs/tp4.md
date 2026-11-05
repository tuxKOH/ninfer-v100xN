# Four-card V100 / NVLink deployment

The four-card results below belong to a separate NVLink host. Local V100X2 controls are labeled
explicitly. QUASAR and the older official NVFP4/native FP8 measurements use different artifacts
and draft windows; keep their results separate.

## v0.2.1: QUASAR concurrent decode

QUASAR TP4 Text/MTP4 now has qualified C2/C4/C8 batched projections, packet/norm handoff,
short-context batch attention and per-request GDN record/norm. The complete 2K/85K coding
task matrix includes TP4, NVLink TP2, local PCIe P2P TP2 and the same-host 1Cat 1.5.1 control.
See [results, acceptance, actual batch and numerical limits](performance.md#tp4-concurrent-packed-batch-decode-2026-11-05).
The binary and Docker assets use this same implementation.

For the benchmark's bounded concurrency configuration:

```bash
./bin/ninfer-serve /path/to/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --tp 4 --devices 0,1,2,3 --max-context 180000 --kv-capacity 180000 \
  --kv-dtype int8 --prefill-chunk 1024 --max-concurrency 8 \
  --no-prefix-reuse --prefix-cache-tokens 0 \
  --spec mtp --draft-tokens 4 --lm-head-draft \
  --volta-long-attention --volta-fast-decode \
  --default-max-tokens 512 --max-output-tokens 512 --no-thinking
```

180K is the shared KV pool, not 180K per lane. Eight short requests fit, but eight 85K
requests require queueing at this capacity. Increase `--kv-capacity` only within available
VRAM; `--kv-capacity auto` resolves capacity at startup. Cold long prefills also stagger
decode readiness even when there is enough KV. Both numerical flags remain default-off;
this command opts in and can change generated outputs. TP4 Vision/DFlash remain unsupported.

## v0.2.0 serving: 256K context and 128K output

Use the Linux SM70 release package, or the equivalent [Docker image](docker.md).
The following starts QUASAR TP4 Text/MTP4, enables both experimental numerical profiles,
and defaults to non-thinking generation:

```bash
./bin/ninfer-serve /path/to/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --model-id Qwen3.8-27B --host 0.0.0.0 --port 8080 \
  --tp 4 --devices 0,1,2,3 --max-context 262144 --kv-dtype int8 \
  --prefill-chunk 2560 --max-concurrency 1 \
  --spec mtp --draft-tokens 4 --lm-head-draft \
  --volta-long-attention --volta-fast-decode \
  --default-max-tokens 131072 --max-output-tokens 131072 --no-thinking
```

The 262144-token ceiling covers prompt plus generation. Output defaults to 131072 and
requests above that hard ceiling return HTTP 400 before model preparation; remaining
context can impose a smaller effective output budget. A client may explicitly enable
thinking. These startup settings do not imply that every request reaches the greedy 85K
benchmark rate below. Both numerical experiments can change output; no new GPQA is claimed.

## Separate 300 W host: QUASAR / MTP4 (2026-10-24)

The new native Linux host has four Tesla V100-SXM2 16 GB cards with all-pairwise NV2,
an E5-2640 v4 with ten exposed CPUs, driver 550.163.01 and approximately 58 GiB RAM.
The deployed binary is built with CUDA 12.8 for SM70. Devices 0–3 already report the
user-authorized **300 W** limit at campaign startup; clocks are not changed. These are
measurements of this host, not power/clock extrapolations of the 220 W host below.

The existing QUASAR v3 `.ninfer` artifact is used directly from the host's model pool.
The public Engine benchmark retains the saved prompt IDs, 180000 capacity, INT8 group-64
KV, chunk 2560, greedy optimized MTP4, CUDA Graphs, one request, no prefix reuse and no
extra warmup. Each row contains two cold requests, each emitting one prefill output and
512 timed committed decode outputs. Decode is `512 / (total_seconds - first_token_seconds)`;
loading and graph priming are excluded from request timing.

| Route | Occupied input | Prefill tok/s | Committed decode wall tok/s | TTFT, s |
|---|---:|---:|---:|---:|
| TP4, long attention + fast decode | 3072 | 3302.15 | 250.707 | 0.9311 |
| TP4, long attention + fast decode | 85000 | **2731.60** | **322.178** | **31.1190** |
| TP4, both experimental profiles off | 3072 | 3287.95 | 245.697 | 0.9351 |
| TP4, both experimental profiles off | 85000 | 2715.22 | 227.019 | 31.3068 |
| TP2, long attention + fast decode | 3072 | 2146.37 | 188.788 | 1.4316 |
| TP2, long attention + fast decode | 85000 | 1518.72 | 199.226 | 55.9700 |

The two 85K TP4 long/fast measurements are **322.138 / 322.218 tok/s**. Each route repeats
its complete output and MTP counters exactly. Both TP4 85K profiles also preserve their
respective earlier 513-ID/counter fixtures. This is supplementary regression evidence,
not proof of universal output equality or model quality; no new GPQA is run. Long attention
and fast decode retain their existing default-off numerical qualifications.

Four-rank collective checks pass before the matrix. At 1 Hz, the TP4 matrix reaches
56°C and shows active SM samples from 1485–1530 MHz. Software power-cap activity occurs;
no sampled hardware thermal/power-brake slowdown or nonzero volatile ECC is recorded.
The three benchmark processes take 99.44 / 101.38 / 150.36 seconds, respectively, including
load, graph priming and four requests per process. Durable reports and health logs are
retained under `profiles/bench/tp4-300w-host-20261024/` locally and the matching directory
under `~/Ninfer/profiles/` remotely.

### No-NVLink traffic: forced host-staging control

This is a **no-P2P / driver-staged simulation on the same cards**, not a measurement of
direct PCIe P2P on physically unlinked cards. A measurement-only runtime interposer reports
peer access unavailable. The Engine disables peer access and qualifies its existing CUDA UVA
D2D staging route. No physical link, driver, clock or power setting is changed; the normal
service does not load this interposer. NVML's per-link data Tx/Rx counters are sampled before
and after the complete benchmark process. All **48 counters across the four cards remain
unchanged**, both in the independent collective qualification and the TP4 whole-model run.
The complete TP2 control also leaves every counter unchanged.

| Route, both experimental profiles on | Occupied input | Prefill tok/s | Committed decode wall tok/s | Prefill loss vs NVLink | Decode loss vs NVLink |
|---|---:|---:|---:|---:|---:|
| TP4, forced host staging | 3072 | 456.94 | 51.585 | 86.16% | 79.42% |
| TP4, forced host staging | 85000 | 456.09 | 72.360 | 83.30% | 77.54% |
| TP2, forced host staging | 3072 | 1406.53 | 108.513 | 34.47% | 42.52% |
| TP2, forced host staging | 85000 | 1122.62 | 119.234 | 26.08% | 40.15% |

The TP2 process takes **198.40 seconds**. At 85K, its mean TTFT is **75.7174 s** and complete
request time **80.0115 s**; decode measures **119.106 / 119.361 tok/s**. Both TP2 occupancies
preserve all **513 output IDs and every MTP counter** of their respective direct-route controls.
The TP2 staged run therefore provides a paired unchanged-continuation control. It uses the
native-Linux default DMA fallback, not the WSL-only automatic mapped-host small-reduction route.

The TP4 process takes **449.72 seconds** for loading, graph priming and four cold requests.
At 85K, mean TTFT is **186.3711 s** and complete request time **193.4468 s**. Its two decode
measurements are **72.391 / 72.330 tok/s**. Each repetition reproduces its output and MTP
counters exactly. Relative to the direct route, the continuations first differ at zero-based
output **167 / 129** for 3K/85K, respectively. At 85K, the staged route accepts **389/490**
drafts versus direct **391/479**, with **123 versus 120** rounds. Thus the table measures
the complete fallback implementation, including disabled peer fusions and changed reduction
association/acceptance; it is not an isolated physical-link bandwidth penalty or an
equal-quality claim. In particular, these losses must not be applied to the local V100X2's
working direct PCIe P2P route.

The initial whole-model probe exposed an inadmissible FP16 norm image on the staged TP4
route. The runtime now chooses that image only when packet storage is qualified, matching
the existing GDN output admission. Without packets it composes the established BF16 norm
and projection. The direct route and the Op's precision/validation requirements remain intact.

The forced-no-peer real-model 85K/MTP4 gate passes in **419.14 seconds**, covering
CUDA Graph and eager execution, cold execution and two exact prefix replays per mode,
replay logits/counters and the committed frontier. The original `ninfer-tp2` container
is restored after testing; devices 0–3 retain the requested **300 W** limits.

## QUASAR NVFP4 / MTP4 at 220 W (2026-10-23–24)

Four Tesla V100-SXM2 16 GB cards, all-pairwise NV2, CUDA 12.8 / SM70. The existing
**220 W per-card power limit and 1200 MHz SM ceiling were retained; no power or clock
setting was changed**. The existing `gpu-guard.service` drop-in sets `PL=220` and
`MAXCLK=1200`; its launcher applies that graphics-clock ceiling.
The published QUASAR NVFP4 v3 artifact is registered as `qwen3.8-27b/quasar-nvfp4`.
Deployment files are under `~/Ninfer` on the four-card host.

The public `ninfer_bench` Engine route uses 180000 context capacity (180032 allocated),
INT8 group-64 KV, chunk 2560, greedy optimized MTP4 and CUDA Graphs. Each point consists of
two cold requests with no prefix reuse or extra warmup. Each request emits one prefill token
and **512 committed timed decode tokens**. Loading and graph priming are outside request
timing. Decode below is committed wall throughput, `512 / (total_seconds - first_token_seconds)`.

### Existing 1Cat installation: same-host MTP4 control

The control uses the host's existing **1cat-vllm 1.5.1** environment at `~/1cat/.venv`
and existing `~/1cat/models/Qwen3.8-27B-QUASAR-NVFP4` checkpoint. Nothing was downloaded,
installed or upgraded. Its original launcher selects DFlash2; this control explicitly selects
greedy **MTP4** to match the requested NInfer workload.

Both engines receive the same saved **85000 token IDs**, TP4, one request, 180000 maximum
context, chunk 2560, CUDA Graphs, disabled prefix reuse, ignored model stops and 513 output
tokens. Both measurements are in process, with load/graph initialization outside request
timing. The first emitted chunk is verified to contain exactly one token; decode counts
the remaining **512 committed tokens**, including host and framework overhead after that
first token. Two complete cold requests are measured:

| Existing 1Cat / TP4 / MTP4 | First request | Second request | Mean |
|---|---:|---:|---:|
| Committed decode wall tok/s | 98.249 | 102.509 | **100.379 ± 3.012** |
| First token, seconds | 35.9527 | 35.9230 | **35.9378** |
| Input tokens / first-token seconds | 2364.22 | 2366.18 | **2365.20** |

1Cat's mean complete request takes **41.0408 s**, versus current NInfer long/fast's
**38.2153 s** for this 85K-input/513-output workload. The current prefill improvements reverse
the preceding 231-stage whole-request comparison, whose NInfer request took 50.9730 s.

This is a workload-aligned control, **not equal arithmetic or equal output quality**.
1Cat uses FP16 compute and E4M3 KV with 2048-token blocks; NInfer uses BF16 boundaries and
INT8 group-64 KV. The existing Hugging Face checkpoint and converted QUASAR v3 container
may also differ in auxiliary projections. Within each engine the two repetitions reproduce
their own IDs. Across engines the first 325 output IDs agree, then continuations diverge.
1Cat reports 124 draft rounds, 496 proposed and 390 accepted tokens per request; its metric
accounting is retained as reported, rather than treated as NInfer's commit counters.

The explicit 1Cat KV allocation is 2.5 GiB per rank, enough for the requested context. Its
initial automatic allocation consumed 8.32 GiB per rank and left insufficient compile
workspace; the retry changes this memory budget only. Successful control initialization
takes 93.45 s and the complete two-request process 190.79 s. Detailed reports and health
logs are retained in `profiles/bench/tp4-push300-20261023/` locally and
`~/Ninfer/profiles/tp4-push300-20261023/` remotely.
Loaded, GPU-active 1Cat samples show 1185–1200 MHz SM clocks, occasional software-power-cap
activity and a maximum temperature of 66°C. The earlier 231.302 tok/s NInfer run stays at
1200 MHz, reaches 69°C and records no sampled power cap. Neither run records hardware thermal slowdown,
nonzero volatile ECC or new Xid/Oops/OOM alerts. Both retain the same 220 W limit.

### Retained TP4 decode optimization

The current 85K-input result is **283.118 / 283.367 committed wall tok/s**, mean
**283.242 ± 0.176**, with both experimental profiles. The normal profile measures
**191.997 / 192.210**, mean **192.104 ± 0.151**. These retain MTP4 and the original
220 W / 1200 MHz limits. **300 tok/s was not reached.** The measurements do not establish
a hardware limit or eliminate the remaining optimization directions.

| TP4 implementation, same 85K fixture | Prefill tok/s | Decode wall tok/s |
|---|---:|---:|
| Initial normal profile | 1740.06 | 113.306 ± 0.148 |
| Repaired activation/communication, normal | 1740.13 | 155.511 ± 0.080 |
| Repaired activation/communication + long/fast | 1745.16 | 216.934 ± 0.154 |
| Paired quarter-shard gate/up + four-group QPN, long/fast | 1743.29 | 231.302 ± 0.272 |
| Same paired/QPN schedule, flags off | 1737.77 | 164.392 ± 0.150 |
| Large collective + Split-D prefill, long/fast | 2065.01 | 231.433 ± 0.150 |
| Projection publication, 64-pack epochs / 16-byte stores | 2258.71 | 238.454 ± 0.171 |
| Published sum/norm, 640 threads | 2253.36 | 241.457 ± 0.185 |
| Verify5 K/V prefetch + FP32 proposal N32768/rank | 2257.05 | 244.655 ± 0.262 |
| Exact BF16→FP16 norm output image, long/fast | 2259.24 | 246.431 ± 0.151 |
| Single-launch small-T GDN controls, long/fast | 2255.67 | 250.824 ± 0.208 |
| Same GDN-controls implementation, flags off | 2246.25 | 178.525 ± 0.172 |
| TP4 GDN projection/record/conv fusion, long/fast | 2258.49 | 258.037 ± 0.373 |
| Same GDN fusion, flags off | 2251.22 | 182.074 ± 0.172 |
| Split-D prefill N64, long/fast | 2336.60 | 260.133 ± 0.300 |
| Same N64 implementation, flags off | 2328.94 | 182.133 ± 0.102 |
| Ordered attention reducer, long/fast | 2337.35 | 262.259 ± 0.350 |
| Same reducer + BF16-rounded SwiGLU FP16 image, long/fast | 2338.95 | 265.004 ± 0.003 |
| Same SwiGLU image implementation, flags off | 2328.01 | 183.583 ± 0.062 |
| GDN gated-norm BF16-rounded FP16 image, long/fast | 2335.13 | 267.487 ± 0.214 |
| Same GDN norm-image implementation, flags off | 2327.87 | 184.815 ± 0.091 |
| Last-CTA GDN record/norm fusion, long/fast | 2334.24 | 268.946 ± 0.079 |
| Same GDN record/norm fusion, flags off | 2329.72 | 185.367 ± 0.094 |
| W8 verifier FP16 input image, long/fast | 2335.07 | 270.694 ± 0.263 |
| Same W8 input-image implementation, flags off | 2326.84 | 186.204 ± 0.124 |
| Direct-peer four-rank argmax, long/fast | 2335.33 | 281.662 ± 0.219 |
| Same direct argmax implementation, flags off | 2326.98 | 191.373 ± 0.132 |
| Direct-peer small metadata broadcast, long/fast | **2334.73** | **283.242 ± 0.176** |
| Same current implementation, flags off | **2327.73** | **192.104 ± 0.151** |

Each row consists of two cold requests at the settings above. These are successive
measurements, not independent per-mechanism speedups. Relative to the preceding paired/QPN
stage, decode improves **22.46%** with long/fast and **16.86%** with both flags off; prefill
improves about **34.2%**. The 244.655 binary was additionally rerun after an interrupted
candidate cleanup and reproduces its original IDs/counters at 244.483 / 244.884 tok/s.
The norm-image step improves that same-session control by approximately **0.71%**.
Single-launch GDN controls subsequently improve the norm-image result by **1.78%**.
TP4 GDN projection/record/conv fusion adds **2.88%** over that 250.824 stage;
its normal-profile gain is **1.99%** over 178.525.
The N64 prefill tile then adds **3.46% / 3.45%** prefill throughput in long/fast and normal.
Its long/fast decode rate is 0.81% higher, but its changed prefill association also changes
acceptance and the saved continuation. This is not an isolated decode-kernel speedup.
The ordered reducer and SwiGLU image then improve the N64 result by **1.87% / 0.80%**
in long/fast and normal, preserving each N64 fixture's complete generation and counters.
The GDN gated-norm image adds a further **0.94% / 0.67%**, again preserving all generation
IDs and speculative counters; no independent decode gain is claimed for its prefill variation.
The last-CTA GDN record/norm fusion adds **0.55% / 0.30%**, followed by the W8 verifier
input image's **0.65% / 0.45%**. Both preserve every fixture ID and speculative counter.

Source mechanisms and numerical boundaries:

- **1Cat-inspired large allreduce:** reduce-scatter/allgather for large payloads; narrow
  collectives retain rank-specialized packet communication. The independent FP64 collective
  oracle and changed-input Graph checks qualify the affected sizes.
- **1Cat / fishlikeX Split-D prefill:** explicitly specialize TP4 D256 attention rather than
  sharing a TP2 launch shape. All causally visible KV remains present. Split-D changes
  floating-point association in the normal and experimental profiles; saved outputs first
  differ from the preceding 231 stage at output **505**. Passing the attention oracle does
  not prove unchanged logits or task accuracy. Provenance and license notices are retained.
  The subsequent N32→N64 prefill tile uses 57856 bytes of opt-in shared memory and applies
  only to the TP4 six-query-head geometry. Its real final chunk (2560 queries ending at
  85000 occupied keys, fragmented pages) passes the unchanged independent FP64 criterion
  on 18 sampled columns across all 24 heads, evaluating every visible key and dimension.
  Exact full-width guards and cache checks also pass. The existing TP1 GPU comparator
  fails its own oracle on this long fixture, so this fixture compares TP4 directly to FP64;
  short fixtures retain full TP1/head-mapping controls. N64 first changes output at
  zero-based index **504 / 505** against the preceding long/fast and normal fixtures.
- **Projection publication and norm:** publish output contributions to the other three ranks
  with 64-pack epochs and 16-byte volatile stores, then use a 640-thread sum/norm schedule.
  Original NVFP4 codes, scales, FP32 divisors and the normal/fast arithmetic boundaries remain.
- **Five-query verifier:** prefetch the next packed K/V tile to L2. FP32 proposal reduction
  uses N32768/rank; it changes draft selection, not target sampling policy.
- **Norm output image:** post-mixer norm writes FP16 image bytes after explicit BF16 rounding,
  retaining **FP32→BF16→FP16**. Quarter-shard A16 gate/up consumes that image directly,
  avoiding a standalone cast launch; there is no extra resident weight image.
- **GDN controls:** a single 320-thread launch handles T2..8 for 12/24 value heads,
  preserving the ten split dot products, warp reduction and split-combine order.
  Independent decoded-input FP64 checks cover T1..9, 128 and 2560 without relaxing criteria.
- **TP4 GDN input/record/conv:** quarter-shard NVFP4 directly writes QKV/Z at T1..8,
  removing the packed projection and two strided copies. For multi-token verification it
  fuses projection, BF16 record and conv/SiLU while leaving source history unchanged.
  A decoded-code FP64 dot oracle covers T1/2/4/5/6/8 under normal and fast scaling;
  the conv oracle applies the public record BF16 cast before its independent FP64 formula.
  Exact composed/fused comparison, source/guard checks and newest-history snapshots
  supplement the numerical oracle. No criterion was relaxed. The four-card operator gate
  takes **13.91 s**.
- **Ordered attention reducer:** the final five max/sum tree levels use warp shuffles with
  the original paired order. Numerator loads prefetch four splits while retaining ascending
  FP32 accumulation and the dynamic active-split count. The real 85K independent FP64
  oracle passes verifier widths one and four through eight without relaxing its criterion.
- **SwiGLU output image:** the explicit **FP32→BF16→FP16** boundary produces a typed
  FP16 activation consumed directly by quarter-shard down projection, removing another
  standalone cast. Independent FP64 SwiGLU/down oracles and exact representation checks
  pass; no BF16 rounding boundary is removed and no extra weight copy is resident.
- **GDN gated-norm output image:** TP4 small-T gated RMSNorm writes the explicitly
  BF16-rounded FP16 activation consumed by the N5120/K1536 output projection. It removes
  the separate cast without changing stored GDN state or the FP32→BF16→FP16 boundary.
  Independent norm/projection oracles and exact representation checks pass without
  relaxing criteria; their four-card checks take **1.41 / 15.14 s**.
- **GDN record/norm fusion:** retain the 192-CTA Dv2/W4 recurrence schedule. Each writer
  completes its BF16 readout/record stores before advancing a per-head completion counter;
  the last CTA evaluates gated RMSNorm and resets that head's counter. State/record boundaries
  and the FP32→BF16→FP16 image remain explicit. An independent FP64 recurrence/norm oracle,
  dense/masked T2..8, exact readout/state/records, changed-input Graphs and counter reuse
  checks pass in **3.92 s** without relaxed criteria. Each Program/rank owns its counters.
- **W8 verifier input image:** materialize the represented FP16 `[K,T]` operands once in
  caller-owned scratch for SM70 N62080/K5120, T3..8. Keep the existing Split4 MMA/reduction,
  BF16 output and one resident weight image; FP16 is the same operand conversion previously
  performed repeatedly inside the kernel. An independent decoded-weight FP64 oracle,
  exact RowSplit controls and changed-input Graph checks pass at widths 1..9, 16 and 33
  in **17.43 s**. This does not enable another artifact or execution domain.
- **Direct-peer TP4 argmax:** one rank-zero kernel reads all four completed score/index
  shards, resolves exact ties by global index before optional ID remapping, and publishes
  the selected IDs to all ranks. Ready/completion joins protect sources and prior output
  consumers under repeated Graph rounds. This adds **4.05% / 2.78%** over the W8-image stage.
- **Direct-peer metadata broadcast:** SM70 TP4 I32 payloads through 4 KiB use one origin
  publication kernel after joining every destination's preceding consumer. Other payloads
  and non-peer domains retain DMA composition. This adds **0.56% / 0.38%** over direct argmax.
  The combined collective suite passes in **6.48 s**, including independent exact CPU argmax
  and relocation oracles, infinities/signed zero, mapped/unmapped T1/5/8/33/129, broadcasts
  around the 1024-word boundary, changed-input Graphs, immediate source/output reuse and guards.

The current long/fast fixture uses **120 rounds, 479 drafted / 391 accepted, one fallback**,
accepted positions `[117, 108, 90, 76]`. The normal fixture uses **122 rounds, 486 drafted /
390 accepted, zero fallbacks**, positions `[118, 107, 90, 75]`. Both repetitions reproduce
their respective IDs/counters. The norm-image, GDN-control and GDN projection/record/conv steps preserve all 513 IDs and every counter
of the qualified 244.655 stage; it does not imply output equality with the older 231 stage,
ordinary attention, or BF16.

Four-card qualification includes independent attention/collective/projection/SwiGLU oracles,
with real quarter-shard shapes, normal/fast scaling and widths through eight where relevant.
The norm image is checked against the independent decoded-input FP64 formula and its explicit
BF16→FP16 representation; gate/up receives the represented FP16 values. Current normal,
actual-85K long/fast and prefix state gates pass in **119.01 / 108.28 / 146.72 s**, respectively.
They cover Graph/eager execution, exact replay logits/counters, speculative committed frontiers,
append/edited/shortened prefixes and saved-frontier restoration. The normal gate additionally
covers sampled routing, teacher-forced positions and B=2 mixed/greedy state.

An earlier eager-only residual/RMSNorm trial read uninitialized scratch. Initcheck identified
it and the branch remains removed; its real-shape residual regression passes independent
FP64, exact Graph/eager and initcheck with zero errors. No numerical criterion was relaxed.
Model-state checks do not establish task accuracy. **No current-combination GPQA was run.**

Current long/fast mean request phases are preparation **0.000723 s**, prefill **36.4068 s**,
first token **36.4077 s**, timed wall decode **1.80764 s**, total **38.2153 s**. The normal
profile averages prefill **36.5162 s**, first token **36.5172 s**, total **39.1824 s**.
The two benchmark processes take **93.05 / 94.78 s**, including load, graph priming,
two requests and teardown. The independent prefill check takes **226.24 s**.
Loaded GPU-active benchmark samples reach **73°C**, with 1170–1200 MHz samples and occasional
existing software-power-cap activity. The state gates reach 74°C. No hardware thermal
slowdown, nonzero volatile ECC or new Xid/Oops/OOM is recorded for these successful cases.
This reduces the old 231-stage full request from **50.9730 s**. Loading and graph priming
remain outside request timing. 1Cat's first token remains faster at **35.9378 s**; its
complete request is **41.0408 s**. Different compute/KV representations and continuations
prevent an equal-quality speedup claim.

Directions retained for further optimization include TP4 residual/collective/RMSNorm fusion,
activation/metadata layouts, deeper attention load overlap, QPN register/warp scheduling,
GDN value tiling, and fixed-MTP4 proposal/tail fusion. The specific tested QPN staging and
occupancy variants, shared-activation dual accumulators, wider W8 materialization, 64-key
attention tile and GDN chunk64 did not demonstrate a whole-model benefit and were withdrawn.
Register-buffered attention preload was slower; shared V-load reuse left partial-kernel
latency essentially unchanged. A packet sentinel-polling probe saved only 0.06–0.13 µs per
isolated collective and was not integrated. These are specific candidates, not rejected directions.
An MTP stem sum/norm fusion preserves IDs/counters but measures 282.938 / 283.156 tok/s,
mean 283.047 versus 283.242; no benefit is established, so it is withdrawn. A Q/K-norm/RoPE
prototype is also withdrawn: its long-position composite oracle was not qualified and its
frequency-override control was not bit-exact. No whole-model speed or quality claim is made
for that prototype; fusion remains a candidate after numerical qualification.
A token-parallel GDN record epilogue is bit-exact but slower (roughly 1–2 µs at T4/5/8),
so that specific variant remains outside production. A private 64-key attention tile remained
slower even after removing spills. Kernel probes do not establish end-to-end speedups.
An INT8 shared-V / Q-reload probe lowers shared memory and admits three blocks per SM,
but spills and extra loads outweigh the occupancy gain. Its tested loop-unroll/split variants
are also slower; none is integrated. This does not rule out other attention schedules.
An additional paired gate/up warp-interleaving probe preserves output bits in both numerical
profiles at T4/5/6, but leaves the 41.984 µs T5 median unchanged and worsens the fast T6
median to 43.008 µs. It is not integrated. A private FP16-PV attention accumulator probe,
including packed shared V and Q-reload variants, measures 173.056–232.448 µs against the
161.792 µs baseline at 85K/T5. Its 0.00184 output relative L2 difference is only against
the existing implementation on the synthetic probe, not an independent oracle or a quality
evaluation. This numerically changed, slower candidate is also excluded. These probes do
not change the qualified 283.242 tok/s result or establish an absolute performance ceiling.

Both `--volta-long-attention` and `--volta-fast-decode` remain **off by default**. The former
changes partitioning/softmax/PV association, the latter NVFP4 scale placement and MTP
arithmetic. They can change logits, output and acceptance. Split-D prefill also changes
floating-point association in the normal profile; there is no new intelligence-score estimate.
TP4 ships in release v0.2.0, separately from the earlier v0.1.2 Vision qualification.

Repair-stage experimental mean request phases: preparation **0.000720 s**, prefill **48.7061 s**,
first token **48.7071 s**, timed decode phase **2.35642 s**, total **51.0672 s**. The process
takes **118.778 s** including load, graph priming, two requests and teardown; loading takes
**15.486 s**, including **11.738 s** upload. The repaired normal process takes **120.935 s**;
mean prefill is **48.8470 s**, first token **48.8479 s**, timed decode **3.28786 s**, total
**52.1403 s**. The two repaired-stage benchmark processes total **239.713 s** (about four minutes).
Decode wall rate includes overhead after the first token and does not count rejected drafts.

During repair-stage loaded, GPU-active telemetry samples all four devices stay at **1200 MHz SM /
877 MHz memory**, with no sampled software power cap or hardware thermal slowdown;
maximum sampled temperature is **64°C** in the final experimental run and **62°C** in the
normal control. Volatile corrected/uncorrected ECC counters stay zero, and the six successful
resumed gates record no new Xid, kernel Oops or OOM in their logged intervals. The host remains
in the same boot throughout those gates. This selection covers active work across both
request phases, not separately aligned phase means. At the retained clock ceiling the
repair-stage result is **1745 prefill / 217 decode tok/s**. Whole-phase linear clock scenarios
would give about **1908 / 237** at 1312 MHz or **2225 / 277** at 1530 MHz; these are optimistic
estimates, not measured unlocked results or achievable-clock promises. Multiplying by
`300/220` is unjustified. No power or clock setting was changed.

The CLI/server, benchmark and standard-name state test are installed in
`~/Ninfer/bin`; the initial benchmark is retained as `ninfer_bench-initial-normal-20261023`.
CLI, server and benchmark `--help` smoke checks pass; no service is started automatically.
This work has not been committed or released. Retained reports, comparison and health logs
are saved locally under `profiles/bench/tp4-prefill-decode-20261023/` and remotely under
`~/Ninfer/profiles/tp4-prefill-decode-20261023/`; the preceding 1Cat/paired-QPN records remain in
`profiles/bench/tp4-push300-20261023/`, and repair-stage records in
`profiles/bench/tp4-resume-20261023/`. Reproduce the experimental benchmark on that host with:

```bash
cd ~/Ninfer
export LD_LIBRARY_PATH="$PWD/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
bin/ninfer_bench \
  --weights models/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --tp 4 --devices 0,1,2,3 --max-ctx 180000 --kv-dtype int8 \
  --prefill-chunk 2560 --spec mtp --draft-tokens 4 --lm-head-draft \
  --volta-long-attention --volta-fast-decode \
  --corpus profiles/deployment/v100-code-85000-iommu-pt.ids \
  -pg 85000,512 --warmup 0 -r 2 --capture-generation -o json \
  --output-file profiles/tp4-resume-20261023/reproduction.json
```

### Initial normal-profile controls

| Host / route | Actual input tokens | Prefill tok/s | Decode wall tok/s | Acceptance |
|---|---:|---:|---:|---:|
| Four-card NV2, TP4, 220 W | 3072 | 2171.72 ± 18.54 | 161.415 ± 0.163 | 83.09% |
| Four-card NV2, TP4, 220 W | 85000 | 1740.06 ± 1.10 | 113.306 ± 0.148 | 77.71% |
| Same remote host, TP2, 220 W | 3072 | 1822.88 ± 6.01 | 190.633 ± 0.100 | 79.39% |
| Same remote host, TP2, 220 W | 85000 | 1267.34 ± 0.47 | 133.273 ± 0.028 | 79.02% |
| Local PCIe PHB, TP2, 300 W | 3072 | 1753.13 ± 30.71 | 215.052 ± 0.139 | 79.39% |
| Local PCIe PHB, TP2, 300 W | 85000 | 1333.77 ± 2.26 | 154.925 ± 0.057 | 79.02% |

Mean ± sample standard deviation. These initial rows use the normal arithmetic profile,
without `--volta-fast-decode` or `--volta-long-attention`. Different TP widths can change logits,
continuations and MTP acceptance; a throughput ratio is not isolated TP scaling or a quality score.
Both repetitions at each point reproduce all 513 output IDs and speculative counters without EOG.
At 85K on the same host, TP4 is 37.30% faster in prefill and 14.98% slower in committed
decode than TP2. Four ranks are not an automatic single-request decode win.

The initial QUASAR TP4 snapshot passes direct/graph collectives, GDN/attention and SwiGLU operators,
the exact load plan, model-state graph/eager and speculative/plain controls, 64 teacher-forced
positions with no argmax disagreements, and B=2 mixed-sampling state controls. The prefix gate
passes exact graph/eager output, logit, acceptance and frontier replay, including partial-round
rollback and concurrent exact hits. That gate disables historical branch archives to isolate
active-lane checkpoints; it is not a TP4 historical-tree or model-quality evaluation. MTP4 graph
residency is explicitly reserved and checked per device, including the supplementary B=2 gates.
The repaired final implementation separately passes these model-state and prefix gates,
as well as the occupied-85K experimental state gate described above.

The same remote TP2 control with both existing experimental switches enabled measures:

| Actual input tokens | Prefill tok/s | Decode wall tok/s | Acceptance |
|---:|---:|---:|---:|
| 3072 | 1823.29 ± 11.78 | 196.099 ± 0.034 | 79.39% |
| 85000 | 1268.73 ± 0.63 | 175.200 ± 0.026 | 79.39% |

These switches change arithmetic and attention association and can change outputs and draft
acceptance. This is a speed control, not a new quality evaluation or a TP4 experimental result.

Mean 85K request phases, in seconds:

| Route | Prepare | Prefill | Decode phase, 512 tokens | Total request |
|---|---:|---:|---:|---:|
| Remote TP4, 220 W | 0.000724 | 48.8490 | 4.5143 | 53.3687 |
| Remote TP2, 220 W | 0.000731 | 67.0698 | 3.8386 | 70.9125 |
| Remote TP2, experimental, 220 W | 0.000721 | 66.9962 | 2.9199 | 69.9195 |
| Local TP2, 300 W | 0.00158 | 63.7292 | 3.3017 | 67.0359 |

The remote TP4 process takes 124.09 seconds for loading, graph priming, both cold requests
and teardown. Model loading is 15.840 seconds, including 11.816 seconds of upload; the first
measured request starts 16.823 seconds after process launch. Saved-token preparation is not
raw-text tokenizer latency.

The six remote benchmark processes take 501.05 seconds in total (8 min 21 s), excluding
model transfer, correctness gates and telemetry setup between processes. Each completed case
is saved and is not repeated when the campaign resumes.

### Initial controls: power-limit interpretation

Read-only GPU telemetry is sampled every 0.2 seconds and aligned with each request's phase
timers; 0.4 seconds at phase boundaries are excluded. During the TP4 85K interior windows,
all four cards remain at **1200 MHz SM / 877 MHz memory**. Mean per-card draw is 174.52 W
during prefill and 124.43 W during decode. No sampled software power cap or hardware thermal
slowdown is active; maximum phase temperature is 62°C. Short spikes between samples remain
unobserved.

The same-host TP2 85K control does encounter the power cap during prefill: 47.88% of
per-device samples report it active, with mean SM clock 1171.2 MHz and draw 211.76 W.
Its decode stays at 1200 MHz / 177.45 W without sampled power or thermal limiting.
If prefill were entirely SM-bound, restoring only its observed uncapped 1200 MHz would
estimate `1267.34 * 1200/1171.2 = 1298.50 tok/s` (about +2.46%); memory/host time makes this
an optimistic correction. This does not justify applying a 36.36% wattage ratio to either phase.
The experimental TP2 85K decode likewise stays at 1200 MHz, averages 197.63 W and records
no power or thermal slowdown. A whole-phase 1530/1200 frequency scenario would estimate
223.38 tok/s for that separate profile; it is subject to the same limitations as the TP4
frequency scenarios below.

Consequently, the TP4 measurements do **not** establish a power-cap loss that can be recovered
by multiplying throughput by `300/220`. With the same observed clocks, the evidence-supported
estimate remains about **1740 prefill / 113 decode tok/s**. A separate optimistic clock-scaling
scenario, assuming the frequency ceiling were relaxed and the entire phase scaled with SM
clock, is:

| Assumed SM clock | Estimated prefill tok/s | Estimated decode tok/s |
|---:|---:|---:|
| 1312 MHz, default application clock | 1902 | 124 |
| 1530 MHz, advertised maximum | 2219 | 144 |

These scenarios are `measured_rate * assumed_clock / 1200`, **not measured 300 W results**.
Neither clock is attainable under the current 1200 MHz ceiling or guaranteed merely by a
higher power limit. Unchanged memory/communication/host time reduces the benefit relative
to this whole-phase linear assumption. No causal 220 W versus 300 W A/B was performed.

Reports, exact commands, per-process elapsed times and health logs are saved locally under
`profiles/bench/tp4-quasar-20261023/` and remotely under
`~/Ninfer/profiles/tp4-quasar-20261023/`.

## Earlier official NVFP4 / native FP8 measurements

### NVLink deployment: TP2 stage

Measured on 2026-10-02 on a second machine with four Tesla V100-SXM2 16 GB cards, all pairwise
NV2, 300 W per card, driver 580.178.04 and CUDA 12.8 / SM70. **NInfer uses only devices 0,1
at TP2 in this table.** The other two GPUs remain idle. Direct NVLink P2P is qualified despite
DMA-FQ host IOMMU domains; no GRUB, driver or reboot changes were needed.

Official Qwen3.8-27B NVFP4 v3, INT8 group-64 KV, 180000 context capacity (180032 allocated),
chunk 2560, greedy optimized MTP3 and CUDA Graphs. Two cold requests per row, no prefix reuse
or extra request warmup; loading and graph priming are excluded. Each request produces one
prefill token plus 512 committed timed decode tokens. Saved prompts retain the final coding
task and assistant suffix. Mean ± sample standard deviation:

| Actual input tokens | Prefill tok/s | Committed decode tok/s | Wall decode tok/s | MTP acceptance |
|---:|---:|---:|---:|---:|
| 3072 | 2101.49 ± 4.03 | 111.19 ± 0.01 | 111.10 ± 0.01 | 72.16% |
| 8192 | 2114.14 ± 6.82 | 109.04 ± 0.03 | 108.91 ± 0.03 | 70.67% |
| 16384 | 2050.33 ± 1.22 | 104.46 ± 0.02 | 104.34 ± 0.02 | 69.00% |
| 32768 | 1887.71 ± 0.55 | 101.99 ± 0.07 | 101.89 ± 0.08 | 74.47% |
| 65536 | 1604.61 ± 3.27 | 90.28 ± 0.03 | 90.23 ± 0.03 | 76.39% |
| 85000 | 1463.73 ± 4.35 | 85.53 ± 0.04 | 85.50 ± 0.05 | 77.11% |

All repetitions reproduce all 513 output IDs and speculative counters, with no EOS/EOG in the measured window.
These fixed coding windows are performance measurements, not scored complete solutions or a
matched motherboard/NVLink A/B. Reports reside on that machine under
`results/tp2-nvlink/`; [bench_tp.py](../tools/v100/bench_tp.py) runs the occupancy, capacity and
plain-decode matrices sequentially through `ninfer_bench`.

Capacity-only sweep: the same **512-token input**, chunk 1024, MTP3, two repetitions and
**256 timed decode tokens** at every capacity (512 timed outputs would overflow the 1024 row).
Every row allocates exactly its requested capacity and reproduces all 257 output IDs and
speculative counters. Acceptance is 70.33% throughout; these rows are not long-context inputs.

| Context capacity | Prefill tok/s | Committed decode tok/s | Wall decode tok/s |
|---:|---:|---:|---:|
| 1024 | 1669.31 ± 29.44 | 112.60 ± 0.04 | 112.56 ± 0.03 |
| 2048 | 1676.79 ± 30.19 | 112.86 ± 0.01 | 112.82 ± 0.01 |
| 4096 | 1666.35 ± 33.44 | 112.70 ± 0.02 | 112.66 ± 0.02 |
| 8192 | 1670.13 ± 31.18 | 112.71 ± 0.02 | 112.67 ± 0.02 |
| 16384 | 1669.02 ± 34.89 | 112.91 ± 0.03 | 112.86 ± 0.03 |
| 32768 | 1668.32 ± 37.79 | 112.81 ± 0.04 | 112.75 ± 0.03 |
| 65536 | 1670.67 ± 27.88 | 112.97 ± 0.01 | 112.90 ± 0.02 |

The ordinary-decode control uses the same saved inputs, capacity 180000, chunk 2560 and
512 timed outputs, but `--spec none --draft-tokens 0` without the draft head:

| Actual input | Plain prefill tok/s | Plain decode tok/s | MTP3 decode tok/s |
|---:|---:|---:|---:|
| 3072 | 2132.37 ± 8.53 | 39.584 ± 0.004 | 111.19 ± 0.01 |
| 85000 | 1480.16 ± 4.93 | 28.349 ± 0.003 | 85.53 ± 0.04 |

Within each control the repeated output IDs agree. MTP/plain outputs match at 85K, but not
at 3K; these speed ratios do not prove universal token parity or a no-quality-loss result.
The [phase accounting](#separate-nvlink-deployment-tp2-stage) separates
model loading, prompt preparation, prefill and committed decode.

### NVLink deployment: NInfer TP4

Measured on 2026-10-03 on the same four-card NV2-linked V100 host above, with all four
devices active. Official Qwen3.8-27B NVFP4 v3, capacity 180000 (180032 allocated), INT8
group-64 KV, chunk 2560, greedy optimized MTP3 and CUDA Graphs. Two cold requests per
point, one prefill output plus 512 committed timed decode outputs, no prefix reuse or
extra warmup. Load/graph priming are outside request timing. Mean ± sample standard deviation:

| Actual input tokens | Prefill tok/s | Committed decode tok/s | Wall decode tok/s | MTP acceptance |
|---:|---:|---:|---:|---:|
| 3072 | 2381.83 ± 37.81 | 125.68 ± 0.08 | 125.42 ± 0.08 | 73.13% |
| 8192 | 2415.01 ± 15.32 | 136.81 ± 0.07 | 136.61 ± 0.07 | 80.62% |
| 16384 | 2381.76 ± 9.29 | 124.09 ± 0.08 | 123.82 ± 0.09 | 72.31% |
| 32768 | 2279.47 ± 2.72 | 117.96 ± 0.02 | 117.73 ± 0.02 | 74.47% |
| 65536 | 2072.25 ± 0.66 | 109.12 ± 0.13 | 108.99 ± 0.13 | 79.30% |
| 85000 | 1957.93 ± 2.16 | 102.60 ± 0.08 | 102.51 ± 0.08 | 78.56% |

Every pair reproduces all 513 output IDs and speculative counters, without EOS/EOG. At
85K, this is 33.76% more prefill and 19.95% more decode throughput than the earlier TP2
snapshot. Acceptance and arithmetic grouping differ, so this is not isolated TP-width scaling
or proof of identical continuations across widths. Reports are in `results/tp4-nvfp4/` on
the remote host; these fixed windows are not scored complete code solutions.

Four-card NVFP4 graph/eager, sampled routing, B=2 MTP and 64 strict teacher-forced positions
pass, with zero argmax disagreements. Prefix checkpoint replay, repeated append, exact
frontiers and partial-round truncation pass exact state controls. A cold-vs-cached choice at
position 28 was a BF16 tie (zero logit deficit); stop-truncation and output-budget truncation
produce identical append outputs, logits and acceptance. Cold re-prefill is not claimed
bit-identical to retained-state execution.

### Four-card 1Cat/vLLM deployment control

For the requested friend-host setup, the isolated launchers under `/home/z/1cat-vllm-v100`
used all four NV2-linked V100-SXM2 cards, the existing OneCat Studio model directories and
loopback ports 6100/6101. The original OneCat UI on port 8888 was not modified. With
180000 maximum context, MTP3 and a 1,570-token benchmark prompt plus 256-token stream:

| 1Cat model | TTFT | streamed decode |
|---|---:|---:|
| Qwen3.8-27B NVFP4 | 0.955 s | 83.03 tok/s |
| Qwen3.8-27B FP8 | 1.590 s | 83.23 tok/s |

These are single short requests through the HTTP streaming client, not 85K occupied-context
measurements and not NInfer TP4 results. The launchers use the model's own FP8/NVFP4 kernels;
NInfer's native block-scaled FP8 identity is measured separately. After the control runs, the
1Cat processes were stopped and the NInfer TP4 API was started on loopback 6200.

The deployed TP4 API launch is `bash tools/v100/serve-tp4.sh /home/z/Models/ninfer/qwen3_8_27b_nvfp4.ninfer`:
loopback port 6200, all four NVLink cards, 180K capacity, MTP3, one active request, 65536
default output budget and prefix reuse. Text-only OpenAI/Responses/Anthropic requests,
unsupported-image rejection and turn-checkpoint replay pass.
The friend can connect through an SSH tunnel; the API is not exposed on a public interface:

```bash
ssh -N -L 6200:127.0.0.1:6200 -p 12031 z@YOUR_HOST
curl http://127.0.0.1:6200/health
```

For a graphical launcher, run `bash tools/v100/ninfer-gui.sh`. The PyQt console exposes the
artifact, TP/device mapping, context/KV capacity, prefill chunk, MTP/DFlash draft window,
CUDA Graph/prefix reuse, concurrency, output limit and sampling defaults, while validating
unsupported combinations before starting `ninfer-serve`. See [`docs/gui.md`](gui.md).

**NVFP4 and native block-128 FP8 TP4 model/state gates, prefix replay, occupancy/capacity
controls and the TP4 HTTP deployment are complete.**

Native FP8 TP4 occupancy (same 180K capacity, INT8 KV, MTP3, two cold repetitions and
512 committed decode tokens) is:

| Actual input | Prefill tok/s | Committed decode tok/s | Wall decode tok/s | Acceptance |
|---:|---:|---:|---:|---:|
| 3072 | 2415.00 ± 13.42 | 98.75 ± 0.02 | 98.59 ± 0.02 | 80.04% |
| 8192 | 2445.95 ± 5.48 | 100.16 ± 0.01 | 100.07 ± 0.02 | 80.44% |
| 16384 | 2404.93 ± 4.30 | 95.48 ± 0.01 | 95.31 ± 0.02 | 77.27% |
| 32768 | 2298.05 ± 0.04 | 89.15 ± 0.01 | 89.02 ± 0.01 | 74.68% |
| 65536 | 2089.18 ± 1.61 | 85.05 ± 0.11 | 84.99 ± 0.11 | 80.22% |
| 85000 | 1972.43 ± 2.89 | 81.53 ± 0.09 | 81.47 ± 0.09 | 80.58% |

The native FP8 capacity sweep (512-token input, 256 committed outputs, chunk 1024) measured:

| Capacity | Prefill tok/s | Decode phase tok/s | Decode wall tok/s |
|---:|---:|---:|---:|
| 1024 | 2003.40 ± 38.05 | 101.52 ± 0.06 | 101.49 ± 0.07 |
| 2048 | 1996.61 ± 50.16 | 101.43 ± 0.08 | 101.40 ± 0.08 |
| 4096 | 1997.52 ± 47.56 | 101.48 ± 0.01 | 101.45 ± 0.01 |
| 8192 | 1986.08 ± 56.70 | 101.12 ± 0.05 | 101.10 ± 0.05 |
| 16384 | 1990.54 ± 54.50 | 101.55 ± 0.01 | 101.52 ± 0.01 |
| 32768 | 2000.66 ± 49.85 | 101.33 ± 0.20 | 101.29 ± 0.20 |
| 65536 | 1999.82 ± 41.27 | 101.26 ± 0.06 | 101.20 ± 0.06 |

All allocations, output IDs and speculative counters repeated exactly. Plain FP8 controls
were 40.98 tok/s at 3072 and 33.12 tok/s at 85000. The FP8 teacher-force gate records three
speculative argmax differences (maximum emitted-logit deficit 0.125), while ordinary cold
re-prefill records one at the same 0.125 bound; this is the documented numerical envelope,
not a claim of token-by-token losslessness.

The same four-card host has passed the production four-rank collective suite and the BF16
quarter-shard projection/residual suite (10 geometry/phase cases, eager plus graph replay,
zero failures). Four-rank NVFP4/row-FP8/native block-FP8 attention and GDN projections, including
quarter-shard BF16 GDN control at T=1/4/128/2560, pass independent FP64 checks. A short four-card
NVFP4 ordinary eager request generates coherent code; this is a smoke check, not a throughput
or quality score. Local SM70 tests pass
the native block-FP8 Linear, fused SwiGLU/residual FP64 oracles, exact TP4 tensor slicing,
GQA 6Q/1KV and GDN 4QK/12V replay. These qualify operators, not a full four-card request.

The Q4 optimized proposal-head quarter shard is now registered and passes its FP64 operator
checks. Parallel proposal selection merges local FP32 winners/I32 IDs instead of gathering the
whole draft vocabulary; target logits remain complete for sampling. Local TP2 MTP graph/eager,
sampled routing, B=2 state and all 64 teacher-forced output positions pass after this change,
with zero argmax disagreements. TP4 graph residency has its own startup budget, now verified
by the four-card NVFP4 MTP startup/state gate.

The native artifact is converted from the friend's existing OneCat FP8 source, preserving
E4M3 codes and every 128×128 BF16 multiplier; no requantization is performed. BF16 embeddings,
output head and MTP stem remain BF16. The result is 31.30 GB (decimal). Its identity is
`qwen3.8-27b/fp8`, separate from the NVFP4 package's row-scaled FP8 projections.
See the [native artifact contract](maintainer/qwen3.8-27b-artifact.md#native-block-128-fp8-textmtp-package).

The selected TP4 domain is Qwen3.8-27B official NVFP4, QUASAR NVFP4 and native FP8 Text/MTP
on SM70. Native FP8 requires TP4; all three TP4 routes exclude Vision and DFlash. The older
official NVFP4/native FP8 graph/eager, MTP commit, prefix-cache, HTTP and occupancy/capacity
measurements in this section were obtained on the separate four-card host.
The existing TP2 and 1Cat tables above are not relabeled as TP4 measurements.

## Request phases and qualification

### Separate NVLink deployment: TP2 stage

The [NVLink deployment table](#nvlink-deployment-tp2-stage) is measured on
2026-10-02 on a different four-V100-SXM2-16GB host, all-pairwise NV2, 300 W per card,
driver 580.178.04 / CUDA 12.8 / SM70. NInfer uses only devices 0,1 at TP2; devices 2,3
are idle. A verified direct NVLink mesh bypasses the PCIe-only translated-IOMMU guard,
then still passes the directed CUDA UVA D2D qualification. No system configuration changed.

The six occupied-context points use the official Qwen3.8-27B NVFP4 v3 artifact, saved final-task
code prompts with 3072/8192/16384/32768/65536/85000 actual input tokens, 180000 capacity
(180032 allocated), INT8 group-64 KV, chunk 2560, greedy optimized MTP3, CUDA Graphs, two
cold requests and no extra warmup/prefix reuse. Each produces one prefill token plus 512
committed timed decode tokens; graphs prime outside measurement. Prefill is 2101.49 tok/s
at 3072 and 1463.73 tok/s at 85000; committed decode is 111.19 and 85.53 tok/s respectively.
Every pair reproduces all 513 output IDs and speculative counters; no EOS/EOG occurs.

The same deployment also completes a capacity-only sweep from 1024 through 65536, with
512 actual input tokens, chunk 1024 and **256** timed decode tokens at every capacity. The
smaller output window leaves sufficient speculative margin in the 1024-capacity row.
All seven requested capacities are allocated exactly. Committed decode is 112.60–112.97 tok/s;
all 257 output IDs and speculative counters repeat exactly at each capacity.

The ordinary control keeps the saved 3072/85000 inputs, capacity 180000, chunk 2560 and
512 timed decode tokens, but disables speculation and omits the optimized draft head.
Its committed decode is 39.584 ± 0.004 / 28.349 ± 0.003 tok/s. MTP/plain output IDs match
at 85000, not at 3072; within each route both repetitions match. This is not universal
speculative/ordinary output parity or a scored quality evaluation.

Mean request-phase seconds for the MTP3 occupancy sweep (model load and graph priming excluded):

| Actual input | Prepare | Prefill | Decode, 512 committed tokens | Total request |
|---:|---:|---:|---:|---:|
| 3072 | 0.000042 | 1.4618 | 4.6048 | 6.0708 |
| 8192 | 0.000075 | 3.8749 | 4.6955 | 8.5763 |
| 16384 | 0.000265 | 7.9909 | 4.9015 | 12.8985 |
| 32768 | 0.000602 | 17.3586 | 5.0202 | 22.3849 |
| 65536 | 0.001199 | 40.8425 | 5.6714 | 46.5187 |
| 85000 | 0.001555 | 58.0711 | 5.9864 | 64.0614 |

The benchmark prepares saved token IDs, so its Prepare column is not raw-text tokenizer latency.
The first occupancy process reports 56.10 s model load, including 53.56 s upload, 21.20 GB
artifact bytes read and 22.46 GB host-to-device traffic; these are startup costs, not request
throughput. Total request also includes boundary/output work outside the two GPU phase timers.

The same four-card host has a separate 1Cat/vLLM control using the existing FP4 and FP8 model
directories (MTP3, maximum context 180000, 1,570-token prompt, 256 streamed output tokens):
NVFP4 measured 0.955 s TTFT and 83.03 tok/s streamed decode; FP8 measured 1.590 s TTFT and
83.23 tok/s. These short-prompt HTTP numbers are not occupied-85K or NInfer-TP4 claims. The
original OneCat UI remained on port 8888; the temporary controls used loopback 6100/6101.

Reports are on that host in `results/tp2-nvlink/`; the corresponding saved commands and
`tools/v100/bench_tp.py` distinguish occupied-context, capacity and plain-decode sweeps.
Wall decode is `n_gen / (total_seconds - first_token_seconds)` and remains separate from
phase decode. These fixed output windows are not a code-quality score, a same-host NVLink A/B,
or TP4 results. Ordinary/streaming API and turn-checkpoint prefix reuse pass a separate smoke.
The TP4 NVFP4 measurements below are separate from the completed native block-128 FP8
qualification reported with them.

The four-card NVLink host separately ran the production multi-device operator checks. The
collective suite covers direct NVLink and staged fallback, uneven gathers, broadcasts and
three graph replays. The quarter-shard BF16 projection suite covers column/row forms at
`3584×5120`, `5120×1536` and `5120×2560`, plus an exactly-once residual-add route at
`5120×1536` and `5120×4352`, for one and four active columns. All ten residual/projection
cases and their graph replays reported zero failures. This proves four-rank transport and
registered BF16 split mechanics; it does not establish a four-rank Engine schedule or a
native block-scaled FP8 decoder.

### Separate NVLink deployment: NInfer TP4

Measured on 2026-10-03 on the same four-V100/NV2 host, all devices 0,1,2,3 active. The
[TP4 occupancy table](#nvlink-deployment-ninfer-tp4) uses the same saved final-task
inputs as the earlier TP2 snapshot: official Qwen3.8 NVFP4 v3, 180000 capacity (180032
allocated), INT8 group-64 KV, chunk 2560, greedy optimized MTP3, CUDA Graphs and two cold
requests per point. One prefill output plus 512 committed timed decode outputs; no prefix
reuse, extra warmup or EOS/EOG. All repetition IDs and acceptance counters agree.

| Actual input | Prepare, s | Prefill, s | Decode, 512 committed tokens, s | Total request, s |
|---:|---:|---:|---:|---:|
| 3072 | 0.000051 | 1.2899 | 4.0739 | 5.3728 |
| 8192 | 0.000108 | 3.3922 | 3.7423 | 7.1409 |
| 16384 | 0.000316 | 6.8790 | 4.1260 | 11.0149 |
| 32768 | 0.000628 | 14.3753 | 4.3404 | 18.7256 |
| 65536 | 0.001211 | 31.6256 | 4.6922 | 36.3251 |
| 85000 | 0.001540 | 43.4132 | 4.9904 | 48.4093 |

Prepare uses saved token IDs, not raw-text tokenization. Load and graph priming are outside
request timing. The first process reports 60.25 s load (56.94 s upload), 21.20 GB artifact
reads and 25.01 GB aggregate host-to-device traffic. Rank 0 reserves 6.25 GB weights,
1.73 GB sequence and 1.61 GB workspace; these are per-device arenas, not four-card sums.
Observed graph residency is 26/22/22/22 MiB across the four ranks. It is measured separately
from the 188 MiB per-device graph allowance.

At 85000 input, prefill is 1957.93 ± 2.16 tok/s, committed decode 102.60 ± 0.08 tok/s and
wall decode 102.51 ± 0.08 tok/s, with 78.56% acceptance. Against the earlier TP2 snapshot,
these are 33.76% / 19.95% prefill/decode increases. Acceptance and arithmetic grouping differ;
this is not isolated width scaling, cross-width token parity or a code-quality score.

The four-card NVFP4 gate passes exact graph/eager and sampled routing, B=2 state and all
64 strict teacher-forced positions, with zero argmax disagreements. Prefix replay/append and
partial-round truncation pass their exact state controls. A cached-vs-cold divergence at
position 28 has zero teacher-forced logit deficit; stop-vs-budget truncation yields identical
append outputs, logits and counters. Cold re-prefill is not claimed bit-identical.
Reports remain on the remote host in `results/tp4-nvfp4/`.

### Native block-128 FP8 TP4

The converted `qwen3_8_27b_fp8.ninfer` artifact was measured on the same four-card NV2 host,
with the same 180000 capacity, INT8 group-64 KV, chunk 2560, optimized MTP3 and two cold
repetitions. The six occupied-context points produced one prefill output plus 512 committed
decode outputs:

| Actual input | Prefill tok/s | Decode phase tok/s | Decode wall tok/s | Acceptance |
|---:|---:|---:|---:|---:|
| 3072 | 2415.00 ± 13.42 | 98.75 ± 0.02 | 98.59 ± 0.02 | 80.04% |
| 8192 | 2445.95 ± 5.48 | 100.16 ± 0.01 | 100.07 ± 0.02 | 80.44% |
| 16384 | 2404.93 ± 4.30 | 95.48 ± 0.01 | 95.31 ± 0.02 | 77.27% |
| 32768 | 2298.05 ± 0.04 | 89.15 ± 0.01 | 89.02 ± 0.01 | 74.68% |
| 65536 | 2089.18 ± 1.61 | 85.05 ± 0.11 | 84.99 ± 0.11 | 80.22% |
| 85000 | 1972.43 ± 2.89 | 81.53 ± 0.09 | 81.47 ± 0.09 | 80.58% |

The capacity sweep uses 512 input tokens, 256 committed outputs and chunk 1024. Decode phase
throughput is 101.12–101.55 tok/s at every requested capacity from 1024 through 65536; all
allocated capacities, output IDs and speculative counters repeat exactly. Plain controls are
40.98 tok/s at 3072 and 33.12 tok/s at 85000. The FP8 state gate and prefix gate pass on all
four ranks. Its numerical rule is ordinary-cold-controlled: three speculative teacher-forced
argmax differences (maximum deficit 0.125) are within the ordinary control's one difference
and 0.125 maximum deficit. This is not a token-by-token losslessness claim.
