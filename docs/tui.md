# NInfer terminal launcher

Run from a local terminal or SSH session with Python 3.11+ and curses (standard library on Linux).
No GUI, pip packages, GPU allocation or driver changes are required to open the launcher.

```bash
bash tools/v100/ninfer-tui.sh
```

The default is **QUASAR NVFP4 / TP2 / MTP4 / 180K**, INT8 KV, prefill chunk 2560,
CUDA Graphs and prefix reuse enabled, 65536 default output tokens, loopback port 8080,
one active request. It selects the existing `build-v100/apps/ninfer-serve`, or the local
`build-v100-tp4/apps/ninfer-serve` when the first is absent. This build-directory name does
not select TP4. Artifact paths are explicit; the TUI itself does not download or convert models.
The [Docker entrypoint](docker.md) downloads QUASAR v3 (or a custom `--model` URL), then opens
this launcher with the downloaded path, container binary and `0.0.0.0` API listen address.

## Controls / 操作

| Key | Action |
|---|---|
| ↑ / ↓ / Tab | select a setting / 选择参数 |
| Enter / Space | edit a value or cycle a switch / 编辑或切换 |
| Ctrl-U while editing | clear the input / 清空输入 |
| `p` | select a preset / 选择预设 |
| `m` | choose an existing known artifact or type an explicit path / 选择模型 |
| `c` | view the exact command / 查看完整命令 |
| `i` | experimental gains and output impact / 实验收益与数值影响 |
| `v` | live and last-request metrics / 实时及最近请求指标 |
| `l` | scroll a server-log snapshot / 滚动查看服务日志 |
| `s` | start or stop this launcher's server / 启动或停止 |
| `r` | stop, then restart the owned child / 重启自己的服务 |
| `u` | hot unload to host RAM, clearing KV/prefix caches / 热卸载到内存 |
| `g` | hot load from host RAM into VRAM / 从内存热加载回显存 |
| `q` / Esc / Ctrl-C | stop the owned server and exit / 停止自己的服务并退出 |

Use a terminal at least **80 columns × 18 rows**. Settings are locked while the child runs;
stop it before changing the model, GPUs or capacities. CUDA IDs are logical: with
`CUDA_VISIBLE_DEVICES=4,6`, enter `0,1`. The API output limit is still clamped to remaining
context. The model's embedded template determines supported thinking effort; this launcher
sets only the server's default thinking on/off, not a universal `xhigh` request setting.
Command, impact and log panels support Up/Down/PgUp/PgDn scrolling; other keys return.
The default thinking setting follows the server (on); existing saved choices take precedence.

`u` / `g` keep the owned server process and its configuration alive. The status row shows model
residency, retained RAM size and transition progress. Unload rejects active requests rather than
cancelling generation. On the first unload the complete original `.ninfer` container is copied to
host RAM, then model/KV/GDN/workspace/graphs are destroyed; CUDA driver context overhead may
remain. Allow the container's byte size plus 1 GiB available host memory. Reload uses the RAM image,
rebuilds the runtime and retains that image for subsequent cycles. KV, prefix history and processed
media caches are cleared, so the next request is cold. RAM-only/loading/offloading states return
HTTP 503 for inference. Failed reload leaves the RAM image available for retry. Stop/restart releases
the image. No power or clock setting changes. See [serving lifecycle API](serving.md#model-residency).

Non-secret preferences are saved after edits and on exit to
`${XDG_CONFIG_HOME:-~/.config}/ninfer/tui.json`. Use `--settings PATH` for a separate file or
`--no-settings` to neither read nor save preferences. Explicit `--profile` is applied after saved
settings, then `--binary`/`--artifact` override paths. `--host` and `--log` override the API
listen address and request JSONL path. Applying a preset clears prior experiment
and sampling overrides. API keys are masked in the editor and command preview and are never saved.

Editable service controls include KV capacity (empty = context, `auto`, or an explicit count),
INT8/BF16 KV, bounded concurrency, thinking preservation, CORS, and all seven sampling overrides
(temperature, top-p, top-k, min-p, presence/frequency penalties and seed). Empty sampling fields
keep model/request defaults. The native context ceiling is 262144; no implicit YaRN is selected.

## Presets

| Preset | Configuration |
|---|---|
| `quasar-mtp4` | default profile TP2 / MTP4 / 180000 / chunk 2560 |
| `quasar-long` | fast numeric profile + long attention / MTP4 / 180000 / ~198.1 tok/s at 85K |
| `quasar-vision` | image/video / TP2 / MTP4 / 98304 / chunk 1024 / visual budget 4096 |
| `quasar-dflash7` | TP2 / DFlash7 / 98304 / chunk 1024 / full proposal head |
| `nvfp4-tp4` | official NVFP4 / four cards / MTP3 / 180000 |
| `fp8-tp4` | native block-128 FP8 / four cards / MTP3 / 180000 |
| `q4-mtp3` | GGUF-derived Q4_K_M / TP2 / MTP3 / 180000 / chunk 4096 |

`Vision ON / OFF` controls `--vision`; it is off in the default Text profile. `Visual token budget`
controls `--vision-max-tokens`. Turning Vision on/off requires stopping and restarting the child.
The same QUASAR v3 artifact contains both Text and Vision; off omits all Vision GPU payloads.
The Vision preset keeps prefix reuse and numerical experiments off initially. Turning Vision on
only disables fast decode and RAM-KV; long attention may stay enabled. A 1024 visual
budget suffices for the chart image; the bundled five-second video requires 4096 in the tested
98K-capacity profile. This budget is independent of text context and aggregate across request media.

`quasar-long` enables both default-off numerical experiments. It selects greedy sampling and
disables prefix reuse to match the 85K timing setup. The normal MTP4 preset keeps model sampling
defaults and prefix reuse. `i` explains the combined ~198.1 tok/s result, the final W8 step's
~0.84% gain, and the prior 25-point logit evidence (9/25 near-lossless, max KL 0.001903103).
The experiment panel also reports the separate four-card 85K/MTP4 result, **231.11 / 231.49
wall tok/s** at 220 W / 1200 MHz. Paired projection/QPN tuning is +6.62% over 216.93 with identical
fixture IDs/counters; the underlying long/fast profiles can still change outputs.
After removing an eager-only residual trial that read uninitialized scratch, the normal,
prefix and actual-85K Graph/eager, replay-logit/counter and committed-frontier checks pass.
IQ/task-accuracy impact is unknown; the current combination has no GPQA result. Gains are not
independent per-switch ratios and do not describe Vision throughput. `Long attention (Text/Vision)`
also supports TP2 images/video with either GPU or CPU vision encoding and complete INT8 KV;
`Fast math (Text only)` excludes Vision. TP2 DFlash admits both experiments,
independently or together: long attention changes target verification, while fast math
changes target NVFP4 arithmetic and retains the native W8/BF16 drafter. Outputs and
acceptance can change; DFlash measurements are separate from the historical MTP rates.
Relative edited paths are resolved from the repository root, matching the child working directory.
The [QUASAR measurements](performance.md) describe its 85K coding workload and quality limits.
TP4 needs four suitable devices and the corresponding artifact; see [TP4/NVLink](tp4.md).
The C++ server remains the authority for artifact identity, admission and available GPU memory.

## Other existing experiments

`Experimental PP = 2` selects the existing layer pipeline: TP1, two device IDs, concurrency 1,
NVFP4/QUASAR Text/None/MTP; no Vision, DFlash or either numerical experiment. PP prefill chunks
are 1..8. There is no established local speed gain to advertise.

`Experimental RAM-KV` is default-off approximate attention, including during long prefill.
The GPU window replaces `--kv-capacity` while enabled; it must be a multiple of 64, at most the
logical context, and at least `chunk+8192` (`chunk*pipeline_chunks+8192` under PP2).
It supports single-request Text/None/MTP on TP1/TP2/PP2, excluding Vision, DFlash, TP4 and long
attention. RAM archive bytes default to 32000000000 and do **not** impose a process RSS limit.
Unretrieved history does not participate in attention, so long-context quality may fall. The
current combination has no applicable speed/GPQA result. `i` labels these limits; see
[RAM-KV semantics](maintainer/paged-kv-cache.md#15-optional-native-ram-kv-experiment).

Explicit paths and command-only mode:

```bash
bash tools/v100/ninfer-tui.sh --profile quasar-long \
  --binary /home/z/ninfer-V100X2/build-v100-tp4/apps/ninfer-serve \
  --artifact /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer

bash tools/v100/ninfer-tui.sh --profile quasar-dflash7 --print-command
```

The UI shows child stdout/stderr, loading/listening state, `/health`, and **one-second Engine
prefill/committed-decode intervals**, not request-phase average rates. Full-precision request
records append to the editable log path (default `profiles/tui-server.jsonl`). It sends SIGTERM
only to its own child, then SIGKILL after five seconds if necessary. External API processes
are untouched; the terminal launcher does not run a background system service.

The main screen also shows live draft acceptance and prefix cache rates. `v` shows the latest
and token-weighted session cache counts, live acceptance and the last completed request's
prefill/decode phase averages, token counts, times and acceptance. Decode phase averages exclude
the first output token produced by prefill. These phase values are separate from the one-second
Engine throughput above. A new launch skips historical records already in the append-only log;
its session counters reset on start/restart, and incomplete/error requests do not count as hits.

`Historical prefix budget` sets `--prefix-cache-tokens` and is saved with preferences.
The recommended default is **384,000 unique page-rounded Main Text tokens**; `0` keeps
only current GPU-prefix reuse. Historical Text/None/MTP branches retain exact packed KV and
complete GDN/tail state in RAM, evicting the oldest saved branch first. QUASAR TP2 INT8 KV
needs up to about **12.8 GiB RAM** for the 384K default main+MTP tokens (about 33.4 GiB at 1M), plus bounded state images and
transfer headroom. This is not a process RSS cap. Vision, DFlash and sparse RAM-KV are excluded.
Long attention admits experimental startup-fixed concurrency **1–8**; all independent requests
share the GPU KV pool, so configured concurrency does not guarantee that many long prompts
can run simultaneously. See [concurrency measurements](performance.md#experimental-long-context-concurrency-2026-10-20).

## CPU Vision (0.1.2)

`Vision encoder GPU / CPU` selects `--vision-device gpu|cpu` when Vision is enabled and persists
in saved settings. CPU keeps tower weights and computation in RAM, uploading only BF16 visual
embeddings before language prefill. Image/video latency increases; CPU encoding happens before
joining the GPU FIFO so existing decode can continue. CPU/GPU arithmetic can change output;
no new GPQA result is available. Long attention admits Vision; fast decode remains text-only.

CPU 视觉模式下，启动器将 API 的媒体准备/排队超时设为 600000 ms（10 分钟）；
命令行直接启动服务时，大图或视频需自行提高 `--pending-timeout-ms`，默认 30 秒可能不够。
