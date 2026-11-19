# Linux V100 prebuilt package

The v0.2.2 release includes a Linux x86_64 / `sm_70` / CUDA 12.8 binary package.
It includes single-card QUASAR text inference, the qualified V100 TP2 routes and QUASAR TP4 Text/MTP with 1–8 active requests, including both optional
experimental profiles. Model weights, the NVIDIA driver, Python and Qt are not included.
See [TP4 measurements and limits](tp4.md); TP4 Vision/DFlash remain unsupported.
Concurrent requests share the configured KV capacity; 180000 tokens admit only two
85000-input / 512-output requests at a time. Additional requests queue.

Use Ubuntu 22.04 or newer, or a compatible Linux distribution whose glibc meets the exact
`minimum_glibc` in `manifest.json`. A working NVIDIA driver is required; use driver 570 or newer
for the CUDA 12.8 build. The separate native Linux TP4 host also passes real serving on
driver 550.163.01 through CUDA minor-version compatibility; this is a measured configuration,
not qualification of every older driver or deployment. The CUDA Toolkit, nvcc and system FFmpeg development
packages are unnecessary when using this package. The package bundles its CUDA runtime,
FFmpeg/curl dependencies, C++ runtime and license notices; it uses the host's glibc and NVIDIA
driver. CPU Vision uses statically linked single-threaded OpenBLAS.
The same release also provides the unmodified `ffmpeg-6.1.2.tar.xz` source for the bundled LGPL
runtime. Its build configuration is recorded in `tools/v100/build_dependencies.sh` in the source archive.

Extract the archive and run from any directory:

```bash
./bin/ninfer --help
./bin/ninfer-serve /path/to/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 --max-context 98304 --kv-dtype int8 \
  --spec mtp --draft-tokens 4 --lm-head-draft \
  --vision --vision-device cpu --vision-max-tokens 4096 --volta-long-attention \
  --pending-timeout-ms 600000
```

Keep `bin/`, `libexec/` and `lib/` together; the wrappers select the bundled libraries and accept
the same arguments as source builds. No `LD_LIBRARY_PATH` setup is needed. Download the exact
QUASAR `.ninfer` v3 artifact separately using the source README's command.

The optional launchers need Python 3.11; GUI additionally needs PyQt5 or PyQt6:

```bash
NINFER_PYTHON=/path/to/python3.11 ./bin/ninfer-tui --artifact /path/to/model.ninfer
NINFER_PYTHON=/path/to/python3.11 ./bin/ninfer-gui --artifact /path/to/model.ninfer
```

Media encoding on CPU is slower and consumes host RAM. Text/model layers, KV and MTP stay on
GPU; only BF16 visual embeddings are uploaded. CPU and GPU visual arithmetic are not
bit-identical. Long attention also changes floating-point association, so both can change
logits or generated output. No new GPQA score or losslessness claim is made.

## Single V100 16 GB

Run the v3 text model on one available V100 with host RAM for the embedding table, GDN checkpoints
and historical KV pages. The measured 85K profile leaves about 95 MiB of free VRAM after startup
on a dedicated card. Select the physical GPU through `CUDA_VISIBLE_DEVICES`.

```bash
CUDA_VISIBLE_DEVICES=0 ./bin/ninfer /path/to/qwen3_8_27b_nvfp4.ninfer \
  --tp 1 --max-context 86016 --prefill-chunk 1280 --kv-dtype int8 \
  --ram-kv-window 9472 --kv-capacity 9472 --prefix-cache-tokens 0 \
  --spec mtp --draft-tokens 4 --lm-head-draft --volta-fast-decode \
  --greedy --no-thinking --prompt "Write a Python palindrome test." --max-new 128
```

For the API, use `./bin/ninfer-serve` and replace `--prompt` and `--max-new` with
`--host 127.0.0.1 --port 8080`. The GPU window uses approximate historical KV retrieval.
Fast decode changes floating-point arithmetic and limits the MTP proposal head to 65,536
artifact rows. Target verification uses all 248,320 vocabulary rows. Generated IDs can change.
The two CUDA Graph runs produced the same 513 IDs as eager on the measured 85K input.
GPQA and long-context retrieval quality remain unmeasured for this profile.

To reproduce the v0.2.2 package after building with the private dependency prefix:

```bash
.venv/bin/python3 tools/release/package_linux.py --build build-v100-tp4 \
  --output dist --version 0.2.2 --dependency-prefix build/_deps/install
```

The packager checks transitive library resolution, glibc symbols, dependency license files and
CLI help. Release qualification additionally runs real inference from a relocated extracted
directory, rather than relying on these help-only checks.
