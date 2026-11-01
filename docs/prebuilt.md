# Linux V100 prebuilt package

The v0.2.1 release includes a Linux x86_64 / `sm_70` / CUDA 12.8 binary package.
It supports the qualified V100 TP2 routes and QUASAR TP4 Text/MTP, including both optional
experimental profiles. Model weights, the NVIDIA driver, Python and Qt are not included.
See [TP4 measurements and limits](tp4.md); TP4 Vision/DFlash remain unsupported.

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

To reproduce the v0.2.1 package after building with the private dependency prefix:

```bash
.venv/bin/python3 tools/release/package_linux.py --build build-v100-tp4 \
  --output dist --version 0.2.1 --dependency-prefix build/_deps/install
```

The packager checks transitive library resolution, glibc symbols, dependency license files and
CLI help. Release qualification additionally runs real inference from a relocated extracted
directory, rather than relying on these help-only checks.
