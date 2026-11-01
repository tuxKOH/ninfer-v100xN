# V100 implementation and verification

## V100X2 changes

- **Q4_K_M prefill:** cooperative SM70 GGML-K block decoding materializes rows into caller-owned
  FP16 workspace and feeds the Volta CUTLASS Tensor-Core GEMM. GDN control projection has an FP32
  output path. The original GGUF Q4_K/Q6_K codes and scales are kept unchanged.
- **Volta FP8/NVFP4 execution:** wide prefill projections decode weights once per call and use
  SM70 CUTLASS, including the prepacked TP2 matrices. NVFP4 decoding reads complete K16 tuples
  cooperatively. Wide TP2 NVFP4 MLP calls at T≥2048 use a 256-column GEMM tile.
  Decode uses QPN Tensor-Core kernels. Both wide prefill and decode keep gate/up
  projections in FP32 through SwiGLU, rounding only the final activation to BF16.
  The prepacked FP8 output head now uses its matching kernel for single-token calls and chunk
  tails too; reading that layout through row-major GEMV was a correctness bug. NVFP4 A4 is not
  supported on V100.
- **TP2 NVFP4 concurrent decode:** real dense TP2 shapes at T9..64 share one K-major FP16
  activation image and consume the existing packed QPN weights directly. Eight logical K
  partitions execute in shape-selected physical phases; paired gate/up emits SwiGLU from one
  CTA. Original normal/fast scale profiles and BF16 Op boundaries remain. This replaces some
  former T>32 CUTLASS calls and can change their output/acceptance. Independent FP64 oracles
  and real C2/C4/C8 Graph/eager MTP4 checks qualify the route; see the
  [concurrency measurements](performance.md#tp2-concurrent-packed-batch-decode-2026-11-02).
- **Long-context decode:** the split-KV attention reducer shares softmax weights and reads eight
  split vectors cooperatively, then accumulates in the original FP32 order. It retains every
  history token, the same KV format and the same partial-output precision. The INT8 attention
  kernel also computes each QK score tile once and shares it across the four output-dimension
  warps, preserving their softmax and PV accumulation order.
- **QUASAR MTP4 KV lookahead:** the default SM70 TP2 five-query, eight-warp verifier preloads the
  next 16-key tile's packed INT8 K/V and FP16 scales into registers while computing the current
  tile. Dequantization still consumes the original codes/scales, with the same split ranges,
  QK contractions, softmax updates, PV arithmetic and BF16 partial outputs for each profile.
  No extra workspace or shared-memory buffer is allocated. Four-query staging remains unchanged
  after mixed measurements. The optional long profile now uses the QPN verifier described below.
  The public GQA suite includes an 85K fragmented-page append/cached comparison against its
  independent FP64 oracle, exact cache/scale checks and allocation/input guards. Local throughput
  and the scope of exact comparisons are recorded in [performance.md](performance.md).
- **TP2 NVFP4 row outputs:** narrow SM70 projections use the existing staging planes for
  contributions and reduce directly into the residual destinations. The rank-0 epilogue preserves
  BF16 projection rounding before residual addition; separate residual-add and reduction-copy
  launches are removed. Direct P2P/DMA collectives, both real row-shard shapes, changed-input
  Graph replay, independent FP64 and real-model MTP4 state checks pass. Subsequent packet push, norm handoff, exact GDN fusion and related changes reach
  198.105 / 198.097 committed wall tok/s at 85K/MTP4 after the final packed-MTP W8 step.
  The user ended the search below 200. See [performance.md](performance.md).
- **Experimental numerical decode:** `--volta-fast-decode` explicitly enables packed NVFP4
  parents with FP32 scale-after-reduction and packed MTP W8 GEMV/QPN arithmetic. It is
  default-off and limited to QUASAR SM70 TP2 Text/None/MTP. The normal route retains
  per-group FP16 scaling. Codes/scales remain unchanged; arithmetic and draft acceptance
  may change. The ~198.1 tok/s result combines this switch with long attention.
- **Optional long attention:** `--volta-long-attention` keeps complete INT8 KV history but
  retunes long TP2 partitions, online softmax and FP32 PV accumulation. It supports experimental 1–8-request
  Text/None/MTP on SM70, excludes DFlash/RAM-KV/Vision, and can change generated output.
  Five-query verification uses four eight-row QK warps with distributed Q fragments and inline
  32-key softmax, followed by eight dimension warps accumulating PV in FP32. The Q staging tile
  is reused for K/V after its fragments become resident in registers. This changes the former
  sixteen-key update/partition association; original INT8 codes/scales and complete history are
  retained. The current combined 85K MTP4 whole-request result is
  198.105 / 198.097 committed wall tok/s; no more candidate sweeps are planned. The latest FP32 proposal Op changes drafts, preserves all 513 measured target IDs, and passes renewed whole-model Graph/eager state qualification. Its current 25-point full-vocabulary comparison matches every argmax but
  fails the near-lossless bounds. It remains default-off; see the
  [current numerical and performance evidence](performance.md#quasar-v3-mtp4-five-query-qpn-verifier-2026-10-18).
  The older GPQA comparison does not qualify this arithmetic profile.
- **DFlash sliding-window attention:** the Volta path uses FP32 split numerators and a warp-per-row
  route once enough rows are in flight; direct tiny windows retain the CTA route. The supported
  local window is 2,048 or 4,096 positions, and the complete-history target KV is unchanged.
- **GDN prefill:** long normalized inputs prepare each Q/K row once in FP32, avoiding repeated
  normalization across state tiles. The sequential FP32 state transition is unchanged. The
  3,072-token TP2 GDN scratch allocation is 24 MiB.
- **TP2 transfers:** collectives use graph-capturable UVA device-to-device copies and two event
  pairs. Startup checks exact transfer bytes in both directions. This host now uses direct
  PCIe P2P after rebooting with `iommu=pt` and identity IOMMU domains. On translated `DMA`/`DMA-FQ`
  domains, startup instead selects verified CUDA-managed staging for PCIe peers. An active
  NVLink mesh is identified from NVML's actual remote PCI endpoints and qualified separately;
  it bypasses the host-IOMMU restriction without changing system settings. Neither path maintains a
  second explicit pinned-host copy route; PCIe P2P is not NVLink. Qualified aligned BF16 SM70 sums use two-epoch packet mailboxes; direct-device TP2 reserves
  640 KiB per channel for up to 64 hidden rows, while other packet domains retain 80 KiB.
  Row-parallel NVFP4 publishes BF16 contributions before fused sum/RMSNorm. Other direct reads
  retire before either source is overwritten. Wide prefill and peer-off collectives keep the DMA route.
- **Experimental PP2:** SM70 Qwen3.8 NVFP4/QUASAR can use `--tp 1 --pp 2 --devices 0,1`.
  Complete layers are split 32/32; chronological prefill chunks run on two stages, while
  decode visits both stages serially. Text/None/MTP and prefix reuse are supported for one
  request; optional sparse RAM-KV is supported, but Vision and DFlash are not. TP2 remains the default recommended route.
  See [the matched pipeline evaluation](experiments.md#experimental-pp2-layer-pipeline).
- **Volta TP2 residuals:** Q5 row shards now select executable SM70 SIMT, fused MMA, or CUTLASS
  routes with the actual peak workspace reserved. BF16 row shards use the SM70 CUTLASS route.
  Both paths pass their independent FP64 operator checks and the two-card composition tests.
- **NVFP4 v3 container:** the reader accepts the upstream `NINFER\x00\x03` single-file
  `qwen3.8-27b` container and projects its physical objects and logical bindings into the
  registered `qwen3.8-27b/nvfp4` identity without repacking weight bytes. The v3
  embedded chat template is preserved verbatim. Text and MTP load through the normal Engine path;
  v3 Vision objects are projected for the existing Vision route. Its optional
  five-layer DFlash2 package is also bound for text-only TP2 experiments.
- **MTP and DFlash:** the Q4_K_M target uses the native MTP route. An optional five-layer BF16
  auxiliary / W8-projection DFlash2 route is integrated with TP2. Greedy MTP and DFlash target
  verification exchange shard argmax values/IDs instead of full logits; DFlash proposal selection
  merges each rank's exact top-16 keys. Sampled or mixed-batch verification retains the full-logit
  route. Removing DFlash's unused
  full-history draft KV pool leaves the all-local v3 route at 98,304-token capacity; 180,000-capacity
  DFlash is not claimed on two 16-GB cards. Narrow BF16 draft projections use direct BF16
  operands with FP32 SIMT accumulation on V100. MTP remains the default.

## Verification

The CUDA 12.8 `sm_70` build was completed with `-j4`. The all-reduce integration test passed,
including exact cross-device byte probes and the 500-iteration 10 KiB microbenchmark. The focused
`nvfp4_a16`, `fp8_a16`, `swiglu_nvfp4`, and `swiglu_fp8` suites passed. Both SwiGLU suites check
full and TP2 shard outputs directly against the same FP64 mathematical oracle, including wide
prefill and caller-workspace sizing. FP8 A8 and NVFP4 A4 suites intentionally
report unsupported on V100 because those tensor-core routes require newer architectures; they are
not part of the V100X2 execution profile.

The decode update also passes `ninfer_gqa_attention_test` and `ninfer_attention_headlocal_test`,
covering BF16/INT8 caches, TP1/TP2, masks, fragmented pages, cache mutation and the full 85K FP64
oracle. FP8 regression cases cover native and prepacked full/TP2 vocabulary heads, single-token
calls and chunk tails. Both FP8 and NVFP4 prefill tests check native/prepacked real matrix shapes
at T=128/1024/4096 against the original represented weights. Q5 LinearAdd passes the FP64 oracle
at both TP2 row-shard extents, including its interior workspace maximum; BF16 Linear passes at
both row and column shards. The two-card LinearAdd and SwiGLU composition suites pass as well.
CLI, server and benchmark are rebuilt with these changes.
