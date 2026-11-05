# V100 decoding 优化：公开项目调研与借鉴清单

调研日期：2026-10-18；源码与试验更新至 2026-11-05。对象包括本工作区的 **2 × V100-SXM2 16 GB、CUDA 12.8、SM70** 和远端四卡 NVLink 的 QUASAR/MTP4 并发优化；本机使用 **PCIe P2P，没有 NVLink**。两种硬件的测量单独记录。

本报告回答“读哪些项目、借鉴什么、落到哪里、如何判断值得保留”。它是研究结论，不修改现行产品契约，也不表示候选已经移植或在本机测得收益。检索覆盖 GitHub 仓库搜索、公开搜索引擎、上游引用及相关分支；下列为本轮能够核实的公开项目，不作全网穷尽声明。

**已有实现仍然进入候选池。** 有 QPN 不等于其布局、解码、预取和寄存器分配已经最优；有 attention、Graph、MTP、P2P 也不等于同方向没有增量。优先级依据本地差距、当前工作负载及外部证据调整，不能仅凭“已经做过”排除。

## 1. 直接结论

最值得持续对照的三个主仓库仍是 **geoffwatts/ninfer-v100、plus1998/Ninfer-V100-Duo、1CatAI/1Cat-vLLM**。新增的重点对象是：

1. **dnv2003/v100-skinny**：QPN2/QPN8 的原始实现、形状相关 split-K、独立累加链和解码开销。
2. **xormal/V100-SM_70-Flash-Attn-2-v1（HOOLIGAN）**：K/V 同时发射、GQA 复用、split-KV 数量与 CTA 数、反量化依赖链。
3. **kamilch1k/voltattn**：量化 KV 为什么会“省显存却不加速”，以及 sub-warp、多位置并行如何改善 Volta 访存。
4. **ggml-org/llama.cpp、ikawrakow/ik_llama.cpp、InternLM/lmdeploy**：保留原量化权重的 decode、GGML K 格式解包、SIMT/WMMA/QPN 路由与 Volta GEMM。
5. **squibloads/vllm-pcie-allreduce、123123213weqw/dual-v100-llama.cpp**：本机尤其相关的 PCIe 小消息通信和完整轮次测量。
6. **Augur-rust/Augur-V100**：近期的 GDN 周边融合、共享词表头和草稿路径优化；采用其机制时仍须保持本项目自己的数值和 artifact 契约。
7. **zhinianqin/marlin_v100、haohervchb/sglang-V100**：前者是明确的 SM70 Marlin 开发工作区，后者是 1Cat push collective 注明的上游来源，也有 27B FP8 的 split-KV/QPN8 路线。

按用户当前优先级，后续实现集中在 **QUASAR NVFP4 decoding**：先研究 **长上下文 INT8 attention 的访存/归约、NVFP4 QPN 的形状专用调度**，再研究 **QUASAR 使用的 W8 词表/MTP、GDN 周边融合、通信、投机轮次成本**。Q4_K_M、普通 groupwise INT4 和原生 FP8 的专用工作保留为低优先级待引入项，暂不投入实现；QUASAR 自身使用的 W8/BF16 组件仍在范围内。这是待验证的选题顺序，不是未经 profile 就认定的瓶颈排序。

本轮验收固定 **MTP4**：85000 实际输入 tokens、180000 context capacity、单请求、
双卡 TP2、INT8 group-64 KV、CUDA Graphs，以 committed wall decode 计数。早期最低目标为
180 tok/s，随后曾追求 **200 tok/s**；2026-10-22 用户取消固定速度目标，改为试验下列六个
QUASAR/MTP4 方向，汇报采用项、未采用原因和输出影响：attention 加载重叠、MTP 尾部融合、
TP2 通信/归一化、NVFP4 QPN 调度、GDN value tile 和固定 MTP4 历史查找。
不继续扩大草稿窗口或测试 DFlash。其他窗口、后端及精度的
相关机制仍留在待引入/继续优化池，但不占用本轮实现优先级。输出变化方案单独记录收益和
质量证据，GPQA 是否运行由用户决定。

## 收尾与未完成项（用户于 2026-10-19 调整范围）

2026-10-21 用户在恢复设备后重新开启 QUASAR TP2 的 prefill/decode 优化；2026-10-22
将交付范围收敛为六项 decode 试验，继续固定 MTP4，不以达到 200 或 250 tok/s 为完成条件。
以下停止说明记录上轮收尾；新结果见[性能记录](performance.md)。

用户决定仅完成当前 MTP W8 布局/分区试验，此后停止追逐 200 tok/s；未达到不再启动下一轮优化。
最终保留 **198.105 / 198.097 tok/s**（均值 198.101），未达 200 后按用户要求停止新候选。
最后一步为 MTP stem sum/norm 融合与五投影 W8 重排/分区；513 个 target IDs 不变，草稿接受位置改变。
完整组合 Graph/eager 状态验证通过；独立数学/精确打包检查通过。以下继续列为待引入/继续优化，未因已有实现或本次负收益而删除：

- 完整历史 INT8 attention 的 QK/PV 分块、跨 head 复用、加载重叠和 softmax/归约布局。
- NVFP4 QPN 的形状调度、metadata/activation 复用及 MLP epilogue；已测较慢的具体变体不等于方向封顶。
- MTP W8 打包与小矩阵调度、共享词表/候选精确重评分、context-conditioned shortlist。
- TP2 PCIe packet 的 producer/consumer 融合、协议和同步；本轮已移除 TP2 MTP stem 中无人读取的
  token 广播，其他通信与小算子融合继续优化。
- GDN 的 value tile、状态/快照及周边融合。
- 固定 MTP4 的 lookup：此 corpus 的 16-token suffix 无命中，其他上下文策略仍待验证。
- 草稿近似注意力：16K/32K 仅 AR 试验因接受率下降，整机均慢于完整历史，未采用；继续待优化。
- 其他精度、其他草稿窗口和 DFlash 机制保持低优先级，本轮不实施。
- 当前数值组合未跑新 GPQA，25 点 full-vocab 对照未通过 near-lossless；行为质量评测由用户决定。
- QUASAR TP2/MTP4 的现有完整视觉/前缀集成检查已通过（4096 容量）；视觉预设默认冷请求，可手动开启前缀。
  更长的视觉占用、视觉质量任务评分仍未测，不当作纯文本 85K 性能证据。

上轮交付转为：保留最佳已验证实现、实验入口显示收益和数值影响、清理确认无用的临时脚本，
并主要适配含/不含视觉的 QUASAR NVFP4。视觉版复用已含 Vision payload 的发布 artifact，
需通过本地视觉集成验证后再删除旧 Unsloth 派生模型。

### 六项 decode 试验（2026-10-22）

六项均已实现候选并进行本机 85K/MTP4 测量，详见[完整结果与验证范围](performance.md#six-quasarmtp4-decode-trials-2026-10-22)。
本轮重新测得基线 196.938 / 197.062 tok/s；保留的 MTP argmax/remap 尾部融合为
197.062 / 197.318，513 个输出 IDs 和全部接受计数不变，尚不能认定整机提速。
attention 预取 188.754、TP2 packet 向量化 193.073、QPN split-4 191.874，均未采用。
GDN 小 value tile 初测 197.308、复测 197.614 / 197.698，已与尾部融合一起保留；
2048 decode tokens 的配对检查为 181.580 → 181.880（约 +0.16%），2049 个输出 IDs 和
全部接受计数一致。收益很小，这些顺序测量不构成统计显著性或通用提升的证明。

固定 MTP4 的八 token 后缀 lookup 提案试验为 187.220 tok/s，14 次命中仅接受 23 个草稿。
该试验保留完整 target 验证，但仍运行 learned drafter；它验证了此提案策略接受效果不佳，
没有验证“命中后跳过 drafter”的完整优化收益。上述具体变体已回滚，整个方向仍保留待优化。
QPN 和 packet 变体均在第 130 个输出首次发生差异，因此也不作为无损方案推荐。
本轮没有运行 GPQA。整数映射融合没有新增浮点运算；GDN 保留逐列 FP32 状态更新、
归约与 BF16 边界，没有引入新的低精度配置。原有实验性 profile 的数值影响仍然存在。

### TP2 并发源码借鉴（2026-11-02）

本轮实际读取 1Cat 的 `csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu`、
`csrc/custom_all_reduce.cuh` 和 `docs/design/sm70_dflash2_long_batch_20260927.md`，
并将适合本地 TP2/MTP4 的机制落到算子层：

- **批次共享 activation supply**：BF16 activation 一次打包为 K-major FP16，供 output CTAs
  复用；权重仍直接消费已有 QPN codes/scales，没有整块 FP16 权重展开或第二份常驻权重。
- **分阶段 K 调度及 gate/up 配对**：T9..64 的真实 TP2 attention/GDN、MLP 和 contraction
  shapes 使用八个 logical K partitions，物理 warps 按形状选择；gate/up 在同 CTA 输出 SwiGLU。
  保留 normal/fast scale profile、原有 BF16 输出边界，各请求的状态和 acceptance 独立。
- **批量 projection→packet→sum/norm**：direct-P2P TP2 的双 epoch channel 扩到 640 KiB，
  使 D5120/T1..64 不再因原 80 KiB 上限退回多步路径。生产者保留 BF16 contribution 边界，
  直接发布 mailbox，消费者融合 sum/RMSNorm；其他 TP domain 和 mapped-host 容量保持原值。

独立 FP64 算子 oracle、输入/权重只读与 guard 检查、changing-input Graph、真实模型 C2/C4/C8
的 normal/longfast、greedy/mixed Graph/eager 完整输出及投机计数检查均通过。
T>32 的旧 CUTLASS 路径与新 batch 归约不同，八并发实测已出现输出/acceptance 差异；
不能称为逐位无损，也不能把全部吞吐变化归因于纯 kernel 时间。当前组合未跑新 GPQA。
[前后吞吐、共同 decode 窗口与完整条件](performance.md#tp2-concurrent-packed-batch-decode-2026-11-02)。

1Cat 文档中的 C1/C4/C8 为 159.324/429.134/624.863 tok/s，扩展倍率约 2.69×/3.92×，
但它使用四张 32 GB V100、TP4、DFlash2 q7、FP16/E4M3 KV、32K 输入及 client all-live 窗口；
本轮为双张 16 GB、TP2/MTP4、BF16/INT8-G64。这里只借鉴机制与扩展趋势，没有实际运行
同机 1Cat，不将异条件数字作为已追平的结论。来源及 MIT/Apache notice 记录在 `NOTICE`。

批量 attention 的 KV/head 复用、GDN 周边、W8 proposal/head 以及混合 prefill/decode 调度仍可
继续优化；本轮采用 batch projection 和 packet 并不表示这些方向已经耗尽。

### TP4 并发源码借鉴（2026-11-05）

本轮主要参考 1Cat，读取其 2026-11-04 main 的实际四卡源码，而不只采用旧文档中的并发倍率：

| 1Cat 源码机制 | NInfer 采用方式及边界 |
|---|---|
| `sm70_turbomind/ops/nvfp4_qpn2_sm70.cu` 的批量 activation supply、分阶段 K 归约 | 将已有 packed batch kernel 扩展到 TP4 的 attention/GDN、gate/up 和 contraction shapes，覆盖 T9..64；保留原 QPN codes/scales、normal/fast profile 和 BF16 输出，不展开第二份权重 |
| `custom_all_reduce.cuh` 的批量 push、`sm70_push_allreduce_gemma_rms_norm_batch` | TP4 device channel 扩至 640 KiB；row projection 直接发布 BF16 contribution，T1..64 支持跨层 sum/norm handoff。T>8 使用较小 CTA 分 panel 处理完整 row，保留 NInfer 的 global-rank 求和顺序和二十个 logical warp norm sums |
| `attention/sm70_grouped_long/kernel/grouped-attention.cu` 的 batch/context routing | C4/C8 已有请求级并行，短上下文减少 KV split。新私有路径以 FP32 累加并合并 PV numerator，输出仍为 BF16；保留完整 packed INT8-G64 KV，不丢历史 keys。分区/association 会改变数值，不能按逐位无损发布 |
| `sm70_turbomind/ops/gdn_verify_sm70.cu` 的整块 verify 与周边融合 | 扩展 NInfer 自有 replay-record + gated norm 到 TP4 B1..8，保持原 FP32 recurrence、BF16 readout/record；completion 按 request/head 隔离。新 zero-valid fixture 暴露并修正了零有效行仍改写首个 key record 的问题 |

**同机运行的对照是 1Cat 1.5.1 发布版**，不将 main 源码版本与实测运行版本混为一谈。
两边均为四卡、MTP4、相同的完整 2048/85000-token 代码任务 IDs、每请求强制输出 512 tokens、greedy、关闭前缀复用。
2K 单独保留完整任务与 assistant prefix，不再直接截短 85K；此前 raw-prefix 结果单独保留，不能跨任务计算软件提升。
现有 1Cat checkpoint 为 AWQ INT4/FP16/E4M3-KV，NInfer 为 QUASAR NVFP4/BF16/INT8-G64；
1Cat 使用 HTTP SSE token IDs 计数，NInfer 的 executor-round 与 all-live sampled 另列。
它们可提供同机速度与扩展参考，不能称为同质量、同算术或相同计时面的胜负。
条件、结果与图表见[TP4 并发测量](performance.md#tp4-concurrent-packed-batch-decode-2026-11-05)。

采用项通过独立算子 oracle 后还须经过真实模型并发 Graph/eager、投机计数与提交状态检查；
数值组合没有新 GPQA。W8 proposal/head、长上下文 batch attention 和进一步的通信/投影调度仍保留
为待继续优化方向，不因本轮已采用同方向机制而删除。

### TP4 源码复核（2026-10-23）

本轮重新读取了 1Cat、SGLang-V100、v100-skinny、geoffwatts、Duo，另外复核
HOOLIGAN 和 voltattn 的实际 kernel。远端现存 1Cat 启动脚本使用 QUASAR + DFlash2，
本轮 NInfer 继续固定用户指定的 MTP4；同机现有 1Cat 1.5.1 也显式改用 MTP4 对跑，
没有下载或重装，更没有把其原启动脚本当作同条件速度结果。

| 实际源码 | TP4 可取之处 / 本地处理 |
|---|---|
| 1Cat `custom_all_reduce.cuh` | 固定 rank 特化和保留本地贡献；本地 BF16 packet256、逐 rank FP32 归约，固定全局顺序。普通 85K wall 从原始 113.306 提升到 145.556 tok/s |
| 1Cat `grouped-attention.cu::GroupedVerifyTraits`、combine kernel | 打包共享 KV、按 SM 数量组织 split、合并时共享权重。TP4 每卡仅一个 KV head，测试较短分区及直接读取合并；属于 default-off long attention，分区变化会改变输出与接受率 |
| 1Cat `nvfp4_qpn2_sm70.cu` / `nvfp4_qpn4_sm70.cu`、v100-skinny `skinny_nvfp4_qpn2` | streaming、FP16 activation 复用、按形状 split-K。TP4 的 3584/4096 输入、4352 gate/up 和 1536/4352 contraction 独立筛选；不能直接沿用 TP2 的参数 |
| SGLang `sm70_nvfp4_moe_decode.cu` 与 MTP worker、1Cat `gdn_verify_sm70.cu` | 整轮 Graph、批量投影、窄 value tile；本地已有对应 Graph 与 FP32-state 窄 tile。保持本地 BF16 边界与 FP32 GDN 状态，继续保留融合与 tile 优化方向 |
| geoffwatts / Duo `mtp_impl.h`、NVFP4 QPN | 复核真实验证与重对齐调度；本地已由同族路径扩展 TP4，不把同源整体当作无需继续优化 |
| HOOLIGAN `split_decode.cu`、voltattn `voltattn.cu` | K/V 同时取数、subwarp 多请求在途和足够 split 并行度；本地长 QPN 已同时读 K/V。更深预取仍待实测，D128 专用布局不直接替代本地 D256/INT8-G64 |

四卡当前保留组合在 85K 输入、MTP4、220 W / 1200 MHz 下实测 **257.773 / 258.300 wall tok/s**，
均值 **258.037**，prefill **2258.49 tok/s**。此前配对 gate/up 与四组 QPN 调度达到 231.302；
本轮继续引入大消息 reduce-scatter/allgather、TP4 Split-D prefill、64-pack 投影发布、640-thread
sum/norm、verify5 下一块 K/V 预取、每 rank N32768 的 FP32 proposal 归约，以及保留
FP32→BF16→FP16 显式舍入的 norm 输出 image，省去独立 cast；随后将 T2..8 的 GDN controls
合为单次 launch，保留 split-10 的点积和合并顺序，相对 norm-image 阶段再提升约 1.78%，
并通过独立 FP64 oracle。TP4 GDN projection/record/conv 随后融合为直接输出，
保留公开 BF16 record 边界，省去中间投影副本与 launch；独立 FP64 点积/conv oracle 通过。
相对 250.824 阶段再提升 2.88%，相对 231 阶段 decode +11.56%，prefill 约 +29.5%。这些是固定任务的顺序测量，不是各机制独立增幅。
Split-D 会改变浮点关联，保存输出与 231 阶段在第 505 个 token 起分叉；norm-image 与后续两步 GDN 融合的全部
513 IDs 和接受计数则与合格的 244.655 阶段相同。不能继续宣称所有阶段输出不变。
普通模式为 **181.952 / 182.196 tok/s**，均值 **182.074**，prefill **2251.22 tok/s**；
普通模式同样使用 Split-D prefill，不代表它与早期普通路径逐位等价。
同机 1Cat 在相同 85000 IDs、180K capacity、chunk2560、MTP4、单请求、513 输出的两次冷请求中
测得 **98.249 / 102.509**，均值 **100.379 wall tok/s**；首字 **35.9378 s**，本组合为
**37.6367 s**。本组合完整请求 **39.6209 s**，1Cat 为 **41.0408 s**。1Cat 使用 FP16/E4M3 KV，
本地使用 BF16/INT8-G64，已有对比在 325 个共享输出 IDs 后分歧；不是同精度或同质量比较。
四卡 collective/attention/投影/SwiGLU 独立 oracle、当前普通状态、前缀及实际 85K long/fast
Graph/eager、重放 logits/接受计数与提交前沿检查通过。旧 eager-only 未初始化残差分支已删除，
真实形状残差回归通过 initcheck。两个实验入口默认关闭，当前组合没有 GPQA，**未达到 300 tok/s**，
也没有证据把当前速度当作硬件极限。
更深预取、activation/metadata 布局与 QPN 调度仍保留优化方向；本轮具体的 staging/occupancy、
共享 activation 双 accumulator、W8 重排、64-key attention tile、GDN chunk64 变体均未证明
整模型收益，已撤回，不能将方向整体排除。
TP4 residual/collective/RMSNorm 融合继续列为低优先级待优化：必须使用真实 residual，且 epoch
域与 packet256 一致，不能直接恢复被删除的试验实现。
最终测量与数值影响以 [TP4 记录](tp4.md) 为准。算子筛选不能替代四卡整请求结果。

## 2. 本地实现读到了什么

调研阶段检查当前工作区文件，包括已有未提交修改；没有复原或覆盖这些修改。阅读了 README、V100 实现/性能文档，以及下表对应代码。调研阶段未启动模型；随后按 QUASAR 优先级进行的实现和本机测量见 [V100 实现](v100-implementation.md) 与 [性能记录](performance.md)。

| 方向 | 本地现状和入口 | 仍然保留的研究方向 |
|---|---|---|
| NVFP4 小矩阵 | [`nvfp4_volta_qpn_gemm.cuh`](../src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh)、[launcher](../src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cu)；已有 QPN、split-K/累加链及部分 FP16 activation 预转换 | 比较真实 TP2 几何上的取数、解码、寄存器寿命、输出融合及各宽度调度，不能把 QPN2 当作尚未引入的新功能 |
| FP8/W8 投影和词表 | [`fp8_volta_qpn_gemm.cuh`](../src/ops/linear/fp8/fp8_volta_qpn_gemm.cuh)、[`w8_volta_qpn_gemm.cuh`](../src/ops/linear/w8/w8_volta_qpn_gemm.cuh) | 继续比较一份权重服务多个 query、输入转换复用、精确 top-1 输出与采样路径的完整成本 |
| Q4_K_M | [`ggml_k.cu`](../src/ops/linear/ggml_k/ggml_k.cu)；T=1..4 走直接解包 SIMT GEMV，T=5..16 走 WMMA 小矩阵，宽 prefill 可走 CUTLASS | **它并没有自动获得 groupwise Q4/NVFP4 的全部 QPN 优化。** MTP3 主要涉及的窄行路径值得单独优化 |
| INT8 attention | [`gqa_attention_decode.cu`](../src/ops/launcher/gqa_attention_decode.cu)、[SIMT](../src/ops/kernel/gqa_attention_decode_i8.cuh)、[Volta TC](../src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh) | 已有 QK 共享、QPN/TC、协同解包与分区归约；仍比较 K/V load 重叠、跨 GQA head 复用、寄存器与 split 代价 |
| GDN | [`recurrent.cuh`](../src/ops/linear_attention/gated_delta_net/recurrent.cuh)、[`gdn_input_proj/`](../src/ops/gdn_input_proj/) | 调整 value tile、合并 norm/cast/小投影/快照写入，保持 FP32 持久状态与 commit/rollback 正确 |
| TP2 通信 | [`allreduce.cu`](../src/ops/common/allreduce.cu)、[peer kernel](../src/ops/kernel/allreduce.cuh)；小 SM70 P2P 消息已有直接 peer-read 求和 | pull/push/本地 staging 的实际轮次代价、事件依赖、与生产者/消费者的可行融合 |
| MTP/Graph | [`mtp_impl.h`](../src/targets/qwen3_6/impl/runtime/mtp_impl.h)、[`program_impl.h`](../src/targets/qwen3_6/impl/runtime/program_impl.h)、[`decode_graph.cpp`](../src/core/decode_graph.cpp) | 草稿数与接受长度的收益模型、尾轮/部分提交的图覆盖、减少重复准备和主机同步 |
| DFlash | [`dflash_impl.h`](../src/targets/qwen3_6/impl/runtime/dflash_impl.h) 与共享投机调度 | 继续优化 drafter/selector/context 更新/target verify 的完整闭环；不能只优化 target 矩阵 |

量化路径必须区分：本地 **groupwise Q4、GGUF Q4_K/Q6_K、NVFP4、W8、原生 block-FP8** 不是同一个 codec。Q4_K 的 scale/minimum 仿射解码也不能直接替换成对称 INT4 解码。

前一轮的中间组合曾测得 **196.471 / 196.433 committed wall tok/s**（两次冷请求；85K 实际输入、180K capacity；当时组合通过整模型 Graph/eager MTP4 状态复验）。已接入 parent FP32 epilogue、输入投影 streaming、TP2 packet push、sum/RMSNorm 融合与跨层交接、MTP shortlist tiled argmax、GDN output activation staging、投影 epilogue 提前发布 packet、精确 split-10 controls 与 projection/conv/record 融合。随后又接入 packed KV vector16、输出投影 streaming、W8 byte-permute 和 FP32 proposal top-1。全部 513 输出 IDs 保持一致；FP32 proposal 将接受计数变为 390/486、122 rounds、零 fallback，此前为 389/490、123 rounds。parent epilogue 与 QPN5 会改变计算关联；组合的 25 点 default 对照为 argmax 25/25、min cosine 0.998831、max KL 0.001903，未过 near-lossless。后续精确组合保持这条测得轨迹，不能据此声称全局无损。FP32 proposal 会改变草稿选择，但不修改 target head/采样政策。W8 预转换整机 193.207，未确认收益，已恢复生产路径；该方向继续待优化。未跑新 GPQA。当前结果及历史阶段见 [performance.md](performance.md)；Q4_K_M 的原对照不能用这些 QUASAR 数字替代。

草稿窗口也要区分：Q4_K_M 验收配置最多三草稿；27B 产品契约上限为七，但当前工作区 [`config.h`](../src/targets/qwen3_6_27b/impl/config.h) 的 `kMaximumMtpDraftTokens` 为五。外部 q7/q15/B8 成绩不能直接当成本地可用配置。

## 3. 三个主要抄考对象：具体抄哪里

### 3.1 geoffwatts/ninfer-v100

[仓库](https://github.com/geoffwatts/ninfer-v100) · [V100 说明](https://github.com/geoffwatts/ninfer-v100/blob/HEAD/docs/v100.md) · [QPN 内核](https://github.com/geoffwatts/ninfer-v100/blob/HEAD/src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh)

这是本地已继承的重要来源，优势是同族 runtime、artifact 和算子边界较接近，适合逐算子比较。已检查其源码；本地的 QPN、GDN 和 MTP 文件与所读版本存在差异，不能把整个分支理解为单向“更先进”。

继续借鉴：

- 在真实 N/K/T 上比较 NVFP4/FP8/W8 路由、activation staging 和融合 epilogue，尤其是 T=1、验证宽度及剩余尾轮。
- 比较 GDN 验证、snapshot 和短宽度 attention 的路径，寻找本地 TP2 整合后留下的重复转换或中间结果写回。
- 研究 context-lookup MTP 的“命中才跳过串行 drafter，target 仍逐 token 验证”机制；本地实现须遵守自己的窗口上限。
- 借鉴按完整 round latency 与 accepted tokens/round 选 K，而非追求更高接受率或更大的 K。

其 README 的 **218.98 tok/s** 是单 V100-PCIe-32GB、2K 上下文、近 99.2% 接受率、K=1 的特定 corpus 结果。它不是本机 85K TP2 的目标值。其上下文复述 lookup 文档可到十五草稿，这部分不能原样并入当前 27B 契约。

### 3.2 plus1998/Ninfer-V100-Duo

[仓库](https://github.com/plus1998/Ninfer-V100-Duo) · [V100 说明](https://github.com/plus1998/Ninfer-V100-Duo/blob/HEAD/docs/v100.md) · [TP2 MTP 调度](https://github.com/plus1998/Ninfer-V100-Duo/blob/HEAD/src/targets/qwen3_6/impl/runtime/mtp_impl.h)

已检查其源码。它同样继承 geoffwatts 分支，双卡 SM70 的投影/GDN/attention/通信组合很接近本地。部分底层文件与本地相同，另一些不同；应比较具体机制与结果，不能把同源优化重复算作独立发现。

继续借鉴：

- 对照两卡 projection shard、归约、GDN 状态与 MTP graph 的衔接，检查各阶段是否重复存取相同小张量。
- 对照 QPN 的宽度选择、不同草稿数下的验证路径，以及 1..当前上限每个宽度的资源占用。
- 参考其 MTP3/4/5 的真实任务结果，理解增加 K 的收益如何随模型和输出类型变化。
- 继续跟进原量化 GGUF 执行中的布局、混合格式和投影融合实现；只借鉴当前已注册 artifact 所需的部分。

其官方 NVFP4 MTP3/5 **138.06/151.45 tok/s** 来自 NVLink 双卡、74-token prompt、131072 capacity 的 pelican/SVG/HTML 自然完成任务；不等于长上下文性能，也不与本机 QUASAR 同权重。Swift/GSQ/Coder390 是另外的 artifact，吞吐不能解释为保持本地权重的内核提升。

### 3.3 1CatAI/1Cat-vLLM（README 正在使用 1Cat-LLM 名称）

[仓库](https://github.com/1CatAI/1Cat-vLLM) · [SM70 算子入口](https://github.com/1CatAI/1Cat-vLLM/tree/HEAD/csrc) · [设计资料](https://github.com/1CatAI/1Cat-vLLM/tree/HEAD/docs/design)

已检查其 SM70 相关源码和多份设计资料。它不是只值得借鉴 prefill：**长上下文小 query attention、量化投影、GDN、DFlash 完整轮次与 TP 通信都是 decoding 的直接参考**。

| 可借鉴项 | 优先读的源码/文档 | 对本地的用途 |
|---|---|---|
| GQA-packed 小 query attention | [`sm70_grouped_long`](https://github.com/1CatAI/1Cat-vLLM/tree/HEAD/csrc/attention/sm70_grouped_long)、[`sm70_dflash2_tail_graphs_20260911.md`](https://github.com/1CatAI/1Cat-vLLM/blob/HEAD/docs/design/sm70_dflash2_tail_graphs_20260911.md) | 多 query/head 共享 KV 读取，联合选择 split、QK/PV 分块及 FP32 分区结果 |
| FP32 分区状态与精度代价 | [`sm70_dflash2_fp32_defaults.md`](https://github.com/1CatAI/1Cat-vLLM/blob/HEAD/docs/design/sm70_dflash2_fp32_defaults.md) | 区分存储量化误差、反量化算术误差和归约误差，避免快路径掩盖精度下降 |
| TP2 verifier 专项 | [`sm70_dflash2_tp2_verifier.md`](https://github.com/1CatAI/1Cat-vLLM/blob/HEAD/docs/design/sm70_dflash2_tp2_verifier.md) | 借鉴直接 packed GDN、缩小 value tile、原 scale 顺序匹配及完整 round A/B |
| GDN 验证 tile | [`sm70_gdn_verify_tile.md`](https://github.com/1CatAI/1Cat-vLLM/blob/HEAD/docs/design/sm70_gdn_verify_tile.md) | 用更短的独立 value-column tile 改善寄存器依赖，同时检查每个状态快照 |
| 小矩阵 QPN8 | [`sm70_qwen38_qpn8_decode.md`](https://github.com/1CatAI/1Cat-vLLM/blob/HEAD/docs/design/sm70_qwen38_qpn8_decode.md) | 融合 gate/up、形状特定 split-K、分清 GPU kernel 服务时间与 host gap |
| push 归约 | [`custom_all_reduce.cuh`](https://github.com/1CatAI/1Cat-vLLM/blob/HEAD/csrc/custom_all_reduce.cuh) | 对照本地 peer-read、消息大小门槛、同步及归约布局；FP16 路径须适配本地 BF16 边界 |
| 草稿尾轮 CUDA Graph | [`sm70_dflash2_tail_graphs_20260911.md`](https://github.com/1CatAI/1Cat-vLLM/blob/HEAD/docs/design/sm70_dflash2_tail_graphs_20260911.md) | 检查接近 capacity、缩短验证宽度和 fallback 的图覆盖及真实长度，而非只测满宽度 |
| lookup 与 DFlash 协作 | [`sm70_dflash2_ngram_hybrid.md`](https://github.com/1CatAI/1Cat-vLLM/blob/HEAD/docs/design/sm70_dflash2_ngram_hybrid.md) | 命中时跳过 query/selector，仍更新 drafter context；低命中率需退出或关闭 |

其 v1.5.1 的 **4K/32K/128K：229.34/201.18/171.43 tok/s** 来自 **4 × V100-SXM2 32GB、TP4、QUASAR+DFlash2 q7、E4M3 target KV**，与本机两张 16GB、INT8-G64 KV、MTP 不同。README 的约 260 tok/s 是另一台四张 16GB 的演示。两者均不是本地提升预测。

特别值得保留的负面结果：其 TP2 verifier 记录中，多个 QPN unroll/寄存器限制/预取方案变慢，某些 allreduce+norm 融合也更慢；ngram hybrid 在低命中 coding 上未通过默认启用的吞吐门槛。**“融合”“更少寄存器”“更多草稿”都只是候选机制。**

### 3.4 当前 decoding 源码复核（2026-10-18）

旧 README 致谢只说明历史来源，不能证明本地已吸收当前 decoding。此轮用远端 HEAD
核对了三个研究副本，并重新沿执行调用链读代码：geoffwatts 为 2026-09-14 的
`perf(sm70): accelerate vision encoding`；Duo 为 2026-10-15；1Cat 为 2026-10-13。
以下结论来自实现，而不是致谢或吞吐表。这里的“已读”只指列出的相关路径。

| 来源和实际入口 | 代码确认的机制 | 本地差异与继续优化项 |
|---|---|---|
| geoffwatts `program_impl.h::lookup_draft` / `decode_mtp_batch`、`mtp_impl.h` | 16-token suffix 倒序找最近匹配，允许重叠续接；与已生成 MTP proposal 一致后才切到较长 lookup 验证，下一轮只生成一个 learned proposal | 本地没有这条 lookup 调度；原版扩大验证窗口，不能照搬到本轮 MTP4。固定四草稿的 lookup/缓存 proposal 机制继续列为候选，先评估真实命中率和 MTP 状态维护成本 |
| Duo `mtp_impl.h::slice_mtp_frame` / `mtp_decode_batch_body`、NVFP4 QPN 内核 | 每 rank 用各自 ingress 和 penalty counters，重复确定性 accept/position 计算；分片 target/AR hidden 配合双卡投影、归约和 Graph | 本地已具备其主体机制并扩展为多 rank；仍比较小张量布局、AR hidden 复制、验证与 alignment 的重复工作，不能因同源而整项排除 |
| 1Cat `nvfp4_qpn4_sm70.cu` 和 `nvfp4_qpn2_sm70.cu` | QPN4 code 用 `__ldcs`，可提前读取下一组 code；QPN2 有 bundled code/scale、按宽度特化、M8 paired gate/up 和共享 activation | 本地已引入按 N7168/8192、K5120 的 code/scale streaming，85K MTP4 两次均值 157.312，输出和接受计数一致。显式 L2 prefetch 具体变体更慢；共享 activation/metadata 布局方向继续保留 |
| 1Cat `gdn_verify_sm70.cu::gdn_verify` | 当前 `cfg=1` 实际选 2 warps × 4 value rows，即每 CTA 8 rows；按 TT=1/2/4/5/8 特化，先发射状态、短窗口 Q/K/V 和 gates 的读取，再计算递推 | 本地 record 为 4 warps × 4 rows，即每 CTA 16 rows，逐 token 取输入；小 tile 和整窗口预取存在可测试增量。保持本地 BF16 输入/record、FP32 状态、归约公式和 commit 边界，不复制其 FP16 staging |
| 1Cat `grouped-attention.cu::GroupedVerifyTraits` 与 packed verifier | q8 把 6 heads × 8 queries 合成 48 rows，N32；80 context splits 对应 80 SM；FP32 softmax/累加配合量化 KV staging | 本地为 TP2 两个 KV heads、T5/30 rows、INT8-G64；现有 QPN5 已复用 KV，仍保留 head packing、分区并行度、FP32 概率和 reduction 的继续优化项，不能照搬 TP4/E4M3 几何 |
| 1Cat `custom_all_reduce.cuh::sm70_cross_device_reduce_1stage_push` | 双 epoch 槽、16-byte volatile push，用保留 FP16 NaN payload 标识空槽，固定 rank 顺序 FP32 求和；当前 kernel 明确限制 world size | 本地为 TP2/BF16/PCIe，需要单独设计资格；本地已有直接 contribution→residual 归约，不等于 push、同步融合或 packet geometry 已最优。已接入 BF16 TP2 双 epoch packet push（173.803），以及 packet sum/norm 与跨层 norm 交接（180.410/180.566）；direct/DMA、可变尺寸链和 Graph/eager 状态通过。传输/融合仍可继续优化 |

所有候选仍按完整 round 和 committed wall throughput 判断。micro 收益不是整模型收益；
改变计算关联或精度的候选须独立记录输出影响，不能以一次相同 token 轨迹声称无损。

## 4. 新增重点项目的可迁移价值

### 4.1 v100-skinny：继续打磨 QPN，而不是止于“已有 QPN”

[源码](https://github.com/dnv2003/v100-skinny/blob/HEAD/kernels/skinny_kernels.cu) · [QPN 研究说明](https://github.com/dnv2003/v100-skinny/blob/HEAD/docs/qpn_race_notes.md)

已读实际 kernel。QPN2 用 warp 内 quadpair 沿 N 分工，在窄 T 时减少空算；不同 SPLITK、NACC 和取数方式改变并行度、依赖链及归约成本。其公开 kernel 表报告 NVFP4 M=8 加权约 **637 GB/s**，旧固定四 warp 路线约 **441 GB/s**，参考 read ceiling 约 879 GB/s。这是其几何和测量下的算子结果。

本地已从该设计获得大量实现，且已有更细的数值边界处理。下一步应逐项比较真实 TP2 shape，重点是 W8/FP8/QUASAR 输入投影、gate/up、down、output 及不同 T 的取舍。源码也记录 **PRMT LUT 比 shift/rebias 慢** 的特定结果；不能只因另一个仓库的 LUT 峰值高就全局替换。

### 4.2 HOOLIGAN：K/V 同时取数和 split-KV 工作量组织

[项目](https://github.com/xormal/V100-SM_70-Flash-Attn-2-v1) · [split_decode.cu](https://github.com/xormal/V100-SM_70-Flash-Attn-2-v1/blob/HEAD/fa2_sm70/csrc/split_decode.cu)

已读 kernel：其 `FA2SM70_DEC_VK` 让 V 与 K 同时发射，避免等 QK 完成才开始独立的 V load。README 报告相应 decode attention microbenchmark 在 32K/256K 分别约 **+16%/+28%**；还展示 unroll 与 head dimension 的寄存器约束，以及用 `B × H × splits` 考虑 CTA 总数的方式。

对本地最有用的是 **同时保持更多 KV 读取在途、减少 scale 重复解码、复用六个 Q head 对同一 KV head 的读取**。其 cache、dtype 和 query/head 布局不是本地 INT8-G64 的直接替代品；已有 QK 共享也不能证明 V-load overlap 已最优。改变分区/softmax 关联的候选要单独做数值与模型检查。

### 4.3 voltattn：先解决并行度，再谈压缩收益

[项目](https://github.com/kamilch1k/voltattn) · [源码](https://github.com/kamilch1k/voltattn/blob/HEAD/src/voltattn.cu)

已读 `attn_q8_mlp` 等实现。其设计从一个 warp 服务一个位置，改为 **8 lanes/位置、每 warp 四个位置在途**，缩短 shuffle 并增加独立 load/softmax 链。

证据要准确读：V100-FHHL-16GB、H=32、D=128、L=16384 下，FP16 优化版约 **0.343 ms**，Q8 优化版约 **0.255 ms**；Q8 的 memory-bandwidth utilization 是 **67.1%**，不是 99%。约 99% 是其较长上下文 **FP16** 控制路径，README 前半还有 Ada 数据。Q8 相对优化后 FP16 的提升约 1.35 倍，而不是把相对较弱 FP16 baseline 的 2.35 倍当成纯压缩收益。

本地 D256、GQA6、Hadamard/INT8-G64 和分页访问均需重新适配。该项目最值得借鉴的是访存组织与对照方法；不能把它的 microbenchmark tokens/s 当成整模型 decode。

### 4.4 PCIe 专项：push/pull 可能比增加理论带宽更重要

[vllm-pcie-allreduce](https://github.com/squibloads/vllm-pcie-allreduce) · [patch](https://github.com/squibloads/vllm-pcie-allreduce/tree/HEAD/patches) · [双 V100 llama.cpp 专项](https://github.com/123123213weqw/dual-v100-llama.cpp)

前者已检查 patch：主要作用是使已有 SM70 push collective 在符合条件的 PCIe mesh 及 CUDA Graph 路径上被选中，**不是一个全新的归约 kernel**。它在四卡混合 NVLink/PCIe 拓扑报告 Flash-Next 单请求约 **75.9 → 91.5 tok/s**，但明确只覆盖 TP4，其测得的跨 root-complex peer store/load 有明显不对称。

对本机的启发是比较 **peer-read 与 peer-write、同步成本、消息大小**，不是照抄其 TP4 路由开关，也不是修改 IOMMU/ACS。第二个项目报告双 V100 matched-precision 通信在 32K/128K/255K 的 warm decode 约 +3%..4%，更接近本地规模，但仍需看其 Q8_0/F16 KV 和具体拓扑。

### 4.5 Augur：GDN 周边、词表和 drafter 都能继续抠

[项目](https://github.com/Augur-rust/Augur-V100) · [GDN glue 报告](https://github.com/Augur-rust/Augur-V100/blob/HEAD/docs/AUGUR_GDN_GLUE_B8_20261016.md)

已进一步读取 `augur_qwen35_gdn_input_b8.cuh`、`augur_qwen35_nvfp4_grouped_decode.cuh`、`augur_decode.rs` 和 shared-head rw4 报告，并核对 MIT/Apache 许可。2026-10-16 报告在单 V100-32GB/Windows 的 B8 DFlash2 路径，把 GDN 周边 norm、stage、小 gate 投影、trim、record copy 合并为少量封闭程序；其 14 段 English-thinking 对照报告 **50.31 → 46.96 ms/round、70.86 → 75.92 tok/s**，并记录相同轨迹。

其 B8 实现用两次 launch 完成 norm→FP16 stage→grouped projection，控制投影使用 FP32 normalized 输入，且有独立 packed B8 布局；本地保留 BF16 控制输入，借鉴 **共用输入、合并极小投影、直接写最终布局和 rollback record** 的机制，不能直接复制数值边界。当前已实现保持 split-10 求和顺序的控制单 kernel，以及 QPN 列 epilogue 的 BF16 projection→conv/record 融合，整模型测得 186.422 tok/s，513 IDs 和接受计数与此前可靠组合一致。shared-head rw4 是 BF16/FP32 SIMT 权重复用，而本地为 W8 head；其 790 GB/s 和 18% 收益不能当作本地预测。grouped NVFP4 的整块 half materialization 也不是本地无运行时权重重打包路线的直接替代。它的 BF16 full head、Q8_0 draft、校准 gate 和概率草稿不等同于本地 W8/BF16 DFlash2 package；不应为了复制数字而改变 artifact 或 gate 参数。

### 4.6 SM70 Marlin 与 SGLang-V100：沿引用链找到的额外主来源

[marlin_v100](https://github.com/zhinianqin/marlin_v100) · [sglang-V100](https://github.com/haohervchb/sglang-V100)

不能因为 **IST-DASLab 的 Marlin 上游要求 SM80** 就排除 Marlin 方向。`zhinianqin/marlin_v100` 明确提供独立 SM70 dense/MoE 工作区、七字段 CTA/warp geometry、split-K、metadata cache 及独立的 repack macro-N。它适合与 QPN 比较不同 token 宽度和矩阵形状；README 列有多种 V100 模型测试，但没有足够条件在此直接推断本地 T=1..4 的收益。其 N/K tile、scale metadata 布局及 epilogue 也可独立借鉴。

`haohervchb/sglang-V100` 是另一条实际存在的 Volta 引擎路线，不能只搜到 `dg1kjd/sglang-sxm2` 就停止。其 README 覆盖四张 V100-32GB/NVLink，报告原生 Qwen3.8-27B FP8 target-only 在 70K/200K 约 **59.1/49.6 tok/s**，对应 CUDA split-KV 与 fused QPN8 gate/up；还记录 DFlash2 的不同上下文结果。1Cat 的 `custom_all_reduce.cuh` 明确注明从它适配 one-shot push collective，因此通信研究应追到该源实现，而不只看下游开关。两仓库本轮均为 README/引用核实，未完整审核其 kernel。

### 4.7 verssionhack / PR #3：调度与会话状态管理（2026-10-22）

[PR #3](https://github.com/tuxKOH/ninfer-V100X2/pull/3) 的完整 diff 为 90 个文件、
新增 10751 行、删除 415 行。其中测试新增 2051 行，文档新增 344 行；家族 runtime
新增 6201 行，包含会话归档、检查点、持久化和 CPU Vision。它不只是一个调度参数，
也没有修改 `src/ops` 下的 decode CUDA 算子。

| 具体机制 | 对当前目标的作用 | 接入或研究状态 |
|---|---|---|
| prefill 步间执行多轮 decode | 已有流在其他请求 prefill 时得到更多输出机会 | 已接入 `--prefill-decode-rounds`，默认 1；本地 85K × 2 / MTP4 对照见下文 |
| 有 decoder 时缩小 prefill slice，无 decoder 时恢复完整 chunk | 缩短一次 prefill 对已有流的连续阻塞；可能牺牲 prefill 和整组吞吐 | **P1 待验证**。PR 同时增加 `PrefillStepResult::service_units`，使小片段不逐次扣完整服务配额；本地接入必须与现有树检查点切分、slice/完整 chunk 切换的 admission 计费一起验证，不能只复制 `min(chunk, slice)` |
| 合并连续物理页传输、固定页内存中转、避免恢复后立即被覆盖的 GDN 导入 | 降低归档/恢复成本，可能减少会话切换停顿 | **P2 保留参考**。本地 `transfer_kv_pages` 已合并连续页；pinned staging 尚未作本地 A/B。属于缓存路径，按用户本轮范围暂不接入 |
| GPU 前缀页引用计数、跨 lane 共享和写时复制 | 共同前缀请求可减少重复 GPU KV 占用 | **P2 保留参考**。与本地不可变 host chunk 共享不同，不应视作树缓存已经覆盖；尚无本地单请求或总吞吐收益证据，本轮不改缓存 |
| RAM parking、消息/tools/response 边界、取消后保留状态、落后分枝恢复 | 避免多会话或客户端重写时重复 prefill | **P2 保留参考**。本地树缓存与其中部分目标重叠，不代表所有状态场景已经等价支持；本轮不移植第二套缓存体系 |
| 磁盘 spill、序列镜像/索引、重启恢复、后台 FIFO 写盘、启动预读/过期清理 | 扩展历史会话保存并减少同步写盘停顿 | **条件候选**。本地没有该磁盘层；PR 的磁盘读取和 GPU 恢复仍在 worker 同步执行，不能称为可与 decode 交错的分步恢复 |
| CPU Vision、GPU Vision 临时空间按需分配、VMM 换出并恢复 KV/state | 以 CPU 工作或搬运时间换取视觉显存空间 | **低优先级保留参考**，服务于视觉/容量目标；没有证据支持纯文本稳态 decode 提速 |
| 队列超时可禁用、prefill 进度、SSE keepalive、模型上下文公告 | 改善长请求服务体验与客户端集成 | **低优先级保留参考**，不作为 decode 算力收益 |

已接入的多轮 decode 在 QUASAR TP2、MTP4、两个同时提交的冷请求（每条 85000 输入、
2048 输出）、chunk 1024 下，把已有流在混合期间的 committed wall 输出速度从
5.026 提高到 17.689 tok/s；该流整个 decode 区间从 22.894 提高到 24.044 tok/s。
代价是后一个请求的 TTFT 增加 5.75 秒，整组吞吐下降约 0.5%。这不是单请求稳态
decode 加速；两组输出 IDs 也有差异，未作无损或质量结论，未跑 GPQA。
完整测量条件和限制见 [性能记录](performance.md#decode-service-during-another-requests-prefill-2026-10-22)。

以上是源码复核和候选归纳；除多轮 decode 外，其余机制未在本地移植验证，
不能把未测候选当成已证明无用，也不能将 PR 作者的测量直接当成本机收益。

## 5. 广覆盖项目清单

核实等级：**S** = 本轮进一步检查了相关源码；**D** = 阅读 README/设计/报告，未完整审核实现；**A** = 主要作算法或工具设计参考，不宣称当前可在 V100 运行。

### 5.1 最直接的 V100 decode、内核和同族实现

| # | 项目 | 核实 | 借鉴点与适用边界 |
|---:|---|:---:|---|
| 1 | [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100) | S | 同族 Volta 投影、状态、MTP；单卡 32GB 与本地 TP2 分开比较 |
| 2 | [plus1998/Ninfer-V100-Duo](https://github.com/plus1998/Ninfer-V100-Duo) | S | TP2 组合、QPN、MTP 窗口；NVLink 及其他 artifact 成绩不能直接外推 |
| 3 | [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | S | 小 query attention、GDN、QPN、DFlash、通信；TP4/E4M3 是另一配置 |
| 4 | [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) | S | QPN2/QPN8、split-K、累加链、精确混合 NVFP4/FP8 执行；已有覆盖仍继续比较 |
| 5 | [kamilch1k/voltattn](https://github.com/kamilch1k/voltattn) | S | Sub-warp 量化 KV decode、内存并行度；本轮看到的实测主几何是 D128 |
| 6 | [xormal/HOOLIGAN](https://github.com/xormal/V100-SM_70-Flash-Attn-2-v1) | S | K/V load overlap、GQA、split/寄存器权衡；移植 INT8-G64 需适配 |
| 7 | [squibloads/vllm-pcie-allreduce](https://github.com/squibloads/vllm-pcie-allreduce) | S | PCIe push 路由、读写不对称；公开资格是 TP4，不能原样声称 TP2 收益 |
| 8 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | D | 原量化 MMVQ、Q4_K/Q6_K 解码、attention/MTP；本地已 vendor 部分 Volta FA，仍应比较后续改进 |
| 9 | [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | D | 量化解包与窄行路径；[Pascal/Volta 表](https://github.com/ikawrakow/ik_llama.cpp/blob/HEAD/docs/pascal-volta.md)明确若干 fallback 和 Graph 限制，不能泛称全部更快 |
| 10 | [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) | D | TurboMind 的 Volta W4A16/MXFP4、流水线/epilogue；适配本地权重而非改为 AWQ |
| 11 | [123123213weqw/dual-v100-llama.cpp](https://github.com/123123213weqw/dual-v100-llama.cpp) | D | 双卡 SM70 kernel/通信实验、MTP/DFlash A/B；优先读 matched-precision 对照 |
| 12 | [Augur-rust/Augur-V100](https://github.com/Augur-rust/Augur-V100) | D | GDN glue、共享 BF16 head、直接 Q8_0 draft；不同 artifact 与采样边界 |
| 13 | [fishlikeX/sm70-attn](https://github.com/fishlikeX/sm70-attn) | D | D256 Split-D、SplitKV3、量化 KV 直读；其 176K +39.9% 主证据是 prefill，8K decode 基本持平 |
| 14 | [taskeee/ninfer-v100-sm70-decode](https://github.com/taskeee/ninfer-v100-sm70-decode) | D | 长上下文 decode 移植、真实 Agent session；ported kernel 改数值，MTP 收益含接受率变化 |
| 15 | [maco1979/nvfp4-v100](https://github.com/maco1979/nvfp4-v100) | D | PRMT-LUT GEMV、paired fusion、DeltaNet/Graph；其 README 称 E8M0 scales，不能当成本地 E4M3 NVFP4 codec |
| 16 | [zhinianqin/marlin_v100](https://github.com/zhinianqin/marlin_v100) | D | 独立 SM70 dense/MoE Marlin、CTA/warp shape、split-K、metadata cache；不是 SM80 上游原版 |
| 17 | [haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) | D | 27B FP8 split-KV、QPN8、DFlash2 和 one-shot push；四卡/NVLink 成绩需与本机分开 |

### 5.2 同源分支、独立 Volta 引擎和额外内核

| # | 项目 | 核实 | 借鉴点与适用边界 |
|---:|---|:---:|---|
| 18 | [JimmyMax/ninfer-v100](https://github.com/JimmyMax/ninfer-v100) | D | 同族窄行 QPN、GGUF 执行、投机开销；其关闭 ngram/adaptive 后更快的结果值得检查 |
| 19 | [ww485000/ninfer-windows-v100](https://github.com/ww485000/ninfer-windows-v100) | D | 单卡 SM70 融合与测量；仅研究 CUDA 增量，Windows 适配不在本任务实现范围 |
| 20 | [Havensky-stack/windows-ninfer-V100X2](https://github.com/Havensky-stack/windows-ninfer-V100X2) | D | 双卡 P2P/投机整合；本项目下游，同源机制不重复算收益 |
| 21 | [taskeee/ninfer-v100-splitd-kernel](https://github.com/taskeee/ninfer-v100-splitd-kernel) | D | 1Cat/fishlikeX D256 FA 的 NInfer 适配经验；直接主收益为 prefill |
| 22 | [gilby/volta-nvfp4](https://github.com/gilby/volta-nvfp4) | D | QPN2 MoE、Graph capture size 对路由的影响；27B dense 主要借鉴 kernel/Graph 机制 |
| 23 | [humanjesse/vllm-v100](https://github.com/humanjesse/vllm-v100) | D | TurboMind、GGUF、数值与 GDN layout 修复；某些 loader 会转 F16/改表示，不能直接沿用 |
| 24 | [dg1kjd/sglang-sxm2](https://github.com/dg1kjd/sglang-sxm2) | D | 多 V100 小消息通信、MTP 和缓存热路径；主模型多为其他 MoE，只取相关机制 |
| 25 | [Haru-neo/qengine](https://github.com/Haru-neo/qengine) | D | 直接 CUDA、gate/up 输入复用、split-K attention；主要实测 CMP 100-210，不能等同正常 V100 |
| 26 | [poisonxa16/pxa](https://github.com/poisonxa16/pxa) | D | Pascal/Volta、tensor split/fused collective、按卡调度；PXQ/PXQN 是其他权重格式，不能替代本地 Q4_K_M |
| 27 | [xeonai44/xllama.cpp](https://github.com/xeonai44/xllama.cpp) | D | 多种 KV codec 的 V100 路由/解包成本；主表是 Qwen3.5-9B，低 bit KV 改质量边界 |
| 28 | [cristianku/redshift](https://github.com/cristianku/redshift) | D | 窄行验证容量、融合反量化；其 supplied tokens/s 不是 accepted/generated tokens/s |
| 29 | [e-c-k-e-r/volta-engine](https://github.com/e-c-k-e-r/volta-engine) | D | 四卡 C++ Volta 引擎、QPN 组合；主要 GLM 等 MoE，且本体标 AGPLv3，代码复用要处理相应许可 |
| 30 | [jajmangold/superl8](https://github.com/jajmangold/superl8) | D | DP4A GEMM/attention、Q4_K 解包；优先级低于正常 V100 HMMA 路线，W8A8 额外量化需验证 |
| 31 | [jajmangold/superl8-serve](https://github.com/jajmangold/superl8-serve) | D | superl8 的服务/投机整合；不可因有 server 就假设所列 kernel 已获本模型验证 |
| 32 | [ai-bond/flash-attention-v100](https://github.com/ai-bond/flash-attention-v100) | D | Volta FA、varlen/KV-cache API 与内核布局；需核实所需 D256/INT8 小 query 路径 |
| 33 | [ZRayZzz/flash-attention-v100](https://github.com/ZRayZzz/flash-attention-v100) | D | 早期 Volta/CUTLASS FA 实现；主要基础参考，不据此预测本地 decode |
| 34 | [zhinianqin/flash-attention-v100](https://github.com/zhinianqin/flash-attention-v100) | D | SM70 FA/vLLM 适配与 kernel 组织；q8k 等标题并非本机端到端 decode 结论 |
| 35 | [sirCamp/flash-attention-legacy](https://github.com/sirCamp/flash-attention-legacy) | D | WMMA/legacy FA；README 所列主要 D64/128，q_len=1 fallback，D256 decode 优先级低 |
| 36 | [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75) | D | GDN D128 前向分块；SM70 只有编译覆盖，实测为 SM75，不能称 V100 实测加速 |
| 37 | [bitexploder/fnfl-v100](https://github.com/bitexploder/fnfl-v100) | D | llama.cpp 派生 Flash-Next Volta 路线；MoE 专用，相关 QPN/attention 可继续筛选 |
| 38 | [christ-pher/bnk](https://github.com/christ-pher/bnk) | D | V100+RAM 的 MoE expert/投机/长上下文；offload 不是本地 dense 驻留 decode 的直接替代 |
| 39 | [Sak-114514/ac922-monster](https://github.com/Sak-114514/ac922-monster) | D | POWER9/NVLink、MoE 与多卡调度；平台及模型不同，作为通信/路由旁证 |
| 40 | [P-J-GROUP-INC-LTD/Vibe100](https://github.com/P-J-GROUP-INC-LTD/Vibe100) | D | 两种 V100 MoE 引擎组合；没有本地 27B dense 的同条件吞吐证据 |
| 41 | [ADA-quart/bonsai2-27b-v100-optimizations](https://github.com/ADA-quart/bonsai2-27b-v100-optimizations) | D | D256、q8 KV 直读、SM70 matvec/MTP；ternary 权重不可用于证明本地 Q4 性能 |
| 42 | [christopherrobertbrooks-tech/volta-bonsai](https://github.com/christopherrobertbrooks-tech/volta-bonsai) | D | ternary Volta kernel；只借鉴解包/分工，权重数学不同 |
| 43 | [JulianFisla/vllm-0.9.2-volta-prefix-caching-fork](https://github.com/JulianFisla/vllm-0.9.2-volta-prefix-caching-fork) | D | 旧 vLLM chunked prefill/prefix 修复；prefix 主要降低 TTFT，非 steady decode 加速证据 |
| 44 | [RomanMoz/power-sglang-cuda124](https://github.com/RomanMoz/power-sglang-cuda124) | D | POWER9/SM70 fallback、Triton 路由；部署兼容不直接等于 decode 更快 |
| 45 | [RomanMoz/power-triton-cuda124](https://github.com/RomanMoz/power-triton-cuda124) | D | 旧架构 codegen/工具链；只有具体 kernel 的编译结果影响当前选择时再深挖 |
| 46 | [boa-collaboration/triton-power9-volta](https://github.com/boa-collaboration/triton-power9-volta) | D | Volta/Triton 与 Mamba SSD；适合状态算子思路参考，非本地已验证路线 |
| 47 | [0ics-srls/1Cat-vLLM](https://github.com/0ics-srls/1Cat-vLLM) | D | 发现为 1Cat 分支；本轮 README 主要继承上游，未建立独立 kernel 优势，低优先级追踪 |

### 5.3 基础库、投机算法和调度参考

这组不全部是 V100 专项；明确列出是为了避免把可借鉴算法误报成已有 V100 kernel。

| # | 项目 | 核实 | 借鉴点与适用边界 |
|---:|---|:---:|---|
| 48 | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | D | 经典 SM70 TensorOp、epilogue、smem 布局；本地已有依赖，仍可比较 narrow/wide tile |
| 49 | [NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer) | D | 传统 V100 FP16 decoder fusion、采样、TP/PP；历史源码参考，不期待直接支持本地 Qwen GDN |
| 50 | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | D | inference kernel injection、融合和通信组织；保留单请求所需机制，不搬分布式框架 |
| 51 | [NVIDIA/nccl](https://github.com/NVIDIA/nccl) | D | 小消息/传输协议和通信对照；不默认认为引入库一定优于现有 TP2 专用实现 |
| 52 | [microsoft/mscclpp](https://github.com/microsoft/mscclpp) | A | GPU-driven communication、semaphore、零拷贝思路；本轮未核实当前版本 SM70 资格 |
| 53 | [Neroued/ninfer](https://github.com/Neroued/ninfer) | D | 同族 Program、投影/词表/状态优化；新卡专用数学路径需重写为 SM70 |
| 54 | [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) | D | 本地 TP2 继承来源，跨卡 Graph/collective/state；RTX 3060/5090 数字不能替代 V100 |
| 55 | [turboderp/exllama](https://github.com/turboderp/exllama) | D | GPTQ 直接 GEMV、解包、融合；格式不同，借鉴算子组织 |
| 56 | [turboderp/exllamav2](https://github.com/turboderp/exllamav2) | D | 低 bit 权重带宽、窄 batch、采样和量化 KV；本轮未建立当前 V100/Qwen3.8 资格 |
| 57 | [AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | D | 旧 CUDA/ExLlama 路由和分组解包；量化工具与本地原码执行分开看 |
| 58 | [qwopqwop200/GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) | D | 旧 CUDA/Triton GPTQ GEMV；仅作低 bit 解包历史参考 |
| 59 | [z-lab/dflash](https://github.com/z-lab/dflash) | A | Block diffusion proposer、验证与训练定义；算法可借鉴，非独立 SM70 成绩 |
| 60 | [snowflakedb/ArcticInference](https://github.com/snowflakedb/ArcticInference) | A | Arctic Suffix Decoding、重复/代码文本的低成本 proposer；需要本地命中率与完整验证 |
| 61 | [hyx1999/SAM-Decoding](https://github.com/hyx1999/SAM-Decoding) | A | Suffix automaton proposal，替代每轮全历史线性 lookup；不能绕过 target |
| 62 | [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding) | A | 无额外训练的并行候选/验证组织；验证算力与状态成本要计入 |
| 63 | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | A | 接受长度、draft/verify 成本及树验证参考；新增模型/草稿权重不属于当前默认实现范围 |
| 64 | [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) | A | 多候选 head、树 attention mask；需要额外模型头，优先级低于已有 MTP |
| 65 | [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA) | A | GDN 代数重排/合理融合；上游主要 Hopper，不能照搬 SM90 warp specialization |
| 66 | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | A | GDN chunk/recurrent 数学、状态布局与融合；本轮未核实当前主线 SM70 codegen |

### 5.4 测量、部署资料和历史分支

| # | 项目 | 核实 | 用途与优先级 |
|---:|---|:---:|---|
| 67 | [pyys/layer-tensor-parallel-bench](https://github.com/pyys/layer-tensor-parallel-bench) | D | V100 layer/tensor split 对照，区分 cold prefill/warm decode；帮助避免把 PP 当单请求 decode 天然加速 |
| 68 | [Efutrrionpy/LlmInferenceOnTaiwania](https://github.com/Efutrrionpy/LlmInferenceOnTaiwania) | D | 多节点 16 V100、FA backend/吞吐；规模不同，低优先级参考 |
| 69 | [KumphanartDansiri/v100-vllm-2026](https://github.com/KumphanartDansiri/v100-vllm-2026) | D | 八卡 V100 部署与 benchmark；用于交叉核对环境，非 kernel 来源 |
| 70 | [redlinedtm-jpg/vllm-v100-2080ti-recipes](https://github.com/redlinedtm-jpg/vllm-v100-2080ti-recipes) | D | pre-Ampere 路由与配方；不同显卡数据分别解读 |
| 71 | [efellowsbg/llm-serving-on-v100](https://github.com/efellowsbg/llm-serving-on-v100) | D | V100 软件栈与测量；部署层资料，低优先级 |
| 72 | [christopherrobertbrooks-tech/legacy-gpu-llm-notes](https://github.com/christopherrobertbrooks-tech/legacy-gpu-llm-notes) | D | 旧卡实测、dtype/PCIe 影响；勿将单卡驻留模型的 PCIe 结论推广到 TP collectives |
| 73 | [jajmangold/vllm-sm70](https://github.com/jajmangold/vllm-sm70) | D | README 已推荐迁往 1Cat；历史补丁来源，优先跟进现行实现 |
| 74 | [dg1kjd/vllm-v100-sxm2-qwen3.5-397b](https://github.com/dg1kjd/vllm-v100-sxm2-qwen3.5-397b) | D | 明确不再维护并推荐 sglang-sxm2；保留拓扑/通信经验，不重复列收益 |

### 5.5 检索到但不能直接当作 V100 decode 实现的项目

| # | 项目 | 核实 | 判定 |
|---:|---|:---:|---|
| 75 | [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | D | 上游 README 要求 CC≥8.0；可以借鉴流水线/布局，SM70 实现看 1Cat/TurboMind/QPN 分支 |
| 76 | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | D | 当前 README 支持表从 Ampere 起；借鉴 paged attention/Graph 设计，不声称原版支持 V100 |
| 77 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | D | 官方主线安装不等于支持 SM70；Volta 实现应看明确的移植分支 |
| 78 | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | D | 当前主线不是本目标 CUDA12.8/SM70 直接候选；借鉴 decoder fusion/Graph 思路 |
| 79 | [facebookresearch/xformers](https://github.com/facebookresearch/xformers) | D | 历史 CUTLASS attention 可能有参考价值，本轮未核实当前包的 SM70 与 D256 decode 资格 |
| 80 | [Supercomputing-System-AI-Lab/VoltanaLLM](https://github.com/Supercomputing-System-AI-Lab/VoltanaLLM) | D | 名称含 Volta，但核心是频率控制/P-D 路由；不属于 V100 专项，也不符合本次不改频率的方向 |

另有 diffusion/图像编辑、ComfyUI 外层插件、普通训练作业、只有部署镜像的命中，本轮没有把它们当作 LLM decoding kernel 来源。纯镜像和未建立独立增量的 forks 也没有逐一包装成新优化。

## 6. 建议的待引入 / 继续优化清单

以下每项都保留“沿已有方向继续优化”的可能；只有实际比较证实无增量时才停止。P0 是首批选题，P1 是下一批，P2 为有条件扩展。代码实现仍应放在现有 Op/家族调度的明确所有者内，不引入第二套模型 runtime。

首项已落实为 QUASAR MTP4 五-query verifier 的下一块 packed K/V 预取，保留各自 attention profile 的原有算术。四-query 的通用替换测量不一致，暂未引入；仍保留按 masking、页表、CTA/寄存器调度继续优化的候选。这个结果也不表示 KV 访存方向已经完成。

随后实验 long profile 的五-query 路径引入了 distributed-Q QPN 和 32-key inline softmax；默认路径保留原预取。独立 85K FP64、缓存/页表/mask 与 Graph/eager MTP4 状态检查通过，整模型及质量边界见上。完整 Q 常驻、64-key tile、历史 mask 简化、原 TC inline-softmax 和本轮 Graph 下界方案未见收益，暂不引入这些具体变体；各方向仍在候选池。

GDN norm+control 融合仍列为**待引入 / 继续优化**，但本轮生产试验已撤回。最初 BF16 control 小基准的 10.608→7.072 µs 与宽松相对误差检查不能代替现有正式标准；公开两卡检查在 T5 的一项 control 上失败。改用未舍入的 FP32 normalized control 后，正式 FP64、parent/split、T1/5/8/9/64 与 Graph/eager 通过；整模型为 152.898 tok/s（一次），约 +0.92%，但输出轨迹变化，且默认 attention 的 25 点比较 min cosine 0.998454、max KL 0.006260，未过 near-lossless。与 long attention 的组合反而更接近原参考（0.999696 / 0.001792），这不证明任务质量改善。当前小收益不足以支持改变默认数值路径，保留融合方向，未跑 GPQA。

TP2 行投影已引入 contribution→residual 的直接归约及 rank-0 BF16 round/add epilogue；去掉单独加法与回拷，正式两卡算子/Graph/整模型检查通过，实测见上。该方向仍可继续优化，尚未完成所有 fusion。peer sum+norm 与输出缓冲交换仍列为待引入 / 继续优化：K5120/T5 的 64 次图序列曾由 21.712 降至 17.088 µs，包含 seed、fixture 的 sum/norm 精确；随后每层 mixer 输出的单 kernel 融合、保留原 norm 归约顺序的两阶段融合分别仅为 148.849、147.702 tok/s（85K/MTP4，各一次），虽均复现全部 513 IDs 和接受计数，但未兑现微基准收益，生产接入已撤回。两卡独立 FP64、旧组合逐位比较、输入/权重 guards 与 Graph 检查通过；两阶段还通过 64 次连续 T5 源输入复用。不能将旧微基准比例叠加到当前 155 结果。NVFP4 PRMT E2M1 解码及预计算 FP16 scale operands 也留在候选池：前者 MLP 持平，少数 input 的 unroll 组合仅有算子收益；后者在本轮形状更慢。二者输出精确比较及抽样 FP64 通过，未作为整模型优化引入。

新的显式 NVFP4 operands 预取也留在候选池，已测 T1/T5 更慢且输出精确。QPN5 packed-KV lookahead 的独立 FP64 与输出精确比较通过，但 85K partial+reducer 从 276.480 µs 变为 320.512 µs（单 block、176 regs）或 435.200 µs（双 block、128 regs 且 spill），这两个变体不引入。FP16 短段累加的私有原型在修正 fragment 解包后仍未过 decoded-weight FP64 标准，没有接入生产或运行 GPQA；后续若研究这一方向，仍需同时处理累加范围、舍入误差与收益。

普通 NVFP4 input unroll4 仍列为待引入 / 继续优化：T5 GDN/attention input 算子从 46.080/39.936 降至 44.032/38.912 µs，公开 Linear 的真实 N7168/8192、K5120、T3/4/5/6/7 独立 FP64、guards、输入/权重不变与 T5 changed-input Graph 检查通过，且与旧 unroll2 输出逐位一致；但整模型两次仅为 150.835/146.667 tok/s，未证明收益，已恢复原展开。随后未改动的 155 基线二进制测得 152.922/154.695 tok/s，四组试验全部 513 IDs 和接受计数一致。测量按时间先后进行且存在波动，因此不能把融合或 unroll 与历史 155 均值的全部差距归因于改动；当前证据支持撤回具体变体，不能排除方向。

QPN5 扩大 KV tile 也保留待引入 / 继续优化：私有 64-key 变体通过完整 85K decoded-input FP64 标准，但 160 regs、约 71 KiB shared memory 限制为单驻留 CTA，partial+reducer 在同轮比较为 388.064 µs，原 32-key 为 277.504 µs，未引入。softmax 分组和 split 边界改变使输出非逐位相同；未作整模型或 GPQA 质量结论。

已有 RAM-KV 的 32768 resident-window 在本轮相同 85K/MTP4 输入上为 136.439 tok/s、68.93% 接受率，慢于完整历史的新实验 attention，且会省略未选历史页，暂不推荐。稀疏/检索方向仍保留，后续重点应是检索与搬运成本、保留策略及长上下文行为，而不是继续缩窗口就认定必然更快。

| 优先级 | 具体候选 | 主要参考 | 本地落点与验收重点 |
|---|---|---|---|
| P0 | **K/V 同时发射、scale 解码复用、增加独立在途位置** | HOOLIGAN、voltattn、1Cat | INT8 attention 的 T=1 和实际验证宽度；D256/GQA6、真实页表/85K；先测算子再测完整 round |
| P0 | **GQA/head/query 联合复用与 split/reducer 改善** | 1Cat、HOOLIGAN、geoff/plus | 对照当前 QPN attention 和默认/实验 profile；split 数影响归约顺序，性能与精度分开验收 |
| P2（暂缓） | **Q4_K/Q6_K 窄行 direct GEMV 的解包/访存/多行复用** | llama.cpp、ik_llama、v100-skinny、maco | `ggml_k.cu`；保留 T=1..4 的读取、header/activation 复用及原码融合 QPN 候选；禁止把 GGML K 当 groupwise Q4；本轮不实施 |
| P0（QUASAR） | **QPN 每种真实形状继续调 SPLITK/NACC、unroll、load、寄存器寿命，并比较 SM70 Marlin** | v100-skinny、marlin_v100、1Cat、geoff/plus | 优先 NVFP4 及其 fused projection，随后 QUASAR 的 W8 组件；其他精度专用调度暂缓。本地已有这些机制，比较其具体组合，避免重复移植 |
| P1 | **norm/cast/多个小投影共用输入，结果直接写目标布局/record** | Augur、1Cat、geoff/plus | GDN input、alpha/beta、post-mixer；检查 dtype 边界、state snapshot、arena 与 Graph 地址稳定 |
| P1 | **GDN value tile 和 packed verifier 的寄存器/依赖链** | 1Cat GDN tile、FlashQLA legacy、geoff/plus | recurrent 与 replay；保持 FP32 persistent state，逐步对照输出和全部状态快照 |
| P1 | **PCIe push vs peer-read，减少小消息 staging/同步成本** | sglang-V100、squibloads、1Cat、dual-v100 | TP2 allreduce；同消息大小、同 BF16 sum、同图依赖；算子变快必须落实到两卡完整轮次 |
| P2（暂缓） | **按完整成本选草稿数，包括低接受率回到少草稿/无草稿** | geoff/plus、v100-skinny、1Cat | 本轮按用户要求固定 MTP4，不扫其他窗口；保留调度成本与接受率分析，其他窗口/自适应待恢复该方向时再研究 |
| P1 | **继续优化 full/optimized vocabulary head、精确 shard top-1** | 1Cat、Augur、v100-skinny | 本地已有 greedy shard argmax，继续研究融合/多行复用；采样不能用未经证明的候选截断代替全分布 |
| P2 | **尾轮、部分接受、fallback 的 Graph 覆盖和多余 fence** | 1Cat、gilby、同族 forks | 检查本地已捕获路径的实际覆盖；保留跨设备生命周期、commit/rollback 和页表更新依赖 |
| P2 | **低开销 ngram/suffix proposer 与模型 drafter 混合** | Arctic、SAM、1Cat、geoff | 在当前草稿上限内先测命中；低命中停用，target/draft 状态仍一致；不直接复制 q15 |
| P2 | **Q4_K 仿射项与 tensor-core 融合解码的专用设计** | QPN、TurboMind、GGML codecs | 保留每块原码、scale、minimum；FP16 operands 是否合适由 oracle/模型门槛判断，不能只检查 packed bytes |
| 条件研究 | **DP4A/W8A8、低 bit KV、稀疏/检索 attention** | superl8、pxa、xllama、qengine | 可能改变输入量化或可见历史，不能作为现有 full-history/INT8 验收的透明替换；只在相应实验契约内研究 |
| 条件研究 | **PP2 与 TP2 的 decode 取舍** | qengine、pyys、sglang-sxm2 | 本地已有 PP2 实验；单请求 decode 两阶段串行，预填充流水收益不能冒充 decoding 收益 |

### 选题时最容易走错的四件事

1. **把低 bit 当作必然更快。** Volta 上解包、scale、转换和 softmax 的 issue/依赖成本可能吃掉全部带宽节省；voltattn 与 xllama 都给出了反例。
2. **把更高接受率当作必然更快。** 应最大化 `E[本轮提交 token 数] / E[本轮耗时]`；drafter、verify、head、状态回放、通信都在分母内。
3. **把单 kernel 数字当作整模型结果。** 本机已有成熟优化，外部从较弱 baseline 得到的百分比通常不能叠加。
4. **把数学等价当作输出不变。** split、融合精度、累加链、softmax 更新改变后可能影响 logits/接受率。要直接对同一个独立 oracle 和本地行为门槛验证。

## 7. 后续实现的最小有效验证

每个候选只收集足以决定保留/放弃的证据，不把调研扩大成全项目 benchmark 工程。

- **数值输入与 oracle**：由公开表示的 BF16/FP32 输入、原始 packed codes/scale/minimum 或原 INT8-G64 cache 解码，独立 FP32/FP64 计算完整公式。不能以另一个快 kernel 为唯一正确性基线。
- **真实几何**：attention 用本地 D256、TP2 Hq12/Hkv2、GQA6，覆盖 T=1 与实际 spec 验证宽度；GDN 用本地 state/heads；projection 用实际 TP2 N/K，而非外部 TP4 的半宽矩阵。
- **状态边界**：只在涉及的范围检查 KV 写入、GDN FP32 state、每个接受前沿、拒绝回退、prefix restore 和 Graph/eager 一致性。访存/布局的 exact 声明用 exact 比较。
- **性能范围**：先算子筛选，只有可改变决策的候选才进入 public Engine 端到端 A/B。报告单请求、occupied tokens、capacity、sampling、K、accepted/drafted、emitted/round、完整 round latency 和 committed wall decode。
- **主要工作负载**：本轮以 QUASAR、TP2、85000 实际输入、180000 capacity、INT8-G64、Graph、greedy optimized-head MTP4 的独立基线比较。Q4_K_M 的原验收条件保留，待恢复该方向时再测。短上下文和 capacity-only sweep 不能替代。
- **质量声明**：改算术的候选通过相应 operator/logit/state/behavior gate 后再讨论默认引入；保持原权重 bytes 或一次 smoke 不是“无额外质量损失”的证明。

实施时不需要先更换为 vLLM/SGLang，也不需要添加新 artifact 或下载新模型。优先把这些项目中可证实的更强机制放入本地既有 Op 和家族调度，再在本机验证。外部源码采用前按具体文件保留许可与 provenance；本报告中的项目级描述不代替文件级许可检查。

本次收尾清理了 112 个过时一次性 Colab 脚本、A100 staging 目录/微检查及空 import 名称文件；
有效生产实现、独立数学测试、实测证据和待优化方向保留。
