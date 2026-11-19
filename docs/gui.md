# NInfer PyQt 控制台

`tools/v100/ninfer_gui.py` 是一个本地 PyQt 控制台。它只管理由窗口启动的
`ninfer-serve` 子进程，不修改驱动、IOMMU、GRUB 或系统服务。窗口关闭时会询问是否
停止它自己启动的 API；已经由其他终端启动的 API 不会被杀掉。

## 安装

在有图形桌面的机器上使用 Python 3.10+，安装 PyQt6（已有 PyQt5 也可以）：

```bash
python3 -m venv .venv-gui
.venv-gui/bin/python -m pip install -r tools/v100/requirements-gui.txt
```

如果系统已经提供 PyQt6/PyQt5，可以直接把 `PYTHON` 指向该解释器。不要把 PyQt 放进
推理运行时的 Python 环境；GUI 只启动 C++ `ninfer-serve`。
启动脚本优先尊重显式 `PYTHON`，否则依次检查 `.venv-gui` 和已安装的 Python 3.10–3.13，
选择已有 Qt 的解释器，不安装任何依赖。

模型入口

GUI 优先查找仓库旁的 `models/` 软链，因此不会复制或纳入 Git 中的几十 GB artifact。
本机可以这样建立入口（目标目录中已有的文件保持原样）：

```bash
ln -s /Models/ninfer-V100X2 models
```

客户机若 artifact 位于用户目录，则对应使用：

```bash
ln -s /home/z/Models/ninfer models
```

软链目标必须包含配置中指定的 `.ninfer` 文件；GUI 会在启动前检查文件是否存在，
不会把一个精度的 artifact 当作另一个精度使用。

## 启动

```bash
bash tools/v100/ninfer-gui.sh
```

也可以显式指定仓库、二进制和模型：

```bash
bash tools/v100/ninfer-gui.sh \
  --repo /home/z/ninfer-V100X2 \
  --binary /home/z/ninfer-V100X2/build-v100/apps/ninfer-serve \
  --artifact /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer
```

客户机若使用独立 artifact 路径，只需在“模型 → 模型文件”中选择该文件。GUI 会记住
非敏感的路径、端口和推理参数；API key 不写入设置文件，只在当前窗口运行期间使用。

## 布局与自定义配置

顶部提供 **卸载到内存 / 加载回显存**。服务保持监听，模型仍是同一个配置；卸载前必须没有活动
请求（包括仍持有响应的请求）。首次卸载会把完整原始 `.ninfer` 容器复制到系统 RAM，成功后释放
模型、KV/GDN、工作区和 CUDA Graph 的 GPU 资源。界面显示 RAM 字节数与切换进度；驱动上下文
仍可能占少量显存。首次复制需要读盘，并要求可用 RAM 至少覆盖容器大小另加 1 GiB 余量。

加载直接读取内存镜像，并重新建立推理运行时，不再次读取 artifact 文件。加载后继续保留这份 RAM
镜像，后续卸载无需再复制；停止/重启服务或关闭它的进程才会释放这份内存。**KV、历史前缀、媒体
处理缓存不跨卸载保留**，下个请求从冷缓存开始。RAM 状态或切换期间推理接口返回 503；有活动
请求时卸载返回 409。加载失败仍保留 RAM 镜像，可以重试。切换操作不会修改功耗或时钟设置。

界面使用中性色和五个参数标签页：模型、推理、采样、服务、实验。右侧保留显存预估和加载
进度，下方可切换指标、日志、命令。实验测量和数值影响放在“实验 → 数值影响与测量说明”，
不放入窗口标题。

“推理”页默认只显示“上下文容量”，KV 自动跟随它。旁边的“独立 KV”展开后可填写单独的整数
或 `auto`；手动整数不得小于上下文容量，`auto` 表示按可用显存选择。收起会清除独立值并恢复
跟随，保存配置时也记录为跟随。加载包含独立值的配置时自动展开，避免隐藏生效的设置。
展开区显示当前手动 KV 的最低值。RAM-KV 实验启用后使用实验页的 GPU window，保留普通设置。
报错、确认及配置对话框使用与主窗口一致的浅色主题。

顶部“配置”是唯一的配置入口，管理用户命名配置：

- **保存为**：保存当前所有非敏感参数，可创建多个名字；同名覆盖会询问。
- **加载**：完整恢复模型路径、TP/PP、上下文、KV、视觉、采样和实验选项；运行时先停止服务。
- **删除**：只删除选中的配置记录，不删除模型文件。
- **导出**：将当前参数导出为一个可携带的 JSON 文件。
- **导入**：检查文件格式、完整字段、类型和取值后保存并加载；不合法文件不覆盖现有配置。

首次创建配置库时加入用户提供的两份默认配置：

| 配置 | 上下文容量 | 视觉 | 长注意力 | 快速数值路径 |
| --- | ---: | --- | --- | --- |
| 视觉-mtp4-248k上下文-长上下文无速度优化 | 248000 | 开，预算1024 | 关 | 关 |
| 无视觉-mtp4-满上下文-长上下文速度优化 | 262144 | 关 | 开 | 关 |

两份均使用 QUASAR、TP2、MTP4、INT8 KV、Graph、前缀复用，KV 跟随上下文；默认输出上限128000，
监听 `0.0.0.0:6198`，CORS 开启，采样参数交由模型或请求决定。长注意力会改变数值归约，
数值与质量依据见下方实验说明；这些容量不是新的速度或显存实测。配置文件位于
[`tools/v100/configs/defaults.json`](../tools/v100/configs/defaults.json)，模型与程序路径相对仓库解析。
它们加入后与自建配置一样可修改、覆盖、删除；已有配置库不被覆盖，删掉的默认记录不会再次出现。
选择配置后点击“加载”才应用，未选择时保留当前参数。

配置库与 Qt 偏好文件放在同一目录，使用 `.profiles.json` 后缀。Linux 默认路径通常是
`~/.config/NInfer/V100X2Server.profiles.json`；`XDG_CONFIG_HOME` 会改变其父目录。
导出的格式是 `ninfer-launcher-config`、版本1，包含名称和完整设置。API key 始终排除，
导入/加载配置也不会替换当前会话中的 key。导入不下载模型，另一台机器可能需要修改文件路径。

## 显存预估与加载进度

选中模型后，GUI 只读取 `.ninfer` 对象目录，不读取完整权重 payload、不创建 CUDA 上下文。
目前覆盖注册的 27B v2 和已支持的 Qwen3.8 v3 目录。预估按每张卡分别显示：

- 权重：按 TP 分片、复制的 embedding/小控制张量、PP 层归属和可选 MTP/DFlash/Vision 筛选。
- KV：按64-token页对齐的预留容量，区分 INT8-G64/BF16、MTP headroom 和 RAM-KV GPU window。
  并发请求共享这份容量，不将整个 KV pool 再乘以并发数。
- 状态、工作区、Graph 与 CUDA 开销：按固定模型形状计算或给出估算区间。

卡片进度条在启动前显示预估占总量的比例，启动后显示 `nvidia-smi` 实际用量；悬停可看分项。
设备总量、当前占用、空闲显存约每5秒刷新，尊重 `CUDA_VISIBLE_DEVICES` 的顺序与 UUID 选择。
预估超过当前空闲显存时只提示，不据此替代引擎的分配判定。无法读取设备时仍可配置和预估。
`KV=auto` 只报告按上下文容量得到的下限，实际容量由引擎在权重上传后选择。

预估不是 allocator 的精确计划，也不代表当前 prompt 的占用。以已保存的 QUASAR TP2/MTP4、
180000容量、INT8-G64、chunk2560记录核对，权重每卡误差小于1 MiB，KV payload 一致；
其他 chunk、视觉预算、并发和 PP2 工作区没有同等精度保证。主机 RAM archive、prefix checkpoints
和媒体缓存不计入 VRAM。
新版 GUI 的同配置启动检查中，每卡预估 13.34–14.39 GiB，`nvidia-smi` 实测分别为
13.84、13.53 GiB，均在区间内。这是启动显存检查，不是新的吞吐测量。

“加载进度”解析服务端实际 `load weights` 百分比。上传完成后仍显示“初始化运行状态”，
预热阶段显示“预热”，这两阶段使用不定进度条；只有收到 API 监听日志才到“就绪”。
百分比表示权重上传阶段，不假设整个启动过程匀速；日志输出间隔也会影响刷新频率。

## 初始设置与视觉开关

新设置默认 **TP2 · QUASAR NVFP4 · MTP4 · 180K**，选择已有的
`/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer`。保留用户已保存的路径和参数。
首次打开时可从默认配置中选择并加载，也可保存当前参数或导入配置 JSON；再次打开恢复上次的非敏感参数。
其他支持的模型和执行方式通过字段配置，所有组合共用同一配置保存、加载入口。

“推理 → 视觉”默认关闭，对应 `--vision`。打开时加载同一 QUASAR v3 文件里的
333 个视觉对象：默认 GPU 模式放主卡，CPU 模式只放主机 RAM；关闭时不分配塔权重和视觉工作区。无需重复下载模型。
“视觉 token 预算”对应 `--vision-max-tokens`，独立于文本上下文容量，按一条请求所有媒体合计。
图片测试用 180K 文本容量、1024 视觉预算；现有五秒视频测试用 98K 容量、4096 视觉预算。
较小预算可能拒绝视频。开关和预算会保存，重启 API 后生效。

勾选视觉会关闭 fast decode、RAM-KV 及前缀复用，但可保留 long attention；可手动重新启用缓存。QUASAR 的 TP2 / MTP4 /
4096 容量现有前缀集成检查已通过，包括视觉桥接、媒体变化/追加、停止和恢复。TP4、DFlash、Q4_K_M 和 native FP8 的不支持组合在启动前被阻止。
GUI 是 API 启动器；图片/视频由客户端通过既有 OpenAI/Anthropic 接口传入。

加载命名配置会替换完整参数，包括视觉、数值实验、RAM-KV、采样和网络设置；API key 保留在
当前会话。运行期间锁定参数及配置加载；先停止服务再编辑，
或使用“重启”按当前配置重新启动。“日志 → 保存日志”可导出窗口内的服务日志。
自己的子进程只有输出 API 监听信息后才显示运行就绪，提前返回的 `/health` 不代表模型加载完成。

## 实验性加速（可能影响输出）

单独分组提供两个默认关闭、会保存选择的启动开关：

- “NVFP4 快速数值路径（MTP / DFlash）”：`--volta-fast-decode`，QUASAR / SM70 / TP2/TP4 / 纯文本 None/MTP，DFlash 仅 TP2；
  NVFP4 全局缩放移到 FP32 K 归约后，并启用 MTP W8 特定分区/归约。
- “完整历史长注意力（文本 / 图片 / 视频）”：`--volta-long-attention`，SM70 27B / TP2/TP4 / INT8 KV / 实验性并发1–8，视觉和 DFlash 仅 TP2；
  保留每个历史 KV，但改变 softmax、PV 和归约顺序。

TP2 DFlash 可单独或同时开启这两项。长注意力作用于目标模型验证，快速数值路径保留
原生 W8/BF16 drafter；数值变化仍可能改变接受率和输出。DFlash 的实测增幅见
[性能记录](performance.md)，不能套用下方 MTP 的历史收益；未做新的质量测评。

本机 TP2 纯文本组合在 85K 实际输入、180K 容量、MTP4、greedy、Graph 下测得 **198.105 / 198.097 committed tok/s**。
最后 W8 步骤比前一组合约 **+0.84%**；不是两个开关各自的独立提升幅度。
远端 TP4 / 220 W / 1200 MHz 同任务为 **231.11 / 231.49 wall tok/s**；配对投影与 QPN 循环
调度比 216.93 组合约 **+6.62%**，该任务全部输出与接受计数不变。两平台和实验数值路径的影响分别看待。
移除读取未初始化暂存区的 eager 残差试验后，TP4 普通、前缀及实际 85K Graph/eager、重放
logits/接受计数与提交前沿检查均通过；这些检查不等于质量无损。
前序组合 25 点全词表检查 argmax 25/25 一致，最小 cosine 0.998830814、最大 KL 0.001903103，
但 near-lossless 仅 9/25 点通过。当前组合未跑 GPQA，不能量化“智力下降百分比”。
提示和分组说明明确这些限制；可将自己的实验组合保存为命名配置。
关闭快速数值路径时仍保留布局、融合及通信优化，保持普通 per-group FP16 缩放。
详细数值依据和剩余优化方向见 [性能记录](performance.md) 与 [调研清单](v100-decode-survey.md)。

“实验 PP”提供已有 `--pp 2` 路径：选择 TP1、填写两个设备 ID、并发1、纯文本 None/MTP；
“PP prefill chunks”范围 1–8。它不支持视觉、DFlash 和上述两个数值实验，没有确定的本机速度增幅。

“实验 RAM-KV”提供已有近似历史检索路径，默认关闭。启用时 `--kv-capacity` 自动等于 GPU token
窗口，暂存的普通 KV 容量设置不变；关闭后恢复普通完整 KV 配置。窗口须为64倍数，不超过逻辑
上下文，至少为 `prefill_chunk+8192`（PP2 为 `prefill_chunk*pipeline_chunks+8192`）。
预算是 RAM archive 字节数，默认 32000000000，**不是进程 RSS 上限**。仅单请求 Text/None/MTP、
TP1/TP2/PP2，排斥视觉、DFlash、TP4、长注意力。它在长 prefill 和 decode 都可能舍弃未检索历史，
因此可能降低长上下文质量；没有可直接套用到当前组合的速度或 GPQA 数据。
模型与硬件最终准入仍由 C++ Engine 验证，详见 [RAM-KV 语义](maintainer/paged-kv-cache.md#15-optional-native-ram-kv-experiment)。

## 远程主机

GUI 本身是本地桌面程序。若客户需要控制远端主机，应在远端图形会话中运行它，或使用
SSH X11/远程桌面；不要让 GUI 直接暴露服务端口。API 默认绑定 loopback。也可以只在
GUI 中复制命令，再通过受控 SSH 会话启动。当前四卡验收主机的命令是：

```bash
bash tools/v100/serve-tp4.sh /home/z/Models/ninfer/qwen3_8_27b_nvfp4.ninfer
```

GUI 的“命令 → 检查连接”只请求配置的 `/health` 地址，不会停止外部进程。

## 推理指标与缓存命中率

“实时草稿接受率”每约 1 秒更新，显示该区间的 target 接受草稿数 / 验证草稿数，
适用于 MTP 和 DFlash；bonus token 不算接受草稿。没有草稿的区间显示“无草稿”，
不是 0% 接受率。
“实时吞吐”显示同一区间的实际 Prefill 和已提交 Decode tok/s，来自服务端 JSONL；
与“上次请求”的整阶段平均值分别显示。

“上次请求”保留最近一次完成请求的 Prefill、Decode tok/s、token 数、阶段耗时和草稿
接受率，新请求进行中不覆盖它。Prefill 按实际计算的输入（总输入减缓存命中）除以
prefill 阶段秒数；Decode 排除 prefill 产生的首个输出 token，按其余已提交输出除以
decode 阶段秒数。它们是阶段平均值，不是 1 秒墙钟区间速度；完整输入复用显示“全缓存”，
没有 decode token 或有效耗时时显示“—”，不显示虚高速度。

窗口显示“最近请求”和“本次启动累计”的前缀缓存命中率，以及命中 / 总输入 token 数。
命中率为实际复用的输入 token 除以总输入 token；累计值按 token 加权，不是各请求百分比
的平均值，也不是 MTP/DFlash 草稿接受率。完整命中显示 100%，冷请求显示 0%。

GUI 为自己启动的 API 自动添加 `--request-log-jsonl` 和 `--log-stats-interval-ms 1000`，
从引擎的区间记录和完成记录读取真实统计，
覆盖 Chat Completions、Responses 和 Anthropic Messages 的流式及非流式请求。
请求完成后更新，重启 API 时清零；未完成或报错的请求不计入累计值。窗口关闭时清理
自己的临时统计日志。只检查外部 API 的 health 不会获取该进程的缓存统计。
“复制命令”会移除 GUI 专属的临时日志参数，关闭窗口后仍可在终端使用。

API 客户端也可读取缓存命中数：OpenAI Chat 使用
`usage.prompt_tokens_details.cached_tokens`（流式需 `stream_options.include_usage:true`），
Responses 使用 `usage.input_tokens_details.cached_tokens`，Anthropic Messages 使用
`usage.cache_read_input_tokens`。详见 [HTTP serving](serving.md)。

普通 Text/MTP 请求还会在 RAM 中保存有界的完整循环状态检查点。下一次输入较短或
只修改后半段时，可恢复公共前缀内最近的检查点，再重算余下部分；命中数只计真正
跳过计算的 token，不把重算部分算进缓存。首次建立检查点需要分配和传输 RAM 数据，
会增加 prefill 耗时；后续请求复用已有缓冲。DFlash、视觉和 RAM-KV 仍使用原有的
完整 frontier / turn / response 检查点规则。

历史分枝缓存的预算在“推理 → 历史缓存预算 / tokens”设置，随命名配置保存/加载。
推荐默认 **384,000 tokens**；0 关闭历史分枝，仅保留当前 GPU 前缀。
旧分枝在覆盖前把原始 KV、各 rank 的 GDN 和 tail hidden 保存到 RAM，
因此 `ABD → ABC → ABDE` 可以回到 ABD 的完整检查点，不局限于上一个请求。
多个对话树共用预算，按最早保存的分枝先淘汰；同一已恢复血统的完整不可变块只计一次。
预算按 Main Text 的64-token页取整，不是进程内存上限：QUASAR TP2 INT8 KV 的384K预算，
主 KV 加 MTP 最多约 **12.8 GiB RAM**；按1M tokens折算约33.4 GiB，还需完整状态检查点
及恢复时的临时余量。
恢复到有完整状态的最近前缀，其后需要重算的 token 不算命中。Vision、DFlash 和稀疏 RAM-KV
不使用此历史分枝缓存。[所有权与预算](maintainer/paged-kv-cache.md#104-historical-branches)。

## CPU 视觉塔（0.1.2）

开启视觉后，在“视觉推理设备”选择 `gpu`（默认）或 `cpu`。CPU 模式只把视觉塔放入主机 RAM，
媒体编码结束后上传 BF16 特征；显存区间不再计入塔权重/编码工作区，但仍计 GPU 特征缓冲。
下方同时显示 CPU 塔权重 RAM 估计，另需媒体工作区与已有缓存 RAM。选项支持设置和配置保存/加载。
图片/视频会更慢，其他已入队请求可继续解码。CPU 与 GPU 计算的数值差异可能改变输出，未跑新 GPQA。
开启视觉只自动关闭 fast decode 和 RAM-KV，不再关闭 long attention；该长注意力实验仍默认关闭。

CPU 视觉模式下，启动器将 API 的媒体准备/排队超时设为 600000 ms（10 分钟）；
命令行直接启动服务时，大图或视频需自行提高 `--pending-timeout-ms`，默认 30 秒可能不够。
