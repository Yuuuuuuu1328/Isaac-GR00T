# GR00T Orin Nano Super 推理阶段耗时测量与分析实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在本地 Jetson Orin Nano Super 上，对 GR00T N1.5 的推理路径建立可复现、可分阶段、可比较的时延测量体系，输出足以支撑后续加速设计的基线、日志和分析结论。

**Architecture:** 先固化实验口径，再在现有 PyTorch 和 TensorRT 两条推理路径上加入统一 profiling hook，把“数据读取/预处理/模型输入准备/backbone/action head/每个 denoising step/后处理”拆开记录；随后跑固定基线和单变量扫描，把结果汇总成结构化日志和分析报告，最后用明确闸门决定下一步优化优先级。

**Tech Stack:** Python, PyTorch, TensorRT, CUDA Event, `time.perf_counter_ns`, `tegrastats`, `pytest`, Markdown, CSV/JSONL

---

## 一、先回答你的核心问题：当前仓库是否已经包含“详细时延测试”

结论：**没有形成你现在需要的“本机、分阶段、可复现、可分析”的时延测试闭环。**

当前仓库里已有的东西：

1. 部署文档里有官方 benchmark 结果和模块级参考数字，但它们是文档说明，不是你本机的可复现实验流水线。
   - `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/README.md`
   - `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/orin/README.md`

2. 推理入口已经很清楚，适合插桩。
   - `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/gr00t_inference.py`
   - `/home/jetson/Desktop/project/Isaac-GR00T/gr00t/model/policy.py`
   - `/home/jetson/Desktop/project/Isaac-GR00T/gr00t/model/gr00t_n1.py`
   - `/home/jetson/Desktop/project/Isaac-GR00T/gr00t/model/backbone/eagle_backbone.py`
   - `/home/jetson/Desktop/project/Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py`
   - `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/trt_model_forward.py`

3. 现有“计时”只有非常粗粒度的总耗时打印，不能支撑加速设计。
   - `/home/jetson/Desktop/project/Isaac-GR00T/scripts/inference_service.py` 只统计 client 到 server 的总耗时。

4. 现有测试不覆盖 latency/profiling。
   - `/home/jetson/Desktop/project/Isaac-GR00T/tests/test_gr00t_inference_cli.py` 只验证 `--help` 不依赖 TensorRT 懒加载。

当前缺失的关键能力：

1. 没有按阶段拆分的 latency 记录。
2. 没有 PyTorch 和 TensorRT 共用的统一日志格式。
3. 没有 warmup / measure / repeat 的标准实验协议。
4. 没有把 Jetson 热状态、频率、功耗与推理时延绑定记录。
5. 没有 open-loop 基线、参数扫描、结果汇总的自动化脚本。
6. 没有用于防回归的 profiling 相关测试。

这意味着：**你现在不能直接依赖仓库现状来做严谨的加速设计，必须先补“测量基础设施”。**

---

## 二、测量目标必须先固定，否则后面的数字没有决策价值

你要测的不是单一“总耗时”，而是两套口径：

### 口径 A：Full Pipeline

包括：

1. 数据集样本读取
2. 视频解码/状态读取
3. `policy.apply_transforms`
4. `model.prepare_input`
5. backbone
6. action head
7. `policy.unapply_transforms`
8. CPU/GPU 同步带来的真实 E2E

用途：回答“真实部署端到端瓶颈在哪里”。

### 口径 B：Model Only

固定同一个已经加载到内存的 `step_data`，或者进一步固定 `normalized_input`，不再重复视频读盘。

用途：回答“模型本身哪里最慢”，避免把视频 IO/解码噪声混进加速结论。

如果不把这两套口径分开，你后面会出现两个常见误判：

1. 以为模型慢，实际上慢在视频解码或 transforms。
2. 以为 TensorRT 收益不大，实际上被外部 IO 和后处理淹没。

---

## 三、最终必须产出的时延字段

最低要求字段如下，所有实验都统一输出：

### 公共元信息

1. 时间戳
2. 机器名
3. JetPack / CUDA / TensorRT / PyTorch 版本
4. 模型路径
5. 推理模式：`pytorch` / `tensorrt`
6. `denoising_steps`
7. `data_config`
8. `video_backend`
9. 是否 warmup
10. 样本 ID / episode ID / step ID

### Full Pipeline 字段

1. `dataset_fetch_ms`
2. `apply_transforms_ms`
3. `prepare_input_ms`
4. `backbone_total_ms`
5. `action_head_total_ms`
6. `postprocess_ms`
7. `e2e_total_ms`

### PyTorch backbone/action head 细分字段

1. `backbone_eagle_total_ms`
2. `backbone_vl_self_attn_ms`
3. `state_encoder_ms`
4. `denoise_total_ms`
5. `denoise_step_{i}_action_encoder_ms`
6. `denoise_step_{i}_dit_ms`
7. `denoise_step_{i}_action_decoder_ms`
8. `denoise_step_{i}_update_ms`

### TensorRT 细分字段

1. `trt_vit_ms`
2. `trt_llm_ms`
3. `trt_vlln_vl_self_attention_ms`
4. `trt_state_encoder_ms`
5. `trt_denoise_total_ms`
6. `trt_step_{i}_action_encoder_ms`
7. `trt_step_{i}_dit_ms`
8. `trt_step_{i}_action_decoder_ms`
9. `trt_step_{i}_update_ms`

### 资源字段

1. `gpu_peak_mem_mb`
2. `cpu_rss_mb`
3. `tegrastats_temp`
4. `tegrastats_power`
5. `tegrastats_freq`

### 汇总字段

1. p50
2. p95
3. p99
4. mean
5. std
6. coefficient of variation

---

## 四、统一实验纪律

在你开始写任何 profiling 代码前，先固定下面这些规则：

1. 固定电源模式和频率。
   - 建议运行 `sudo nvpmodel -m 0`
   - 建议运行 `sudo jetson_clocks`

2. 固定环境。
   - 同一 conda 环境
   - 关闭无关后台进程
   - 测量时不要同时下载模型或编译 engine

3. 固定样本。
   - 先从 `demo_data/robot_sim.PickNPlace` 选 1 个固定 step 做基线
   - 再扩展到 10 到 20 个固定 step 做稳定性统计

4. 固定实验协议。
   - warmup 20 次
   - 正式测量 50 次
   - 每种配置重复 3 轮

5. GPU 计时和 CPU 计时要分开。
   - CPU 段用 `time.perf_counter_ns`
   - GPU 段用 `torch.cuda.Event(enable_timing=True)`

6. 不允许只看单次结果。
   - 最终报告必须至少给出 p50/p95/p99

7. 先做 open-loop，再谈 closed-loop。
   - 你现在的第一目标是建立可靠基线，不是先追控制回路最优。

---

## 五、实施顺序

下面是推荐的闭环顺序。顺序不能乱。

### Task 1：冻结测量口径和实验矩阵

**Files:**
- Modify: `/home/jetson/Desktop/project/dox/plan.md`
- Create: `/home/jetson/Desktop/project/dox/latency_measurement_spec.md`

**目标：**
先把“测什么、不测什么、怎么比较”写死，避免后续改来改去导致数字不可比。

**步骤：**
1. 定义两套口径：`full_pipeline` 和 `model_only`。
2. 定义默认基线配置：
   - `inference_mode=pytorch`
   - `denoising_steps=4`
   - 固定 `dataset_path`
   - 固定 `data_config`
   - 固定 `video_backend`
3. 定义日志 schema。
4. 定义 warmup / measure / repeat 次数。
5. 定义通过标准和回滚标准。

**通过标准：**
团队内任何人只看文档，不看代码，也知道如何重复你的实验。

---

### Task 2：先补统一 profiling 基础设施

**Files:**
- Create: `/home/jetson/Desktop/project/Isaac-GR00T/gr00t/utils/profiling.py`
- Create: `/home/jetson/Desktop/project/Isaac-GR00T/tests/test_profiling_utils.py`

**目标：**
先有统一的计时器和日志结构，再碰业务逻辑。

**建议实现：**
1. 提供 `StageTimer`，支持 CPU 段和 CUDA 段。
2. 提供 `ProfilingSession`，保存单次推理的所有 stage 结果。
3. 提供 `to_dict()`，输出扁平化 JSON 记录。
4. 提供 `aggregate_latency_records()`，计算 p50/p95/p99/mean/std。

**验证：**
1. `pytest tests/test_profiling_utils.py -v`
2. 验证 CPU 计时字段存在。
3. 验证 CUDA 不可用时能自动降级，不崩。

---

### Task 3：在 policy 层建立端到端骨架计时

**Files:**
- Modify: `/home/jetson/Desktop/project/Isaac-GR00T/gr00t/model/policy.py`
- Modify: `/home/jetson/Desktop/project/Isaac-GR00T/gr00t/model/gr00t_n1.py`
- Create: `/home/jetson/Desktop/project/Isaac-GR00T/tests/test_policy_latency_schema.py`

**目标：**
先拿到稳定的顶层分段：
`apply_transforms -> prepare_input -> model.get_action -> unapply_transforms -> e2e`

**步骤：**
1. 给 `Gr00tPolicy.get_action()` 增加可选 profiling 参数。
2. 给 `_get_action_from_normalized_input()` 增加 profiling 透传。
3. 给 `GR00T_N1_5.get_action()` 增加 profiling 透传。
4. 保持默认行为不变，未开启 profiling 时零侵入。

**注意：**
不要一开始就把所有细节塞进 policy；policy 只负责总骨架。

**验证：**
1. 运行现有 CLI，确认默认输出不变。
2. 新增测试验证 profiling 开启后返回结构化字段。

---

### Task 4：在 PyTorch 路径上做细粒度插桩

**Files:**
- Modify: `/home/jetson/Desktop/project/Isaac-GR00T/gr00t/model/backbone/eagle_backbone.py`
- Modify: `/home/jetson/Desktop/project/Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py`
- Create: `/home/jetson/Desktop/project/Isaac-GR00T/tests/test_pytorch_latency_breakdown.py`

**目标：**
把当前 PyTorch 推理路径拆到足够支撑优化决策的粒度。

**必须测到的 PyTorch 阶段：**
1. backbone 总耗时
2. `process_backbone_output`
3. `state_encoder`
4. denoising loop 总耗时
5. 每个 denoising step 的：
   - `action_encoder`
   - `DiT`
   - `action_decoder`
   - `update`

**特别要求：**
1. 每个 denoising step 都要编号，后续才能判断是否前几步/后几步耗时分布异常。
2. 不要只统计 loop 总时长；DreamZero 路线后续需要你比较 step 级收益。

**验证：**
1. `python deployment_scripts/gr00t_inference.py --inference-mode=pytorch`
2. 输出日志中必须能看到 step 级字段。

---

### Task 5：在 TensorRT 路径上做对称插桩

**Files:**
- Modify: `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/trt_model_forward.py`
- Modify: `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/gr00t_inference.py`
- Create: `/home/jetson/Desktop/project/Isaac-GR00T/tests/test_tensorrt_latency_schema.py`

**目标：**
让 TensorRT 结果能和 PyTorch 结果按同一个 schema 对齐比较。

**必须测到的 TensorRT 阶段：**
1. `vit_engine`
2. `llm_engine`
3. `vlln_vl_self_attention_engine`
4. `state_encoder_engine`
5. 每个 denoising step 的：
   - `action_encoder_engine`
   - `DiT_engine`
   - `action_decoder_engine`
   - `update`

**特别要求：**
1. 不要把 TRT engine 内部算子级 profiling 和 pipeline 级 profiling 混为一谈。
2. 记录 PyTorch <-> TensorRT 数据搬运开销，因为文档已经说明 pipeline latency 会比纯 `trtexec` benchmark 更长。

**验证：**
1. `python deployment_scripts/gr00t_inference.py --inference-mode=tensorrt ...`
2. 与 PyTorch 输出同一套主字段。

---

### Task 6：写独立 benchmark runner，不要把实验逻辑堆进现有 demo CLI

**Files:**
- Create: `/home/jetson/Desktop/project/Isaac-GR00T/scripts/benchmark_gr00t_latency.py`
- Create: `/home/jetson/Desktop/project/Isaac-GR00T/tests/test_benchmark_gr00t_latency_cli.py`

**目标：**
把“跑实验”和“单次 demo 推理”彻底分离。

**这个脚本必须支持：**
1. `--inference-mode pytorch|tensorrt`
2. `--measure-scope full_pipeline|model_only`
3. `--warmup-runs`
4. `--measure-runs`
5. `--repeat`
6. `--denoising-steps`
7. `--output-jsonl`
8. `--output-csv`
9. `--sample-index`
10. `--sample-count`

**输出要求：**
1. 每次推理一条 JSONL
2. 每个实验配置一个 summary CSV
3. 控制台打印 p50/p95/p99

**验证：**
1. 单样本跑通
2. 多次重复跑通
3. 无 profiling 时和有 profiling 时功能一致

---

### Task 7：并行采集 Jetson 热状态和资源状态

**Files:**
- Create: `/home/jetson/Desktop/project/Isaac-GR00T/scripts/capture_tegrastats.py`
- Create: `/home/jetson/Desktop/project/Isaac-GR00T/tests/test_tegrastats_parser.py`

**目标：**
避免把热降频、功耗波动误判成模型优化收益或回退。

**步骤：**
1. 启动 benchmark 前启动 `tegrastats` 采集。
2. 按时间戳把 `tegrastats` 和推理记录对齐。
3. 输出：
   - 温度
   - 功耗
   - GPU/EMC 频率
   - RAM 使用

**验证：**
1. benchmark 结束后能生成对应资源日志。
2. 能从日志里看出是否发生热降频。

---

### Task 8：先跑基线，再做单变量扫描

**Files:**
- Create: `/home/jetson/Desktop/project/dox/latency_baseline_results.md`
- Create: `/home/jetson/Desktop/project/dox/latency_sweep_results.md`

**目标：**
先得到“可信基线”，再做 DreamZero 导向的优化优先级排序。

**实验顺序必须是：**
1. PyTorch baseline，`denoising_steps=4`
2. TensorRT baseline，`denoising_steps=4`
3. PyTorch `denoising_steps: 4 -> 3 -> 2 -> 1`
4. TensorRT `denoising_steps: 4 -> 3 -> 2 -> 1`
5. 如果 `torch.compile` 在你当前 Jetson 软件栈上稳定，再单独做 compile 分支
6. 如果前面确认 action head 占主导，再继续做 DreamZero 风格策略

**每一步只能改一个变量。**

**基线输出必须回答：**
1. E2E 最大瓶颈在 IO、backbone、还是 denoising loop
2. TensorRT 到底压缩了哪些阶段
3. 随 denoising step 减少，收益是否接近线性
4. p95/p99 是否异常放大

---

### Task 9：把“数字”翻译成“加速路线”

**Files:**
- Create: `/home/jetson/Desktop/project/dox/latency_analysis_and_next_actions.md`

**目标：**
把测量结果转化为后续工程决策，而不是停留在报表。

**决策规则建议：**
1. 如果 `dataset_fetch_ms + apply_transforms_ms` 占比超过 30%，优先优化视频解码、缓存和 transforms。
2. 如果 `backbone_total_ms` 占比超过 50%，优先考虑 backbone 路径的 compile / precision / TRT 化。
3. 如果 `denoise_total_ms` 占比最高，优先做：
   - TensorRT action head
   - denoising step 扫描
   - DreamZero 风格 caching / async overlap
4. 如果 p95 和 p50 差距过大，先查热状态、同步点、CPU 抖动，不要急着改模型。
5. 如果 TensorRT 只让 `DiT` 变快，但 E2E 收益很小，说明外部环节已经成为新瓶颈。

**这一步的输出不是“想法”，而是明确优先级列表。**

---

## 六、你后面真正应该优先看的插桩点

下面这些位置是本仓库最关键的插桩点，优先级从高到低：

1. `Gr00tPolicy.get_action`
   - 作用：拿到顶层 E2E 骨架

2. `GR00T_N1_5.get_action`
   - 作用：拆开 `prepare_input / backbone / action_head`

3. `EagleBackbone.forward` 或 `forward_eagle`
   - 作用：拆 backbone 主耗时

4. `FlowmatchingActionHead.process_backbone_output`
   - 作用：单独看 action head 前置开销

5. `FlowmatchingActionHead.get_action`
   - 作用：逐步拆 denoising loop

6. `deployment_scripts/trt_model_forward.py`
   - 作用：得到 TRT 模块级数据，并与 PyTorch 对齐

7. `deployment_scripts/gr00t_inference.py`
   - 作用：保留 demo 用法，但不应承载大规模 benchmark 逻辑

---

## 七、风险和避免方式

### 风险 1：CUDA 异步导致计时失真

避免方式：

1. GPU 段用 CUDA Event
2. 在读取结果前统一 `torch.cuda.synchronize()`
3. 不要用裸 `time.time()` 统计 GPU 核函数

### 风险 2：视频解码噪声掩盖模型瓶颈

避免方式：

1. 分开 `full_pipeline` 和 `model_only`
2. 固定样本
3. 预加载样本后重复测

### 风险 3：热降频污染结论

避免方式：

1. 记录 `tegrastats`
2. 长时压测至少 20 分钟
3. 单看单次最小值没有意义

### 风险 4：插桩本身引入太大开销

避免方式：

1. profiling 默认关闭
2. profiling 开启时只做轻量日志收集
3. 分析时比较“带 profiling”和“不带 profiling”的偏差

### 风险 5：把官方文档 benchmark 当成本机真值

避免方式：

1. 官方表格只作为参考上限
2. 所有决策以本机 Orin Nano Super 实测为准

---

## 八、完成标准

这件事完成时，必须同时满足下面 8 条：

1. 你可以一条命令跑 PyTorch 分阶段测时。
2. 你可以一条命令跑 TensorRT 分阶段测时。
3. 你能得到 JSONL 原始日志。
4. 你能得到 CSV 汇总表。
5. 你能看到 p50/p95/p99。
6. 你能知道每个 denoising step 的耗时分布。
7. 你能把时延和 Jetson 热状态对上。
8. 你能明确给出下一步加速优先级，而不是凭感觉改。

---

## 九、推荐的第一批实际执行命令

当你开始实现这个计划时，建议按这个顺序执行：

1. 环境稳态
   - `sudo nvpmodel -m 0`
   - `sudo jetson_clocks`

2. 先确认当前 PyTorch 路径可跑
   - `python deployment_scripts/gr00t_inference.py --inference-mode=pytorch`

3. 再确认当前 TensorRT 路径可跑
   - `python deployment_scripts/gr00t_inference.py --inference-mode=tensorrt --trt-engine-path gr00t_engine`

4. 实现 profiling 基础设施后，先跑单样本 model-only

5. 再跑 full-pipeline

6. 再做 `denoising_steps` 扫描

7. 最后再进入 DreamZero 路线的 async overlap / caching / smoothing 设计

---

## 十、最后的判断

基于当前仓库现状，你现在最应该做的不是立刻讨论 DreamZero 的高级优化细节，而是：

1. 先把 **本机 Orin Nano Super 的真实分阶段基线** 测出来。
2. 先确认 **瓶颈究竟在 IO、backbone、还是 denoising loop**。
3. 先把 **PyTorch 和 TensorRT 的收益拆清楚**。
4. 再决定是先做 `denoising_steps`、TRT、compile，还是 DreamZero 风格改造。

如果跳过这一步，后面的“推理加速设计”很容易变成没有基线、没有证据、没有闭环的试错。

---

## 2026-03-11 本轮实现前分析补充

1. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant` 当前不存在。
   - 这意味着本轮如果按你的要求在该目录下实现，本质上是先创建新目录，再放入独立脚本。

2. `/home/jetson/Desktop/project/Isaac-GR00T` 当前工作树不是干净的。
   - 已存在未提交修改：
     - `deployment_scripts/export_onnx.py`
     - `deployment_scripts/gr00t_inference.py`
     - `gr00t/model/backbone/eagle2_hg_model/modeling_eagle2_5_vl.py`
     - `gr00t/model/backbone/eagle_backbone.py`
     - `pyproject.toml`
     - `scripts/inference_service.py`
   - 已存在未跟踪文件：
     - `tests/test_eagle_backbone_logits.py`
     - `tests/test_gr00t_inference_cli.py`

3. 基于上面的状态，本轮实现策略应优先选择：
   - 只新增 `deployment_scripts/ant` 下的独立本地推理/测时脚本
   - 只追加更新 `dox/plan.md`
   - 尽量不修改已有推理主路径源码

4. 只有当“独立脚本方案无法获得需要的阶段级 timing 数据”时，才考虑修改仓库其他文件。
   - 如果走到这一步，需要先向用户说明原因：
     - 独立脚本无法无侵入拿到 `FlowmatchingActionHead.get_action()` 内部每个 denoising step 的阶段耗时
     - 如果要拿 step 级细分结果，通常需要在 `policy.py`、`gr00t_n1.py`、`flow_matching_action_head.py` 或 `trt_model_forward.py` 中加 profiling hook

5. 用户已明确选择优先路线：
   - 先做“不修改仓库现有源码”的独立本地推理脚本
   - 目标优先级是：
     - 本地推理可跑通
     - 能拿到 `full_pipeline`
     - 能拿到 `model_only`
     - 能拿到 `backbone_total_ms`
     - 能拿到 `action_head_total_ms`
   - 暂不强求 step 级 denoising profiling

---

## 2026-03-11 独立脚本实现方案比较

### 方案 A：最小包装脚本

思路：

1. 在 `deployment_scripts/ant` 下新建一个独立脚本。
2. 直接复用 `Gr00tPolicy`、`LeRobotSingleDataset` 和现有 TensorRT setup 入口。
3. 只测：
   - dataset fetch
   - `policy.get_action`
   - 整体 E2E

优点：

1. 改动最小。
2. 风险最低。
3. 很容易快速跑通。

缺点：

1. 拿不到 `backbone_total_ms` 和 `action_head_total_ms`。
2. 对后续加速设计帮助不够。

结论：

1. 不推荐。
2. 原因是它虽然“可跑”，但还不够支撑瓶颈判断。

### 方案 B：独立脚本 + 运行时 monkey patch 分段计时

思路：

1. 在 `deployment_scripts/ant` 下创建独立脚本与本地工具模块。
2. 不修改仓库已有源码文件。
3. 在脚本内部通过 monkey patch 包装以下运行时对象：
   - `policy.apply_transforms`
   - `policy._get_unnormalized_action`
   - `policy.model.backbone.forward`
   - `policy.model.action_head.get_action`
4. 由包装器记录：
   - `dataset_fetch_ms`
   - `apply_transforms_ms`
   - `backbone_total_ms`
   - `action_head_total_ms`
   - `postprocess_ms`
   - `e2e_total_ms`
5. 同时支持：
   - `full_pipeline`
   - `model_only`
   - `pytorch`
   - `tensorrt`
   - warmup / measure / repeat
   - JSONL 输出
   - 控制台 summary 输出

优点：

1. 满足你当前选的“先不改仓库现有源码”约束。
2. 能拿到当前阶段最有价值的分段数据。
3. 逻辑上和后面真正加 profiling hook 的方向一致，不会推翻前期工作。
4. 可行性高。

缺点：

1. 计时粒度只能到 `backbone` 和 `action_head` 总层级。
2. 拿不到每个 denoising step 的内部细分。
3. monkey patch 需要仔细写，避免破坏原有调用方式。

结论：

1. 推荐作为本轮实现方案。
2. 这是“约束内信息量最大”的方案。

### 方案 C：脚本内重写完整推理流程

思路：

1. 不走 `policy.get_action()`。
2. 在新脚本里自己拼装：
   - dataset
   - transforms
   - `prepare_input`
   - backbone
   - action head
   - unapply

优点：

1. 看起来可以更灵活。

缺点：

1. 会复制仓库现有推理路径。
2. 很容易和主逻辑漂移。
3. 一旦上游行为改动，脚本很容易失真。
4. 调试成本高。

结论：

1. 不推荐。
2. 这是高维护成本方案。

---

## 2026-03-11 推荐设计草案

本轮推荐采用 **方案 B：独立脚本 + 运行时 monkey patch 分段计时**。

预期新增文件：

1. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/local_inference_profiler.py`
2. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/profile_utils.py`
3. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/__init__.py`

职责划分：

1. `profile_utils.py`
   - 提供轻量计时器
   - 提供单次记录结构
   - 提供 summary 聚合函数
   - 提供 JSONL 写出

2. `local_inference_profiler.py`
   - 解析 CLI 参数
   - 构造 policy / dataset
   - 按需启用 TensorRT
   - 在运行时包装关键函数
   - 执行 warmup / measure / repeat
   - 打印 summary
   - 输出 JSONL

3. `__init__.py`
   - 保持目录可导入

本轮脚本的保证范围：

1. 可以跑本地 PyTorch 推理。
2. 如果本机已有可用 engine，也可以跑 TensorRT 推理。
3. 可以输出 `full_pipeline` 和 `model_only` 两种口径。
4. 可以输出：
   - `dataset_fetch_ms`
   - `apply_transforms_ms`
   - `backbone_total_ms`
   - `action_head_total_ms`
   - `postprocess_ms`
   - `e2e_total_ms`
5. 可以输出 p50 / p95 / p99 / mean。

本轮明确不做的事情：

1. 不修改 `policy.py`。
2. 不修改 `gr00t_n1.py`。
3. 不修改 `flow_matching_action_head.py`。
4. 不做每个 denoising step 的细分。
5. 不做 `tegrastats` 采集。
6. 不做 closed-loop。

后续升级路径：

1. 如果这版结果表明 `action_head_total_ms` 是主瓶颈，再申请改仓库主路径，补 step 级 hook。
2. 如果 `dataset_fetch_ms` 或 `apply_transforms_ms` 占比异常，再继续拆外部 IO 路径。

---

## 2026-03-11 用户改用方案 C 后的需求收敛

用户最新要求是改用 **方案 C：脚本内重写完整推理流程**，并且目标粒度调整为下面三大类与其子阶段：

1. transforms
   - `state transform`
   - `action transform`
   - `image preprocess`
   - `text tokenization`

2. Eagle-2.5 backbone
   - `vision encoder`
   - `llm`

3. action head
   - `features process`
   - `state encoder`
   - `action encoder`
   - `dit block`
   - `action decoder`

同时用户要求确认：当前仓库是否已经包含这种测试工具。

结论：**当前仓库没有现成工具提供上述粒度的本地推理时延测试。**

证据：

1. transforms 侧：
   - `GR00TTransform` 中已经存在可自然分段的函数边界：
     - `_prepare_language`
     - `_prepare_state`
     - `_prepare_action`
     - `_apply_vlm_processing`
   - 但仓库没有任何 profiling 包装或 benchmark 脚本把它们系统性计时。

2. backbone 侧：
   - `EagleBackbone.forward_eagle()` 当前把 Eagle 模型整体跑完后直接取 `hidden_states`。
   - 现有 `forward_eagle()` 没有把 `vision encoder` 和 `llm` 分开计时的工具。
   - 文档里只有 benchmark 表，不是本地自动化测试工具。

3. action head 侧：
   - `FlowmatchingActionHead.get_action()` 中存在明确阶段边界：
     - `process_backbone_output`
     - `state_encoder`
     - `action_encoder`
     - `model`（DiT）
     - `action_decoder`
   - 但仓库没有脚本把这些阶段在真实本地推理里逐项测出来。

4. 全仓搜索结果：
   - 没有发现针对这些阶段的 `torch.profiler` / `cudaEvent` / `perf_counter` 级 profiling 流水线。
   - 现有主要是文档 benchmark，以及个别脚本里的粗粒度总耗时打印。

---

## 2026-03-11 方案 C 的可行性判断

方案 C 现在是可行的，原因是你要求的三大类延迟都能映射到稳定的函数边界：

### 1. transforms 粒度映射

可以测成下面四段：

1. `state transform`
   - 对应 `StateActionToTensor` / `StateActionSinCosTransform` / `StateActionTransform` / `ConcatTransform` 里和 `state` 相关的调用

2. `action transform`
   - 对应 `StateActionToTensor` / `StateActionTransform` / `ConcatTransform` 里和 `action` 相关的调用

3. `image preprocess`
   - 对应视频变换链和 `GR00TTransform._prepare_video`
   - 外加 `_apply_vlm_processing` 里图像转 `PIL`、`process_vision_info` 的图像部分

4. `text tokenization`
   - 对应 `GR00TTransform._prepare_language`
   - `apply_chat_template`
   - `eagle_processor(...)` 的 tokenizer 路径

注意：

1. transforms 当前是“组合式调用”，不是天然按这四类输出。
2. 方案 C 下应在新脚本里显式拆开这些步骤，而不是直接整段调用 `policy.apply_transforms()`。

### 2. Eagle-2.5 backbone 粒度映射

目标拆成两段：

1. `vision encoder`
   - 对应 Eagle 模型视觉塔和后续视觉 projector/mlp 路径

2. `llm`
   - 对应语言模型主干前向

注意：

1. `EagleBackbone.forward_eagle()` 当前是整体调用 `self.eagle_model(...)`。
2. 如果坚持“不改仓库源码”，脚本里需要优先尝试读取内部模块并自行分段调用。
3. 如果运行时发现 Eagle 内部结构与预期不一致，可能需要退回“近似边界”：
   - 例如把 `vision tower + mlp1` 记为 `vision encoder`
   - 把 `language_model` 记为 `llm`

### 3. action head 粒度映射

这一部分最清晰，可以直接在脚本里按 `FlowmatchingActionHead.get_action()` 的逻辑重写并分段计时：

1. `features process`
   - 对应 `process_backbone_output`

2. `state encoder`
   - 对应 `state_encoder`

3. `action encoder`
   - 对应 denoising loop 中的 `action_encoder`

4. `dit block`
   - 对应 denoising loop 中的 `self.model(...)`

5. `action decoder`
   - 对应 denoising loop 中的 `action_decoder`

因此：

1. action head 的目标粒度在方案 C 下可行性最高。
2. 这部分最适合作为本轮主输出之一。

---

## 2026-03-11 方案 C 的新推荐设计

推荐保持“只新增 `deployment_scripts/ant` 下文件，不修改其它源码文件”的前提下，实现一套**本地重组式 profiler**。

建议新增文件：

1. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/profile_utils.py`
2. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/local_inference_breakdown.py`
3. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/__init__.py`
4. `/home/jetson/Desktop/project/Isaac-GR00T/tests/test_ant_profile_utils.py`

核心思路：

1. 脚本自行加载 dataset、data config、policy。
2. 不直接调用 `policy.get_action()` 做黑盒统计。
3. 而是在脚本里按现有仓库推理逻辑重组为三段：
   - transforms
   - backbone
   - action head
4. 每段内再按用户要求拆分子阶段并计时。

输出字段至少包含：

1. `transform_state_ms`
2. `transform_action_ms`
3. `transform_image_preprocess_ms`
4. `transform_text_tokenization_ms`
5. `backbone_vision_encoder_ms`
6. `backbone_llm_ms`
7. `action_head_features_process_ms`
8. `action_head_state_encoder_ms`
9. `action_head_action_encoder_ms`
10. `action_head_dit_block_ms`
11. `action_head_action_decoder_ms`
12. `backbone_total_ms`
13. `action_head_total_ms`
14. `transform_total_ms`
15. `e2e_total_ms`

这版设计的边界：

1. 本轮优先做 PyTorch 本地推理路径。
2. 先保证单样本和重复测量可跑。
3. 先保证 summary 和 JSONL 落盘。
4. 暂不把 TensorRT 也做到同样粒度。

原因：

1. 你现在新增的粒度已经明显深入到了模型内部。
2. 先把 PyTorch 路径做准，比同时追 PyTorch+TensorRT 更稳。
3. 等这版稳定后，再扩展 TensorRT 才不会把问题源头混在一起。

---

## 2026-03-11 关于 `e2e_total_ms` 是否会被插桩污染的补充结论

会。

原因很直接：

1. 如果在推理过程中加入大量阶段计时、字段组装、日志写出、控制台打印，那么这些观测动作本身就会增加总耗时。
2. 你现在要求的粒度已经深入到 transforms / backbone / action head 子阶段，插桩密度较高，`e2e_total_ms` 很容易高于“干净路径”的真实端到端时延。
3. 因此，**带细分插桩的 `e2e_total_ms` 只能用于分解占比分析，不应被当成最终真实 E2E 基线。**

结论：

1. 需要保留两套脚本。
2. 两套脚本的职责必须严格分开。

### 脚本 A：细分 profiler

用途：

1. 分析瓶颈结构。
2. 输出三大类及其子阶段耗时。
3. 允许有少量观测开销。

特点：

1. 会记录大量中间阶段。
2. `e2e_total_ms` 是“带观测开销的 e2e”。
3. 适合回答“哪一段最慢”。

### 脚本 B：纯净 e2e benchmark

用途：

1. 获取尽可能接近真实部署的端到端总时延。
2. 作为优化前后最终对比基线。

特点：

1. 不做细粒度分段。
2. 不做中途打印。
3. 不在热路径里组装复杂日志。
4. 只测：
   - dataset fetch（可选）
   - 整体推理调用
   - 整体后处理
   - 最终 `e2e_total_ms`

### 本轮实现新增调整

除了前面方案 C 的细分脚本外，再新增一个独立脚本：

1. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/local_inference_breakdown.py`
   - 负责细粒度时延拆解

2. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/local_inference_e2e.py`
   - 只负责纯净 e2e benchmark

3. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/profile_utils.py`
   - 公共计时与汇总工具

### 两套结果的使用规则

1. 判断瓶颈时，使用 `local_inference_breakdown.py`。
2. 判断最终优化收益时，使用 `local_inference_e2e.py`。
3. 不允许拿 breakdown 脚本里的 `e2e_total_ms` 直接当最终宣传数字。
4. 最终报告里必须同时给出：
   - `breakdown_e2e_total_ms`
   - `clean_e2e_total_ms`

### 实现原则

1. `local_inference_breakdown.py` 允许中文注释和结构化 JSONL 输出。
2. `local_inference_e2e.py` 要尽量保持热路径最短。
3. `local_inference_e2e.py` 默认不输出中间阶段。
4. 两个脚本使用同一套模型加载参数和数据源，避免口径漂移。

---

## 2026-03-11 本轮实际实现与验证结果

本轮已新增文件：

1. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/__init__.py`
2. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/profile_utils.py`
3. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/local_inference_breakdown.py`
4. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/local_inference_e2e.py`
5. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/test_profile_utils.py`
6. `/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/test_ant_cli.py`

### 已实现能力

1. `local_inference_breakdown.py`
   - 采用方案 C，脚本内重组推理链路
   - 目标输出：
     - `transform_state_ms`
     - `transform_action_ms`
     - `transform_image_preprocess_ms`
     - `transform_text_tokenization_ms`
     - `backbone_vision_encoder_ms`
     - `backbone_llm_ms`
     - `action_head_features_process_ms`
     - `action_head_state_encoder_ms`
     - `action_head_action_encoder_ms`
     - `action_head_dit_block_ms`
     - `action_head_action_decoder_ms`
     - `transform_total_ms`
     - `backbone_total_ms`
     - `action_head_total_ms`
     - `postprocess_ms`
     - `e2e_total_ms`

2. `local_inference_e2e.py`
   - 只保留纯净 E2E 基准路径
   - 不做中途细分输出
   - 用于和 breakdown 脚本分离对照

3. `profile_utils.py`
   - `LatencyRecord`
   - `compute_percentile`
   - `summarize_latency_records`
   - `write_jsonl`
   - `measure_wall_time_ms`
   - `measure_cuda_time_ms`

### 设计上的关键处理

1. 两个脚本都做了惰性导入。
   - 这样 `--help` 不依赖 `numpy` / `torch` / `gr00t` 是否已装在当前默认解释器里。

2. `breakdown` 和 `e2e` 分离。
   - 避免细粒度插桩污染最终纯净总时延。

3. 没有修改仓库现有主路径源码文件。
   - 本轮只新增了 `deployment_scripts/ant` 下的脚本和测试。
   - 外加按用户要求继续追加更新 `dox/plan.md`。

### 已完成验证

1. 标准库单元测试：
   - 命令：
     - `python -m unittest /home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/test_profile_utils.py /home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/test_ant_cli.py`
   - 结果：
     - `Ran 5 tests in 0.223s`
     - `OK`

2. 语法编译检查：
   - 命令：
     - `python -m py_compile /home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/profile_utils.py /home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/local_inference_breakdown.py /home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/local_inference_e2e.py /home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/test_profile_utils.py /home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/test_ant_cli.py`
   - 结果：
     - 通过，无报错

3. `gr00t-orin` 环境依赖检查：
   - 命令：
     - `/home/jetson/Desktop/soft_env/miniconda3/envs/gr00t-orin/bin/python -c "import torch, numpy, gr00t; print('torch=', torch.__version__); print('numpy=', numpy.__version__); print('gr00t=', gr00t.__file__)"`
   - 结果：
     - `torch= 2.8.0`
     - `numpy= 1.26.4`
     - `gr00t= /home/jetson/Desktop/project/Isaac-GR00T/gr00t/__init__.py`

4. 两个脚本在 `gr00t-orin` 环境中的 CLI 检查：
   - 命令：
     - `/home/jetson/Desktop/soft_env/miniconda3/envs/gr00t-orin/bin/python .../local_inference_breakdown.py --help`
     - `/home/jetson/Desktop/soft_env/miniconda3/envs/gr00t-orin/bin/python .../local_inference_e2e.py --help`
   - 结果：
     - 两者均正常输出帮助信息

### 真实运行探测结果

对 `local_inference_e2e.py` 做了 1 次真实运行探测：

1. 命令：
   - `/home/jetson/Desktop/soft_env/miniconda3/envs/gr00t-orin/bin/python /home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/local_inference_e2e.py --warmup-runs 0 --measure-runs 1 --sample-index 0`

2. 结果：
   - 脚本成功进入：
     - processor 初始化
     - 本地模型路径加载
     - checkpoint shard 加载
   - 但在 `model.to(cuda)` 阶段失败

3. 失败根因：
   - `NvMapMemAllocInternalTagged ... error 12`
   - `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED ... CUDACachingAllocator`
   - 结合先前分析，这里仍然指向 **Jetson Orin Nano Super 在加载 GR00T N1.5 3B 到 CUDA 时内存不足**

4. 结论：
   - 本轮新增脚本的结构和导入路径已经打通
   - 当前未能完成真实 GPU 推理，不是因为脚本接口错误，而是因为模型装载阶段的设备内存限制

### 当前状态判断

1. 现在已经具备“脚本级工具基础设施”：
   - breakdown 脚本
   - 纯净 e2e 脚本
   - 公共聚合工具
   - 最小自动化测试

2. 当前仍未获得真实 GPU 时延数字。
   - 阻塞原因是 `GR00T-N1.5-3B` 在本机 CUDA 装载阶段触发内存错误。

3. 下一步如果要继续拿到真实时延数据，优先方向应是：
   - 想办法先让推理跑起来
   - 然后再用本轮新增脚本采集时延

### 2026-03-11 15:37:27 CST `local_inference_breakdown.py` `pixel_values` KeyError 分析与修复

1. 现象：
   - 用户执行：
     - `python /home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/local_inference_breakdown.py`
   - 报错位置：
     - `deployment_scripts/ant/local_inference_breakdown.py:338`
     - `backbone.eagle_model.extract_feature(eagle_input["pixel_values"])`
   - 直接异常：
     - `KeyError: 'pixel_values'`

2. 根因定位：
   - 正常主链路里，`GR00TTransform.apply_batch()` 会走 `gr00t/model/transforms.py` 的 `collate()`，把 Eagle processor 输出包装成 `eagle_input_ids`、`eagle_attention_mask`、`eagle_pixel_values`、`eagle_image_sizes`。
   - 但 `local_inference_breakdown.py` 为了做细粒度 profiling，自行实现了 `_apply_gr00t_transform_with_breakdown()`，它直接把 processor 输出裸写成 `input_ids`、`attention_mask`、`pixel_values`、`image_sizes`。
   - 后续 `_run_backbone_breakdown()` 又按正式 backbone 协议只提取 `eagle_` 前缀字段，因此得到的 `eagle_input` 为空，访问 `eagle_input["pixel_values"]` 时触发 KeyError。
   - 结论：问题不在模型或数据集，而在 breakdown 脚本自定义 transform 输出格式和正式推理链路不一致。

3. 修复：
   - 在 `deployment_scripts/ant/local_inference_breakdown.py` 新增 `_prefix_eagle_batch_keys()`。
   - 让 `_apply_gr00t_transform_with_breakdown()` 返回与正式 `collate()` 一致的 Eagle 键名，即统一改为 `eagle_*` 前缀。

4. 回归验证：
   - 新增单元测试：`deployment_scripts/ant/test_local_inference_breakdown.py`
   - 先验证红灯：修复前测试失败，结果显示返回 batch 中只有裸 `pixel_values/input_ids/...`，没有 `eagle_` 前缀。
   - 修复后执行：
     - `python -m unittest deployment_scripts.ant.test_local_inference_breakdown deployment_scripts.ant.test_profile_utils deployment_scripts.ant.test_ant_cli -v`
   - 结果：
     - 6 个测试全部通过。

5. 真实脚本复测：
   - 执行：
     - `python deployment_scripts/ant/local_inference_breakdown.py --warmup-runs 0 --measure-runs 1`
   - 结果：
     - 不再复现本次 `KeyError: 'pixel_values'` 路径；流程继续推进到模型搬运 CUDA 阶段。
     - 本次实机运行随后在 `model.to(cuda)` 处失败，错误为：
       - `NvMapMemAllocInternalTagged ... error 12`
       - `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED ... CUDACachingAllocator`
   - 判断：
     - 本轮 `pixel_values` 键名问题已经修掉。
     - 当前新的阻塞点是 Jetson 侧 CUDA/NVML 内存分配问题，而不是 breakdown 脚本的 Eagle 输入结构错误。
