# Isaac GR00T VLA 微调全流程说明

本文不是泛泛介绍，而是直接以你当前遇到的问题为例，解释为什么你现在不能直接拿默认权重跑 `new_embodiment + new_interaction_group`，以及如果要把这条链路补齐，端到端需要准备什么、执行什么、会生成什么产物、最终如何部署。

---

## 1. 先说你当前的真实情况

你现在手里有三类东西：

1. 代码仓库  
   `/home/jetson/Desktop/project/Isaac-GR00T`

2. 基座模型权重  
   `/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B`

3. 别人给你的新配置定义  
   `/home/jetson/Desktop/project/ossfs/node_59823209/workspace/gr00t`

其中，第 3 类配置里定义了：

- `new_embodiment`
- `new_interaction_group`
- `new_interaction_group_future`

而你没有拿到的是：

- 这个新机器人对应的 finetuned checkpoint
- finetune 过程中生成的 `experiment_cfg/metadata.json`

这正是你现在启动时报错的根因。

---

## 2. 为什么默认权重不能直接跑 `new_embodiment`

很多人第一次接触 GR00T 都会有同一个误解：

> 代码里已经有 `EmbodimentTag.NEW_EMBODIMENT`，  
> data config 里也已经有 `new_interaction_group`，  
> 那我为什么不能直接用 `GR00T-N1.5-3B` 这个默认权重推理？

原因是：`有配置定义` 不等于 `有可直接部署的模型资产`。

### 2.1 代码层面发生了什么

`Gr00tPolicy` 在加载模型时，不只加载权重，还会去读：

`<model_path>/experiment_cfg/metadata.json`

这个逻辑在：

- `gr00t/model/policy.py`

里面做的事情是：

1. 打开 `experiment_cfg/metadata.json`
2. 按 `embodiment_tag` 取对应的统计信息
3. 用这些统计信息给 transform 设置归一化/反归一化参数

如果你传的是：

- `embodiment_tag = new_embodiment`

那么它就一定会去 `metadata.json` 里找：

- `new_embodiment`

这一项。

### 2.2 你当前的 base checkpoint 里有什么

你当前基座模型路径：

`/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B/experiment_cfg/metadata.json`

这个文件里目前只有：

- `gr1`
- `oxe_droid`
- `agibot_genie1`

没有：

- `new_embodiment`

所以一启动就会报：

```text
ValueError: No metadata found for embodiment tag: new_embodiment
```

### 2.3 更深一层的原因

即使你手工伪造一个 `new_embodiment` 键，也不代表就能正确推理。因为 `metadata.json` 不只是一个名字注册表，它里面还保存了：

- 状态各维度的 `min/max/mean/std/q01/q99`
- 动作各维度的 `min/max/mean/std/q01/q99`
- 视频分辨率信息
- 状态/动作每个字段的 shape 和语义

这些信息决定了：

- 输入怎么归一化
- 输出怎么反归一化
- `state.single_arm` / `action.single_arm` 到底长什么样

如果这些统计不对应你的数据，哪怕权重能勉强跑，输出也基本不可信。

### 2.4 再直白一点

你现在拥有的是：

- 一个通用基座模型
- 一套新的机器人模态定义

你缺失的是：

- 让这个模型学会你这个机器人控制空间的后训练结果
- 以及和这套训练数据对应的统计信息

所以不能直接部署。

---

## 3. `new_embodiment`、`new_interaction_group`、`metadata` 分别是什么

这一节非常关键。很多人把这三者混为一谈。

### 3.1 `embodiment_tag` 是什么

`embodiment_tag` 是“这个数据/模型属于哪类机器人控制头”的标签。

在：

- `gr00t/data/embodiment_tags.py`

里可以看到：

- `gr1`
- `oxe_droid`
- `agibot_genie1`
- `new_embodiment`

其中：

- `new_embodiment` 的意思不是“已经支持你的机器人”
- 它更像是“给新机器人 finetune 预留的一个入口”

也就是说：

- 代码架构支持你去训练它
- 但默认 checkpoint 并没有自动变成你机器人的模型

### 3.2 `data_config` 是什么

`data_config` 决定的是：

- 模型训练/推理时要消费哪些模态
- 每种模态的时间 horizon 是多少
- 使用哪些 transform
- 状态和动作如何拼接

你现在外部配置里的 `NewInteractionGroup` 定义是：

- `video_keys = ["video.ego_view"]`
- `state_keys = ["state.single_arm"]`
- `action_keys = ["action.single_arm"]`
- `language_keys = ["annotation.task_index"]`
- `observation_indices = [0]`
- `action_indices = list(range(16))`

这意味着它要求的原生模态协议是：

- 单相机：`video.ego_view`
- 单臂状态：`state.single_arm`
- 单臂动作：`action.single_arm`
- 任务条件：`annotation.task_index`
- 动作 horizon：16 步

注意：

- 这是单臂配置，不是双臂配置
- 如果你后面想左右臂都共用一个模型，那通常意味着“每次请求喂一个单臂状态，输出一个单臂动作”，而不是把它当作一个双臂联动模型

### 3.3 `meta/modality.json` 是什么

`meta/modality.json` 是数据集侧的“字段解释器”。

它告诉数据加载器：

- `observation.state` 这条大向量里，哪些维度属于 `single_arm`
- `action` 这条大向量里，哪些维度属于 `single_arm`
- 视频文件 `observation.images.ego_view` 要映射成哪个标准视频键
- `task_index` 要映射成哪个 annotation 键

也就是说：

- `data_config` 决定模型希望看到什么键
- `modality.json` 决定你数据集里原始字段怎么被翻译成这些键

### 3.4 `metadata.json` 是什么

`experiment_cfg/metadata.json` 不是你手写的主配置，它是训练时根据数据集自动总结出来的统计文件。

它本质上是一个“训练数据统计快照”，至少包含：

- `embodiment_tag`
- `modalities`
- `statistics`

其中 `statistics` 里会有：

- `state.<key>.max/min/mean/std/q01/q99`
- `action.<key>.max/min/mean/std/q01/q99`

这是部署时必须要用到的，因为推理前后都要做 transform。

### 3.5 三者关系一句话总结

- `data_config` 定义模型接口
- `modality.json` 定义数据如何映射到这个接口
- `metadata.json` 定义这个接口上的统计量和归一化参数

三者必须一致。

---

## 4. 微调到底改了什么

微调不是只改一个 `config` 文件，它至少会改两类核心资产。

### 4.1 改模型参数

`scripts/gr00t_finetune.py` 会加载基座模型，然后基于你的 `data_config` 检查：

- action horizon 是否匹配
- action dim 是否匹配

如果不匹配，它会：

1. 重新创建 action head
2. 尽可能拷贝老权重
3. 对新增或不兼容的维度随机初始化
4. 在你的数据上继续训练

这意味着微调真正发生的是：

- projector / diffusion head 的参数更新
- 可选地，visual / llm 也可以更新
- 新控制空间需要学习的部分会从头学

默认参数里：

- `tune_projector = True`
- `tune_diffusion_model = True`
- `tune_visual = False`
- `tune_llm = False`

这是一种比较合理的起点：先适配动作层，不轻易动大 backbone。

### 4.2 生成新的 `metadata.json`

训练不是只产出权重。`gr00t/experiment/runner.py` 会把训练数据集的 `metadata` 写到：

`<output_dir>/experiment_cfg/metadata.json`

并且每次保存 checkpoint 时，还会复制到：

`<output_dir>/checkpoint-<step>/experiment_cfg/metadata.json`

所以 `metadata.json` 正常情况下不是你手工维护的，而是训练产物。

### 4.3 因此，“微调后有什么区别”

微调前你有：

- 通用基础知识
- 但没有你这个机器人、你这个状态维度、你这个动作分布、你这个语言任务空间的适配

微调后你会得到：

- 针对你机器人控制空间更新过的模型参数
- 对应你数据集统计的 `metadata.json`
- 可直接部署的 checkpoint 目录

这就是“为什么我不能只拿默认权重跑”的完整答案。

---

## 5. 你现在要准备哪些东西

如果你要从 0 开始把 `new_interaction_group` 真正训出来，最少需要准备 6 类东西。

### 5.1 基座模型

你已经有了：

`/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B`

它是初始化起点，不是最终部署模型。

### 5.2 一个可被 Python 导入的 `data_config`

你的主仓库本地 `gr00t/experiment/data_config.py` 里当前没有 `new_interaction_group`。你有两种做法：

1. 同步外部定义到本地仓库  
2. 不改仓库，单独新建一个 Python 模块，然后在训练命令里用 `module:ClassName`

GR00T 本身支持第二种方式。`load_data_config()` 支持：

```text
custom_module.submodule:NewInteractionGroup
```

这点很重要，因为你不一定非要把外部配置硬拷进主仓库。

### 5.3 数据集

训练数据必须是 LeRobot compatible 结构，至少包含：

```text
<DATASET_PATH>/
├─ meta/
│  ├─ info.json
│  ├─ episodes.jsonl
│  ├─ tasks.jsonl
│  └─ modality.json
├─ data/
│  └─ chunk-000/
│     ├─ episode_000000.parquet
│     └─ ...
└─ videos/
   └─ chunk-000/
      └─ observation.images.ego_view/
         ├─ episode_000000.mp4
         └─ ...
```

### 5.4 符合 `new_interaction_group` 的 `modality.json`

因为你现在的模态是：

- `video.ego_view`
- `state.single_arm`
- `action.single_arm`
- `annotation.task_index`

所以最小化的 `meta/modality.json` 可以长这样：

```json
{
  "state": {
    "single_arm": {
      "start": 0,
      "end": 6,
      "original_key": "observation.state"
    }
  },
  "action": {
    "single_arm": {
      "start": 0,
      "end": 6,
      "original_key": "action"
    }
  },
  "video": {
    "ego_view": {
      "original_key": "observation.images.ego_view"
    }
  },
  "annotation": {
    "task_index": {
      "original_key": "task_index"
    }
  }
}
```

这里 `end = 6` 只是示例。你必须根据自己的真实状态维度和动作维度来定。

### 5.5 与 `modality.json` 一致的 parquet 列

每个 episode 的 parquet 至少要有：

- `observation.state`
- `action`
- `timestamp`
- `task_index`
- `episode_index`
- `index`

其中：

- `observation.state` 是一维向量
- `action` 是一维向量
- `task_index` 对应 `meta/tasks.jsonl` 里的任务描述

### 5.6 足够像样的数据量

这个仓库没有一个“放之四海而皆准”的最小样本数，因为结果高度依赖任务复杂度、动作空间和数据质量。

但工程上你至少要明白：

- 几十条演示通常只够验证链路，不足以得到稳定策略
- 几百条到上千条高质量演示才更像真正可用的起点
- 如果动作空间变化很大、场景变化大、任务多，所需数据会明显增加

---

## 6. 以你当前场景为例，应该怎么准备数据

你当前目标是单臂 `new_interaction_group`。最实用的思路是：

### 6.1 先把“控制语义”定死

你要先决定：

- `state.single_arm` 是什么
- `action.single_arm` 是什么

例如：

1. 6 维关节角
2. 5 维关节角 + 1 维 gripper
3. 6 维末端位姿增量
4. 其他自定义控制空间

无论你选哪种，训练集和部署时的输入语义必须完全一致。

这是你当前最需要警惕的点：

你之前的 `aistudio.py` 是把 5 维 `joint_angles` 手动补一个 0 再喂给模型。  
这只是兼容旧接口的业务层 hack，不应直接当成正式训练定义。

严谨做法是：

- 如果模型要学 6 维，就让数据集里的 `state.single_arm` 和 `action.single_arm` 真正稳定地定义成 6 维
- 不要把“线上临时补零”当作训练语义本身

### 6.2 视频键要和配置一致

既然 `new_interaction_group` 用的是：

- `video.ego_view`

那你的数据视频目录和 `modality.json` 映射就要对应到：

- `observation.images.ego_view`

### 6.3 语言键要和配置一致

你现在外部配置使用的是：

- `annotation.task_index`

这意味着训练数据里要能取到 `task_index`，并能通过 `meta/tasks.jsonl` 还原出任务文本。

例如：

`meta/tasks.jsonl`

```json
{"task_index": 0, "task": "pick up the object and place it into the bin"}
{"task_index": 1, "task": "move the block to the left tray"}
```

parquet 里：

```text
task_index = 0
```

---

## 7. 训练前的推荐检查步骤

在你真正跑训练之前，先做这 4 个检查。

### 7.1 检查数据能否被加载

先确认数据目录结构没问题，视频和 parquet 都在。

### 7.2 检查 `modality.json` 是否和 `data_config` 对齐

重点检查：

- `state.single_arm` 是否真的存在
- `action.single_arm` 是否真的存在
- `video.ego_view` 是否真的存在
- `annotation.task_index` 是否真的存在

### 7.3 检查 state/action 维度是否稳定

最怕的不是“数据少”，而是：

- 有的 episode 是 5 维
- 有的 episode 是 6 维
- 有的 action 是绝对量
- 有的是增量

这类数据会直接把训练意义打散。

### 7.4 检查任务标签是否正确

`task_index` 和 `tasks.jsonl` 必须一一对应，不能错位。

---

## 8. 如何实际发起微调

下面给一个最接近你当前情况的训练示例。

假设你创建了一个外部 config 模块：

`custom_data_configs/new_interaction_group_config.py`

里面定义了：

- `NewInteractionGroup`

那么训练命令可以写成：

```bash
python scripts/gr00t_finetune.py \
  --dataset-path /path/to/your_dataset \
  --base-model-path /home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B \
  --output-dir /path/to/output/new_interaction_group_run \
  --data-config custom_data_configs.new_interaction_group_config:NewInteractionGroup \
  --embodiment-tag new_embodiment \
  --num-gpus 1 \
  --batch-size 16 \
  --max-steps 10000 \
  --save-steps 1000 \
  --video-backend torchcodec \
  --tune-projector True \
  --tune-diffusion-model True \
  --tune-visual False \
  --tune-llm False
```

如果你已经把 `new_interaction_group` 同步进本地 `gr00t/experiment/data_config.py`，那也可以直接写：

```bash
python scripts/gr00t_finetune.py \
  --dataset-path /path/to/your_dataset \
  --base-model-path /home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B \
  --output-dir /path/to/output/new_interaction_group_run \
  --data-config new_interaction_group \
  --embodiment-tag new_embodiment \
  --num-gpus 1 \
  --batch-size 16 \
  --max-steps 10000 \
  --save-steps 1000 \
  --video-backend torchcodec
```

### 8.1 这些参数分别是什么意思

- `--dataset-path`  
  训练数据目录，可以传一个或多个

- `--base-model-path`  
  基座模型，作为初始化起点

- `--output-dir`  
  所有 checkpoint、metadata、日志的输出目录

- `--data-config`  
  模型的模态定义和 transform 定义

- `--embodiment-tag new_embodiment`  
  明确告诉训练器：这是一个新机器人适配任务

- `--batch-size` / `--max-steps`  
  训练强度和时长

- `--video-backend`  
  视频解码后端，取决于你的视频编码和环境

- `--tune-*`  
  决定微调哪些子模块

### 8.2 从工程经验上给你的默认建议

初次训练建议先从下面组合开始：

- `tune_projector = True`
- `tune_diffusion_model = True`
- `tune_visual = False`
- `tune_llm = False`

原因：

- 这是适配新控制空间最常见、风险也相对小的起点
- 显存压力相对更可控
- 不容易因为训练数据量不够而把 backbone 带偏

---

## 9. 训练过程中会生成哪些产物

假设输出目录是：

`/path/to/output/new_interaction_group_run`

那么你通常会得到几类产物。

### 9.1 训练目录根部

根目录会包含：

- 训练最终保存的模型权重
- `experiment_cfg/`
- `runs/` 或 wandb 相关日志
- trainer state 等训练状态文件

### 9.2 `experiment_cfg/metadata.json`

这是最关键的产物之一。

它是从训练数据自动总结出来的，至少会记录：

- `new_embodiment`
- 这个 embodiment 对应的 `modalities`
- 这个 embodiment 对应的 `statistics`

之后部署 `Gr00tPolicy` 时，就会读取这里的内容。

### 9.3 `checkpoint-<step>/`

每次到 `save_steps` 时，会生成：

- `checkpoint-1000/`
- `checkpoint-2000/`
- ...

每个 checkpoint 里除了模型权重，还会带上：

- `checkpoint-<step>/experiment_cfg/metadata.json`

这一步由 `CheckpointFormatCallback` 完成，目的就是让 checkpoint 自包含、可直接部署。

### 9.4 可视化和日志

如果 `report_to=tensorboard`，会得到：

- `runs/`

如果 `report_to=wandb`，会在输出目录写：

- wandb 运行信息

这些日志帮助你看 loss、训练是否发散、收敛速度等。

---

## 10. 如何判断训练是不是成功了

不要只看“程序跑完了”。至少做下面 3 层验证。

### 10.1 第一层：能不能正常加载 checkpoint

拿某个 checkpoint 路径，例如：

`/path/to/output/new_interaction_group_run/checkpoint-5000`

看它是否包含：

- 模型权重
- `experiment_cfg/metadata.json`

然后确认用：

- `embodiment_tag = new_embodiment`
- `data_config = new_interaction_group`

能成功构造 `Gr00tPolicy`

### 10.2 第二层：开环评估

可以参考仓库的 `eval_policy.py` 思路，对训练集或验证集样本做开环可视化。

典型调用方式是：

```bash
python scripts/eval_policy.py --plot \
  --embodiment_tag new_embodiment \
  --model_path /path/to/output/new_interaction_group_run/checkpoint-5000 \
  --data_config custom_data_configs.new_interaction_group_config:NewInteractionGroup \
  --dataset_path /path/to/your_dataset \
  --modality_keys single_arm
```

如果你已经把 config 同步进本地，也可以直接用：

```bash
--data_config new_interaction_group
```

### 10.3 第三层：真实部署验证

开环只说明“拟合得像不像”。  
真正有效要看部署到机器人后的闭环表现。

---

## 11. 训练完成后如何部署

部署逻辑要点很简单：

1. 用 finetuned checkpoint，不要再用 base checkpoint
2. `embodiment_tag` 必须传 `new_embodiment`
3. `data_config` 必须和训练时一致
4. 输入模态必须和训练时一致

如果你训练得到的 checkpoint 在：

`/path/to/output/new_interaction_group_run/checkpoint-5000`

那么部署时，模型路径就应该传这个 checkpoint，而不是：

`/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B`

这就是官方部署文档里一直强调“传 finetuned checkpoint”的原因。

---

## 12. 你这次报错到底说明了什么

结合你现在的场景，你可以这样理解这次异常：

### 12.1 不是代码里没有 `new_embodiment`

代码里其实有。

### 12.2 也不是你传参拼错了

你传 `new_embodiment` 本身没有问题。

### 12.3 真正的问题是“资产链不闭环”

你有：

- `new_interaction_group` 配置定义
- `new_embodiment` 标签定义

但你没有：

- 对这个 embodiment 训练出来的 checkpoint
- 对这个 embodiment 自动生成的 `metadata.json`

所以部署链断在模型资产这一层，而不是断在 HTTP 服务层。

---

## 13. 微调后 TRT 会发生什么

这一点和你当前项目也直接相关。

TRT engine 不是微调的直接产物。通常顺序应该是：

1. 先用 PyTorch 跑通 finetuned checkpoint
2. 确认输入输出模态和动作维度都正确
3. 再基于这个 finetuned checkpoint 导出 ONNX / TensorRT
4. 最终部署时加载新的 TRT engine

这意味着：

- 你现在手里的旧 TRT engine  
  `/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_fp16`

未必能直接对应未来新的 finetuned 模型。

因为一旦微调后的 action head、输入签名、内部配置发生变化，原来的 engine 很可能就不再是正确部署资产。

严谨做法是：

- 把 TRT engine 视为“部署优化产物”
- 不要把它当作“训练前就固定不变的资产”

---

## 14. 什么时候可以不微调

这也是一个很重要的边界。

如果你的机器人和任务完全落在 base checkpoint 已支持的 embodiment 范围内，而且：

- 观测字段一致
- 控制空间一致
- 归一化语义一致
- 数据分布也接近

那你可以尝试直接推理。

例如：

- 用 `gr1` 对 `gr1`
- 用 `oxe_droid` 对 `oxe_droid`

但你现在不是这种情况。你现在用的是：

- `new_embodiment`
- `new_interaction_group`

这在语义上已经明确表示“我要适配一个新的机器人/新控制空间”。  
这类情况默认就应该走 finetune，而不是假设 base checkpoint 自动可用。

---

## 15. 一条最实用的落地路线

如果你现在真的要把这件事推进，我建议按这个顺序做。

### 第 1 步：先定义清楚真实控制空间

确认：

- `state.single_arm` 维度
- `action.single_arm` 维度
- 是关节空间还是末端空间
- 是否包含 gripper

### 第 2 步：整理数据集

把录制数据整理成 LeRobot compatible 格式，并补齐：

- `meta/modality.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`

### 第 3 步：让 `data_config` 可被训练脚本导入

两种方式二选一：

- 同步进主仓库 `gr00t/experiment/data_config.py`
- 或者单独写成 `custom_module:ClassName`

### 第 4 步：先用小数据和小步数冒烟

比如：

- 1 个数据集
- `max_steps = 1000`
- `save_steps = 200`

先验证链路是否闭环：

- 数据能加载
- 训练能启动
- checkpoint 能保存
- checkpoint 里有 `experiment_cfg/metadata.json`
- `Gr00tPolicy` 能用 `new_embodiment` 正常加载这个 checkpoint

### 第 5 步：再扩大训练

等冒烟通过后，再去调：

- batch size
- max steps
- 学习率
- 是否要开 `tune_visual`
- 是否要混多个数据集

### 第 6 步：最后再做 TensorRT

先把 PyTorch 部署稳定，再导出 TRT engine。  
不要反过来。

---

## 16. 你现在最容易踩的误区

### 误区 1：以为有 `new_embodiment` 枚举就代表有现成权重

不是。  
这只代表代码支持你去训练和加载这种 tag。

### 误区 2：以为只缺一个 config 文件

不是。  
你缺的是 finetuned checkpoint 和对应 metadata。

### 误区 3：手工复制别人的 `metadata.json` 就能解决

大多数情况下不行。  
统计量必须对应你的真实训练数据，否则 transform 会错。

### 误区 4：线上补零等兼容逻辑可以直接当训练定义

不建议。  
训练语义要明确且稳定，不能依赖接口层临时修补。

### 误区 5：微调完原来的 TRT engine 一定还能用

不一定。  
应该按新的 finetuned 资产重新验证或重新导出。

---

## 17. 最后给你的结论

把你当前的问题压缩成一句话：

> 你现在只有“新机器人的接口定义”，没有“新机器人训练后的模型资产”。

所以：

- 代码能识别 `new_embodiment`
- 配置能定义 `new_interaction_group`
- 但 base checkpoint 不能直接替代 finetuned checkpoint

如果你要真正跑起来，必须补齐这条链：

1. 准备符合 `new_interaction_group` 的 LeRobot 数据集  
2. 用 `new_embodiment` + `new_interaction_group` 发起 finetune  
3. 得到新的 checkpoint 和 `experiment_cfg/metadata.json`  
4. 用这个 finetuned checkpoint 做 PyTorch 推理  
5. 验证无误后再做 TRT 加速部署

到这一步，整个“配置 -> 数据 -> 训练 -> metadata -> checkpoint -> 推理 -> TRT”闭环才算成立。

---

## 18. 你接下来可以直接做什么

如果你现在马上要推进，最现实的动作顺序是：

1. 先确认你希望 `state.single_arm` / `action.single_arm` 的精确定义和维度
2. 整理一版最小可训练数据集
3. 我再帮你补一份专门针对 `new_interaction_group` 的 `modality.json` 模板
4. 然后我可以继续帮你写：
   - 自定义 `data_config` 模块
   - 冒烟训练命令
   - 数据校验脚本
   - 训练后 checkpoint 加载验证脚本

这样你就不是“知道原理”，而是可以真正把第一版新模型训出来。
