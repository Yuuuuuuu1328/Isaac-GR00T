# GR00T 端到端推理过程详解

本文按当前仓库里最真实、最常用的一条默认推理路径来解释一次任务下，多次推理请求的端到端过程。主线不是抽象论文图，而是你现在这套代码实际在走的路径：

- 模型：`GR00T-N1.5-3B`
- policy 包装：`gr00t/model/policy.py`
- 主模型：`gr00t/model/gr00t_n1.py`
- backbone：`gr00t/model/backbone/eagle_backbone.py`
- action head：`gr00t/model/action_head/flow_matching_action_head.py`
- 数据配置：`fourier_gr1_arms_only`
- 视觉输入：单路 `video.ego_view`
- 状态 horizon：`1`
- 动作 horizon：`16`
- 默认动作去噪步数：`4`

文中会先按这条真实路径把输入、输出、每一步张量形状和维度变化写清楚，再在每个关键点补一句“哪些是当前配置固定值，哪些不是固定值”。最后一节会分析推理加速机会。

## 0. 单次请求总流程图

先给出一张“单次 `policy.get_action()` 请求”总流程图。下面这张图只保留主干，不展开每个 helper：

```text
┌─────────────────────────────────────────────────────────────────────┐
│ 外部调用                                                            │
│ policy.get_action(observations)                                     │
└─────────────────────────────────────────────────────────────────────┘
                                |
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ 原始输入                                                            │
│ video.ego_view + state.* + task language                            │
│ 当前真实路径示例:                                                   │
│   video.ego_view = (1,256,256,3)                                    │
│   raw state total = 26 dims                                         │
│   task text = "pick ... place ..."                                  │
└─────────────────────────────────────────────────────────────────────┘
                                |
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ Policy 预处理                                                       │
│ 1. 自动补 batch 维                                                  │
│ 2. numpy 化                                                         │
│ 3. apply_transforms()                                               │
└─────────────────────────────────────────────────────────────────────┘
                                |
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ 变换后模型输入                                                      │
│ state=(1,1,64)                                                      │
│ state_mask=(1,1,64)                                                 │
│ eagle_input_ids=(1,296)                                             │
│ eagle_attention_mask=(1,296)                                        │
│ eagle_pixel_values=(1,3,224,224)                                    │
│ eagle_image_sizes=(1,2)                                             │
│ embodiment_id=(1,)                                                  │
└─────────────────────────────────────────────────────────────────────┘
                                |
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ 主模型                                                              │
│ GR00T_N1_5.get_action()                                             │
│   -> prepare_input()                                                │
│   -> backbone(...)                                                  │
│   -> action_head.get_action(...)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                |
                 ┌──────────────┴──────────────┐
                 v                             v
┌──────────────────────────────┐  ┌──────────────────────────────────┐
│ EagleBackbone                │  │ FlowmatchingActionHead           │
│ 图像+语言 -> 条件特征        │  │ 条件特征+state -> 16步动作       │
│ backbone_features=(B,N,2048) │  │ action_pred=(1,16,32)            │
└──────────────────────────────┘  └──────────────────────────────────┘
                 \                             /
                  \                           /
                   \_________________________/
                                |
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ Policy 后处理                                                       │
│ unapply_transforms({"action": action_pred})                         │
│ -> 拆回真实动作字典                                                 │
└─────────────────────────────────────────────────────────────────────┘
                                |
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ 最终输出                                                            │
│ action.left_arm   = (16,7)                                          │
│ action.right_arm  = (16,7)                                          │
│ action.left_hand  = (16,6)                                          │
│ action.right_hand = (16,6)                                          │
└─────────────────────────────────────────────────────────────────────┘
```

如果只记一句话，这张图表达的是：

- 先把“图像 + 状态 + 语言”变成 Eagle 条件特征
- 再从随机动作轨迹出发做去噪式动作生成
- 最后把统一的 `16x32` 内部动作张量拆回真实动作字典

## 1. 一次任务和多次推理请求是什么关系

先区分两个概念：

- 一次任务：例如一句语言指令，`pick the pear from the counter and place it in the plate`
- 一次推理请求：在某个时刻把当前观测送进 policy，得到一段未来动作

在 GR00T 里，这两者不是一回事。

一次任务通常持续多个控制周期。任务描述可以在一段时间内保持不变，但每次推理请求都会带入“新的当前观测”，重新预测未来动作序列。也就是说，系统不是“任务只推理一次”，而是：

1. 任务描述固定
2. 当前相机图像和当前机器人状态不断变化
3. 每次请求重新跑一遍完整推理链
4. 每次输出未来 `16` 步动作

所以从控制视角看，GR00T 更像是一个“反复被调用的条件动作生成器”。

## 2. 当前真实路径的原始输入是什么

### 2.1 原始输入字段

对于 `fourier_gr1_arms_only`，当前配置关心以下输入模态：

- 视频：`video.ego_view`
- 状态：`state.left_arm`
- 状态：`state.right_arm`
- 状态：`state.left_hand`
- 状态：`state.right_hand`
- 语言：`annotation.human.action.task_description`

训练时数据里还会有动作标签：

- `action.left_arm`
- `action.right_arm`
- `action.left_hand`
- `action.right_hand`

但推理时，模型最终要输出动作，不需要外部再提供目标动作。

### 2.2 一个真实 demo 样本的原始形状

我基于仓库里的 `demo_data/robot_sim.PickNPlace` 实际读取到的单样本原始形状如下：

```text
video.ego_view                          (1, 256, 256, 3)   uint8
state.left_arm                         (1, 7)             float64
state.right_arm                        (1, 7)             float64
state.left_hand                        (1, 6)             float64
state.right_hand                       (1, 6)             float64
action.left_arm                        (16, 7)            float64
action.right_arm                       (16, 7)            float64
action.left_hand                       (16, 6)            float64
action.right_hand                      (16, 6)            float64
annotation.human.action.task_description ['pick the pear from the counter and place it in the plate']
```

这里可以先得到两个很重要的事实：

- 单帧视觉 observation horizon 是 `1`
- 原始状态总维度是 `7 + 7 + 6 + 6 = 26`
- 原始动作每步总维度也是 `26`
- 模型每次不是输出一步动作，而是输出 `16` 步动作

## 3. policy 的输入输出接口长什么样

### 3.1 外部调用接口

外部最常见入口是 `Gr00tPolicy.get_action(observations)`。

调用者传入的 `observations` 在语义上是：

- 当前图像
- 当前机器人状态
- 当前任务语言

返回的是动作字典，最后会被拆回：

- `action.left_arm`
- `action.right_arm`
- `action.left_hand`
- `action.right_hand`

### 3.2 batched 和 unbatched 的区别

`Gr00tPolicy.get_action()` 同时支持 batched 和 unbatched 输入。

如果输入不是 batch，它会先自动加 batch 维。也就是说，单请求在进入 transform 之前，会从：

- 图像：`(T, H, W, C)` 或当前配置下的单键字典

变成：

- 图像：`(B, T, H, W, C)` 风格的数据组织

在当前路径里，实际经过 concat 后视频会进一步整理成：

- `video = (B, T, V, H, W, C)`

其中：

- `B=1`
- `T=1`
- `V=1`

## 4. 从原始 observation 到模型输入：transform 链路

这一段是理解整条推理链最关键的地方。当前数据配置 `fourier_gr1_arms_only` 的 transform 顺序是：

1. `VideoToTensor`
2. `VideoCrop(scale=0.95)`
3. `VideoResize(224, 224)`
4. `VideoColorJitter(...)`
5. `VideoToNumpy`
6. `StateActionToTensor` for state
7. `StateActionSinCosTransform` for state
8. `StateActionToTensor` for action
9. `StateActionTransform(min_max)` for action
10. `ConcatTransform`
11. `GR00TTransform`

其中训练和推理都共用这条总链路，但推理时 transform 会被设成 `eval()`，因此虽然配置里有 `VideoColorJitter`，实际在线推理不应再把图像随机扰动成训练增强模式。

### 4.1 视频路径的形状变化

原始单视角图像：

```text
video.ego_view = (1, 256, 256, 3)
```

这表示：

- `T=1`
- `H=256`
- `W=256`
- `C=3`

`ConcatTransform` 会把多相机视角拼成统一的 `video` 键。当前只有一路 `ego_view`，所以拼完后逻辑形状是：

```text
video = (T=1, V=1, H=256, W=256, C=3)
```

加上 batch 后可理解成：

```text
video = (B=1, T=1, V=1, H=256, W=256, C=3)
```

到 `GR00TTransform._prepare_video()` 时会执行重排：

```text
(T, V, H, W, C) -> (V, T, C, H, W)
```

所以单样本内部图像张量会变成：

```text
(1, 1, 3, 224, 224)
```

注意这里的 `224x224` 是 resize 后的结果。

随后 Eagle processor 会把图像整理成模型真正消费的视觉输入。实际跑出来的 policy 变换结果是：

```text
eagle_pixel_values = (1, 3, 224, 224)
eagle_image_sizes  = (1, 2)
```

这里当前只有一张图，所以 batch 后还是一张视觉输入。

### 4.2 状态路径的维度变化

原始状态维度：

- `left_arm = 7`
- `right_arm = 7`
- `left_hand = 6`
- `right_hand = 6`

总计：

```text
26 dims
```

然后状态先经过 `StateActionToTensor`，只是把 `numpy.ndarray` 变成 `torch.Tensor`。

接着经过 `StateActionSinCosTransform`。这一步对每个状态维做：

```text
x -> [sin(x), cos(x)]
```

所以状态维数翻倍：

```text
26 -> 52
```

随后 `ConcatTransform` 按顺序把四个状态键拼起来，得到：

```text
state = (T=1, D=52)
```

再进入 `GR00TTransform._prepare_state()`，它会把状态 pad 到 `max_state_dim=64`，并生成 mask：

```text
state      = (1, 64)
state_mask = (1, 64)
```

policy 实际 batched 后，验证得到的真实模型输入形状是：

```text
state      = (1, 1, 64)
state_mask = (1, 1, 64)
```

这三维分别是：

- `B=1`
- `state_horizon=1`
- `max_state_dim=64`

其中只有前 `52` 维是真实状态，后 `12` 维是 padding。

### 4.3 语言路径的变化

当前语言键是：

```text
annotation.human.action.task_description
```

在 `GR00TTransform` 中，语言会与图像一起打包成 Eagle conversation。也就是说，语言不是单独走一个“文本编码器 API”，而是被构造成多模态聊天模板：

1. 把图像包装成 `{"type": "image", ...}`
2. 把任务文本包装成 `{"type": "text", ...}`
3. 拼成一轮 user message
4. 调 Eagle processor 的 `apply_chat_template(...)`
5. 再经过 tokenizer

对于当前 demo 样本，真实 policy 变换后的文本相关张量形状是：

```text
eagle_input_ids      = (1, 296)
eagle_attention_mask = (1, 296)
```

这里的 `296` 不是模型永远固定常数，它取决于：

- 任务文本长度
- chat template
- 图像 token 占位形式
- processor 的实现

但在当前样本、当前 processor 下，真实观测到的是 `296`。

### 4.4 动作路径在推理时是什么角色

当前 transform 链里也定义了 action 相关变换，但要注意推理和训练的区别：

- 训练时：数据里有真实动作标签，transform 会把它们归一化并送进 action head 做监督学习
- 推理时：action head 自己从高斯噪声初始化动作轨迹，再逐步去噪生成动作

因此，推理路径最终真正依赖的是：

- state
- image
- language
- embodiment_id

而不是外部提供的目标 action。

## 5. policy 变换完成后，模型真正拿到什么

当前默认路径下，`Gr00tPolicy.apply_transforms()` 之后，模型实际拿到的主要键和值形状是：

```text
state                 (1, 1, 64)      torch.float64
state_mask            (1, 1, 64)      torch.bool
eagle_input_ids       (1, 296)        torch.int64
eagle_attention_mask  (1, 296)        torch.int64
eagle_pixel_values    (1, 3, 224, 224) torch.float32
eagle_image_sizes     (1, 2)          torch.int64
embodiment_id         (1,)            torch.int64
```

这一步非常重要，因为它说明最终模型输入已经不再是“原始相机图像和若干状态键”，而是：

- 一组 Eagle 所需的文本和视觉张量
- 一个统一 pad 过的 state 张量
- 一个 embodiment 索引

## 6. `Gr00tPolicy.get_action()` 到 `GR00T_N1_5.get_action()` 的执行过程

### 6.1 `Gr00tPolicy.get_action()`

整体逻辑可以概括成：

1. 复制输入，避免原地修改
2. 如果不是 batch，自动加 batch 维
3. 把输入值尽量转成 `numpy.ndarray`
4. 调 `apply_transforms()`
5. 调 `_get_action_from_normalized_input()`
6. 再调 `unapply_transforms()` 把模型输出拆回动作字典

### 6.2 `_get_action_from_normalized_input()`

这里会进入：

```python
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    model_pred = self.model.get_action(normalized_input)
```

也就是说当前默认 PyTorch 推理路径是：

- `inference_mode`
- CUDA autocast
- 计算 dtype 以 `bfloat16` 为主

然后取出：

```text
model_pred["action_pred"]
```

它的预期形状是：

```text
(B, action_horizon, action_dim)
```

在当前默认配置下就是：

```text
(1, 16, 32)
```

## 7. `GR00T_N1_5.prepare_input()` 做了什么

进入主模型后，第一步不是直接跑 backbone，而是先把输入拆成两部分：

- `backbone_inputs`
- `action_inputs`

这一步目前本质上还是把同一份 batch 包成两个 `BatchFeature`，方便：

- backbone 读取 Eagle 相关输入
- action head 读取 state / embodiment / action 相关输入

然后模型会把张量搬到目标 device，并对浮点张量 cast 到 action head 当前 dtype。

因此，到了真正前向时，你可以把它理解成：

- 文本和图像张量喂给 Eagle backbone
- 状态和 embodiment id 喂给 action head

## 8. backbone：Eagle 是怎么把图像和语言变成条件特征的

### 8.1 backbone 输入

`EagleBackbone.forward_eagle()` 会把所有以 `eagle_` 开头的键取出来，去掉前缀后送给 Eagle 模型。

也就是类似：

- `eagle_input_ids -> input_ids`
- `eagle_attention_mask -> attention_mask`
- `eagle_pixel_values -> pixel_values`

它还会删除 `image_sizes` 再调用 Eagle。

### 8.2 Eagle 输出

当前代码调用 Eagle 时指定：

- `output_hidden_states=True`
- `return_dict=True`
- `logits_to_keep=1`

然后读取：

```text
eagle_output.hidden_states[self.select_layer]
```

这一步的含义是：

- GR00T 当前并不关心 Eagle 的文本生成 logits
- 它关心的是 Eagle 中间层隐藏状态，把它当作视觉语言条件特征

### 8.3 backbone 输出形状

backbone 输出最后统一成：

```text
backbone_features        (B, N, H_backbone)
backbone_attention_mask  (B, N)
```

当前 checkpoint 里有两个相关配置：

- Eagle 隐藏维度原始是 `2048`
- 当前 checkpoint 的 `backbone_cfg.project_to_dim = None`

因此在这条默认路径下，backbone 特征维保持为：

```text
H_backbone = 2048
```

这里的 `N` 不是固定常数，它是 Eagle 最终保留下来的多模态 token 序列长度，会随着文本长度和视觉 token 组织方式变化。

## 9. action head：为什么输出是 16 步动作

### 9.1 action head 的核心角色

action head 不是简单 MLP 一次性把 state 映射成动作。它更像一个“条件生成器”，根据：

- Eagle 给出的视觉语言条件特征
- 当前机器人状态
- embodiment id
- 当前动作轨迹猜测值
- 当前去噪时间步

逐步更新动作序列。

### 9.2 关键配置

当前 checkpoint 里的关键 action head 配置是：

```text
action_horizon          = 16
action_dim              = 32
input_embedding_dim     = 1536
hidden_size             = 1024
num_target_vision_tokens = 32
num_inference_timesteps = 4
```

这意味着：

- 每次请求输出 `16` 个未来动作 token
- 每个动作 token 在模型内部是 `32` 维
- state / action / future token 会被嵌入到 `1536` 维
- diffusion transformer 内部隐藏维度是 `1024`
- 推理时默认做 `4` 次迭代更新

### 9.3 state 是怎么进 action head 的

action head 会先取：

```text
action_input.state = (B, 1, 64)
```

然后通过 `state_encoder` 把它编码到：

```text
state_features = (B, 1, 1536)
```

这里的 `1` 对应 state horizon。

### 9.4 动作轨迹是怎么初始化的

推理时没有目标动作输入，所以 action head 会自己初始化：

```python
actions = torch.randn(
    size=(batch_size, action_horizon, action_dim),
    dtype=vl_embs.dtype,
    device=device,
)
```

在当前配置下就是：

```text
actions = (1, 16, 32)
```

也就是说，推理一开始不是从零动作开始，而是从高斯噪声动作轨迹开始。

### 9.5 每一步去噪里发生了什么

对每一个去噪步 `t`，都会做下面几件事。

第一步，构造当前时间步：

```text
timesteps_tensor = (B,)
```

当前例子里是：

```text
(1,)
```

第二步，把当前动作轨迹 `actions=(B,16,32)` 送进 `action_encoder`，得到：

```text
action_features = (B, 16, 1536)
```

第三步，如果开启位置编码，则给每个动作位置加上位置嵌入，形状不变。

第四步，取出一组可学习的 future tokens：

```text
future_tokens = (B, 32, 1536)
```

第五步，把三段序列拼起来：

- `state_features = (B, 1, 1536)`
- `future_tokens = (B, 32, 1536)`
- `action_features = (B, 16, 1536)`

拼接后得到：

```text
sa_embs = (B, 49, 1536)
```

当前 `49 = 1 + 32 + 16`。

第六步，把 `sa_embs` 作为主序列，把 backbone 的视觉语言特征作为 cross-attention 条件，送进 diffusion transformer：

```text
model_output = DiT(
    hidden_states=sa_embs,
    encoder_hidden_states=vl_embs,
    timestep=timesteps_tensor,
)
```

输出仍然是与主序列长度对应的一组隐藏表示。

第七步，经过 `action_decoder` 把隐藏表示映射回动作空间：

```text
pred = (B, 49, 32)
```

然后只取最后 `16` 个位置，对应动作段：

```text
pred_velocity = (B, 16, 32)
```

第八步，用 Euler 更新动作轨迹：

```text
actions = actions + dt * pred_velocity
```

循环 `4` 次以后，最终得到：

```text
action_pred = (B, 16, 32)
```

这就是模型内部“归一化动作空间”里的最终输出。

### 9.6 action head 去噪循环图

上面是逐项解释。下面给一张“单次请求内部、单个 action head 推理”的纯文本循环图：

```text
初始条件:
  vl_embs         = backbone 输出条件特征 = (B,N,2048)
  state           = (B,1,64)
  embodiment_id   = (B,)
  actions         = randn(B,16,32)

进入去噪循环, 当前默认 num_inference_timesteps = 4:

  Step 0
    actions(0) = randn(B,16,32)
       |
       +--> action_encoder(actions(0), t=0) -> action_features=(B,16,1536)
       +--> state_encoder(state)            -> state_features=(B,1,1536)
       +--> future_tokens                   -> (B,32,1536)
       +--> concat                          -> sa_embs=(B,49,1536)
       +--> DiT(hidden_states=sa_embs, encoder_hidden_states=vl_embs)
       +--> action_decoder                  -> pred=(B,49,32)
       +--> slice last 16                   -> pred_velocity=(B,16,32)
       +--> Euler update                    -> actions(1)

  Step 1
    actions(1)
       |
       +--> 重复同样流程 -> actions(2)

  Step 2
    actions(2)
       |
       +--> 重复同样流程 -> actions(3)

  Step 3
    actions(3)
       |
       +--> 重复同样流程 -> actions(4)

循环结束:
  action_pred = actions(4) = (B,16,32)
```

这张图有两个重点：

- action head 推理不是“一次线性映射”，而是“从噪声轨迹开始，连续更新多次”
- 如果你把推理步数从 `4` 降到 `3` 或 `2`，本质上就是把这段循环缩短

## 10. 从 `32` 维模型动作回到真实动作字典

当前模型输出的不是最终机器人 API 直接使用的多键动作字典，而是统一动作张量：

```text
action_pred = (1, 16, 32)
```

接下来 `Gr00tPolicy._get_unnormalized_action()` 会调：

```text
self.unapply_transforms({"action": normalized_action.cpu()})
```

反向链路的关键步骤是：

1. `GR00TTransform.unapply` 不负责把 `32` 维还原成多键动作，它主要是上游模型专用封装
2. `ConcatTransform.unapply` 根据原始 action 键顺序把统一动作张量切回：
   - `action.left_arm`
   - `action.right_arm`
   - `action.left_hand`
   - `action.right_hand`
3. `StateActionTransform.unapply` 把各 action 子段从归一化空间反变换回原动作数值空间

对于当前配置，动作维度拆分关系是：

- `action.left_arm = 7`
- `action.right_arm = 7`
- `action.left_hand = 6`
- `action.right_hand = 6`

总计：

```text
26 real action dims
```

所以虽然模型内部 action token 是 `32` 维，但真正有效的动作维度只有前 `26` 维，后面的维度是 padding，并通过 mask 或反向拆分逻辑被忽略掉。

最终 unapply 后，单请求输出会回到类似：

```text
action.left_arm   (16, 7)
action.right_arm  (16, 7)
action.left_hand  (16, 6)
action.right_hand (16, 6)
```

这就是“未来 16 步动作计划”的字典形式。

## 11. 多次推理请求时，哪些东西会重复算，哪些不会

这是理解在线控制效率的关键。

### 11.1 一般会保持不变的东西

在同一个任务连续执行期间，往往这些东西不变：

- 任务语言本身
- 相机数量和相机键名
- data config
- state/action horizon
- embodiment id
- 模型参数

### 11.2 每次请求都会变的东西

每个控制周期通常都会变化：

- 当前图像
- 当前状态
- 最终视觉语言 token 序列长度中的一部分细节
- backbone 输出特征
- action head 的随机初始化噪声
- 生成出的未来动作

### 11.3 当前代码里每次请求会重跑哪些步骤

按当前实现，只要你调用一次 `policy.get_action()`，这几段基本都会重跑：

1. observation 复制和 batch 处理
2. 图像裁剪、resize、格式转换
3. 语言和图像重新组织成 Eagle conversation
4. tokenizer / processor
5. Eagle backbone 前向
6. action head 整个去噪循环
7. 反归一化和动作字典重建

所以“一次任务，多次请求”的本质是：

- 语言语义可以相同
- 但推理图几乎每次都完整再跑一遍

### 11.4 同一任务下多次请求时间线图

下面这张图强调的是：“任务可保持不变，但请求是反复发生的”。

```text
任务开始
  |
  | 任务语言固定:
  | "pick the pear from the counter and place it in the plate"
  |
  +--------------------------------------------------------------+
                                                                 |
时刻 t0                                                          |
  当前观测 obs(t0)                                                |
    - image(t0)                                                  |
    - state(t0)                                                  |
    - same task text                                             |
  -> policy.get_action(obs(t0))                                  |
  -> 输出未来 16 步动作 A(t0)                                    |
                                                                 |
时刻 t1                                                          |
  当前观测 obs(t1)                                                |
    - image(t1)                                                  |
    - state(t1)                                                  |
    - same task text                                             |
  -> policy.get_action(obs(t1))                                  |
  -> 输出未来 16 步动作 A(t1)                                    |
                                                                 |
时刻 t2                                                          |
  当前观测 obs(t2)                                                |
    - image(t2)                                                  |
    - state(t2)                                                  |
    - same task text                                             |
  -> policy.get_action(obs(t2))                                  |
  -> 输出未来 16 步动作 A(t2)                                    |
                                                                 |
时刻 t3 ...                                                      |
  重复同样过程                                                   |
                                                                 |
  +--------------------------------------------------------------+

要点:
  1. 任务描述可以不变
  2. 图像和状态通常每次都变
  3. 每次请求都重新跑完整推理链
  4. 每次输出的是一段未来动作, 不是单步动作
```

## 12. 一条完整推理链的维度总表

下面把当前默认路径最关键的维度放在一张表里。

| 阶段 | 张量/字段 | 当前真实路径形状 |
|---|---|---|
| 原始输入 | `video.ego_view` | `(1, 256, 256, 3)` |
| 原始输入 | `state.left_arm` | `(1, 7)` |
| 原始输入 | `state.right_arm` | `(1, 7)` |
| 原始输入 | `state.left_hand` | `(1, 6)` |
| 原始输入 | `state.right_hand` | `(1, 6)` |
| 原始动作标签 | 总动作维度 | `16 x 26` |
| state sin/cos 后 | 总状态维度 | `1 x 52` |
| pad 后 state | `state` | `(1, 64)` 单样本内部 |
| batched 模型输入 | `state` | `(1, 1, 64)` |
| batched 模型输入 | `state_mask` | `(1, 1, 64)` |
| batched 模型输入 | `eagle_input_ids` | `(1, 296)` |
| batched 模型输入 | `eagle_attention_mask` | `(1, 296)` |
| batched 模型输入 | `eagle_pixel_values` | `(1, 3, 224, 224)` |
| backbone 输出 | `backbone_features` | `(B, N, 2048)` |
| action 初始噪声 | `actions` | `(1, 16, 32)` |
| state encoder 输出 | `state_features` | `(1, 1, 1536)` |
| action encoder 输出 | `action_features` | `(1, 16, 1536)` |
| future tokens | `future_tokens` | `(1, 32, 1536)` |
| 拼接主序列 | `sa_embs` | `(1, 49, 1536)` |
| action decoder 输出 | `pred` | `(1, 49, 32)` |
| 最终模型输出 | `action_pred` | `(1, 16, 32)` |
| unapply 后 | `action.left_arm` | `(16, 7)` |
| unapply 后 | `action.right_arm` | `(16, 7)` |
| unapply 后 | `action.left_hand` | `(16, 6)` |
| unapply 后 | `action.right_hand` | `(16, 6)` |

## 13. 哪些维度是当前路径固定的，哪些不是

为了避免把“当前 demo 路径”误认为“GR00T 永远如此”，这里单独说明。

### 13.1 当前这条路径里基本固定的

如果你不改模型和 data config，下面这些当前是固定的：

- `state_horizon = 1`
- `action_horizon = 16`
- `max_state_dim = 64`
- `action_dim = 32`
- `input_embedding_dim = 1536`
- `hidden_size = 1024`
- `num_target_vision_tokens = 32`
- `num_inference_timesteps = 4`
- 当前只有一个视觉视角 `V=1`

### 13.2 不是固定常数的

下面这些可能变：

- `eagle_input_ids` 的序列长度，例如当前样本是 `296`
- `backbone_features` 里的 `N`
- 原始图像分辨率
- 原始 state/action 总维数
- 相机数量
- 任务文本长度

一旦你换 data config、换 embodiment、换相机、换 checkpoint，这些数字都可能变化。

## 14. 从系统角度看，哪里最值得做推理加速

这一节只分析“高可行性、不会改变主逻辑太多”的方向。

### 14.1 最值得先做的低风险优化

#### 1. 缓存不变的语言模板和 tokenizer 前处理

同一个任务里，任务描述通常不变，但当前实现每次请求都会：

- 重新构造 Eagle conversation
- 重新 `apply_chat_template`
- 重新 tokenizer

这部分对单次时延可能不是最大头，但它是纯重复工作，尤其在高频请求时完全可以缓存。

最直接的思路是：

- 当任务语言不变、相机数量不变时
- 缓存静态文本模板
- 尽量只替换当前图像部分

#### 2. 复用视觉预处理中的固定部分

当前每次都做：

- 图像裁剪
- resize
- numpy / tensor / PIL 转换
- processor 的图像预处理

如果相机输入分辨率和数量固定，这里可以做两类优化：

- 减少 Python 层对象转换，特别是 `numpy -> PIL -> processor`
- 尽量让图像前处理更连续，少做格式来回切换

这一段通常是很现实的 CPU 侧优化点。

#### 3. 减少 action head 的去噪步数

当前默认 `num_inference_timesteps = 4`。这已经不算很大，但 action head 的时间复杂度几乎仍然会随着步数线性增长。

如果你的质量允许，可以评估：

- `4 -> 3`
- `4 -> 2`

代价是动作质量可能下降，但这是最直接、最可量化的加速手段之一。

#### 4. PyTorch 路径下使用 `torch.compile`

如果你跑的是 PyTorch 本地推理路径，而不是 TensorRT，那么 `torch.compile` 可以作为一个可选实验项。

它更适合：

- 固定输入结构
- 重复调用较多
- warmup 成本可接受

但它不应混进 ONNX 导出和 TensorRT 量化链路。

#### 5. 对重复请求做更细的 profile 拆解

你现在已经有 `ant` 目录下的分段脚本，这是很好的基础。下一步最值得量化的是：

- transform 时间
- Eagle backbone 时间
- action head 总时间
- action head 每个 denoise step 时间

只有拆开以后，后面的优化才不会盲目。

### 14.2 中等风险但潜在收益不错的优化

#### 1. 把稳定子图优先交给 TensorRT

如果你最终跑 Jetson Orin，本质上最有潜力的仍然是 TensorRT 化。

优先级上通常是：

1. 视觉 backbone 子图
2. action head 中结构稳定、shape 稳定的部分

但要注意：

- TRT 收益建立在图足够稳定之上
- 文本处理和部分前后处理仍会留在 Python / PyTorch 侧

#### 2. 减少 Python 控制流和对象构造

当前推理路径里有不少 Python 层操作：

- dict 复制
- numpy 转换
- PIL 构造
- 多次键重排
- BatchFeature 封装

单次看不大，但高频调用时累计明显。把它们合并、缓存或前移，通常能得到比较稳的收益。

#### 3. 预分配和复用中间 buffer

例如 action head 里每次都会重新创建：

- 初始噪声动作
- timestep tensor
- 某些辅助张量

在固定 batch、固定 horizon 的在线场景下，可以考虑预分配并复用一部分 buffer，减少频繁的小张量分配。

### 14.3 不建议的方向

#### 1. 在导出 ONNX 之前先 `torch.compile`

这通常对 TensorRT 没什么帮助，反而可能让导出更不稳定。

应该保持：

- PyTorch 推理路线，单独试 `torch.compile`
- TensorRT 路线，直接从未 compile 的模型导出 ONNX

#### 2. 把“真实闭环成功率”和“本地代理指标”混成一个指标

这不是推理加速问题，但会直接污染评测结论。没有仿真环境或真实机器人时，只能诚实地写：

- open-loop 误差
- 平滑性
- proxy success rate

而不是声称真实 closed-loop success rate。

#### 3. 盲目压缩精度或改量化而不先做误差对照

在 Jetson 上，量化当然可能带来加速，但如果没有：

- open-loop 误差
- 平滑性
- proxy success rate

这些对照，你很难判断加速后的动作质量有没有明显恶化。

## 15. 如果你把这条链路记成一句话

可以记成下面这句：

> GR00T 在每次请求里先把“当前图像 + 当前状态 + 任务语言”变成 Eagle 多模态条件特征，再让 action head 从一段随机动作轨迹出发做多步去噪，最后把统一的 `16x32` 动作张量反变换成真实机器人动作字典。

如果再压缩成工程视角，就是：

> 每次请求都会完整重跑“预处理 -> Eagle backbone -> diffusion-style action head -> 动作反归一化”，其中 action head 输出的是未来 `16` 步动作，而不是单步动作。
