# Aistudio Client Random Benchmark Design

**Goal**

让 [`scripts/aistudio_client_example.py`](/home/jetson/Desktop/project/Isaac-GR00T/scripts/aistudio_client_example.py) 在零参数情况下也能直接执行，通过代码内随机生成 Aistudio 兼容输入，完成默认 `5` 次 warmup 和 `10` 次正式测试。

**Architecture**

客户端不再依赖现成数据集作为默认输入，而是在脚本内生成随机 `joint_angles`、`predicted_coords_2d`、`history_angles` 和随机 RGB 图像。脚本会将随机图像直接编码为 JPEG framebuffer，构造成与服务端完全兼容的请求体，随后执行 warmup 和正式测试循环，并打印每次正式测试的客户端 RTT、服务端总时延和服务端推理时延，以及最终均值汇总。

**Behavior**

- 零参数可运行。
- 默认参数：
  - `host=127.0.0.1`
  - `port=8000`
  - `timeout_ms=15000`
  - `warmup_runs=5`
  - `measure_runs=10`
  - `seed=0`
- 默认请求数据：
  - `joint_angles`: 5 维随机浮点
  - `predicted_coords_2d`: 2 维随机浮点
  - `history_angles`: 默认包含当前 joint angles 的一条历史
  - `image`: 随机生成的 `640x480x3` RGB 图像
  - `device_id`: `dev-random`
  - `request_id`: 基础前缀 `req-0001`，多轮时自动追加 `warmup/run` 后缀

**Compatibility**

- 服务端请求体格式保持不变：
  `{"query": "<json-string>"}`
- 如果用户显式传入 `--image-path`、`--joint-angles`、`--predicted-coords-2d`、`--history-angles`，则优先使用用户输入覆盖随机默认值。
- 服务端返回体解析方式不变，客户端继续从响应头读取：
  - `X-Server-Total-Ms`
  - `X-Inference-Ms`

**Output**

- Warmup 阶段打印简要进度，不做汇总统计。
- 正式测试阶段每次打印：
  - `phase`
  - `run_index`
  - `request_id`
  - `client_round_trip_ms`
  - `server_total_ms`
  - `inference_ms`
- 测试结束打印均值汇总。

**Testing**

- 锁定 parser 默认值支持零参数。
- 锁定缺省输入时能生成随机且兼容的请求体。
- 锁定 `main()` 会按默认 `5 + 10` 次调用，并打印正式测试与汇总信息。
