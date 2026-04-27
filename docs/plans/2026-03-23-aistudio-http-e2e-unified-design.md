# Aistudio HTTP E2E Unified Script Design

**Goal**

新增一个单文件脚本，将当前 `scripts/aistudio_c.py` 与 `scripts/aistudio_s.py` 的核心能力整合为 `--role server|client` 两种运行模式，在不改变现有服务协议输入/输出格式的前提下，提供可直接用于微调模型在线推理时延评估的精简链路。

**First-Principles Constraints**

1. 协议兼容优先于重构便利：请求体与响应体格式必须保持不变，避免影响现有调用方。
2. 时延口径可解释优先：指标必须对应明确物理意义（客户端 RTT、服务端纯推理、差值网络时延）。
3. 最小必要复杂度：仅保留 FastAPI HTTP 与 e2e 推理路径，移除 breakdown 相关在线逻辑。
4. 默认参数可复现：固定模型、引擎、数据配置、机器人标签等默认值，确保新设备验证时开箱可跑。

**Scope**

This design applies to:
- 新增 `scripts/aistudio_http_e2e.py`，统一 server/client 两种角色。
- server 仅提供 FastAPI `/predict` 接口。
- client 仅做 HTTP 压测与 jsonl 指标落盘。
- 推理路径参考 `deployment_scripts/ant/local_inference_new_interaction.py` 的 e2e 模式。

This design does not change:
- `scripts/aistudio_c.py` 与 `scripts/aistudio_s.py` 原文件（保留以便回退）。
- Aistudio 兼容请求体格式 `{"query": "<json-string>"}`。
- Aistudio 兼容响应体结构 `resultCode/errorMessage/resultMap`。

**Fixed Defaults (User-Confirmed)**

- `DEFAULT_MODEL_PATH = "/home/jetson/Desktop/project/new_model/left_hand_v2_1223"`
- `DEFAULT_OSSFS_WORKSPACE = "/home/jetson/Desktop/project/ossfs/node_59823209/workspace"`
- `DEFAULT_TRT_ENGINE_PATH = "/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16"`
- `DEFAULT_DATA_CONFIG = "new_interaction_group"`
- `DEFAULT_EMBODIMENT_TAG = "new_embodiment"`
- `DEFAULT_RESPONSE_ACTION_HORIZON = 14`
- `DEFAULT_TASK_PROMPT = "Move to center the book in view. Do nothing if no book is present."`
- `DEFAULT_BACKEND = "tensorrt"` with choices `["pytorch", "tensorrt"]`
- `DEFAULT_DENOISING_STEPS = 4`
- `DEFAULT_VIT_DTYPE = "fp16"`
- `DEFAULT_LLM_DTYPE = "fp16"`
- `DEFAULT_DIT_DTYPE = "fp16"`
- `DEFAULT_SERVER_HOST = "0.0.0.0"`
- `DEFAULT_SERVER_PORT = 8000`
- `DEFAULT_CLIENT_HOST = "127.0.0.1"`
- `DEFAULT_CLIENT_PORT = 8000`
- `DEFAULT_TIMEOUT_MS = 15000`
- `DEFAULT_WARMUP_RUNS = 5`
- `DEFAULT_MEASURE_RUNS = 10`
- `DEFAULT_SEED = 0`
- `DEFAULT_OUTPUT_JSONL = "/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/output/aistudio_http_e2e_result.jsonl"`

**Runtime Architecture**

1. `--role server`
- 加载 ossfs GR00T runtime（路径可配，默认固定）。
- 初始化 `Gr00tPolicy`，可选 TensorRT 引擎挂载。
- 启动 FastAPI，暴露 `/predict`、`/health/live`、`/health/ready`、`/metadata`。
- 在 `/predict` 内计算：
  - `server_total_ms`: 路由总耗时。
  - `pure_inference_ms`: 仅 `policy.get_action(...)` 包围计时（按 `inference_raw.py` 语义）。
  - `server_overhead_ms`: `server_total_ms - pure_inference_ms`。
- 在响应头返回：
  - `X-Server-Total-Ms`
  - `X-Pure-Inference-Ms`
  - `X-Inference-Ms` (legacy alias)
  - `X-Server-Overhead-Ms`

2. `--role client`
- 构造兼容请求 `{"query": "<json-string>"}`，字段保持与现有 client 一致。
- 执行 warmup（不计入统计，不写 jsonl）。
- 执行 measure runs：
  - 记录 `total_latency_ms`（客户端 HTTP 往返）。
  - 从 header 读取 `pure_inference_ms`（缺失时回退策略与现有 client 对齐）。
  - 计算 `network_latency_ms = total - inference`。
  - 计算单次 `inference_proportion_pct`。
- 控制台输出单次指标，并在末尾按固定格式输出平均指标。
- 仅 client 落盘 jsonl，写入每次 run 记录 + 1 条 summary 记录。

**Data Contract**

Request (`POST /predict`):
```json
{
  "query": "{\"joint_angles\":\"[...]\",\"predicted_coords_2d\":\"[...]\",\"history_angles\":\"[...]\",\"framebuffer\":[...],\"framebuffer_size\":12345,\"device_id\":\"...\",\"request_id\":\"...\"}"
}
```

Response:
```json
{
  "resultCode": 0,
  "errorMessage": "ok",
  "resultMap": {
    "action_sequence": "[[...]]",
    "joint_angles": "[...]",
    "predicted_coords_2d": "[...]",
    "history_angles": "[...]",
    "is_success": true,
    "error": "",
    "key_infos": "key_infos",
    "device_id": "...",
    "request_id": "..."
  }
}
```

**Latency Semantics**

- `total_latency_ms` (client): 单次 HTTP 请求完整往返时间。
- `pure_inference_ms` (server header): 仅模型 `get_action` 调用耗时。
- `network_latency_ms`: `max(total_latency_ms - pure_inference_ms, 0)`.
- `inference_proportion_pct`: `(pure_inference_ms / total_latency_ms) * 100`.

Average output format is fixed:
```python
print("\n=== Average Latency ===")
print(f"📊 Average total latency: {total_latency_sum / actual_runs:.5f} ms")
print(f"⚡ Average inference latency: {infer_latency_sum / actual_runs:.5f} ms")
print(f"🌐 Average network latency: {network_latency_sum / actual_runs:.5f} ms")
print(f"📈 Average inference proportion: {(infer_latency_sum / total_latency_sum) * 100:.2f}%")
```

**JSONL Output Contract (Client Only)**

Run record (one per measured request):
- `record_type = "run"`
- `timestamp`
- `config` (full args snapshot)
- `request_meta` (`phase=run`, `iteration`, `request_id`, `device_id`, `http_status`, `result_code`)
- `latency` (`total_latency_ms`, `pure_inference_ms`, `network_latency_ms`, `inference_proportion_pct`)
- `server_headers` (timing headers as parsed values)
- `response` (resultCode/errorMessage/resultMap)

Summary record (appended once at the end):
- `record_type = "summary"`
- `timestamp`
- `config` (full args snapshot)
- `summary_metrics` (average total/inference/network/proportion)
- `run_count`, `success_count`, `failure_count`
- `output_jsonl`

**Error Handling**

- server 任意异常返回兼容错误响应结构，不破坏字段形状。
- client 单次请求异常不终止整体 benchmark；该次写失败记录，继续后续 run。
- 若服务端 header 缺失或非法，client 使用回退逻辑保证统计可继续并显式写入当前解析状态。

**Testing Strategy**

- 新增脚本单测覆盖：
  - parser 默认参数固定值。
  - payload 构造与协议格式。
  - `/predict` 输出结构 + latency headers。
  - client run-only jsonl 与 summary 追加行为。
  - 平均时延输出格式和计算口径。
- 重点回归：
  - `tests/test_aistudio_service.py`
  - `tests/test_aistudio_client_example.py`

**Commit Note**

当前仓库存在大量非本任务改动；本阶段先落地设计文档与实施计划，不在本设计步骤提交 commit。
