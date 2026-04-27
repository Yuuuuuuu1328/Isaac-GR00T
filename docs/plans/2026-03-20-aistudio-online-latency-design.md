# Aistudio Online Latency Design

**Goal**

为 [`scripts/aistudio_test.py`](/home/jetson/Desktop/project/Isaac-GR00T/scripts/aistudio_test.py) 增加在线请求时延观测，分别覆盖服务端总处理时延、服务端纯推理时延，以及客户端 HTTP 往返时延，同时保持现有请求体和响应体格式不变。

**Architecture**

服务端在 `/predict` 路由层记录单次请求的总处理时延，并在模型执行入口内部记录纯推理时延。两项服务端指标会打印到控制台、追加写入 [`deployment_scripts/ant/output/online_result.jsonl`](/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/output/online_result.jsonl)，并通过 HTTP 响应头暴露给客户端。客户端示例脚本单独测量整次 HTTP 往返时延，读取服务端响应头中的两项时延并一起打印。

**Data Flow**

1. 客户端脚本读取图片和控制输入，构造与现有服务兼容的 `{"query": "<json-string>"}` 载荷。
2. 服务端进入 `/predict` 路由时记录 `request_start_ns`。
3. `AistudioRuntime.predict(...)` 内部在 `_predict_previous_actions(...)` 周围记录 `inference_ms`。
4. 路由在拿到返回结果后计算 `server_total_ms`，打印请求日志并将记录追加到 `online_result.jsonl`。
5. 路由响应体继续返回当前 Aistudio JSON；额外通过响应头暴露：
   `X-Server-Total-Ms`
   `X-Inference-Ms`
6. 客户端测量本地 `client_round_trip_ms`，再结合响应头打印三类时延。

**Compatibility**

- 请求体格式不变。
- 响应体 JSON 结构不变。
- 仅新增响应头，不要求现有调用方修改解析逻辑。
- TensorRT 和 PyTorch 两个后端都记录时延；TensorRT 仍走当前 breakdown 执行路径。

**Error Handling**

- 即使推理异常，服务端仍记录本次请求的 `server_total_ms`。
- 如果异常发生在推理前，`inference_ms` 为空或 `0.0`，并随错误记录一起写入 JSONL。
- JSONL 采用逐行追加；目录不存在时自动创建。
- 客户端如果拿不到响应头，仍打印 `client_round_trip_ms` 和响应体。

**Logging Format**

每行 JSONL 使用统一结构：

```json
{
  "meta": {
    "timestamp": "2026-03-20T00:00:00+00:00",
    "request_id": "req-1",
    "device_id": "dev-1",
    "backend": "tensorrt",
    "result_code": 0
  },
  "metrics": {
    "server_total_ms": 12.3456,
    "inference_ms": 8.7654
  }
}
```

**Testing**

- 服务测试锁定：
  - 响应体不变
  - 响应头包含服务端时延
  - 服务端逐行追加写入 JSONL
  - 控制台打印单次时延
- 客户端测试锁定：
  - 输入参数能生成兼容请求
  - 能打印 `client_round_trip_ms`
  - 能读取并打印服务端响应头
