from __future__ import annotations

import argparse
import ast
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

from deployment_scripts.ant import ossfs_gr00t_runtime as _ossfs_runtime_helper
from deployment_scripts.trt_model_forward import setup_tensorrt_engines
from deployment_scripts.ant.profile_utils import measure_cuda_time_ms
try:
    from websocket import create_connection as websocket_create_connection
except ModuleNotFoundError:  # pragma: no cover - handled by runtime/tests
    websocket_create_connection = None


DEFAULT_MODEL_PATH = "/home/jetson/Desktop/project/new_model/left_hand_v2_1223"
DEFAULT_OSSFS_WORKSPACE = "/home/jetson/Desktop/project/ossfs/node_59823209/workspace"
DEFAULT_TRT_ENGINE_PATH = (
    "/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16"
)
DEFAULT_TASK_PROMPT = "book"
DEFAULT_DATA_CONFIG = "new_interaction_group"
DEFAULT_EMBODIMENT_TAG = "new_embodiment"
DEFAULT_RESPONSE_ACTION_HORIZON = 14
DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_CLIENT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_WS_PATH = "/ws/predict"
DEFAULT_TIMEOUT_MS = 15000
DEFAULT_WARMUP_RUNS = 10
DEFAULT_MEASURE_RUNS = 20
DEFAULT_SEED = 0

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSONL = (
    _REPO_ROOT / "scripts/results/aistudio_http_e2e_result.jsonl"
)


def _to_ascii_text(value: Any) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Aistudio WebSocket e2e server/client script")
    parser.add_argument("--role", type=str, choices=["server", "client"], default="server")
    parser.add_argument("--host", type=str, default="")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ws-path", type=str, default=DEFAULT_WS_PATH)

    parser.add_argument("--backend", type=str, choices=["pytorch", "tensorrt"], default="tensorrt")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ossfs-workspace", type=str, default=DEFAULT_OSSFS_WORKSPACE)
    parser.add_argument("--trt-engine-path", type=str, default=DEFAULT_TRT_ENGINE_PATH)
    parser.add_argument("--data-config", type=str, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--embodiment-tag", type=str, default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--task-prompt", type=str, default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--response-action-horizon", type=int, default=DEFAULT_RESPONSE_ACTION_HORIZON)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--vit-dtype", type=str, choices=["fp16", "fp8", "int8"], default="fp16")
    parser.add_argument(
        "--llm-dtype", type=str, choices=["fp16", "nvfp4", "fp8", "int8"], default="fp16"
    )
    parser.add_argument("--dit-dtype", type=str, choices=["fp16", "fp8", "int8"], default="fp16")

    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--joint-angles", type=str, default="")
    parser.add_argument("--predicted-coords-2d", type=str, default="")
    parser.add_argument("--history-angles", type=str, default="")
    parser.add_argument("--device-id", type=str, default="dev-random")
    parser.add_argument("--request-id", type=str, default="req-0001")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--measure-runs", type=int, default=DEFAULT_MEASURE_RUNS)
    parser.add_argument("--output-jsonl", type=str, default=str(DEFAULT_OUTPUT_JSONL))
    return parser


def parse_literal_list(raw_value: str) -> list:
    value = ast.literal_eval(raw_value)
    if not isinstance(value, list):
        raise ValueError(f"Expected list literal, got: {raw_value}")
    return value


def encode_image_to_framebuffer(image_path: str | Path) -> tuple[list[int], int]:
    import cv2

    image_path = Path(image_path).expanduser()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image as JPEG: {image_path}")
    framebuffer = encoded.astype("uint8").tolist()
    return framebuffer, int(encoded.size)


def generate_random_rgb_image(
    rng: np.random.Generator,
    *,
    height: int = 640,
    width: int = 480,
) -> np.ndarray:
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def encode_rgb_to_framebuffer(image_rgb: np.ndarray) -> tuple[list[int], int]:
    import cv2

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode random image as JPEG")
    framebuffer = encoded.astype("uint8").tolist()
    return framebuffer, int(encoded.size)


def build_iteration_request_id(base_request_id: str, phase: str, iteration_index: int) -> str:
    return f"{base_request_id}-{phase}-{iteration_index:02d}"


def _resolve_joint_angles(args: argparse.Namespace, rng: np.random.Generator) -> list[float]:
    if args.joint_angles:
        return parse_literal_list(args.joint_angles)
    return [round(float(value), 4) for value in rng.uniform(-1.57, 1.57, size=5).tolist()]


def _resolve_predicted_coords(args: argparse.Namespace, rng: np.random.Generator) -> list[float]:
    if args.predicted_coords_2d:
        return parse_literal_list(args.predicted_coords_2d)
    return [round(float(value), 4) for value in rng.uniform(0.0, 1.0, size=2).tolist()]


def _resolve_history_angles(
    args: argparse.Namespace,
    joint_angles: list[float],
) -> list[list[float]]:
    if args.history_angles:
        return parse_literal_list(args.history_angles)
    return [joint_angles]


def _resolve_framebuffer(
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    if args.image_path:
        return encode_image_to_framebuffer(args.image_path)
    return encode_rgb_to_framebuffer(generate_random_rgb_image(rng))


def build_request_payload(
    args: argparse.Namespace,
    *,
    request_id: str | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    rng = rng or np.random.default_rng()
    joint_angles = _resolve_joint_angles(args, rng)
    predicted_coords_2d = _resolve_predicted_coords(args, rng)
    history_angles = _resolve_history_angles(args, joint_angles)
    framebuffer, framebuffer_size = _resolve_framebuffer(args, rng)
    inner = {
        "joint_angles": str(joint_angles),
        "predicted_coords_2d": str(predicted_coords_2d),
        "history_angles": str(history_angles),
        "framebuffer": framebuffer,
        "framebuffer_size": framebuffer_size,
        "device_id": args.device_id,
        "request_id": request_id or args.request_id,
    }
    return {"query": json.dumps(inner, ensure_ascii=False)}


def send_request(
    *,
    host: str,
    port: int,
    ws_path: str,
    payload: dict,
    timeout_ms: int,
):
    if websocket_create_connection is None:
        raise RuntimeError("websocket-client is required to run client benchmark")

    ws_path = ws_path if ws_path.startswith("/") else f"/{ws_path}"
    start_ns = time.perf_counter_ns()
    connection = websocket_create_connection(
        f"ws://{host}:{port}{ws_path}",
        timeout=timeout_ms / 1000.0,
    )
    try:
        connection.send(json.dumps(payload, ensure_ascii=False))
        raw_message = connection.recv()
    finally:
        try:
            connection.close()
        except Exception:
            pass
    end_ns = time.perf_counter_ns()
    client_round_trip_ms = round((end_ns - start_ns) / 1_000_000.0, 4)
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    envelope = json.loads(raw_message)
    if not isinstance(envelope, dict):
        raise ValueError("WebSocket response must be a JSON object")
    body = envelope.get("result", {})
    if not isinstance(body, dict):
        raise ValueError("WebSocket response missing object result")
    latency = envelope.get("latency", {})
    if not isinstance(latency, dict):
        latency = {}
    response = SimpleNamespace(
        status_code=int(envelope.get("status_code", 200) or 200),
        headers=build_latency_headers(
            server_total_ms=_coerce_latency_ms(latency.get("server_total_ms", 0.0)),
            pure_inference_ms=_coerce_latency_ms(latency.get("pure_inference_ms", 0.0)),
            server_overhead_ms=_coerce_latency_ms(latency.get("server_overhead_ms", 0.0)),
        ),
        latency_breakdown={
            "preprocess_ms": _coerce_latency_ms(latency.get("preprocess_ms", 0.0)),
            "postprocess_ms": _coerce_latency_ms(latency.get("postprocess_ms", 0.0)),
        },
        json=lambda: body,
    )
    return response, client_round_trip_ms


def _parse_list_like(value: Any):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def parse_aistudio_query(payload: dict) -> dict:
    if "query" not in payload:
        raise ValueError("Missing query field")

    raw_query = payload["query"]
    if isinstance(raw_query, bytes):
        raw_query = raw_query.decode("utf-8")

    parsed = json.loads(raw_query)
    parsed["joint_angles"] = _parse_list_like(parsed.get("joint_angles", []))
    parsed["predicted_coords_2d"] = _parse_list_like(parsed.get("predicted_coords_2d", []))
    parsed["history_angles"] = _parse_list_like(parsed.get("history_angles", []))
    return parsed


def build_aistudio_response(
    *,
    result_map: dict,
    result_code: int = 0,
    error_message: str = "ok",
) -> dict:
    return {
        "resultCode": result_code,
        "errorMessage": error_message,
        "resultMap": result_map,
    }


def build_aistudio_error_response(
    *,
    error_message: str,
    request_id: str = "",
    device_id: str = "",
    joint_angles: list[Any] | None = None,
    predicted_coords_2d: list[Any] | None = None,
    history_angles: list[Any] | None = None,
) -> dict:
    error_message = _to_ascii_text(error_message)
    return build_aistudio_response(
        result_code=1,
        error_message=error_message,
        result_map={
            "action_sequence": "",
            "joint_angles": json.dumps(joint_angles or []),
            "predicted_coords_2d": json.dumps(predicted_coords_2d or []),
            "history_angles": json.dumps(history_angles or []),
            "is_success": False,
            "error": error_message,
            "key_infos": "key_infos",
            "device_id": device_id,
            "request_id": request_id,
        },
    )


def normalize_single_arm_state(joint_angles: list[float] | tuple[float, ...]) -> np.ndarray:
    if len(joint_angles) == 5:
        normalized = [*joint_angles, 0.0]
    elif len(joint_angles) == 6:
        normalized = list(joint_angles)
    else:
        raise ValueError("joint_angles must contain 5 or 6 values")
    return np.array(normalized, dtype=np.float32)


def decode_framebuffer(framebuffer: list[int], framebuffer_size: int = 0) -> np.ndarray:
    del framebuffer_size
    import cv2

    jpeg_array = np.array(framebuffer, dtype=np.uint8)
    framebuffer_data = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
    if framebuffer_data is None:
        raise ValueError("Failed to decode framebuffer")
    frame_bgr = cv2.resize(framebuffer_data, (480, 640))
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_model_observation(
    *,
    frame_rgb: np.ndarray,
    robot_state: np.ndarray,
    task_prompt: str,
) -> dict:
    return {
        "video.ego_view": frame_rgb,
        "annotation.task_index": task_prompt,
        "state.single_arm": robot_state,
    }


def build_model_batch(
    *,
    frame_rgb: np.ndarray,
    robot_state: np.ndarray,
    task_prompt: str,
) -> dict:
    observation = build_model_observation(
        frame_rgb=frame_rgb,
        robot_state=robot_state,
        task_prompt=task_prompt,
    )
    return {
        "video.ego_view": observation["video.ego_view"][np.newaxis, ...],
        "annotation.task_index": [observation["annotation.task_index"]],
        "state.single_arm": observation["state.single_arm"][np.newaxis, ...],
    }


def postprocess_actions(
    previous_actions: np.ndarray,
    robot_state: np.ndarray,
    *,
    response_action_horizon: int,
) -> tuple[str, bool]:
    integrated_actions = np.cumsum(previous_actions[:, :6], axis=0) + robot_state[:6]
    rounded_actions = [[round(float(x), 4) for x in row] for row in integrated_actions.tolist()]
    action_string = json.dumps(rounded_actions[:response_action_horizon])
    return action_string, bool(integrated_actions.size >= 6)


def _coerce_latency_ms(value: Any, *, default: float = 0.0) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return round(default, 4)


def build_latency_headers(
    *,
    server_total_ms: float,
    pure_inference_ms: float,
    server_overhead_ms: float,
) -> dict[str, str]:
    return {
        "X-Server-Total-Ms": str(_coerce_latency_ms(server_total_ms)),
        "X-Pure-Inference-Ms": str(_coerce_latency_ms(pure_inference_ms)),
        "X-Inference-Ms": str(_coerce_latency_ms(pure_inference_ms)),
        "X-Server-Overhead-Ms": str(_coerce_latency_ms(server_overhead_ms)),
    }


@dataclass
class AistudioRuntime:
    policy: object
    task_prompt: str
    response_action_horizon: int
    metadata: dict = field(default_factory=dict)
    ready: bool = True

    def _predict_previous_actions(
        self,
        *,
        model_batch: dict[str, Any],
    ) -> tuple[np.ndarray, float]:
        def _infer_func():
            return self.policy.get_action(model_batch)

        action_result, gpu_infer_ms = measure_cuda_time_ms(_infer_func)
        previous_actions = action_result["action.single_arm"]
        return previous_actions, gpu_infer_ms

    def predict_with_timing(self, payload: dict) -> tuple[dict, dict[str, Any]]:
        parsed: dict[str, Any] | None = None
        get_action_ms = 0.0
        preprocess_ms = 0.0
        postprocess_ms = 0.0
        try:
            t_pre_start = time.perf_counter_ns()
            parsed = parse_aistudio_query(payload)
            frame_rgb = decode_framebuffer(
                parsed.get("framebuffer", []),
                parsed.get("framebuffer_size", 0),
            )
            robot_state = normalize_single_arm_state(parsed["joint_angles"])
            batch = build_model_batch(
                frame_rgb=frame_rgb,
                robot_state=robot_state,
                task_prompt=self.task_prompt,
            )
            preprocess_ms = round(
                (time.perf_counter_ns() - t_pre_start) / 1_000_000.0, 4
            )

            previous_actions, get_action_ms = self._predict_previous_actions(model_batch=batch)

            t_post_start = time.perf_counter_ns()
            action_string, is_success = postprocess_actions(
                previous_actions,
                robot_state,
                response_action_horizon=self.response_action_horizon,
            )
            postprocess_ms = round(
                (time.perf_counter_ns() - t_post_start) / 1_000_000.0, 4
            )

            result_map = {
                "action_sequence": action_string,
                "joint_angles": json.dumps(parsed["joint_angles"]),
                "predicted_coords_2d": json.dumps(parsed.get("predicted_coords_2d", [])),
                "history_angles": json.dumps(parsed.get("history_angles", [])),
                "is_success": is_success,
                "error": "",
                "key_infos": "key_infos",
                "device_id": parsed.get("device_id", ""),
                "request_id": parsed.get("request_id", ""),
            }
            return build_aistudio_response(result_map=result_map), {
                "request_id": str(parsed.get("request_id", "")),
                "device_id": str(parsed.get("device_id", "")),
                "backend": str(self.metadata.get("backend", "")),
                "get_action_ms": get_action_ms,
                "preprocess_ms": preprocess_ms,
                "postprocess_ms": postprocess_ms,
            }
        except Exception as exc:
            request_id = "" if parsed is None else str(parsed.get("request_id", ""))
            device_id = "" if parsed is None else str(parsed.get("device_id", ""))
            return (
                build_aistudio_error_response(
                    error_message=str(exc),
                    request_id=request_id,
                    device_id=device_id,
                    joint_angles=[] if parsed is None else parsed.get("joint_angles", []),
                    predicted_coords_2d=[] if parsed is None else parsed.get("predicted_coords_2d", []),
                    history_angles=[] if parsed is None else parsed.get("history_angles", []),
                ),
                {
                    "request_id": request_id,
                    "device_id": device_id,
                    "backend": str(self.metadata.get("backend", "")),
                    "get_action_ms": get_action_ms,
                    "preprocess_ms": preprocess_ms,
                    "postprocess_ms": postprocess_ms,
                },
            )


def create_app(*, runtime, ws_path: str = DEFAULT_WS_PATH):
    ws_path = ws_path if ws_path.startswith("/") else f"/{ws_path}"
    app = FastAPI(title="Aistudio WebSocket e2e Service")

    @app.websocket(ws_path)
    async def ws_predict(websocket: WebSocket):
        await websocket.accept()
        # 外层循环：保证单个连接异常后，服务依然存活，继续处理下一次请求
        while True:
            try:
                # 接收消息
                raw_payload = await websocket.receive_text()
                
                # 业务处理
                request_start_ns = time.perf_counter_ns()
                try:
                    payload = json.loads(raw_payload)
                    response_body, timing = runtime.predict_with_timing(payload)
                except Exception as exc:
                    response_body = build_aistudio_error_response(error_message=str(exc))
                    timing = {
                        "get_action_ms": 0.0,
                        "preprocess_ms": 0.0,
                        "postprocess_ms": 0.0,
                    }

                server_total_ms = round((time.perf_counter_ns() - request_start_ns) / 1_000_000.0, 4)
                get_action_ms = _coerce_latency_ms(timing.get("get_action_ms", 0.0))
                preprocess_ms = _coerce_latency_ms(timing.get("preprocess_ms", 0.0))
                postprocess_ms = _coerce_latency_ms(timing.get("postprocess_ms", 0.0))
                server_overhead_ms = max(
                    round(server_total_ms - get_action_ms - preprocess_ms - postprocess_ms, 4),
                    0.0,
                )

                await websocket.send_json({
                    "status_code": 200,
                    "result": response_body,
                    "latency": {
                        "server_total_ms": _coerce_latency_ms(server_total_ms),
                        "get_action_ms": get_action_ms,
                        "preprocess_ms": preprocess_ms,
                        "postprocess_ms": postprocess_ms,
                        "server_overhead_ms": _coerce_latency_ms(server_overhead_ms),
                    },
                })

            # 客户端断开连接 → 只退出当前连接，服务不关闭
            except WebSocketDisconnect:
                break

            # 任何其他异常 → 打印日志，但服务继续运行
            except Exception as e:
                print(f"[WebSocket 异常] 连接处理出错，服务继续运行: {str(e)}")
                continue


    @app.get("/health/live")
    def health_live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def health_ready():
        return {"status": "ok", "ready": bool(getattr(runtime, "ready", False))}

    @app.get("/metadata")
    def metadata():
        return dict(getattr(runtime, "metadata", {}))

    return app


def _resolve_data_config(data_config_map: dict[str, Any], name: str):
    if name not in data_config_map:
        raise ValueError(
            f"Local ossfs gr00t does not expose data_config '{name}'. "
            f"Available options: {sorted(data_config_map.keys())}"
        )
    return data_config_map[name]


def build_runtime(args: argparse.Namespace) -> AistudioRuntime:
    import torch

    # Allow CLI override for the local ossfs workspace.
    _ossfs_runtime_helper._OSSFS_WORKSPACE = Path(args.ossfs_workspace)
    ossfs_runtime = _ossfs_runtime_helper._import_local_gr00t_runtime()
    data_config = _resolve_data_config(ossfs_runtime.DATA_CONFIG_MAP, args.data_config)
    policy = ossfs_runtime.Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trt_engine_path = None
    if args.backend == "tensorrt":
        trt_engine_path = str(Path(args.trt_engine_path).expanduser())
        setup_tensorrt_engines(
            policy,
            trt_engine_path,
            vit_dtype=args.vit_dtype,
            llm_dtype=args.llm_dtype,
            dit_dtype=args.dit_dtype,
        )
    return AistudioRuntime(
        policy=policy,
        task_prompt=args.task_prompt,
        response_action_horizon=args.response_action_horizon,
        metadata={
            "backend": args.backend,
            "model_path": args.model_path,
            "data_config": args.data_config,
            "embodiment_tag": args.embodiment_tag,
            "trt_engine_path": trt_engine_path,
            "ossfs_workspace": args.ossfs_workspace,
            "response_action_horizon": args.response_action_horizon,
            "vit_dtype": args.vit_dtype,
            "llm_dtype": args.llm_dtype,
            "dit_dtype": args.dit_dtype,
            "denoising_steps": args.denoising_steps,
        },
        ready=True,
    )


def run_server(app, *, host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is required to run the WebSocket service") from exc
    uvicorn.run(app, host=host, port=port)


def _parse_latency_header(response, header_name: str) -> float:
    try:
        return round(float(response.headers.get(header_name, "nan")), 4)
    except (TypeError, ValueError):
        return float("nan")


def _resolve_nonzero_inference_ms(
    *,
    pure_inference_ms: float,
    server_total_ms: float,
    client_round_trip_ms: float,
) -> float:
    candidates = (pure_inference_ms, server_total_ms, client_round_trip_ms)
    for candidate in candidates:
        if np.isnan(candidate):
            continue
        if candidate > 0.0:
            return round(candidate, 4)
    return 0.0


def _append_jsonl_record(output_path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="ascii") as file_obj:
        file_obj.write(json.dumps(_json_safe(record), ensure_ascii=True) + "\n")


def _host_for_role(args: argparse.Namespace) -> str:
    if args.host:
        return args.host
    if args.role == "server":
        return DEFAULT_SERVER_HOST
    return DEFAULT_CLIENT_HOST


def _fetch_server_metadata(host: str, port: int) -> dict:
    """Fetch server metadata via HTTP to record actual server config."""
    import urllib.request
    try:
        url = f"http://{host}:{port}/metadata"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"Warning: could not fetch server meta {exc}")
        return {}


def run_client_benchmark(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    host = _host_for_role(args)
    total_latency_sum = 0.0
    infer_latency_sum = 0.0
    network_latency_sum = 0.0
    preprocess_latency_sum = 0.0
    postprocess_latency_sum = 0.0
    success_count = 0
    failure_count = 0
    config_snapshot = _json_safe(vars(args))
    server_metadata = _fetch_server_metadata(host, args.port)
    if server_metadata:
        config_snapshot["server"] = server_metadata
        for key in ("backend", "vit_dtype", "llm_dtype", "dit_dtype",
                    "model_path", "trt_engine_path", "data_config",
                    "embodiment_tag", "denoising_steps"):
            if key in server_metadata:
                config_snapshot[key] = server_metadata[key]

    for warmup_index in range(args.warmup_runs):
        request_id = build_iteration_request_id(args.request_id, "warmup", warmup_index)
        payload = build_request_payload(args, request_id=request_id, rng=rng)
        try:
            send_request(
                host=host,
                port=args.port,
                ws_path=args.ws_path,
                payload=payload,
                timeout_ms=args.timeout_ms,
            )
        except Exception:
            # Warmup errors are ignored by design; run phase records persistent metrics.
            pass

    for run_index in range(args.measure_runs):
        request_id = build_iteration_request_id(args.request_id, "run", run_index)
        payload = build_request_payload(args, request_id=request_id, rng=rng)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            response, total_latency_ms = send_request(
                host=host,
                port=args.port,
                ws_path=args.ws_path,
                payload=payload,
                timeout_ms=args.timeout_ms,
            )
            body = response.json()
            result_code = int(body.get("resultCode", 1))
            actual_request_id = body.get("resultMap", {}).get("request_id", request_id)

            server_total_ms = _parse_latency_header(response, "X-Server-Total-Ms")
            pure_inference_ms = _parse_latency_header(response, "X-Pure-Inference-Ms")
            if np.isnan(pure_inference_ms):
                pure_inference_ms = _parse_latency_header(response, "X-Inference-Ms")
            pure_inference_ms = _resolve_nonzero_inference_ms(
                server_total_ms=server_total_ms,
                pure_inference_ms=pure_inference_ms,
                client_round_trip_ms=total_latency_ms,
            )
            breakdown = getattr(response, "latency_breakdown", {})
            preprocess_ms = _coerce_latency_ms(breakdown.get("preprocess_ms", 0.0))
            postprocess_ms = _coerce_latency_ms(breakdown.get("postprocess_ms", 0.0))
            network_latency_ms = round(max(total_latency_ms - pure_inference_ms, 0.0), 4)
            inference_proportion_pct = (
                round((pure_inference_ms / total_latency_ms) * 100.0, 2)
                if total_latency_ms > 0.0
                else 0.0
            )

            total_latency_sum += total_latency_ms
            infer_latency_sum += pure_inference_ms
            network_latency_sum += network_latency_ms
            preprocess_latency_sum += preprocess_ms
            postprocess_latency_sum += postprocess_ms
            if result_code == 0:
                success_count += 1
            else:
                failure_count += 1

            _append_jsonl_record(
                args.output_jsonl,
                {
                    "record_type": "run",
                    "timestamp": timestamp,
                    "config": config_snapshot,
                    "request_meta": {
                        "phase": "run",
                        "iteration": run_index,
                        "request_id": actual_request_id,
                        "device_id": args.device_id,
                        "http_status": int(getattr(response, "status_code", 0) or 0),
                        "result_code": result_code,
                    },
                    "latency": {
                        "total_latency_ms": round(total_latency_ms, 4),
                        "pure_inference_ms": round(pure_inference_ms, 4),
                        "preprocess_ms": preprocess_ms,
                        "postprocess_ms": postprocess_ms,
                        "network_latency_ms": round(network_latency_ms, 4),
                        "inference_proportion_pct": inference_proportion_pct,
                    },
                    "server_headers": {
                        "X-Server-Total-Ms": server_total_ms,
                        "X-Pure-Inference-Ms": pure_inference_ms,
                        "X-Inference-Ms": pure_inference_ms,
                        "X-Server-Overhead-Ms": _parse_latency_header(
                            response, "X-Server-Overhead-Ms"
                        ),
                    },
                    "response": {
                        "resultCode": result_code,
                        "errorMessage": _to_ascii_text(body.get("errorMessage", "")),
                        "resultMap": body.get("resultMap", {}),
                    },
                },
            )
        except Exception as exc:
            failure_count += 1
            _append_jsonl_record(
                args.output_jsonl,
                {
                    "record_type": "run",
                    "timestamp": timestamp,
                    "config": config_snapshot,
                    "request_meta": {
                        "phase": "run",
                        "iteration": run_index,
                        "request_id": request_id,
                        "device_id": args.device_id,
                        "http_status": 0,
                        "result_code": 1,
                    },
                    "latency": {
                        "total_latency_ms": 0.0,
                        "pure_inference_ms": 0.0,
                        "network_latency_ms": 0.0,
                        "inference_proportion_pct": 0.0,
                    },
                    "server_headers": {},
                    "response": {
                        "resultCode": 1,
                        "errorMessage": _to_ascii_text(exc),
                        "resultMap": {},
                    },
                },
            )

    run_count = max(args.measure_runs, 1)
    average_total = total_latency_sum / run_count
    average_inference = infer_latency_sum / run_count
    average_network = network_latency_sum / run_count
    average_preprocess = preprocess_latency_sum / run_count
    average_postprocess = postprocess_latency_sum / run_count
    average_proportion = (
        (infer_latency_sum / total_latency_sum) * 100.0 if total_latency_sum > 0.0 else 0.0
    )

    average_server_total = average_preprocess + average_inference + average_postprocess

    print("\n=== Average Latency ===")
    print(f"  Total (client round-trip):     {average_total:.4f} ms")
    print(f"  Server total:                  {average_server_total:.4f} ms")
    print(f"    Preprocess (CPU):            {average_preprocess:.4f} ms")
    print(f"    get_action (CUDA evt):       {average_inference:.4f} ms")
    print(f"    Postprocess (CPU):           {average_postprocess:.4f} ms")
    print(f"  Network:                       {average_network:.4f} ms")
    print(f"  Inference proportion:          {average_proportion:.2f}%")

    _append_jsonl_record(
        args.output_jsonl,
        {
            "record_type": "summary",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": config_snapshot,
            "summary_metrics": {
                "average_total_latency_ms": round(average_total, 4),
                "average_inference_latency_ms": round(average_inference, 4),
                "average_preprocess_ms": round(average_preprocess, 4),
                "average_postprocess_ms": round(average_postprocess, 4),
                "average_network_latency_ms": round(average_network, 4),
                "average_inference_proportion_pct": round(average_proportion, 2),
            },
            "run_count": int(args.measure_runs),
            "success_count": success_count,
            "failure_count": failure_count,
            "output_jsonl": str(args.output_jsonl),
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.role == "server":
        runtime = build_runtime(args)
        app = create_app(runtime=runtime, ws_path=args.ws_path)
        run_server(app, host=_host_for_role(args), port=args.port)
        return 0
    if args.role == "client":
        return run_client_benchmark(args)
    raise ValueError(f"Unsupported role: {args.role}")


if __name__ == "__main__":
    raise SystemExit(main())
