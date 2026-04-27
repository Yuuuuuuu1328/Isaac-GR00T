from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _ensure_repo_root_on_path()
_AISTUDIO_TASK_PROMPT = "Move to center the book in view. Do nothing if no book is present."


@dataclass(frozen=True)
class AistudioEnginePaths:
    left: str
    right: str


@dataclass(frozen=True)
class AistudioModelPaths:
    left: str
    right: str


@dataclass(frozen=True)
class AistudioPolicyRouter:
    left_policy: object
    right_policy: object

    def select_policy(self, joint_angles: list[float]) -> object:
        return self.left_policy if joint_angles[3] < 0 else self.right_policy


class StandardServiceAdapter:
    def __init__(self, *, policy: object, metadata: dict):
        self.policy = policy
        self._metadata = metadata

    def predict(self, observation: dict) -> dict:
        return self.policy.get_action(observation)

    def modality_config(self) -> dict:
        return self.policy.get_modality_config()

    def metadata(self) -> dict:
        return dict(self._metadata)

    def ping(self) -> dict:
        return {"status": "ok"}

    def echo(self, payload: dict) -> dict:
        return payload


@dataclass
class AistudioServiceAdapter:
    left_policy: object
    right_policy: object
    task_prompt: str = _AISTUDIO_TASK_PROMPT
    metadata_payload: dict | None = None

    def __post_init__(self):
        self.router = AistudioPolicyRouter(self.left_policy, self.right_policy)

    def metadata(self) -> dict:
        return dict(self.metadata_payload or {"service_mode": "aistudio"})

    def ping(self) -> dict:
        return {"status": "ok"}

    def echo(self, payload: dict) -> dict:
        return payload

    def predict(self, payload: dict) -> dict:
        parsed: dict[str, Any] | None = None
        try:
            parsed = parse_aistudio_query(payload)
            frame_rgb = decode_aistudio_framebuffer(
                parsed["framebuffer"],
                parsed.get("framebuffer_size", 0),
            )
            batch, robot_state = build_aistudio_batch(
                frame_rgb=frame_rgb,
                joint_angles=parsed["joint_angles"],
                task_prompt=self.task_prompt,
            )
            policy = self.router.select_policy(parsed["joint_angles"])
            previous_actions = policy.get_action(batch)["action.single_arm"]
            action_string, is_success = postprocess_aistudio_actions(previous_actions, robot_state)
            result_map = {
                "action_sequence": action_string,
                "joint_angles": json.dumps(parsed["joint_angles"]),
                "predicted_coords_2d": json.dumps(parsed["predicted_coords_2d"]),
                "history_angles": json.dumps(parsed["history_angles"]),
                "is_success": is_success,
                "error": "",
                "key_infos": "key_infos",
                "device_id": parsed.get("device_id", ""),
                "request_id": parsed.get("request_id", ""),
            }
            return build_aistudio_response(result_map=result_map)
        except Exception as exc:
            return build_aistudio_error_response(
                error_message=str(exc),
                request_id="" if parsed is None else str(parsed.get("request_id", "")),
                device_id="" if parsed is None else str(parsed.get("device_id", "")),
                joint_angles=[] if parsed is None else parsed.get("joint_angles", []),
                predicted_coords_2d=[] if parsed is None else parsed.get("predicted_coords_2d", []),
                history_angles=[] if parsed is None else parsed.get("history_angles", []),
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ANT 在线推理服务入口")
    parser.add_argument("--role", type=str, choices=["server", "client"], required=True)
    parser.add_argument("--service-mode", type=str, choices=["standard", "aistudio"], default="standard")
    parser.add_argument("--transport", type=str, choices=["http", "zmq"], default="http")
    parser.add_argument("--backend", type=str, choices=["pytorch", "tensorrt"], default="pytorch")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B",
    )
    parser.add_argument("--data-config", type=str, default="fourier_gr1_arms_only")
    parser.add_argument("--embodiment-tag", type=str, default="gr1")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-token", type=str, default=None)
    parser.add_argument("--request-timeout-ms", type=int, default=15000)
    parser.add_argument("--request-json", type=str, default="")
    parser.add_argument("--request-file", type=str, default="")
    parser.add_argument("--latency-endpoint", type=str, choices=["act", "ping", "echo"], default="act")
    parser.add_argument("--trt-engine-path", type=str, default=str(_REPO_ROOT / "gr00t_engine_fp16"))
    parser.add_argument("--vit-dtype", type=str, choices=["fp16", "fp8", "int8"], default="fp16")
    parser.add_argument("--llm-dtype", type=str, choices=["fp16", "nvfp4", "fp8", "int8"], default="fp16")
    parser.add_argument("--dit-dtype", type=str, choices=["fp16", "fp8", "int8"], default="fp16")
    parser.add_argument("--left-model-path", type=str, default="")
    parser.add_argument("--right-model-path", type=str, default="")
    parser.add_argument("--left-trt-engine-path", type=str, default="")
    parser.add_argument("--right-trt-engine-path", type=str, default="")
    parser.add_argument("--aistudio-data-config", type=str, default="new_interaction_group")
    parser.add_argument("--aistudio-embodiment-tag", type=str, default="new_embodiment")
    parser.add_argument("--aistudio-task-prompt", type=str, default=_AISTUDIO_TASK_PROMPT)
    return parser


def _resolve_repo_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def resolve_aistudio_model_paths(args: argparse.Namespace) -> AistudioModelPaths:
    model_root = Path(args.model_path).expanduser()
    left_path = (
        str(Path(args.left_model_path).expanduser())
        if args.left_model_path
        else str(model_root / "left_hand_v2_1223")
    )
    right_path = (
        str(Path(args.right_model_path).expanduser())
        if args.right_model_path
        else str(model_root / "right_hand_v2_1223")
    )
    return AistudioModelPaths(left=left_path, right=right_path)


def resolve_aistudio_engine_paths(args: argparse.Namespace) -> AistudioEnginePaths:
    default_path = _resolve_repo_path(args.trt_engine_path)
    left_path = _resolve_repo_path(args.left_trt_engine_path) if args.left_trt_engine_path else default_path
    right_path = (
        _resolve_repo_path(args.right_trt_engine_path) if args.right_trt_engine_path else default_path
    )
    return AistudioEnginePaths(left=left_path, right=right_path)


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.service_mode == "aistudio" and args.transport == "zmq":
        raise ValueError("Aistudio mode only supports HTTP transport")
    if args.request_json and args.request_file:
        raise ValueError("Only one of --request-json or --request-file can be specified")
    return args


def _parse_list_like(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def parse_aistudio_query(payload: dict) -> dict:
    if "query" not in payload:
        raise ValueError("Missing query field")

    raw_query = payload["query"]
    if isinstance(raw_query, bytes):
        raw_query = raw_query.decode()
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


def build_aistudio_batch(
    *,
    frame_rgb: np.ndarray,
    joint_angles: list[float],
    task_prompt: str = _AISTUDIO_TASK_PROMPT,
) -> tuple[dict, np.ndarray]:
    robot_state = np.array([*joint_angles[:5], 0], dtype=np.float32)
    batch = {
        "video.ego_view": frame_rgb[np.newaxis, ...],
        "annotation.task_index": [task_prompt],
        "state.single_arm": robot_state[np.newaxis, ...],
    }
    return batch, robot_state


def postprocess_aistudio_actions(
    previous_actions: np.ndarray,
    robot_state: np.ndarray,
) -> tuple[str, bool]:
    integrated_actions = np.cumsum(previous_actions[:, :6], axis=0) + robot_state[:6]
    rounded_actions = [[round(float(x), 4) for x in row] for row in integrated_actions[:14]]
    return json.dumps(rounded_actions), bool(integrated_actions.size >= 5)


def decode_aistudio_framebuffer(framebuffer: list[int], framebuffer_size: int = 0) -> np.ndarray:
    del framebuffer_size

    import cv2

    jpeg_array = np.array(framebuffer, dtype=np.uint8)
    framebuffer_data = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
    if framebuffer_data is None:
        raise ValueError("Failed to decode framebuffer")
    frame_bgr = cv2.resize(framebuffer_data, (480, 640))
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _numpy_json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _estimate_payload_size_bytes(payload: Any) -> int:
    return len(json.dumps(payload, default=_numpy_json_default).encode("utf-8"))


def _timing_meta(start_ns: int, *, payload: Any = None) -> dict:
    end_ns = time.perf_counter_ns()
    return {
        "server_received_ns": start_ns,
        "server_sent_ns": end_ns,
        "server_process_ms": round((end_ns - start_ns) / 1_000_000, 4),
        "payload_size_bytes": 0 if payload is None else _estimate_payload_size_bytes(payload),
    }


def _authorize_http_request(request, api_token: str | None) -> None:
    if api_token is None:
        return

    from fastapi import HTTPException

    bearer_token = None
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:]

    request_token = request.headers.get("x-api-token") or bearer_token
    if request_token != api_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


def create_http_app(*, service_mode: str, adapter, api_token: str | None = None):
    import json_numpy
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    json_numpy.patch()
    app = FastAPI(title="ANT Online Inference Service")

    if service_mode == "standard":

        @app.post("/v1/act")
        @app.post("/act")
        def _predict(payload: dict, request: Request):
            _authorize_http_request(request, api_token)
            if "observation" not in payload:
                raise HTTPException(status_code=400, detail="Missing observation field")
            start_ns = time.perf_counter_ns()
            action = adapter.predict(payload["observation"])
            return JSONResponse(
                content={
                    "action": action,
                    "meta": {
                        **adapter.metadata(),
                        **_timing_meta(start_ns, payload=payload),
                    },
                }
            )

        @app.get("/health")
        @app.get("/v1/health/live")
        def _health_live(request: Request):
            _authorize_http_request(request, api_token)
            return {"status": "ok", "service_mode": "standard"}

        @app.get("/v1/health/ready")
        def _health_ready(request: Request):
            _authorize_http_request(request, api_token)
            return {"status": "ok", "ready": True}

        @app.get("/v1/metadata")
        def _metadata(request: Request):
            _authorize_http_request(request, api_token)
            return JSONResponse(content=adapter.metadata())

        @app.get("/v1/latency/ping")
        def _ping(request: Request):
            _authorize_http_request(request, api_token)
            start_ns = time.perf_counter_ns()
            return JSONResponse(content={**adapter.ping(), **_timing_meta(start_ns)})

        @app.post("/v1/latency/echo")
        def _echo(payload: dict, request: Request):
            _authorize_http_request(request, api_token)
            start_ns = time.perf_counter_ns()
            return JSONResponse(
                content={
                    "echo": adapter.echo(payload),
                    "meta": _timing_meta(start_ns, payload=payload),
                }
            )

        return app

    if service_mode == "aistudio":

        @app.post("/v1/aistudio/predict")
        def _predict(payload: dict, request: Request):
            _authorize_http_request(request, api_token)
            return JSONResponse(content=adapter.predict(payload))

        @app.get("/v1/aistudio/health")
        def _health(request: Request):
            _authorize_http_request(request, api_token)
            return {"status": "ok", "service_mode": "aistudio"}

        @app.get("/v1/aistudio/latency/ping")
        def _ping(request: Request):
            _authorize_http_request(request, api_token)
            start_ns = time.perf_counter_ns()
            return JSONResponse(content={**adapter.ping(), **_timing_meta(start_ns)})

        @app.post("/v1/aistudio/latency/echo")
        def _echo(payload: dict, request: Request):
            _authorize_http_request(request, api_token)
            start_ns = time.perf_counter_ns()
            return JSONResponse(
                content={
                    "echo": adapter.echo(payload),
                    "meta": _timing_meta(start_ns, payload=payload),
                }
            )

        return app

    raise ValueError(f"Unsupported service mode: {service_mode}")


def register_standard_zmq_endpoints(server, adapter) -> None:
    server.register_endpoint("get_action", adapter.predict)
    server.register_endpoint("get_modality_config", adapter.modality_config, requires_input=False)
    server.register_endpoint("get_server_info", adapter.metadata, requires_input=False)
    server.register_endpoint("echo", adapter.echo)


def _load_data_config_by_name(name: str):
    from gr00t.experiment.data_config import DATA_CONFIG_MAP, load_data_config

    try:
        return load_data_config(name)
    except Exception as exc:
        available = ", ".join(sorted(DATA_CONFIG_MAP.keys()))
        raise ValueError(
            f"Unknown data config '{name}'. Available configs in this checkout: {available}"
        ) from exc


def _build_policy(
    *,
    model_path: str,
    data_config_name: str,
    embodiment_tag: str,
    denoising_steps: int,
):
    import torch
    from gr00t.model.policy import Gr00tPolicy

    data_config = _load_data_config_by_name(data_config_name)
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=denoising_steps,
        device=device,
    )
    return policy, modality_config


def _setup_tensorrt(policy: object, *, engine_path: str, args: argparse.Namespace) -> str:
    import torch
    from deployment_scripts.trt_model_forward import setup_tensorrt_engines

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT online service requires CUDA to be available")

    resolved_path = _resolve_repo_path(engine_path)
    if not Path(resolved_path).exists():
        raise FileNotFoundError(f"TensorRT engine directory does not exist: {resolved_path}")

    setup_tensorrt_engines(
        policy,
        resolved_path,
        vit_dtype=args.vit_dtype,
        llm_dtype=args.llm_dtype,
        dit_dtype=args.dit_dtype,
    )
    return resolved_path


def build_service_adapter(args: argparse.Namespace):
    if args.service_mode == "standard":
        policy, modality_config = _build_policy(
            model_path=args.model_path,
            data_config_name=args.data_config,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )
        trt_engine_path = None
        if args.backend == "tensorrt":
            trt_engine_path = _setup_tensorrt(policy, engine_path=args.trt_engine_path, args=args)
        metadata = {
            "service_mode": "standard",
            "backend": args.backend,
            "model_path": args.model_path,
            "data_config": args.data_config,
            "embodiment_tag": args.embodiment_tag,
            "transport": args.transport,
            "trt_engine_path": trt_engine_path,
            "modality_keys": sorted(modality_config.keys()),
        }
        return StandardServiceAdapter(policy=policy, metadata=metadata)

    if args.service_mode == "aistudio":
        model_paths = resolve_aistudio_model_paths(args)
        engine_paths = resolve_aistudio_engine_paths(args)
        left_policy, _ = _build_policy(
            model_path=model_paths.left,
            data_config_name=args.aistudio_data_config,
            embodiment_tag=args.aistudio_embodiment_tag,
            denoising_steps=args.denoising_steps,
        )
        right_policy, _ = _build_policy(
            model_path=model_paths.right,
            data_config_name=args.aistudio_data_config,
            embodiment_tag=args.aistudio_embodiment_tag,
            denoising_steps=args.denoising_steps,
        )
        left_engine_path = None
        right_engine_path = None
        if args.backend == "tensorrt":
            left_engine_path = _setup_tensorrt(left_policy, engine_path=engine_paths.left, args=args)
            right_engine_path = _setup_tensorrt(right_policy, engine_path=engine_paths.right, args=args)
        return AistudioServiceAdapter(
            left_policy=left_policy,
            right_policy=right_policy,
            task_prompt=args.aistudio_task_prompt,
            metadata_payload={
                "service_mode": "aistudio",
                "backend": args.backend,
                "transport": args.transport,
                "left_model_path": model_paths.left,
                "right_model_path": model_paths.right,
                "aistudio_data_config": args.aistudio_data_config,
                "aistudio_embodiment_tag": args.aistudio_embodiment_tag,
                "left_trt_engine_path": left_engine_path,
                "right_trt_engine_path": right_engine_path,
            },
        )

    raise ValueError(f"Unsupported service mode: {args.service_mode}")


def create_standard_zmq_server(adapter, *, host: str, port: int, api_token: str | None):
    from gr00t.eval.service import BaseInferenceServer

    server = BaseInferenceServer(host=host, port=port, api_token=api_token)
    register_standard_zmq_endpoints(server, adapter)
    return server


def run_http_server(app, *, host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required to run the HTTP server. Install uvicorn to serve ANT online inference over HTTP."
        ) from exc

    uvicorn.run(app, host=host, port=port)


def run_zmq_server(server) -> None:
    server.run()


def _load_request_payload(args: argparse.Namespace, *, required: bool) -> dict:
    if args.request_json:
        return json.loads(args.request_json)
    if args.request_file:
        request_path = Path(args.request_file).expanduser()
        return json.loads(request_path.read_text(encoding="utf-8"))
    if required:
        raise ValueError("Client requests require --request-json or --request-file")
    return {}


def _authorization_headers(api_token: str | None) -> dict[str, str]:
    if api_token is None:
        return {}
    return {
        "Authorization": f"Bearer {api_token}",
        "X-API-Token": api_token,
    }


def _print_response(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_numpy_json_default))


def _run_standard_http_client(args: argparse.Namespace) -> dict:
    import json_numpy
    import requests

    json_numpy.patch()
    base_url = f"http://{args.host}:{args.port}"
    headers = {"Content-Type": "application/json", **_authorization_headers(args.api_token)}

    if args.latency_endpoint == "ping":
        start_ns = time.perf_counter_ns()
        response = requests.get(
            f"{base_url}/v1/latency/ping",
            headers=headers,
            timeout=args.request_timeout_ms / 1000,
        )
    elif args.latency_endpoint == "echo":
        payload = _load_request_payload(args, required=False)
        start_ns = time.perf_counter_ns()
        response = requests.post(
            f"{base_url}/v1/latency/echo",
            headers=headers,
            data=json.dumps(payload, default=_numpy_json_default),
            timeout=args.request_timeout_ms / 1000,
        )
    else:
        payload = _load_request_payload(args, required=True)
        start_ns = time.perf_counter_ns()
        response = requests.post(
            f"{base_url}/v1/act",
            headers=headers,
            data=json.dumps(payload, default=_numpy_json_default),
            timeout=args.request_timeout_ms / 1000,
        )

    response.raise_for_status()
    result = response.json()
    result["client_rtt_ms"] = round((time.perf_counter_ns() - start_ns) / 1_000_000, 4)
    return result


def _run_aistudio_http_client(args: argparse.Namespace) -> dict:
    import requests

    base_url = f"http://{args.host}:{args.port}"
    headers = {"Content-Type": "application/json", **_authorization_headers(args.api_token)}

    if args.latency_endpoint == "ping":
        start_ns = time.perf_counter_ns()
        response = requests.get(
            f"{base_url}/v1/aistudio/latency/ping",
            headers=headers,
            timeout=args.request_timeout_ms / 1000,
        )
    elif args.latency_endpoint == "echo":
        payload = _load_request_payload(args, required=False)
        start_ns = time.perf_counter_ns()
        response = requests.post(
            f"{base_url}/v1/aistudio/latency/echo",
            headers=headers,
            data=json.dumps(payload),
            timeout=args.request_timeout_ms / 1000,
        )
    else:
        payload = _load_request_payload(args, required=True)
        start_ns = time.perf_counter_ns()
        response = requests.post(
            f"{base_url}/v1/aistudio/predict",
            headers=headers,
            data=json.dumps(payload),
            timeout=args.request_timeout_ms / 1000,
        )

    response.raise_for_status()
    result = response.json()
    result["client_rtt_ms"] = round((time.perf_counter_ns() - start_ns) / 1_000_000, 4)
    return result


def _run_standard_zmq_client(args: argparse.Namespace) -> dict:
    from gr00t.eval.service import BaseInferenceClient

    client = BaseInferenceClient(
        host=args.host,
        port=args.port,
        timeout_ms=args.request_timeout_ms,
        api_token=args.api_token,
    )
    try:
        if args.latency_endpoint == "ping":
            start_ns = time.perf_counter_ns()
            result = client.call_endpoint("ping", requires_input=False)
        elif args.latency_endpoint == "echo":
            payload = _load_request_payload(args, required=False)
            start_ns = time.perf_counter_ns()
            result = client.call_endpoint("echo", payload)
        else:
            payload = _load_request_payload(args, required=True)
            if "observation" not in payload:
                raise ValueError("Standard client payload must contain an observation field")
            start_ns = time.perf_counter_ns()
            result = client.call_endpoint("get_action", payload["observation"])
        result["client_rtt_ms"] = round((time.perf_counter_ns() - start_ns) / 1_000_000, 4)
        return result
    finally:
        client.socket.close()
        client.context.term()


def run_client(args: argparse.Namespace) -> dict:
    if args.service_mode == "standard" and args.transport == "http":
        return _run_standard_http_client(args)
    if args.service_mode == "standard" and args.transport == "zmq":
        return _run_standard_zmq_client(args)
    if args.service_mode == "aistudio" and args.transport == "http":
        return _run_aistudio_http_client(args)
    raise ValueError(
        f"Unsupported client combination: service_mode={args.service_mode}, transport={args.transport}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = validate_args(parser.parse_args(argv))

    if args.role == "server":
        adapter = build_service_adapter(args)
        if args.transport == "http":
            app = create_http_app(
                service_mode=args.service_mode,
                adapter=adapter,
                api_token=args.api_token,
            )
            run_http_server(app, host=args.host, port=args.port)
            return 0

        server = create_standard_zmq_server(
            adapter,
            host=args.host,
            port=args.port,
            api_token=args.api_token,
        )
        run_zmq_server(server)
        return 0

    if args.role == "client":
        _print_response(run_client(args))
        return 0

    raise ValueError(f"Unsupported role: {args.role}")


if __name__ == "__main__":
    raise SystemExit(main())
