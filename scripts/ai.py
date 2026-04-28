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
from deployment_scripts.ant.profile_utils import measure_cuda_time_ms, sync_cuda_if_needed
try:
    from websocket import create_connection as websocket_create_connection
except ModuleNotFoundError:  # pragma: no cover - handled by runtime/tests
    websocket_create_connection = None


DEFAULT_MODEL_PATH = "/home/ningjiang/nj/ckpt/gr00t/left_hand_v2_1223"
DEFAULT_OSSFS_WORKSPACE = "/home/ningjiang/nj/ossfs/node_59823209/workspace"
DEFAULT_TRT_ENGINE_PATH = (
    "/home/ningjiang/nj/Isaac-GR00T/gr00t_engine_int8_new09"
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
    server_latency = envelope.get("latency", {})
    if not isinstance(server_latency, dict):
        server_latency = {}
    response = SimpleNamespace(
        status_code=int(envelope.get("status_code", 200) or 200),
        server_latency=server_latency,
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



def _install_timing_hooks(policy) -> None:
    import torch

    _orig_forward = policy._get_action_from_normalized_input
    _orig_transforms = policy.apply_transforms

    policy._detail_timing = {}

    def _timed_forward(normalized_input):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = _orig_forward(normalized_input)
        end.record()
        policy._timing_fwd_start = start
        policy._timing_fwd_end = end
        return result

    def _timed_transforms(obs):
        t0 = time.perf_counter_ns()
        result = _orig_transforms(obs)
        policy._timing_transform_ns = time.perf_counter_ns() - t0
        return result

    policy._get_action_from_normalized_input = _timed_forward
    policy.apply_transforms = _timed_transforms
    policy._timing_fwd_start = None
    policy._timing_fwd_end = None
    policy._timing_transform_ns = 0

    _install_transform_timing_hooks(policy)
    _install_backbone_timing_hooks(policy)
    _install_action_head_timing_hooks(policy)


def _install_transform_timing_hooks(policy) -> None:
    transforms = getattr(policy, "_modality_transform", None)
    if transforms is None or not hasattr(transforms, "transforms"):
        return

    for i, transform in enumerate(transforms.transforms):
        _orig_apply = transform.apply
        cls_name = type(transform).__name__

        def _make_timed(orig, name, idx):
            def _timed(data):
                t0 = time.perf_counter_ns()
                result = orig(data)
                elapsed_ns = time.perf_counter_ns() - t0
                policy._detail_timing[f"transform_{idx}_{name}_ns"] = elapsed_ns
                return result
            return _timed

        object.__setattr__(transform, "apply", _make_timed(_orig_apply, cls_name, i))


def _install_backbone_timing_hooks(policy) -> None:
    import torch

    model = getattr(policy, "model", None)
    if model is None:
        return
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return

    _orig_backbone_forward = backbone.forward

    is_trt = hasattr(backbone, "vit_engine")

    if is_trt:
        def _timed_backbone_forward(vl_input):
            detail = policy._detail_timing

            _orig_vit_fwd = backbone.vit_engine.forward
            _orig_llm_fwd = backbone.llm_engine.forward

            vit_start = torch.cuda.Event(enable_timing=True)
            vit_end = torch.cuda.Event(enable_timing=True)
            llm_start = torch.cuda.Event(enable_timing=True)
            llm_end = torch.cuda.Event(enable_timing=True)

            def _timed_vit(*args, **kwargs):
                vit_start.record()
                res = _orig_vit_fwd(*args, **kwargs)
                vit_end.record()
                return res

            def _timed_llm(*args, **kwargs):
                llm_start.record()
                res = _orig_llm_fwd(*args, **kwargs)
                llm_end.record()
                return res

            backbone.vit_engine.forward = _timed_vit
            backbone.llm_engine.forward = _timed_llm
            try:
                result = _orig_backbone_forward(vl_input)
            finally:
                backbone.vit_engine.forward = _orig_vit_fwd
                backbone.llm_engine.forward = _orig_llm_fwd

            detail["_vit_start"] = vit_start
            detail["_vit_end"] = vit_end
            detail["_llm_start"] = llm_start
            detail["_llm_end"] = llm_end
            return result
    else:
        def _timed_backbone_forward(vl_input):
            detail = policy._detail_timing

            orig_forward_eagle = backbone.forward_eagle

            def _timed_forward_eagle(vl_in):
                eagle_model = backbone.eagle_model
                _orig_eagle_call = eagle_model.forward

                vit_start = torch.cuda.Event(enable_timing=True)
                vit_end = torch.cuda.Event(enable_timing=True)
                llm_start = torch.cuda.Event(enable_timing=True)
                llm_end = torch.cuda.Event(enable_timing=True)

                def _timed_eagle_call(**kwargs):
                    vision_model = eagle_model.vision_model
                    language_model = eagle_model.language_model

                    _orig_vision_fwd = vision_model.forward
                    _orig_language_fwd = language_model.forward

                    def _timed_vision(*a, **kw):
                        vit_start.record()
                        res = _orig_vision_fwd(*a, **kw)
                        vit_end.record()
                        return res

                    def _timed_language(*a, **kw):
                        llm_start.record()
                        res = _orig_language_fwd(*a, **kw)
                        llm_end.record()
                        return res

                    vision_model.forward = _timed_vision
                    language_model.forward = _timed_language
                    try:
                        res = _orig_eagle_call(**kwargs)
                    finally:
                        vision_model.forward = _orig_vision_fwd
                        language_model.forward = _orig_language_fwd
                    return res

                eagle_model.forward = _timed_eagle_call
                try:
                    result = orig_forward_eagle(vl_in)
                finally:
                    eagle_model.forward = _orig_eagle_call

                detail["_vit_start"] = vit_start
                detail["_vit_end"] = vit_end
                detail["_llm_start"] = llm_start
                detail["_llm_end"] = llm_end
                return result

            backbone.forward_eagle = _timed_forward_eagle
            try:
                result = _orig_backbone_forward(vl_input)
            finally:
                backbone.forward_eagle = orig_forward_eagle
            return result

    backbone.forward = _timed_backbone_forward



def _install_action_head_timing_hooks(policy) -> None:
    import torch

    model = getattr(policy, "model", None)
    if model is None:
        return
    action_head = getattr(model, "action_head", None)
    if action_head is None:
        return

    _orig_get_action = action_head.get_action
    is_trt = hasattr(action_head, "DiT_engine")

    if is_trt:
        def _timed_get_action(backbone_output, action_input):
            detail = policy._detail_timing

            for key in ("features_process", "state_encoder", "action_encoder",
                        "dit_block", "action_decoder"):
                detail[f"_{key}_events"] = []

            _orig_vlln = action_head.vlln_vl_self_attention_engine.forward
            _orig_state = action_head.state_encoder_engine.forward
            _orig_ae = action_head.action_encoder_engine.forward
            _orig_dit = action_head.DiT_engine.forward
            _orig_ad = action_head.action_decoder_engine.forward

            def _wrap(orig, name):
                def _timed(*args, **kwargs):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    res = orig(*args, **kwargs)
                    e.record()
                    detail[f"_{name}_events"].append((s, e))
                    return res
                return _timed

            action_head.vlln_vl_self_attention_engine.forward = _wrap(_orig_vlln, "features_process")
            action_head.state_encoder_engine.forward = _wrap(_orig_state, "state_encoder")
            action_head.action_encoder_engine.forward = _wrap(_orig_ae, "action_encoder")
            action_head.DiT_engine.forward = _wrap(_orig_dit, "dit_block")
            action_head.action_decoder_engine.forward = _wrap(_orig_ad, "action_decoder")

            try:
                result = _orig_get_action(backbone_output, action_input)
            finally:
                action_head.vlln_vl_self_attention_engine.forward = _orig_vlln
                action_head.state_encoder_engine.forward = _orig_state
                action_head.action_encoder_engine.forward = _orig_ae
                action_head.DiT_engine.forward = _orig_dit
                action_head.action_decoder_engine.forward = _orig_ad

            return result
    else:
        def _timed_get_action(backbone_output, action_input):
            detail = policy._detail_timing

            for key in ("features_process", "state_encoder", "action_encoder",
                        "dit_block", "action_decoder"):
                detail[f"_{key}_events"] = []

            _orig_process = action_head.process_backbone_output
            _orig_state_enc = action_head.state_encoder.forward
            _orig_action_enc = action_head.action_encoder.forward
            _orig_dit_fwd = action_head.model.forward
            _orig_action_dec = action_head.action_decoder.forward

            def _wrap(orig, name):
                def _timed(*args, **kwargs):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    res = orig(*args, **kwargs)
                    e.record()
                    detail[f"_{name}_events"].append((s, e))
                    return res
                return _timed

            action_head.process_backbone_output = _wrap(_orig_process, "features_process")
            action_head.state_encoder.forward = _wrap(_orig_state_enc, "state_encoder")
            action_head.action_encoder.forward = _wrap(_orig_action_enc, "action_encoder")
            action_head.model.forward = _wrap(_orig_dit_fwd, "dit_block")
            action_head.action_decoder.forward = _wrap(_orig_action_dec, "action_decoder")

            try:
                result = _orig_get_action(backbone_output, action_input)
            finally:
                action_head.process_backbone_output = _orig_process
                action_head.state_encoder.forward = _orig_state_enc
                action_head.action_encoder.forward = _orig_action_enc
                action_head.model.forward = _orig_dit_fwd
                action_head.action_decoder.forward = _orig_action_dec

            return result

    action_head.get_action = _timed_get_action


def _collect_detail_timing(policy) -> dict[str, float]:
    import torch

    detail = getattr(policy, "_detail_timing", {})
    result: dict[str, float] = {}

    torch.cuda.synchronize()

    # Transform sub-timings
    state_action_ns = 0
    image_preprocess_ns = 0
    text_tokenization_ns = 0
    for key, val in detail.items():
        if not key.startswith("transform_") or not key.endswith("_ns"):
            continue
        name_lower = key.lower()
        if "stateaction" in name_lower or "stateactiontotensor" in name_lower:
            state_action_ns += val
        elif "video" in name_lower or "videotensor" in name_lower or "videotransform" in name_lower:
            image_preprocess_ns += val
        elif "concat" in name_lower:
            pass
        else:
            text_tokenization_ns += val

    result["state_action_transform_ms"] = round(state_action_ns / 1_000_000.0, 4)
    result["image_preprocess_ms"] = round(image_preprocess_ns / 1_000_000.0, 4)
    result["text_tokenization_ms"] = round(text_tokenization_ns / 1_000_000.0, 4)

    # Backbone sub-timings (Vision Encoder, LLM)
    if "_vit_start" in detail and "_vit_end" in detail:
        try:
            result["vision_encoder_ms"] = round(
                float(detail["_vit_start"].elapsed_time(detail["_vit_end"])), 4
            )
        except RuntimeError:
            result["vision_encoder_ms"] = 0.0
    else:
        result["vision_encoder_ms"] = 0.0

    if "_llm_start" in detail and "_llm_end" in detail:
        try:
            result["llm_ms"] = round(
                float(detail["_llm_start"].elapsed_time(detail["_llm_end"])), 4
            )
        except RuntimeError:
            result["llm_ms"] = 0.0
    else:
        result["llm_ms"] = 0.0

    # Action head sub-timings
    for comp_name in ("features_process", "state_encoder", "action_encoder",
                      "dit_block", "action_decoder"):
        events = detail.get(f"_{comp_name}_events", [])
        total_ms = 0.0
        for s, e in events:
            try:
                total_ms += float(s.elapsed_time(e))
            except RuntimeError:
                pass
        result[f"{comp_name}_ms"] = round(total_ms, 4)

    return result


def _empty_timing() -> dict[str, float]:
    return {
        "preprocess_ms": 0.0,
        "get_action_ms": 0.0,
        "model_forward_ms": 0.0,
        "transform_ms": 0.0,
        "postprocess_ms": 0.0,
        "state_action_transform_ms": 0.0,
        "image_preprocess_ms": 0.0,
        "text_tokenization_ms": 0.0,
        "vision_encoder_ms": 0.0,
        "llm_ms": 0.0,
        "features_process_ms": 0.0,
        "state_encoder_ms": 0.0,
        "action_encoder_ms": 0.0,
        "dit_block_ms": 0.0,
        "action_decoder_ms": 0.0,
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
    ) -> tuple[np.ndarray, dict[str, float]]:
        if hasattr(self.policy, "_detail_timing"):
            self.policy._detail_timing.clear()

        action_result, get_action_ms = measure_cuda_time_ms(
            lambda: self.policy.get_action(model_batch)
        )

        model_forward_ms = 0.0
        if (
            getattr(self.policy, "_timing_fwd_start", None) is not None
            and getattr(self.policy, "_timing_fwd_end", None) is not None
        ):
            model_forward_ms = round(
                float(self.policy._timing_fwd_start.elapsed_time(self.policy._timing_fwd_end)), 4
            )
        transform_ms = round(
            getattr(self.policy, "_timing_transform_ns", 0) / 1_000_000.0, 4
        )

        detail = _collect_detail_timing(self.policy)

        timing = {
            "get_action_ms": get_action_ms,
            "model_forward_ms": model_forward_ms,
            "transform_ms": transform_ms,
        }
        timing.update(detail)
        return action_result["action.single_arm"], timing

    def predict_with_timing(self, payload: dict) -> tuple[dict, dict[str, Any]]:
        import traceback as _tb

        parsed: dict[str, Any] | None = None
        timing = _empty_timing()
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
            timing["preprocess_ms"] = round(
                (time.perf_counter_ns() - t_pre_start) / 1_000_000.0, 4
            )

            t_infer_start = time.perf_counter_ns()
            try:
                previous_actions, infer_timing = self._predict_previous_actions(model_batch=batch)
                timing.update(infer_timing)
            except Exception as infer_exc:
                infer_wall_ms = round(
                    (time.perf_counter_ns() - t_infer_start) / 1_000_000.0, 4
                )
                timing["get_action_ms"] = infer_wall_ms
                print(f"[predict_with_timing] Inference error: {infer_exc}")
                _tb.print_exc()
                raise

            t_post_start = time.perf_counter_ns()
            action_string, is_success = postprocess_actions(
                previous_actions,
                robot_state,
                response_action_horizon=self.response_action_horizon,
            )
            timing["postprocess_ms"] = round(
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
            return build_aistudio_response(result_map=result_map), timing
        except Exception as exc:
            print(f"[predict_with_timing] Error: {exc}")
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
                timing,
            )


def create_app(*, runtime, ws_path: str = DEFAULT_WS_PATH):
    ws_path = ws_path if ws_path.startswith("/") else f"/{ws_path}"
    app = FastAPI(title="Aistudio WebSocket e2e Service")

    @app.websocket(ws_path)
    async def ws_predict(websocket: WebSocket):
        await websocket.accept()
        while True:
            try:
                raw_payload = await websocket.receive_text()

                request_start_ns = time.perf_counter_ns()
                try:
                    payload = json.loads(raw_payload)
                    response_body, timing = runtime.predict_with_timing(payload)
                except Exception as exc:
                    response_body = build_aistudio_error_response(error_message=str(exc))
                    timing = _empty_timing()

                server_total_ms = round(
                    (time.perf_counter_ns() - request_start_ns) / 1_000_000.0, 4
                )
                preprocess_ms = _coerce_latency_ms(timing.get("preprocess_ms", 0.0))
                get_action_ms = _coerce_latency_ms(timing.get("get_action_ms", 0.0))
                model_forward_ms = _coerce_latency_ms(timing.get("model_forward_ms", 0.0))
                transform_ms = _coerce_latency_ms(timing.get("transform_ms", 0.0))
                postprocess_ms = _coerce_latency_ms(timing.get("postprocess_ms", 0.0))
                server_overhead_ms = max(
                    round(server_total_ms - preprocess_ms - get_action_ms - postprocess_ms, 4),
                    0.0,
                )

                state_action_transform_ms = _coerce_latency_ms(timing.get("state_action_transform_ms", 0.0))
                image_preprocess_ms = _coerce_latency_ms(timing.get("image_preprocess_ms", 0.0))
                text_tokenization_ms = _coerce_latency_ms(timing.get("text_tokenization_ms", 0.0))
                vision_encoder_ms = _coerce_latency_ms(timing.get("vision_encoder_ms", 0.0))
                llm_ms = _coerce_latency_ms(timing.get("llm_ms", 0.0))
                features_process_ms = _coerce_latency_ms(timing.get("features_process_ms", 0.0))
                state_encoder_ms = _coerce_latency_ms(timing.get("state_encoder_ms", 0.0))
                action_encoder_ms = _coerce_latency_ms(timing.get("action_encoder_ms", 0.0))
                dit_block_ms = _coerce_latency_ms(timing.get("dit_block_ms", 0.0))
                action_decoder_ms = _coerce_latency_ms(timing.get("action_decoder_ms", 0.0))

                await websocket.send_json({
                    "status_code": 200,
                    "result": response_body,
                    "latency": {
                        "server_total_ms": _coerce_latency_ms(server_total_ms),
                        "preprocess_ms": preprocess_ms,
                        "get_action_ms": get_action_ms,
                        "model_forward_ms": model_forward_ms,
                        "transform_ms": transform_ms,
                        "postprocess_ms": postprocess_ms,
                        "server_overhead_ms": _coerce_latency_ms(server_overhead_ms),
                        "state_action_transform_ms": state_action_transform_ms,
                        "image_preprocess_ms": image_preprocess_ms,
                        "text_tokenization_ms": text_tokenization_ms,
                        "vision_encoder_ms": vision_encoder_ms,
                        "llm_ms": llm_ms,
                        "features_process_ms": features_process_ms,
                        "state_encoder_ms": state_encoder_ms,
                        "action_encoder_ms": action_encoder_ms,
                        "dit_block_ms": dit_block_ms,
                        "action_decoder_ms": action_decoder_ms,
                    },
                })

            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"[WebSocket error] {str(e)}")
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


_LATENCY_KEYS = (
    "client_total_ms",
    "server_total_ms",
    "preprocess_ms",
    "get_action_ms",
    "model_forward_ms",
    "transform_ms",
    "postprocess_ms",
    "server_overhead_ms",
    "network_rtt_ms",
    "state_action_transform_ms",
    "image_preprocess_ms",
    "text_tokenization_ms",
    "vision_encoder_ms",
    "llm_ms",
    "features_process_ms",
    "state_encoder_ms",
    "action_encoder_ms",
    "dit_block_ms",
    "action_decoder_ms",
)


def run_client_benchmark(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    host = _host_for_role(args)
    sums: dict[str, float] = {k: 0.0 for k in _LATENCY_KEYS}
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
                host=host, port=args.port, ws_path=args.ws_path,
                payload=payload, timeout_ms=args.timeout_ms,
            )
        except Exception:
            pass

    for run_index in range(args.measure_runs):
        request_id = build_iteration_request_id(args.request_id, "run", run_index)
        payload = build_request_payload(args, request_id=request_id, rng=rng)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            response, client_total_ms = send_request(
                host=host, port=args.port, ws_path=args.ws_path,
                payload=payload, timeout_ms=args.timeout_ms,
            )
            body = response.json()
            result_code = int(body.get("resultCode", 1))
            actual_request_id = body.get("resultMap", {}).get("request_id", request_id)

            sl = response.server_latency
            server_total_ms = _coerce_latency_ms(sl.get("server_total_ms", 0.0))
            preprocess_ms = _coerce_latency_ms(sl.get("preprocess_ms", 0.0))
            get_action_ms = _coerce_latency_ms(sl.get("get_action_ms", 0.0))
            model_forward_ms = _coerce_latency_ms(sl.get("model_forward_ms", 0.0))
            transform_ms = _coerce_latency_ms(sl.get("transform_ms", 0.0))
            postprocess_ms = _coerce_latency_ms(sl.get("postprocess_ms", 0.0))
            server_overhead_ms = _coerce_latency_ms(sl.get("server_overhead_ms", 0.0))
            network_rtt_ms = round(max(client_total_ms - server_total_ms, 0.0), 4)
            state_action_transform_ms = _coerce_latency_ms(sl.get("state_action_transform_ms", 0.0))
            image_preprocess_ms = _coerce_latency_ms(sl.get("image_preprocess_ms", 0.0))
            text_tokenization_ms = _coerce_latency_ms(sl.get("text_tokenization_ms", 0.0))
            vision_encoder_ms = _coerce_latency_ms(sl.get("vision_encoder_ms", 0.0))
            llm_ms = _coerce_latency_ms(sl.get("llm_ms", 0.0))
            features_process_ms = _coerce_latency_ms(sl.get("features_process_ms", 0.0))
            state_encoder_ms = _coerce_latency_ms(sl.get("state_encoder_ms", 0.0))
            action_encoder_ms = _coerce_latency_ms(sl.get("action_encoder_ms", 0.0))
            dit_block_ms = _coerce_latency_ms(sl.get("dit_block_ms", 0.0))
            action_decoder_ms = _coerce_latency_ms(sl.get("action_decoder_ms", 0.0))

            row = {
                "client_total_ms": client_total_ms,
                "server_total_ms": server_total_ms,
                "preprocess_ms": preprocess_ms,
                "get_action_ms": get_action_ms,
                "model_forward_ms": model_forward_ms,
                "transform_ms": transform_ms,
                "postprocess_ms": postprocess_ms,
                "server_overhead_ms": server_overhead_ms,
                "network_rtt_ms": network_rtt_ms,
                "state_action_transform_ms": state_action_transform_ms,
                "image_preprocess_ms": image_preprocess_ms,
                "text_tokenization_ms": text_tokenization_ms,
                "vision_encoder_ms": vision_encoder_ms,
                "llm_ms": llm_ms,
                "features_process_ms": features_process_ms,
                "state_encoder_ms": state_encoder_ms,
                "action_encoder_ms": action_encoder_ms,
                "dit_block_ms": dit_block_ms,
                "action_decoder_ms": action_decoder_ms,
            }
            for k in _LATENCY_KEYS:
                sums[k] += row[k]

            if result_code == 0:
                success_count += 1
            else:
                failure_count += 1
                error_msg = body.get("errorMessage", "")
                result_map = body.get("resultMap", {})
                error_detail = result_map.get("error", "") if isinstance(result_map, dict) else ""
                print(
                    f"  [run {run_index}] FAILED: "
                    f"{error_msg or error_detail or 'unknown error'}"
                )

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
                    "latency": row,
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
                    "latency": {k: 0.0 for k in _LATENCY_KEYS},
                    "response": {
                        "resultCode": 1,
                        "errorMessage": _to_ascii_text(exc),
                        "resultMap": {},
                    },
                },
            )

    n = max(args.measure_runs, 1)
    avg = {k: round(v / n, 4) for k, v in sums.items()}
    other_ms = round(
        max(avg["get_action_ms"] - avg["model_forward_ms"] - avg["transform_ms"], 0.0), 4
    )
    gpu_ratio = (
        round(avg["model_forward_ms"] / avg["client_total_ms"] * 100.0, 2)
        if avg["client_total_ms"] > 0.0 else 0.0
    )

    W = 20
    print(f"\n=== Average Latency ({args.measure_runs} runs, {success_count} ok / {failure_count} failed) ===")
    print(f"  Client round-trip:        {avg['client_total_ms']:>{W}.4f} ms")
    print(f"  +-- Network RTT:          {avg['network_rtt_ms']:>{W}.4f} ms  (client_total - server_total)")
    print(f"  +-- Server total:         {avg['server_total_ms']:>{W}.4f} ms  (wall clock)")
    print(f"      +-- Preprocess:       {avg['preprocess_ms']:>{W}.4f} ms  (JSON parse + JPEG decode + cv2 + batch)")
    print(f"      +-- get_action:       {avg['get_action_ms']:>{W}.4f} ms  (CUDA events, transforms + GPU)")
    print(f"      |   +-- Transforms:   {avg['transform_ms']:>{W}.4f} ms")
    print(f"      |   |   +-- State/Action Transform:{avg['state_action_transform_ms']:>{W-8}.4f} ms")
    print(f"      |   |   +-- Image Preprocess:     {avg['image_preprocess_ms']:>{W-8}.4f} ms")
    print(f"      |   |   +-- Text Tokenization:    {avg['text_tokenization_ms']:>{W-8}.4f} ms")
    print(f"      |   +-- Backbone:     {avg['vision_encoder_ms'] + avg['llm_ms']:>{W}.4f} ms")
    print(f"      |   |   +-- Vision Encoder:       {avg['vision_encoder_ms']:>{W-8}.4f} ms")
    print(f"      |   |   +-- LLM:                  {avg['llm_ms']:>{W-8}.4f} ms")
    print(f"      |   +-- Action Head:  {avg['features_process_ms'] + avg['state_encoder_ms'] + avg['action_encoder_ms'] + avg['dit_block_ms'] + avg['action_decoder_ms']:>{W}.4f} ms")
    print(f"      |   |   +-- Features Process:     {avg['features_process_ms']:>{W-8}.4f} ms")
    print(f"      |   |   +-- State Encoder:        {avg['state_encoder_ms']:>{W-8}.4f} ms")
    print(f"      |   |   +-- Action Encoder:       {avg['action_encoder_ms']:>{W-8}.4f} ms")
    print(f"      |   |   +-- DiT Block:            {avg['dit_block_ms']:>{W-8}.4f} ms")
    print(f"      |   |   +-- Action Decoder:       {avg['action_decoder_ms']:>{W-8}.4f} ms")
    print(f"      |   +-- Other:        {other_ms:>{W}.4f} ms  (batch prep + unapply + squeeze)")
    print(f"      +-- Postprocess:      {avg['postprocess_ms']:>{W}.4f} ms  (action integration)")
    print(f"      +-- Server overhead:  {avg['server_overhead_ms']:>{W}.4f} ms  (JSON parse + framework)")
    print(f"  GPU inference ratio:      {gpu_ratio:>{W}.2f} %  (model_forward / client_total)")

    _append_jsonl_record(
        args.output_jsonl,
        {
            "record_type": "summary",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": config_snapshot,
            "summary_metrics": {f"avg_{k}": avg[k] for k in _LATENCY_KEYS},
            "derived": {
                "avg_other_ms": other_ms,
                "gpu_inference_ratio_pct": gpu_ratio,
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
        _install_timing_hooks(runtime.policy)
        app = create_app(runtime=runtime, ws_path=args.ws_path)
        run_server(app, host=_host_for_role(args), port=args.port)
        return 0
    if args.role == "client":
        return run_client_benchmark(args)
    raise ValueError(f"Unsupported role: {args.role}")


if __name__ == "__main__":
    raise SystemExit(main())
