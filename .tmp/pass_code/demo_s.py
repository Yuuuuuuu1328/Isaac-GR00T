from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from deployment_scripts.ant.profile_utils import LatencyRecord


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - runtime dependency
    torch = None


DEFAULT_MODEL_PATH = "/home/jetson/Desktop/project/new_model/left_hand_v2_1223"
DEFAULT_TRT_ENGINE_PATH = "/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16_bs1_len283"
DEFAULT_TASK_PROMPT = "Move to center the book in view. Do nothing if no book is present."
DEFAULT_DATA_CONFIG = "new_interaction_group"
DEFAULT_EMBODIMENT_TAG = "new_embodiment"
DEFAULT_RESPONSE_ACTION_HORIZON = 14
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ONLINE_RESULT_JSONL = _REPO_ROOT / "deployment_scripts/ant/output/online_result_demo.jsonl"


def _cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _parse_list_like(value: Any):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _resolve_data_config(data_config_map: dict[str, Any], name: str):
    if name not in data_config_map:
        raise ValueError(
            f"Local ossfs gr00t does not expose data_config '{name}'. "
            f"Available options: {sorted(data_config_map.keys())}"
        )
    return data_config_map[name]


def parse_aistudio_query(payload: dict) -> dict:
    if "query" not in payload:
        raise ValueError("Missing query field")

    raw_query = payload["query"]
    if isinstance(raw_query, bytes):
        raw_query = raw_query.decode("latin-1", errors="replace")

    if isinstance(raw_query, dict):
        parsed = dict(raw_query)
    else:
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
        "errorMessage": _safe_ascii_text(error_message),
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
    safe_message = _safe_ascii_text(error_message)
    return build_aistudio_response(
        result_code=1,
        error_message=safe_message,
        result_map={
            "action_sequence": "",
            "joint_angles": json.dumps(joint_angles or []),
            "predicted_coords_2d": json.dumps(predicted_coords_2d or []),
            "history_angles": json.dumps(history_angles or []),
            "is_success": False,
            "error": safe_message,
            "key_infos": "key_infos",
            "device_id": device_id,
            "request_id": request_id,
        },
    )


def append_online_latency_record(
    output_path: str | Path,
    *,
    meta: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    record = LatencyRecord(meta=meta, metrics=metrics)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="ascii", errors="backslashreplace") as file_obj:
        file_obj.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")


def _safe_ascii_text(value: Any) -> str:
    text = str(value)
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


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


def measure_inference_latency_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    if _cuda_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = fn()
        end_event.record()
        torch.cuda.synchronize()
        return result, round(float(start_event.elapsed_time(end_event)), 4)

    wall_start_ns = time.perf_counter_ns()
    result = fn()
    wall_ms = round((time.perf_counter_ns() - wall_start_ns) / 1_000_000.0, 4)
    return result, wall_ms


@dataclass
class LatencySummaryTracker:
    total_latency_sum_ms: float = 0.0
    infer_latency_sum_ms: float = 0.0
    network_latency_sum_ms: float = 0.0
    sample_count: int = 0
    total_request_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(
        self,
        *,
        total_latency_ms: float,
        infer_latency_ms: float,
        network_latency_ms: float,
        include_in_average: bool,
    ) -> dict[str, float]:
        with self._lock:
            self.total_request_count += 1
            if include_in_average:
                self.sample_count += 1
                self.total_latency_sum_ms += total_latency_ms
                self.infer_latency_sum_ms += infer_latency_ms
                self.network_latency_sum_ms += network_latency_ms

            if self.sample_count == 0:
                average_total_ms = 0.0
                average_inference_ms = 0.0
                average_network_ms = 0.0
                inference_ratio_pct = 0.0
            else:
                average_total_ms = self.total_latency_sum_ms / self.sample_count
                average_inference_ms = self.infer_latency_sum_ms / self.sample_count
                average_network_ms = self.network_latency_sum_ms / self.sample_count
                inference_ratio_pct = (
                    (self.infer_latency_sum_ms / self.total_latency_sum_ms) * 100.0
                    if self.total_latency_sum_ms > 0.0
                    else 0.0
                )

            return {
                "sample_count": self.sample_count,
                "total_request_count": self.total_request_count,
                "average_total_latency_ms": round(average_total_ms, 4),
                "average_inference_latency_ms": round(average_inference_ms, 4),
                "average_network_latency_ms": round(average_network_ms, 4),
                "average_inference_proportion_pct": round(inference_ratio_pct, 2),
            }


def print_average_latency_log(summary: dict[str, float]) -> None:
    print("\n=== Average Latency ===")
    print(
        "Successful samples: "
        f"{int(summary.get('sample_count', 0))}/{int(summary.get('total_request_count', 0))}"
    )
    print(f"Average total latency: {summary['average_total_latency_ms']:.3f} ms")
    print(f"Average inference latency: {summary['average_inference_latency_ms']:.3f} ms")
    print(f"Average network latency: {summary['average_network_latency_ms']:.3f} ms")
    print(f"Average inference proportion: {summary['average_inference_proportion_pct']:.2f}%")


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


def _coerce_action_matrix(previous_actions: Any) -> np.ndarray:
    action_matrix = np.asarray(previous_actions, dtype=np.float32)

    if action_matrix.ndim == 3:
        if action_matrix.shape[0] != 1:
            raise ValueError(
                "Expected batched actions with batch size 1 when ndim=3, "
                f"got shape={tuple(action_matrix.shape)}"
            )
        action_matrix = action_matrix[0]

    if action_matrix.ndim != 2:
        raise ValueError(f"Expected action matrix with ndim=2, got shape={tuple(action_matrix.shape)}")

    if action_matrix.shape[1] < 6:
        raise ValueError(
            f"Expected action dimension >= 6 for single_arm, got shape={tuple(action_matrix.shape)}"
        )

    return action_matrix


def postprocess_actions(
    previous_actions: np.ndarray,
    robot_state: np.ndarray,
    *,
    response_action_horizon: int,
) -> tuple[str, bool, list[list[float]]]:
    action_matrix = _coerce_action_matrix(previous_actions)
    integrated_actions = np.cumsum(action_matrix[:, :6], axis=0) + robot_state[:6]
    rounded_actions = [[round(float(x), 4) for x in row] for row in integrated_actions.tolist()]
    action_string = json.dumps(rounded_actions[:response_action_horizon])
    return action_string, bool(integrated_actions.size >= 6), rounded_actions


def _extract_metric_ms(metrics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = _coerce_latency_ms(metrics.get(key), default=0.0)
        if value > 0.0:
            return value
    return 0.0


@dataclass
class AistudioRuntime:
    policy: object
    task_prompt: str
    response_action_horizon: int
    action_runner: Callable[[dict[str, Any]], Any] | None = None
    metadata: dict = field(default_factory=dict)
    ready: bool = True

    def _predict_previous_actions(
        self,
        *,
        model_observation: dict[str, Any],
        model_batch: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.action_runner is not None:
            runner_result = self.action_runner(model_observation)
            if (
                isinstance(runner_result, tuple)
                and len(runner_result) == 2
                and isinstance(runner_result[1], dict)
            ):
                return runner_result[0], dict(runner_result[1])
            return runner_result, dict(getattr(self.action_runner, "last_metrics", {}))
        return self.policy.get_action(model_batch)["action.single_arm"], {}

    def predict_with_timing(self, payload: dict) -> tuple[dict, dict[str, Any]]:
        parsed: dict[str, Any] | None = None
        pure_inference_ms = 0.0
        fallback_inference_ms = 0.0
        try:
            parsed = parse_aistudio_query(payload)
            frame_rgb = decode_framebuffer(
                parsed.get("framebuffer", []),
                parsed.get("framebuffer_size", 0),
            )
            robot_state = normalize_single_arm_state(parsed["joint_angles"])
            model_observation = build_model_observation(
                frame_rgb=frame_rgb,
                robot_state=robot_state,
                task_prompt=self.task_prompt,
            )
            batch = build_model_batch(
                frame_rgb=frame_rgb,
                robot_state=robot_state,
                task_prompt=self.task_prompt,
            )
            inference_stage_start_ns = time.perf_counter_ns()
            runner_metrics: dict[str, Any] = {}
            try:
                (previous_actions, runner_metrics), measured_ms = measure_inference_latency_ms(
                    lambda: self._predict_previous_actions(
                        model_observation=model_observation,
                        model_batch=batch,
                    )
                )
            except Exception:
                fallback_inference_ms = round(
                    (time.perf_counter_ns() - inference_stage_start_ns) / 1_000_000.0,
                    4,
                )
                raise

            metrics_inference_ms = _extract_metric_ms(
                runner_metrics,
                "e2e_total_ms",
                "inference_total_ms",
                "pure_inference_ms",
                "inference_ms",
            )
            measured_ms = _coerce_latency_ms(measured_ms, default=0.0)
            pure_inference_ms = max(metrics_inference_ms, measured_ms)
            if pure_inference_ms <= 0.0:
                pure_inference_ms = round(
                    (time.perf_counter_ns() - inference_stage_start_ns) / 1_000_000.0,
                    4,
                )

            action_string, is_success, _ = postprocess_actions(
                previous_actions,
                robot_state,
                response_action_horizon=self.response_action_horizon,
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
                "pure_inference_ms": round(pure_inference_ms, 4),
                "inference_ms": round(pure_inference_ms, 4),
            }
        except Exception as exc:
            request_id = "" if parsed is None else str(parsed.get("request_id", ""))
            device_id = "" if parsed is None else str(parsed.get("device_id", ""))
            if pure_inference_ms <= 0.0 and fallback_inference_ms > 0.0:
                pure_inference_ms = round(fallback_inference_ms, 4)
            error_message = _safe_ascii_text(f"{type(exc).__name__}: {exc}")
            print(f"[demo_s][error] request_id={request_id} device_id={device_id} error={error_message}")
            return (
                build_aistudio_error_response(
                    error_message=error_message,
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
                    "pure_inference_ms": round(max(pure_inference_ms, 0.0), 4),
                    "inference_ms": round(max(pure_inference_ms, 0.0), 4),
                },
            )

    def predict(self, payload: dict) -> dict:
        response, _ = self.predict_with_timing(payload)
        return response


def create_app(*, runtime, online_result_path: str | Path = _DEFAULT_ONLINE_RESULT_JSONL):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Aistudio Compatibility Service")
    latency_summary_tracker = LatencySummaryTracker()

    @app.post("/predict")
    def predict(payload: dict):
        request_start_ns = time.perf_counter_ns()
        response_body, timing = runtime.predict_with_timing(payload)
        server_total_ms = round((time.perf_counter_ns() - request_start_ns) / 1_000_000.0, 4)
        pure_inference_ms = _coerce_latency_ms(
            timing.get("pure_inference_ms", timing.get("inference_ms", 0.0))
        )
        if pure_inference_ms <= 0.0 and server_total_ms > 0.0:
            pure_inference_ms = server_total_ms
        server_overhead_ms = max(round(server_total_ms - pure_inference_ms, 4), 0.0)
        request_id = str(timing.get("request_id", ""))
        device_id = str(timing.get("device_id", ""))
        backend = str(timing.get("backend", ""))
        result_code = int(response_body.get("resultCode", 1))
        error_message = _safe_ascii_text(response_body.get("errorMessage", ""))
        include_in_average = result_code == 0

        summary = latency_summary_tracker.update(
            total_latency_ms=server_total_ms,
            infer_latency_ms=pure_inference_ms,
            network_latency_ms=server_overhead_ms,
            include_in_average=include_in_average,
        )
        append_online_latency_record(
            online_result_path,
            meta={
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "request_id": request_id,
                "device_id": device_id,
                "backend": backend,
                "result_code": result_code,
                "error_message": error_message,
            },
            metrics=summary,
        )
        print_average_latency_log(summary)
        return JSONResponse(
            content=response_body,
            headers=build_latency_headers(
                server_total_ms=server_total_ms,
                pure_inference_ms=pure_inference_ms,
                server_overhead_ms=server_overhead_ms,
            ),
        )

    @app.get("/health/live")
    def health_live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def health_ready():
        return {"status": "ok", "ready": bool(getattr(runtime, "ready", False))}

    @app.get("/metadata")
    def metadata():
        return JSONResponse(content=dict(getattr(runtime, "metadata", {})))

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aistudio FastAPI compatibility service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", type=str, choices=["pytorch", "tensorrt"], default="tensorrt")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--trt-engine-path", type=str, default=DEFAULT_TRT_ENGINE_PATH)
    parser.add_argument("--data-config", type=str, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--embodiment-tag", type=str, default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--task-prompt", type=str, default=DEFAULT_TASK_PROMPT)
    parser.add_argument(
        "--response-action-horizon",
        type=int,
        default=DEFAULT_RESPONSE_ACTION_HORIZON,
    )
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--vit-dtype", type=str, choices=["fp16", "fp8", "int8"], default="fp16")
    parser.add_argument(
        "--llm-dtype",
        type=str,
        choices=["fp16", "nvfp4", "fp8", "int8"],
        default="fp16",
    )
    parser.add_argument("--dit-dtype", type=str, choices=["fp16", "fp8", "int8"], default="fp16")
    return parser


def build_runtime(args: argparse.Namespace) -> AistudioRuntime:
    from deployment_scripts.ant import ossfs_gr00t_runtime as ossfs_runtime_helper

    ossfs_runtime = ossfs_runtime_helper._import_local_gr00t_runtime()
    data_config = _resolve_data_config(ossfs_runtime.DATA_CONFIG_MAP, args.data_config)
    policy = ossfs_runtime.Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=args.denoising_steps,
        device="cuda" if _cuda_available() else "cpu",
    )

    trt_engine_path = None
    action_runner = None
    if args.backend == "tensorrt":
        from deployment_scripts.ant.local_inference_tensorrt_breakdown import run_breakdown_inference
        from deployment_scripts.trt_model_forward import setup_tensorrt_engines

        trt_engine_path = str(Path(args.trt_engine_path).expanduser())
        setup_tensorrt_engines(
            policy,
            trt_engine_path,
            vit_dtype=args.vit_dtype,
            llm_dtype=args.llm_dtype,
            dit_dtype=args.dit_dtype,
        )

        def _run_tensorrt_breakdown(raw_observation: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
            normalized_observation = dict(raw_observation)
            # local_inference_breakdown.py requires every modality to become batched
            # after unsqueeze_dict_values(); Python str would stay unbatched.
            task_key = "annotation.task_index"
            if isinstance(normalized_observation.get(task_key), str):
                normalized_observation[task_key] = np.array(normalized_observation[task_key])
            action_map, metrics = run_breakdown_inference(
                policy,
                normalized_observation,
                ossfs_runtime.unsqueeze_dict_values,
                compute_dtype=ossfs_runtime.COMPUTE_DTYPE,
            )
            actions = action_map.get("action.single_arm", action_map.get("action"))
            if actions is None:
                raise KeyError(
                    "TensorRT breakdown result missing action key; expected 'action.single_arm' or 'action'"
                )
            return actions, dict(metrics)

        action_runner = _run_tensorrt_breakdown

    return AistudioRuntime(
        policy=policy,
        task_prompt=args.task_prompt,
        response_action_horizon=args.response_action_horizon,
        action_runner=action_runner,
        metadata={
            "backend": args.backend,
            "model_path": args.model_path,
            "data_config": args.data_config,
            "embodiment_tag": args.embodiment_tag,
            "trt_engine_path": trt_engine_path,
            "response_action_horizon": args.response_action_horizon,
        },
        ready=True,
    )


def run_server(app, *, host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("uvicorn is required to run the Aistudio FastAPI service") from exc

    uvicorn.run(app, host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = build_runtime(args)
    app = create_app(runtime=runtime)
    run_server(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
