from __future__ import annotations

import argparse
import ast
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - handled in tests/runtime
    requests = SimpleNamespace(post=None)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ONLINE_RESULT_JSONL = _REPO_ROOT / "deployment_scripts/ant/output/online_result.jsonl"


def _to_ascii_text(value: object) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Aistudio HTTP client example")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--joint-angles", type=str, default="")
    parser.add_argument("--predicted-coords-2d", type=str, default="")
    parser.add_argument("--history-angles", type=str, default="")
    parser.add_argument("--device-id", type=str, default="dev-random")
    parser.add_argument("--request-id", type=str, default="req-0001")
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--measure-runs", type=int, default=10)
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
    payload: dict,
    timeout_ms: int,
):
    if getattr(requests, "post", None) is None:
        raise RuntimeError("requests is required to run the Aistudio client example")

    start_ns = time.perf_counter_ns()
    response = requests.post(
        f"http://{host}:{port}/predict",
        json=payload,
        timeout=timeout_ms / 1000.0,
    )
    end_ns = time.perf_counter_ns()
    client_round_trip_ms = round((end_ns - start_ns) / 1_000_000.0, 4)
    return response, client_round_trip_ms


def _parse_latency_header(response, header_name: str) -> float:
    try:
        return round(float(response.headers.get(header_name, "nan")), 4)
    except (TypeError, ValueError):
        return float("nan")


def _latency_delta_ms(total_ms: float, component_ms: float) -> float:
    if np.isnan(total_ms) or np.isnan(component_ms):
        return float("nan")
    return round(max(total_ms - component_ms, 0.0), 4)


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


def _append_online_output_record(
    output_path: str | Path,
    *,
    meta: dict[str, object],
    response: dict[str, object],
    metrics: dict[str, float] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "meta": meta,
        "response": response,
        "metrics": metrics or {},
    }
    with output_path.open("a", encoding="ascii") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")


def run_benchmark(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    total_latency_sum_ms = 0.0
    infer_latency_sum_ms = 0.0
    network_latency_sum_ms = 0.0

    for warmup_index in range(args.warmup_runs):
        request_id = build_iteration_request_id(args.request_id, "warmup", warmup_index)
        payload = build_request_payload(args, request_id=request_id, rng=rng)
        try:
            response, _ = send_request(
                host=args.host,
                port=args.port,
                payload=payload,
                timeout_ms=args.timeout_ms,
            )
        except Exception as exc:
            _append_online_output_record(
                _DEFAULT_ONLINE_RESULT_JSONL,
                meta={
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "request_id": request_id,
                    "device_id": args.device_id,
                    "backend": "",
                    "phase": "warmup",
                    "iteration": warmup_index,
                    "result_code": 1,
                    "http_status_code": 0,
                },
                response={
                    "resultCode": 1,
                    "errorMessage": _to_ascii_text(exc),
                    "resultMap": {},
                },
            )
            print(
                f"[warmup {warmup_index + 1}/{args.warmup_runs}] "
                f"request_id={request_id} "
                f"error={_to_ascii_text(exc)}"
            )
            continue
        try:
            body = response.json()
        except Exception as exc:
            body = {"resultCode": 1, "errorMessage": _to_ascii_text(exc), "resultMap": {}}
        actual_request_id = body.get("resultMap", {}).get("request_id", request_id)
        _append_online_output_record(
            _DEFAULT_ONLINE_RESULT_JSONL,
            meta={
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "request_id": actual_request_id,
                "device_id": args.device_id,
                "backend": "",
                "phase": "warmup",
                "iteration": warmup_index,
                "result_code": int(body.get("resultCode", 1)),
                "http_status_code": int(getattr(response, "status_code", 0) or 0),
            },
            response={
                "resultCode": int(body.get("resultCode", 1)),
                "errorMessage": _to_ascii_text(body.get("errorMessage", "")),
                "resultMap": body.get("resultMap", {}),
            },
        )
        print(f"[warmup {warmup_index + 1}/{args.warmup_runs}] request_id={actual_request_id}")

    for run_index in range(args.measure_runs):
        request_id = build_iteration_request_id(args.request_id, "run", run_index)
        payload = build_request_payload(args, request_id=request_id, rng=rng)
        try:
            response, client_round_trip_ms = send_request(
                host=args.host,
                port=args.port,
                payload=payload,
                timeout_ms=args.timeout_ms,
            )
        except Exception as exc:
            _append_online_output_record(
                _DEFAULT_ONLINE_RESULT_JSONL,
                meta={
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "request_id": request_id,
                    "device_id": args.device_id,
                    "backend": "",
                    "phase": "run",
                    "iteration": run_index,
                    "result_code": 1,
                    "http_status_code": 0,
                },
                response={
                    "resultCode": 1,
                    "errorMessage": _to_ascii_text(exc),
                    "resultMap": {},
                },
                metrics={
                    "client_round_trip_ms": 0.0,
                    "server_total_ms": float("nan"),
                    "pure_inference_ms": 0.0,
                    "network_ms": 0.0,
                },
            )
            print(
                f"[run {run_index + 1}/{args.measure_runs}] "
                f"request_id={request_id} "
                f"resultCode=1 "
                f"errorMessage={_to_ascii_text(exc)}"
            )
            continue
        try:
            body = response.json()
        except Exception as exc:
            body = {"resultCode": 1, "errorMessage": _to_ascii_text(exc), "resultMap": {}}
        actual_request_id = body.get("resultMap", {}).get("request_id", request_id)
        result_code = int(body.get("resultCode", 1))
        if result_code != 0:
            print(
                f"[run {run_index + 1}/{args.measure_runs}] "
                f"request_id={actual_request_id} "
                f"resultCode={result_code} "
                f"errorMessage={_to_ascii_text(body.get('errorMessage', ''))}"
            )
        server_total_ms = _parse_latency_header(response, "X-Server-Total-Ms")
        pure_inference_ms = _parse_latency_header(response, "X-Pure-Inference-Ms")
        if np.isnan(pure_inference_ms):
            pure_inference_ms = _parse_latency_header(response, "X-Inference-Ms")
        pure_inference_ms = _resolve_nonzero_inference_ms(
            pure_inference_ms=pure_inference_ms,
            server_total_ms=server_total_ms,
            client_round_trip_ms=client_round_trip_ms,
        )
        network_ms = _latency_delta_ms(client_round_trip_ms, pure_inference_ms)
        _append_online_output_record(
            _DEFAULT_ONLINE_RESULT_JSONL,
            meta={
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "request_id": actual_request_id,
                "device_id": args.device_id,
                "backend": "",
                "phase": "run",
                "iteration": run_index,
                "result_code": result_code,
                "http_status_code": int(getattr(response, "status_code", 0) or 0),
            },
            response={
                "resultCode": result_code,
                "errorMessage": _to_ascii_text(body.get("errorMessage", "")),
                "resultMap": body.get("resultMap", {}),
            },
            metrics={
                "client_round_trip_ms": client_round_trip_ms,
                "server_total_ms": server_total_ms,
                "pure_inference_ms": pure_inference_ms,
                "network_ms": network_ms,
            },
        )

        total_latency_sum_ms += client_round_trip_ms
        infer_latency_sum_ms += pure_inference_ms
        network_latency_sum_ms += network_ms
        print(
            f"[run {run_index + 1}/{args.measure_runs}] "
            f"request_id={actual_request_id} "
            f"client_round_trip_ms={client_round_trip_ms:.4f} "
            f"server_total_ms={server_total_ms:.4f} "
            f"pure_inference_ms={pure_inference_ms:.4f} "
            f"network_ms={network_ms:.4f}"
        )

    run_count = max(args.measure_runs, 1)
    average_total_latency_ms = round(total_latency_sum_ms / run_count, 4)
    average_inference_latency_ms = round(infer_latency_sum_ms / run_count, 4)
    average_network_latency_ms = round(network_latency_sum_ms / run_count, 4)
    average_inference_proportion_pct = (
        round((infer_latency_sum_ms / total_latency_sum_ms) * 100.0, 2)
        if total_latency_sum_ms > 0.0
        else 0.0
    )

    print("\n=== Average Latency ===")
    print(f"Average total latency: {average_total_latency_ms:.3f} ms")
    print(f"Average inference latency: {average_inference_latency_ms:.3f} ms")
    print(f"Average network latency: {average_network_latency_ms:.3f} ms")
    print(f"Average inference proportion: {average_inference_proportion_pct:.2f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
