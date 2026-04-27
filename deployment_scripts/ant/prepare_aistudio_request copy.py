from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _ensure_repo_root_on_path()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成严格兼容 Aistudio 请求格式的 JSON 文件")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--joint-angles", type=str, default="[0.1, 0.2, 0.3, -0.4, 0.5]")
    parser.add_argument("--joint4", type=float, default=None)
    parser.add_argument("--predicted-coords-2d", type=str, default="[0.5, 0.5]")
    parser.add_argument("--history-angles", type=str, default="[]")
    parser.add_argument("--device-id", type=str, default="dev-1")
    parser.add_argument("--request-id", type=str, default="req-0001")
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--pixel-value", type=int, default=0)
    return parser


def _parse_literal_list(raw_value: str):
    value = ast.literal_eval(raw_value)
    if not isinstance(value, list):
        raise ValueError(f"Expected list literal, got: {raw_value}")
    return value


def build_request_payload(args: argparse.Namespace) -> dict:
    import cv2

    joint_angles = _parse_literal_list(args.joint_angles)
    if len(joint_angles) != 5:
        raise ValueError("--joint-angles must contain exactly 5 values")
    if args.joint4 is not None:
        joint_angles[3] = args.joint4

    predicted_coords_2d = _parse_literal_list(args.predicted_coords_2d)
    history_angles = _parse_literal_list(args.history_angles)

    image = np.full(
        (args.height, args.width, 3),
        fill_value=np.uint8(args.pixel_value),
        dtype=np.uint8,
    )
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode the test image as JPEG")

    inner = {
        "joint_angles": str(joint_angles),
        "predicted_coords_2d": str(predicted_coords_2d),
        "history_angles": str(history_angles),
        "framebuffer": encoded.astype(np.uint8).tolist(),
        "framebuffer_size": int(encoded.size),
        "device_id": args.device_id,
        "request_id": args.request_id,
    }
    return {"query": json.dumps(inner, ensure_ascii=False)}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = build_request_payload(args)
    output_path = Path(args.output).expanduser()
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
