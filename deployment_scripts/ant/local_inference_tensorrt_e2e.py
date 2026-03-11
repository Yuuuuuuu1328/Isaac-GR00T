from __future__ import annotations

import argparse
import copy
import os
import socket
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _ensure_repo_root_on_path()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 GR00T TensorRT 纯净 E2E 时延测试脚本")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B",
    )
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--data-config", type=str, default="fourier_gr1_arms_only")
    parser.add_argument("--embodiment-tag", type=str, default="gr1")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--video-backend", type=str, choices=["decord", "torchcodec"], default="decord")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measure-runs", type=int, default=30)
    parser.add_argument("--trt-engine-path", type=str, default=str(_REPO_ROOT / "gr00t_engine"))
    parser.add_argument("--vit-dtype", type=str, choices=["fp16", "fp8"], default="fp16")
    parser.add_argument("--llm-dtype", type=str, choices=["fp16", "nvfp4", "fp8"], default="fp16")
    parser.add_argument("--dit-dtype", type=str, choices=["fp16", "fp8"], default="fp16")
    parser.add_argument("--output-jsonl", type=str, default="")
    return parser


def _resolve_repo_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def _load_runtime(args: argparse.Namespace):
    import torch
    import gr00t
    from deployment_scripts.trt_model_forward import setup_tensorrt_engines
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT 本地测试要求 CUDA 可用")

    repo_path = os.path.dirname(os.path.dirname(gr00t.__file__))
    dataset_path = args.dataset_path or os.path.join(repo_path, "demo_data/robot_sim.PickNPlace")
    trt_engine_path = _resolve_repo_path(args.trt_engine_path)
    if not Path(trt_engine_path).exists():
        raise FileNotFoundError(f"TensorRT engine 目录不存在: {trt_engine_path}")

    data_config = load_data_config(args.data_config)
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device="cuda",
    )
    setup_tensorrt_engines(
        policy,
        trt_engine_path,
        vit_dtype=args.vit_dtype,
        llm_dtype=args.llm_dtype,
        dit_dtype=args.dit_dtype,
    )
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=policy.modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )
    return policy, dataset, trt_engine_path, unsqueeze_dict_values


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    from deployment_scripts.ant.profile_utils import (
        LatencyRecord,
        format_summary,
        measure_cuda_time_ms,
        summarize_latency_records,
        write_jsonl,
    )

    policy, dataset, trt_engine_path, _ = _load_runtime(args)
    base_step_data = dataset[args.sample_index]
    records: list[LatencyRecord] = []

    for _ in range(args.warmup_runs):
        step_data = copy.deepcopy(base_step_data)
        _, _ = measure_cuda_time_ms(lambda: policy.get_action(step_data))

    for run_index in range(args.measure_runs):
        step_data = copy.deepcopy(base_step_data)
        _, e2e_total_ms = measure_cuda_time_ms(lambda: policy.get_action(step_data))
        records.append(
            LatencyRecord(
                meta={
                    "host": socket.gethostname(),
                    "script": "local_inference_tensorrt_e2e.py",
                    "run_index": run_index,
                    "sample_index": args.sample_index,
                    "denoising_steps": args.denoising_steps,
                    "trt_engine_path": trt_engine_path,
                    "vit_dtype": args.vit_dtype,
                    "llm_dtype": args.llm_dtype,
                    "dit_dtype": args.dit_dtype,
                },
                metrics={"e2e_total_ms": e2e_total_ms},
            )
        )

    summary = summarize_latency_records(records)
    print(format_summary(summary))
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
