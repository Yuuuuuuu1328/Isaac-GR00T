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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 GR00T 纯净 E2E 时延测试脚本")
    parser.add_argument("--model-path", type=str, default="/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--data-config", type=str, default="fourier_gr1_arms_only")
    parser.add_argument("--embodiment-tag", type=str, default="gr1")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--video-backend", type=str, choices=["decord", "torchcodec"], default="decord")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measure-runs", type=int, default=30)
    parser.add_argument("--output-jsonl", type=str, default="")
    return parser


def _load_runtime(args: argparse.Namespace):
    import torch
    import gr00t
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    repo_path = os.path.dirname(os.path.dirname(gr00t.__file__))
    dataset_path = args.dataset_path or os.path.join(repo_path, "demo_data/robot_sim.PickNPlace")

    data_config = load_data_config(args.data_config)
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=policy.modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )
    return policy, dataset


def main() -> int:
    _ensure_repo_root_on_path()
    parser = build_parser()
    args = parser.parse_args()

    from deployment_scripts.ant.profile_utils import (
        LatencyRecord,
        format_summary,
        measure_cuda_time_ms,
        summarize_latency_records,
        write_jsonl,
    )

    policy, dataset = _load_runtime(args)
    base_step_data = dataset[args.sample_index]
    records: list[LatencyRecord] = []

    # 纯净 E2E 脚本避免热路径内的细粒度打印，只测整次推理调用。
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
                    "script": "local_inference_e2e.py",
                    "run_index": run_index,
                    "sample_index": args.sample_index,
                    "denoising_steps": args.denoising_steps,
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
