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

from deployment_scripts.ant.quality_metrics import (  # noqa: E402
    add_optional_eval_args,
    collect_optional_metrics,
    merge_optional_metrics,
)
from deployment_scripts.ant.system_metrics import (  # noqa: E402
    TegrastatsSampler,
    collect_torch_peak_memory_metrics,
    reset_torch_peak_memory_stats,
)
from deployment_scripts.ant.torch_compile_utils import (  # noqa: E402
    add_torch_compile_arg,
    enable_torch_compile_for_e2e,
)


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
    return add_torch_compile_arg(add_optional_eval_args(parser))


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
    from deployment_scripts.ant.local_inference import run_legacy_main

    return run_legacy_main(
        build_parser_fn=build_parser,
        backend="pytorch",
        mode="e2e",
        script_name="local_inference_e2e.py",
    )


if __name__ == "__main__":
    raise SystemExit(main())
