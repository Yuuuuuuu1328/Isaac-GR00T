from __future__ import annotations

import argparse
import copy
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root

_REPO_ROOT = _ensure_repo_root_on_path()

from deployment_scripts.ant.profile_utils import (  # noqa: E402
    LatencyRecord,
    format_summary,
    measure_cuda_time_ms,
    summarize_latency_records,
    write_jsonl,
)
from deployment_scripts.ant import ossfs_gr00t_runtime as _ossfs_runtime_helper  # noqa: E402
from deployment_scripts.ant.torch_compile_utils import (  # noqa: E402
    add_torch_compile_arg,
    enable_torch_compile_for_breakdown,
    enable_torch_compile_for_e2e,
)

_TRT_VIT_DTYPE_CHOICES = ["fp16", "fp8", "int8"]
_TRT_LLM_DTYPE_CHOICES = ["fp16", "nvfp4", "fp8", "int8"]
_BACKEND_CHOICES = ["pytorch", "tensorrt"]
_MODE_CHOICES = ["e2e", "breakdown"]
_DEFAULT_EXPERIMENT_RESULT_JSONL = (
    _REPO_ROOT / "deployment_scripts/ant/output/local_inference_new_interaction_result.jsonl"
)
_OSSFS_WORKSPACE = _ossfs_runtime_helper._OSSFS_WORKSPACE
_ensure_ossfs_gr00t_on_path = _ossfs_runtime_helper._ensure_ossfs_gr00t_on_path
_import_local_gr00t_runtime = _ossfs_runtime_helper._import_local_gr00t_runtime


def _resolve_data_config(data_config_map, name: str):
    if name not in data_config_map:
        raise ValueError(
            f"Local ossfs gr00t does not expose data_config '{name}'. "
            f"Available options: {sorted(data_config_map.keys())}"
        )
    return data_config_map[name]


def _resolve_repo_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def _add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/jetson/Desktop/project/new_model/left_hand_v2_1223",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/jetson/Desktop/project/Isaac-GR00T/demo_data/new_interaction_group/Real_test_data_0305",
    )
    parser.add_argument("--data-config", type=str, default="new_interaction_group")
    parser.add_argument("--embodiment-tag", type=str, default="new_embodiment")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--video-backend", type=str, choices=["decord"], default="decord")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measure-runs", type=int, default=30)
    parser.add_argument("--output-jsonl", type=str, default="")
    return parser


def _add_tensorrt_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--trt-engine-path",
        type=str,
        default="/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16",
    )
    parser.add_argument("--vit-dtype", type=str, choices=_TRT_VIT_DTYPE_CHOICES, default="fp16")
    parser.add_argument("--llm-dtype", type=str, choices=_TRT_LLM_DTYPE_CHOICES, default="fp16")
    parser.add_argument("--dit-dtype", type=str, choices=_TRT_VIT_DTYPE_CHOICES, default="fp16")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="本地 new_interaction_group 统一推理时延测试脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", type=str, choices=_BACKEND_CHOICES, default="tensorrt")
    parser.add_argument("--mode", type=str, choices=_MODE_CHOICES, default="e2e")
    parser = _add_common_args(parser)
    parser = add_torch_compile_arg(parser)
    return _add_tensorrt_args(parser)


def _build_runtime(args: argparse.Namespace, *, use_tensorrt: bool):
    import torch

    ossfs_runtime = _import_local_gr00t_runtime()

    data_config = _resolve_data_config(ossfs_runtime.DATA_CONFIG_MAP, args.data_config)
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_tensorrt and device != "cuda":
        raise RuntimeError("TensorRT 本地测试要求 CUDA 可用")

    policy = ossfs_runtime.Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device=device,
    )

    trt_engine_path = None
    if use_tensorrt:
        from deployment_scripts.trt_model_forward import setup_tensorrt_engines

        trt_engine_path = _resolve_repo_path(args.trt_engine_path)
        if not Path(trt_engine_path).exists():
            raise FileNotFoundError(f"TensorRT engine 目录不存在: {trt_engine_path}")
        setup_tensorrt_engines(
            policy,
            trt_engine_path,
            vit_dtype=args.vit_dtype,
            llm_dtype=args.llm_dtype,
            dit_dtype=args.dit_dtype,
        )

    dataset = ossfs_runtime.LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=policy.modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )
    return (
        policy,
        dataset,
        ossfs_runtime.COMPUTE_DTYPE,
        ossfs_runtime.unsqueeze_dict_values,
        trt_engine_path,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _build_experiment_result(
    args: argparse.Namespace,
    script_name: str,
    summary: dict[str, Any],
    records: list[LatencyRecord],
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "script": script_name,
        "args": _json_safe(vars(args)),
        "summary": _json_safe(summary),
        "records": [_json_safe(record.to_dict()) for record in records],
    }


def _append_experiment_result_jsonl(output_path: str | Path, result: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")


def _base_meta(args: argparse.Namespace, script_name: str, run_index: int) -> dict[str, object]:
    return {
        "host": socket.gethostname(),
        "script": script_name,
        "run_index": run_index,
        "sample_index": args.sample_index,
        "denoising_steps": args.denoising_steps,
    }


def _run_profile_loop(
    *,
    args: argparse.Namespace,
    policy,
    dataset,
    script_name: str,
    warmup_once,
    measure_once,
    build_meta,
) -> int:
    base_step_data = dataset[args.sample_index]
    records: list[LatencyRecord] = []

    for _ in range(args.warmup_runs):
        warmup_once(copy.deepcopy(base_step_data))

    for run_index in range(args.measure_runs):
        metrics = measure_once(copy.deepcopy(base_step_data))
        records.append(LatencyRecord(meta=build_meta(run_index), metrics=metrics))

    summary = summarize_latency_records(records)
    print(format_summary(summary))
    _append_experiment_result_jsonl(
        _DEFAULT_EXPERIMENT_RESULT_JSONL,
        _build_experiment_result(args, script_name, summary, records),
    )
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, records)
    return 0


def _run_pytorch_e2e(args: argparse.Namespace, script_name: str) -> int:
    policy, dataset, _, _, _ = _build_runtime(args, use_tensorrt=False)
    enable_torch_compile_for_e2e(policy, enabled=args.use_torch_compile)

    def _warmup(step_data: dict) -> None:
        _, _ = measure_cuda_time_ms(lambda: policy.get_action(step_data))

    def _measure(step_data: dict) -> dict[str, float]:
        _, e2e_total_ms = measure_cuda_time_ms(lambda: policy.get_action(step_data))
        return {"e2e_total_ms": e2e_total_ms}

    def _meta(run_index: int) -> dict[str, object]:
        return {
            **_base_meta(args, script_name, run_index),
            "use_torch_compile": args.use_torch_compile,
        }

    return _run_profile_loop(
        args=args,
        policy=policy,
        dataset=dataset,
        script_name=script_name,
        warmup_once=_warmup,
        measure_once=_measure,
        build_meta=_meta,
    )


def _run_pytorch_breakdown(args: argparse.Namespace, script_name: str) -> int:
    from deployment_scripts.ant.local_inference_breakdown import _run_single_breakdown

    policy, dataset, compute_dtype, unsqueeze_dict_values, _ = _build_runtime(
        args, use_tensorrt=False
    )
    enable_torch_compile_for_breakdown(policy, enabled=args.use_torch_compile)

    def _warmup(step_data: dict) -> None:
        _ = _run_single_breakdown(policy, step_data, compute_dtype, unsqueeze_dict_values)

    def _measure(step_data: dict) -> dict[str, float]:
        return dict(_run_single_breakdown(policy, step_data, compute_dtype, unsqueeze_dict_values))

    def _meta(run_index: int) -> dict[str, object]:
        return {
            **_base_meta(args, script_name, run_index),
            "use_torch_compile": args.use_torch_compile,
        }

    return _run_profile_loop(
        args=args,
        policy=policy,
        dataset=dataset,
        script_name=script_name,
        warmup_once=_warmup,
        measure_once=_measure,
        build_meta=_meta,
    )


def _run_tensorrt_e2e(args: argparse.Namespace, script_name: str) -> int:
    policy, dataset, _, _, trt_engine_path = _build_runtime(args, use_tensorrt=True)

    def _warmup(step_data: dict) -> None:
        _, _ = measure_cuda_time_ms(lambda: policy.get_action(step_data))

    def _measure(step_data: dict) -> dict[str, float]:
        _, e2e_total_ms = measure_cuda_time_ms(lambda: policy.get_action(step_data))
        return {"e2e_total_ms": e2e_total_ms}

    def _meta(run_index: int) -> dict[str, object]:
        return {
            **_base_meta(args, script_name, run_index),
            "trt_engine_path": trt_engine_path,
            "vit_dtype": args.vit_dtype,
            "llm_dtype": args.llm_dtype,
            "dit_dtype": args.dit_dtype,
        }

    return _run_profile_loop(
        args=args,
        policy=policy,
        dataset=dataset,
        script_name=script_name,
        warmup_once=_warmup,
        measure_once=_measure,
        build_meta=_meta,
    )


def _run_tensorrt_breakdown(args: argparse.Namespace, script_name: str) -> int:
    from deployment_scripts.ant.local_inference_tensorrt_breakdown import _run_single_breakdown

    policy, dataset, compute_dtype, unsqueeze_dict_values, trt_engine_path = _build_runtime(
        args, use_tensorrt=True
    )

    def _warmup(step_data: dict) -> None:
        _ = _run_single_breakdown(
            policy,
            step_data,
            unsqueeze_dict_values,
            compute_dtype=compute_dtype,
        )

    def _measure(step_data: dict) -> dict[str, float]:
        return dict(
            _run_single_breakdown(
                policy,
                step_data,
                unsqueeze_dict_values,
                compute_dtype=compute_dtype,
            )
        )

    def _meta(run_index: int) -> dict[str, object]:
        return {
            **_base_meta(args, script_name, run_index),
            "trt_engine_path": trt_engine_path,
            "vit_dtype": args.vit_dtype,
            "llm_dtype": args.llm_dtype,
            "dit_dtype": args.dit_dtype,
        }

    return _run_profile_loop(
        args=args,
        policy=policy,
        dataset=dataset,
        script_name=script_name,
        warmup_once=_warmup,
        measure_once=_measure,
        build_meta=_meta,
    )


def run_with_args(
    args: argparse.Namespace,
    *,
    backend: str | None = None,
    mode: str | None = None,
    script_name: str = "local_inference_new_interaction.py",
) -> int:
    normalized_backend = backend or getattr(args, "backend", "tensorrt")
    normalized_mode = mode or getattr(args, "mode", "e2e")
    setattr(args, "backend", normalized_backend)
    setattr(args, "mode", normalized_mode)

    if args.backend == "pytorch" and args.mode == "e2e":
        return _run_pytorch_e2e(args, script_name)
    if args.backend == "pytorch" and args.mode == "breakdown":
        return _run_pytorch_breakdown(args, script_name)
    if args.backend == "tensorrt" and args.mode == "e2e":
        return _run_tensorrt_e2e(args, script_name)
    if args.backend == "tensorrt" and args.mode == "breakdown":
        return _run_tensorrt_breakdown(args, script_name)
    raise ValueError(f"Unsupported backend/mode combination: {args.backend}/{args.mode}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_with_args(args, script_name="local_inference_new_interaction.py")


if __name__ == "__main__":
    raise SystemExit(main())
