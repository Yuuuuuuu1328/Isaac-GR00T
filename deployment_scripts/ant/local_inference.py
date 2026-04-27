from __future__ import annotations

import argparse
import copy
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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
    enable_torch_compile_for_breakdown,
    enable_torch_compile_for_e2e,
)

_TRT_VIT_DTYPE_CHOICES = ["fp16", "fp8", "int8"]
_TRT_LLM_DTYPE_CHOICES = ["fp16", "nvfp4", "fp8", "int8"]
_BACKEND_CHOICES = ["pytorch", "tensorrt"]
_MODE_CHOICES = ["e2e", "breakdown"]
_DEFAULT_EXPERIMENT_RESULT_JSONL = _REPO_ROOT / "deployment_scripts/ant/output/local_inference_result.jsonl"


def _default_trt_engine_path(mode: str) -> str:
    engine_dir = "gr00t_engine_fp16" if mode == "e2e" else "gr00t_engine"
    return str(_REPO_ROOT / engine_dir)


def _add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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


def _add_tensorrt_args(
    parser: argparse.ArgumentParser,
    *,
    trt_engine_path_default: str | None,
) -> argparse.ArgumentParser:
    parser.add_argument("--trt-engine-path", type=str, default=trt_engine_path_default)
    parser.add_argument("--vit-dtype", type=str, choices=_TRT_VIT_DTYPE_CHOICES, default="fp16")
    parser.add_argument("--llm-dtype", type=str, choices=_TRT_LLM_DTYPE_CHOICES, default="fp16")
    parser.add_argument("--dit-dtype", type=str, choices=_TRT_VIT_DTYPE_CHOICES, default="fp16")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 GR00T 统一推理时延测试脚本")
    parser.add_argument("--backend", type=str, choices=_BACKEND_CHOICES, default="pytorch")
    parser.add_argument("--mode", type=str, choices=_MODE_CHOICES, default="e2e")
    parser = _add_common_args(parser)
    parser = add_torch_compile_arg(parser)
    parser = _add_tensorrt_args(parser, trt_engine_path_default=None)
    return add_optional_eval_args(parser)


def run_legacy_main(
    *,
    build_parser_fn: Callable[[], argparse.ArgumentParser],
    backend: str,
    mode: str,
    script_name: str,
    argv: list[str] | None = None,
) -> int:
    parser = build_parser_fn()
    args = parser.parse_args(argv)
    return run_with_args(args, backend=backend, mode=mode, script_name=script_name)


def _normalize_args(
    args: argparse.Namespace,
    *,
    backend: str | None = None,
    mode: str | None = None,
) -> argparse.Namespace:
    normalized_backend = backend or getattr(args, "backend", "pytorch")
    normalized_mode = mode or getattr(args, "mode", "e2e")
    setattr(args, "backend", normalized_backend)
    setattr(args, "mode", normalized_mode)

    if normalized_backend == "tensorrt" and getattr(args, "trt_engine_path", None) is None:
        args.trt_engine_path = _default_trt_engine_path(normalized_mode)
    return args


def _run_profile_loop(
    *,
    args: argparse.Namespace,
    policy,
    dataset,
    script_name: str,
    warmup_once: Callable[[dict], None],
    measure_once: Callable[[dict], dict[str, float]],
    build_meta: Callable[[int], dict[str, object]],
) -> int:
    base_step_data = dataset[args.sample_index]
    records: list[LatencyRecord] = []
    system_metrics: dict[str, float] = {}
    system_sampler = None

    if args.measure_system:
        reset_torch_peak_memory_stats()
        system_sampler = TegrastatsSampler(interval_ms=args.tegrastats_interval_ms).start()

    for _ in range(args.warmup_runs):
        warmup_once(copy.deepcopy(base_step_data))

    for run_index in range(args.measure_runs):
        metrics = measure_once(copy.deepcopy(base_step_data))
        records.append(LatencyRecord(meta=build_meta(run_index), metrics=metrics))

    if system_sampler is not None:
        system_metrics = merge_optional_metrics(
            collect_torch_peak_memory_metrics(),
            system_sampler.stop(),
        )

    quality_metrics = collect_optional_metrics(policy, dataset, args)
    if system_metrics or quality_metrics:
        for record in records:
            record.metrics = merge_optional_metrics(record.metrics, quality_metrics, system_metrics)

    summary = summarize_latency_records(records)
    print(format_summary(summary))
    _append_experiment_result_jsonl(
        _DEFAULT_EXPERIMENT_RESULT_JSONL,
        _build_experiment_result(args, script_name, summary, records),
    )
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, records)
    return 0


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


def _run_pytorch_e2e(args: argparse.Namespace, script_name: str) -> int:
    from deployment_scripts.ant.local_inference_e2e import _load_runtime

    policy, dataset = _load_runtime(args)
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
    from deployment_scripts.ant.local_inference_breakdown import (
        _load_runtime,
        _run_single_breakdown,
    )

    policy, dataset, compute_dtype, unsqueeze_dict_values = _load_runtime(args)
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
    from deployment_scripts.ant.local_inference_tensorrt_e2e import _load_runtime

    policy, dataset, trt_engine_path, _ = _load_runtime(args)

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
    from deployment_scripts.ant.local_inference_tensorrt_e2e import _load_runtime

    policy, dataset, trt_engine_path, unsqueeze_dict_values = _load_runtime(args)

    def _warmup(step_data: dict) -> None:
        _ = _run_single_breakdown(policy, step_data, unsqueeze_dict_values)

    def _measure(step_data: dict) -> dict[str, float]:
        return dict(_run_single_breakdown(policy, step_data, unsqueeze_dict_values))

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
    script_name: str = "local_inference.py",
) -> int:
    args = _normalize_args(args, backend=backend, mode=mode)

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
    return run_with_args(args, script_name="local_inference.py")


if __name__ == "__main__":
    raise SystemExit(main())
