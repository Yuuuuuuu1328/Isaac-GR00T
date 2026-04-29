from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class LatencyRecord:
    meta: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"meta": self.meta, "metrics": self.metrics}


def sync_cuda_if_needed() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def compute_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values 不能为空")
    if len(values) == 1:
        return round(values[0], 4)

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(sorted_values[lower], 4)

    weight = index - lower
    result = sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    return round(result, 4)


def summarize_latency_records(records: list[LatencyRecord]) -> dict[str, Any]:
    if not records:
        return {"runs": 0}

    metric_names = sorted({name for record in records for name in record.metrics})
    summary: dict[str, Any] = {"runs": len(records)}

    for metric_name in metric_names:
        values = [record.metrics[metric_name] for record in records if metric_name in record.metrics]
        if not values:
            continue

        mean = sum(values) / len(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary[metric_name] = {
            "mean": round(mean, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "p50": compute_percentile(values, 50),
            "p95": compute_percentile(values, 95),
            "p99": compute_percentile(values, 99),
            "std": round(std, 4),
        }
    return summary


def write_jsonl(output_path: str | Path, records: list[LatencyRecord]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"runs={summary.get('runs', 0)}"]
    for metric_name, metric_summary in summary.items():
        if metric_name == "runs":
            continue
        unit = ""
        if "_ms" in metric_name:
            unit = " ms"
        elif "_mb" in metric_name:
            unit = " MB"
        elif "_mw" in metric_name:
            unit = " mW"
        elif "_w" in metric_name:
            unit = " W"
        elif "_pct" in metric_name:
            unit = "%"
        elif "_fps" in metric_name:
            unit = " FPS"
        elif "_mhz" in metric_name:
            unit = " MHz"
        elif "_c" in metric_name:
            unit = " C"
        lines.append(
            (
                f"{metric_name}: mean={metric_summary['mean']:.4f}{unit}, "
                f"p50={metric_summary['p50']:.4f}{unit}, "
                f"p95={metric_summary['p95']:.4f}{unit}, "
                f"p99={metric_summary['p99']:.4f}{unit}"
            )
        )
    return "\n".join(lines)


def measure_wall_time_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    start_ns = time.perf_counter_ns()
    result = fn()
    end_ns = time.perf_counter_ns()
    return result, round((end_ns - start_ns) / 1_000_000.0, 4)


def measure_cuda_time_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    try:
        import torch
    except ModuleNotFoundError:
        return measure_wall_time_ms(fn)

    if not torch.cuda.is_available():
        return measure_wall_time_ms(fn)

    sync_cuda_if_needed()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = fn()
    end_event.record()
    torch.cuda.synchronize()
    return result, round(float(start_event.elapsed_time(end_event)), 4)
