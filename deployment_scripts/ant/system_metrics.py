from __future__ import annotations

import re
import shutil
import subprocess
import threading
from typing import Iterable


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _max(values: list[float]) -> float:
    return round(max(values), 4) if values else 0.0


def reset_torch_peak_memory_stats() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def collect_torch_peak_memory_metrics() -> dict[str, float]:
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "torch_peak_memory_allocated_mb": 0.0,
            "torch_peak_memory_reserved_mb": 0.0,
        }

    if not torch.cuda.is_available():
        return {
            "torch_peak_memory_allocated_mb": 0.0,
            "torch_peak_memory_reserved_mb": 0.0,
        }

    return {
        "torch_peak_memory_allocated_mb": round(
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 4
        ),
        "torch_peak_memory_reserved_mb": round(
            torch.cuda.max_memory_reserved() / (1024.0 * 1024.0), 4
        ),
    }


def parse_tegrastats_lines(lines: Iterable[str]) -> dict[str, float]:
    ram_used: list[float] = []
    gr3d_freq: list[float] = []
    emc_freq: list[float] = []
    cpu_temp: list[float] = []
    gpu_temp: list[float] = []
    power_by_rail: dict[str, list[float]] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if ram_match:
            ram_used.append(float(ram_match.group(1)))

        for metric_name, target in (
            ("GR3D_FREQ", gr3d_freq),
            ("EMC_FREQ", emc_freq),
        ):
            match = re.search(rf"{metric_name}\s+(\d+)%", line)
            if match:
                target.append(float(match.group(1)))

        cpu_match = re.search(r"CPU@([0-9.]+)C", line)
        if cpu_match:
            cpu_temp.append(float(cpu_match.group(1)))

        gpu_match = re.search(r"GPU@([0-9.]+)C", line)
        if gpu_match:
            gpu_temp.append(float(gpu_match.group(1)))

        for rail_name, current_mw, _average_mw in re.findall(
            r"(VDD_[A-Z0-9_]+)\s+(\d+)mW(?:/(\d+)mW)?",
            line,
        ):
            power_by_rail.setdefault(rail_name.lower(), []).append(float(current_mw))

    metrics: dict[str, float] = {"tegrastats_available": 1.0 if any([ram_used, gr3d_freq, emc_freq, cpu_temp, gpu_temp, power_by_rail]) else 0.0}
    if ram_used:
        metrics["system_peak_memory_used_mb"] = _max(ram_used)
    if gr3d_freq:
        metrics["tegrastats_gr3d_freq_pct_mean"] = _mean(gr3d_freq)
        metrics["tegrastats_gr3d_freq_pct_max"] = _max(gr3d_freq)
    if emc_freq:
        metrics["tegrastats_emc_freq_pct_mean"] = _mean(emc_freq)
        metrics["tegrastats_emc_freq_pct_max"] = _max(emc_freq)
    if cpu_temp:
        metrics["tegrastats_cpu_temp_c_mean"] = _mean(cpu_temp)
    if gpu_temp:
        metrics["tegrastats_gpu_temp_c_mean"] = _mean(gpu_temp)

    for rail_name, values in power_by_rail.items():
        metrics[f"tegrastats_{rail_name}_mw_mean"] = _mean(values)
        metrics[f"tegrastats_{rail_name}_mw_max"] = _max(values)

    if gr3d_freq:
        session_max = max(gr3d_freq)
        metrics["tegrastats_possible_throttle"] = (
            1.0 if session_max > 0 and any(value < 0.8 * session_max for value in gr3d_freq) else 0.0
        )
    return metrics


class TegrastatsSampler:
    def __init__(self, interval_ms: int = 250):
        self.interval_ms = interval_ms
        self.lines: list[str] = []
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._available = shutil.which("tegrastats") is not None

    def start(self) -> "TegrastatsSampler":
        if not self._available:
            return self
        self._process = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()
        return self

    def _read_stdout(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            self.lines.append(line)

    def stop(self) -> dict[str, float]:
        if not self._available:
            return {"tegrastats_available": 0.0}
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
        return parse_tegrastats_lines(self.lines)
