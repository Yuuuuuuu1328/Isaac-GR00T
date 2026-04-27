from __future__ import annotations

import argparse
from typing import Any

import numpy as np


def add_optional_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--measure-system", action="store_true")
    parser.add_argument("--measure-open-loop", action="store_true")
    parser.add_argument("--measure-smoothness", action="store_true")
    parser.add_argument("--measure-proxy-success", action="store_true")
    parser.add_argument("--open-loop-trajs", type=int, default=1)
    parser.add_argument("--open-loop-steps", type=int, default=150)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--proxy-rmse-threshold", type=float, default=0.05)
    parser.add_argument("--proxy-first-step-threshold", type=float, default=0.05)
    parser.add_argument("--proxy-sign-flip-threshold", type=float, default=0.20)
    parser.add_argument("--proxy-pullback-threshold", type=float, default=0.20)
    parser.add_argument("--tegrastats-interval-ms", type=int, default=250)
    return parser


def merge_optional_metrics(*metric_groups: dict[str, float] | None) -> dict[str, float]:
    merged: dict[str, float] = {}
    for metrics in metric_groups:
        if not metrics:
            continue
        merged.update(metrics)
    return merged


def _round(value: float) -> float:
    return round(float(value), 4)


def compute_action_error_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    if predicted.shape != ground_truth.shape:
        raise ValueError("predicted 和 ground_truth 形状必须一致")
    if predicted.ndim != 3:
        raise ValueError("action 指标要求输入形状为 [chunks, horizon, action_dim]")

    diff = predicted - ground_truth
    abs_diff = np.abs(diff)
    sq_diff = diff**2
    metrics = {
        "open_loop_mse": _round(np.mean(sq_diff)),
        "open_loop_rmse": _round(np.sqrt(np.mean(sq_diff))),
        "open_loop_mae": _round(np.mean(abs_diff)),
        "open_loop_first_step_rmse": _round(np.sqrt(np.mean(sq_diff[:, 0, :]))),
        "open_loop_evaluated_chunks": float(predicted.shape[0]),
        "open_loop_evaluated_steps": float(predicted.shape[0] * predicted.shape[1]),
    }
    for horizon_index in range(predicted.shape[1]):
        metrics[f"open_loop_horizon_pos_{horizon_index}_rmse"] = _round(
            np.sqrt(np.mean(sq_diff[:, horizon_index, :]))
        )
    return metrics


def compute_action_smoothness(predicted: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float32)
    if predicted.ndim != 3:
        raise ValueError("smoothness 指标要求输入形状为 [chunks, horizon, action_dim]")
    if predicted.shape[1] < 2:
        return {
            "smoothness_first_diff_norm_mean": 0.0,
            "smoothness_first_diff_norm_max": 0.0,
            "smoothness_second_diff_norm_mean": 0.0,
            "smoothness_second_diff_norm_max": 0.0,
            "smoothness_sign_flip_ratio": 0.0,
            "smoothness_pullback_ratio": 0.0,
        }

    first_diff = np.diff(predicted, axis=1)
    first_diff_norm = np.linalg.norm(first_diff, axis=2)

    if predicted.shape[1] >= 3:
        second_diff = np.diff(predicted, n=2, axis=1)
        second_diff_norm = np.linalg.norm(second_diff, axis=2)
    else:
        second_diff_norm = np.zeros((predicted.shape[0], 0), dtype=np.float32)

    if first_diff.shape[1] >= 2:
        prev_diff = first_diff[:, :-1, :]
        next_diff = first_diff[:, 1:, :]
        valid_sign = (np.abs(prev_diff) > eps) & (np.abs(next_diff) > eps)
        sign_flips = (np.sign(prev_diff) * np.sign(next_diff)) < 0
        sign_flip_ratio = (
            float(sign_flips[valid_sign].mean()) if np.any(valid_sign) else 0.0
        )
    else:
        sign_flip_ratio = 0.0

    net_displacement = predicted[:, -1, :] - predicted[:, 0, :]
    valid_pullback = (np.abs(first_diff) > eps) & (np.abs(net_displacement[:, None, :]) > eps)
    pullback = (np.sign(first_diff) * np.sign(net_displacement[:, None, :])) < 0
    pullback_ratio = float(pullback[valid_pullback].mean()) if np.any(valid_pullback) else 0.0

    return {
        "smoothness_first_diff_norm_mean": _round(first_diff_norm.mean()),
        "smoothness_first_diff_norm_max": _round(first_diff_norm.max()),
        "smoothness_second_diff_norm_mean": _round(second_diff_norm.mean() if second_diff_norm.size else 0.0),
        "smoothness_second_diff_norm_max": _round(second_diff_norm.max() if second_diff_norm.size else 0.0),
        "smoothness_sign_flip_ratio": _round(sign_flip_ratio),
        "smoothness_pullback_ratio": _round(pullback_ratio),
    }


def compute_proxy_success(
    metrics: dict[str, float],
    rmse_threshold: float,
    first_step_threshold: float,
    sign_flip_threshold: float | None = None,
    pullback_threshold: float | None = None,
) -> dict[str, float]:
    success = (
        metrics.get("open_loop_rmse", float("inf")) <= rmse_threshold
        and metrics.get("open_loop_first_step_rmse", float("inf")) <= first_step_threshold
    )
    if sign_flip_threshold is not None and "smoothness_sign_flip_ratio" in metrics:
        success = success and metrics["smoothness_sign_flip_ratio"] <= sign_flip_threshold
    if pullback_threshold is not None and "smoothness_pullback_ratio" in metrics:
        success = success and metrics["smoothness_pullback_ratio"] <= pullback_threshold
    return {"proxy_success_rate": 1.0 if success else 0.0}


def _infer_action_keys(step_data: dict[str, Any]) -> list[str]:
    return sorted(key for key in step_data if key.startswith("action."))


def _flatten_ground_truth_chunk(
    step_data: dict[str, Any],
    action_keys: list[str],
    action_horizon: int,
) -> np.ndarray:
    chunk = []
    for horizon_index in range(action_horizon):
        chunk.append(
            np.concatenate(
                [
                    np.atleast_1d(np.asarray(step_data[key][horizon_index], dtype=np.float32))
                    for key in action_keys
                ],
                axis=0,
            )
        )
    return np.stack(chunk, axis=0)


def _flatten_predicted_chunk(
    action_chunk: dict[str, Any],
    action_keys: list[str],
    action_horizon: int,
) -> np.ndarray:
    if "actions" in action_chunk:
        actions = np.asarray(action_chunk["actions"], dtype=np.float32)
        return actions[:action_horizon]

    chunk = []
    for horizon_index in range(action_horizon):
        chunk.append(
            np.concatenate(
                [
                    np.atleast_1d(np.asarray(action_chunk[key][horizon_index], dtype=np.float32))
                    for key in action_keys
                ],
                axis=0,
            )
        )
    return np.stack(chunk, axis=0)


def _collect_action_chunks(policy, dataset, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(dataset, "get_step_data") or not hasattr(dataset, "trajectory_lengths"):
        raise RuntimeError("当前 dataset 不支持 open-loop trajectory 评测")

    max_trajs = min(int(args.open_loop_trajs), len(dataset.trajectory_lengths))
    predicted_chunks = []
    ground_truth_chunks = []

    for traj_id in range(max_trajs):
        traj_steps = min(int(args.open_loop_steps), int(dataset.trajectory_lengths[traj_id]))
        if traj_steps <= 0:
            continue
        step_data = dataset.get_step_data(traj_id, 0)
        action_keys = _infer_action_keys(step_data)
        if not action_keys:
            raise RuntimeError("未在 dataset step_data 中找到 action.* 字段")
        action_horizon = min(int(args.action_horizon), len(step_data[action_keys[0]]))
        if action_horizon <= 0:
            raise RuntimeError("action_horizon 必须大于 0")

        step_index = 0
        while step_index + action_horizon <= traj_steps:
            step_data = dataset.get_step_data(traj_id, step_index)
            predicted_chunk = _flatten_predicted_chunk(
                policy.get_action(step_data),
                action_keys,
                action_horizon,
            )
            ground_truth_chunk = _flatten_ground_truth_chunk(step_data, action_keys, action_horizon)
            predicted_chunks.append(predicted_chunk)
            ground_truth_chunks.append(ground_truth_chunk)
            step_index += action_horizon

    if not predicted_chunks:
        raise RuntimeError("没有收集到完整 action chunk，无法计算离线指标")

    return np.stack(predicted_chunks, axis=0), np.stack(ground_truth_chunks, axis=0)


def collect_optional_metrics(policy, dataset, args: argparse.Namespace) -> dict[str, float]:
    need_quality = args.measure_open_loop or args.measure_smoothness or args.measure_proxy_success
    if not need_quality:
        return {}

    predicted, ground_truth = _collect_action_chunks(policy, dataset, args)
    error_metrics = (
        compute_action_error_metrics(predicted, ground_truth)
        if args.measure_open_loop or args.measure_proxy_success
        else {}
    )
    smoothness_metrics = (
        compute_action_smoothness(predicted)
        if args.measure_smoothness or args.measure_proxy_success
        else {}
    )
    proxy_metrics = (
        compute_proxy_success(
            merge_optional_metrics(error_metrics, smoothness_metrics),
            rmse_threshold=args.proxy_rmse_threshold,
            first_step_threshold=args.proxy_first_step_threshold,
            sign_flip_threshold=args.proxy_sign_flip_threshold,
            pullback_threshold=args.proxy_pullback_threshold,
        )
        if args.measure_proxy_success
        else {}
    )
    return merge_optional_metrics(error_metrics, smoothness_metrics, proxy_metrics)
