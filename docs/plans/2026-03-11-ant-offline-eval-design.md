# Ant Offline Evaluation Design

**Date:** 2026-03-11

**Goal:** Extend the four local ANT inference benchmark scripts so they can report offline quality and system telemetry on Jetson Orin without pretending to measure true closed-loop success.

## Scope

This design applies to:

- `deployment_scripts/ant/local_inference_e2e.py`
- `deployment_scripts/ant/local_inference_breakdown.py`
- `deployment_scripts/ant/local_inference_tensorrt_e2e.py`
- `deployment_scripts/ant/local_inference_tensorrt_breakdown.py`

The current scripts already share a JSONL-friendly `LatencyRecord` output format. The extension keeps that format and adds optional metric namespaces instead of introducing a separate reporting path.

## Non-Goals

- No claim of true `closed_loop_success_rate` without a simulator or physical robot.
- No dependency on RoboCasa or real robot hardware for the default local evaluation path.
- No large refactor of the TensorRT forward path beyond what is needed to collect metrics.

## Metric Boundaries

The extended scripts will expose three metric groups:

1. `latency/system`
   - End-to-end and breakdown latency that already exist.
   - Torch CUDA peak memory where available.
   - Jetson `tegrastats` telemetry sampled during the benchmark window.

2. `offline_quality`
   - Open-loop action error over dataset trajectories.
   - First-step error and per-horizon-position error so chunk quality is visible beyond a single average.

3. `behavior_proxy`
   - Action smoothness proxies computed from predicted action chunks.
   - `proxy_success_rate`, explicitly named as a proxy, not a real closed-loop success metric.

## Why Proxy Success Exists

The user currently has no simulation environment and no robot connection. That makes real closed-loop measurement impossible. Instead, the scripts will optionally compute a dataset-backed `proxy_success_rate` using thresholded offline quality checks. The name must remain `proxy_success_rate` everywhere in code and output.

## Architecture

Add two shared utility modules under `deployment_scripts/ant`:

- `system_metrics.py`
  - Wrap Torch CUDA memory peak collection.
  - Spawn and parse `tegrastats` sampling around a benchmark window.
  - Aggregate power, temperature, memory, and frequency telemetry into scalar metrics.

- `quality_metrics.py`
  - Compute open-loop error summaries from dataset trajectories.
  - Compute smoothness proxies from predicted action chunks.
  - Compute `proxy_success_rate` from configured thresholds.

The four existing scripts will opt in to these modules through CLI flags rather than always paying the runtime cost.

## CLI Design

All four scripts will gain a common family of flags:

- `--measure-system`
- `--measure-open-loop`
- `--measure-smoothness`
- `--measure-proxy-success`
- `--open-loop-trajs`
- `--open-loop-steps`
- `--action-horizon`
- `--proxy-rmse-threshold`
- `--proxy-first-step-threshold`
- `--tegrastats-interval-ms`

The default behavior remains latency-only, preserving the current fast path.

## System Metric Semantics

### GPU Peak Memory

Two sources are kept separate:

- `torch_peak_memory_allocated_mb`
- `torch_peak_memory_reserved_mb`
- `system_peak_memory_used_mb`

Torch values are exact only for Torch-managed allocations. On TensorRT they are incomplete, so the report keeps both Torch and system-derived values instead of collapsing them into a single misleading number.

### Tegrastats

`tegrastats` sampling will run in a subprocess during warmup+measure or during the exact measured loop, then parse the captured lines into:

- `tegrastats_gr3d_freq_pct_mean`
- `tegrastats_gr3d_freq_pct_max`
- `tegrastats_emc_freq_pct_mean`
- `tegrastats_gpu_temp_c_mean`
- `tegrastats_cpu_temp_c_mean`
- `tegrastats_vdd_total_mw_mean`
- `tegrastats_vdd_total_mw_max`
- `tegrastats_ram_used_mb_max`

`throttled` should not be inferred from a single line. Instead the parser will compute a conservative boolean such as `tegrastats_possible_throttle` when observed GPU or EMC frequency stays materially below the observed session max while the workload is active.

## Offline Quality Metrics

Open-loop metrics will be computed over dataset trajectories, reusing the same policy loading path already used by the ANT scripts:

- `open_loop_mse`
- `open_loop_rmse`
- `open_loop_mae`
- `open_loop_first_step_rmse`
- `open_loop_horizon_pos_{i}_rmse`

These metrics should be based on unnormalized action space, matching the current repository evaluation utilities.

## Smoothness Proxies

For each predicted action chunk, compute:

- first-order delta norm mean and max
- second-order delta norm mean and max
- sign-flip ratio per action dimension, then mean across dimensions
- pullback ratio

`pullback ratio` is defined as the fraction of local increments inside a chunk whose sign disagrees with that dimension's net displacement across the chunk. This is a reproducible approximation of "回拉".

## Proxy Success Definition

`proxy_success_rate` will be based on thresholding offline trajectory statistics, not on an environment reward:

- success if whole-trajectory RMSE <= `proxy_rmse_threshold`
- and first-step RMSE <= `proxy_first_step_threshold`
- and smoothness constraints are within configured limits when smoothness measurement is enabled

This keeps the value interpretable while remaining honest about what it is.

## Error Handling

- If `tegrastats` is unavailable, scripts should continue and emit a clear metadata flag such as `tegrastats_available=false`.
- If CUDA is unavailable, Torch peak memory metrics become `null` or are omitted.
- If a user asks for `proxy_success_rate` without open-loop inputs, the script should either compute the needed offline metrics automatically or fail with a precise message.

## Testing Strategy

Use TDD with narrow unit tests first:

- parser tests for the new CLI flags
- telemetry parser tests using canned `tegrastats` lines
- smoothness and proxy-success pure-function tests
- one integration-style test that checks new metrics are merged into `LatencyRecord`

Avoid tests that require actual TensorRT engines, Jetson hardware, or long trajectory runs.
