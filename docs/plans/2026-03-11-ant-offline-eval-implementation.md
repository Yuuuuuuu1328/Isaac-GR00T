# Ant Offline Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend ANT local inference benchmark scripts with Jetson system telemetry, offline quality metrics, smoothness analysis, and dataset-backed `proxy_success_rate`.

**Architecture:** Keep the four existing ANT benchmark scripts as the user-facing entry points. Add small shared modules for system telemetry and offline quality so PyTorch and TensorRT paths reuse the same metric logic and keep `LatencyRecord` output stable. Preserve latency-only defaults and gate new work behind explicit CLI flags.

**Tech Stack:** Python, argparse, PyTorch CUDA stats, Jetson `tegrastats`, NumPy, unittest, subprocess-free unit tests

---

### Task 1: Add failing parser tests for the new evaluation flags

**Files:**
- Modify: `deployment_scripts/ant/test_ant_cli.py`
- Modify: `deployment_scripts/ant/test_local_inference_breakdown.py`
- Modify: `deployment_scripts/ant/test_local_inference_tensorrt.py`

**Step 1: Write the failing test**

```python
def test_e2e_parser_exposes_offline_eval_flags(self):
    args = build_parser().parse_args([])
    self.assertFalse(args.measure_system)
    self.assertFalse(args.measure_open_loop)
    self.assertFalse(args.measure_smoothness)
    self.assertFalse(args.measure_proxy_success)
    self.assertEqual(args.open_loop_trajs, 1)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest deployment_scripts/ant/test_local_inference_tensorrt.py::TensorRTCliParserTest::test_e2e_parser_exposes_offline_eval_flags -v`
Expected: FAIL because the parser does not expose the new flags yet.

**Step 3: Write minimal implementation**

Add the shared parser arguments to all four benchmark scripts.

**Step 4: Run test to verify it passes**

Run: `python -m pytest deployment_scripts/ant/test_local_inference_tensorrt.py::TensorRTCliParserTest::test_e2e_parser_exposes_offline_eval_flags -v`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py deployment_scripts/ant/local_inference_e2e.py deployment_scripts/ant/local_inference_breakdown.py deployment_scripts/ant/local_inference_tensorrt_e2e.py deployment_scripts/ant/local_inference_tensorrt_breakdown.py
git commit -m "feat: add offline evaluation flags to ant benchmarks"
```

### Task 2: Add failing tests for pure quality metrics

**Files:**
- Create: `deployment_scripts/ant/test_quality_metrics.py`
- Create: `deployment_scripts/ant/quality_metrics.py`

**Step 1: Write the failing test**

```python
def test_compute_action_smoothness_reports_pullback_ratio():
    pred = np.array([[[0.0], [1.0], [0.2]]], dtype=np.float32)
    metrics = compute_action_smoothness(pred)
    assert metrics["smoothness_pullback_ratio"] > 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest deployment_scripts/ant/test_quality_metrics.py::test_compute_action_smoothness_reports_pullback_ratio -v`
Expected: FAIL because `quality_metrics.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `flatten_action_dict()`
- `compute_action_error_metrics()`
- `compute_action_smoothness()`
- `compute_proxy_success()`

**Step 4: Run test to verify it passes**

Run: `python -m pytest deployment_scripts/ant/test_quality_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/test_quality_metrics.py deployment_scripts/ant/quality_metrics.py
git commit -m "feat: add ant offline quality metrics"
```

### Task 3: Add failing tests for Jetson telemetry parsing

**Files:**
- Create: `deployment_scripts/ant/test_system_metrics.py`
- Create: `deployment_scripts/ant/system_metrics.py`

**Step 1: Write the failing test**

```python
def test_parse_tegrastats_lines_extracts_power_temp_and_freq():
    lines = [
        "RAM 220/62841MB ... GR3D_FREQ 87% EMC_FREQ 61% CPU@58.5C GPU@62C VDD_GPU_SOC 10342mW/9988mW",
    ]
    metrics = parse_tegrastats_lines(lines)
    assert metrics["tegrastats_gr3d_freq_pct_max"] == 87.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest deployment_scripts/ant/test_system_metrics.py::test_parse_tegrastats_lines_extracts_power_temp_and_freq -v`
Expected: FAIL because `system_metrics.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `parse_tegrastats_lines()`
- `collect_torch_peak_memory_metrics()`
- `TegrastatsSampler`

**Step 4: Run test to verify it passes**

Run: `python -m pytest deployment_scripts/ant/test_system_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/test_system_metrics.py deployment_scripts/ant/system_metrics.py
git commit -m "feat: add ant system telemetry helpers"
```

### Task 4: Add failing integration tests for record merging

**Files:**
- Modify: `deployment_scripts/ant/test_local_inference_breakdown.py`
- Modify: `deployment_scripts/ant/test_local_inference_tensorrt.py`

**Step 1: Write the failing test**

```python
def test_merge_optional_metrics_preserves_existing_latency_fields():
    merged = merge_optional_metrics(
        {"e2e_total_ms": 10.0},
        {"open_loop_rmse": 0.5},
        {"smoothness_pullback_ratio": 0.2},
        {"tegrastats_gr3d_freq_pct_max": 87.0},
    )
    assert merged["e2e_total_ms"] == 10.0
    assert merged["open_loop_rmse"] == 0.5
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest deployment_scripts/ant/test_local_inference_breakdown.py::test_merge_optional_metrics_preserves_existing_latency_fields -v`
Expected: FAIL because the merge helper does not exist yet.

**Step 3: Write minimal implementation**

Add a shared merge helper and wire optional metric collection into the benchmark scripts.

**Step 4: Run test to verify it passes**

Run: `python -m pytest deployment_scripts/ant/test_local_inference_breakdown.py::test_merge_optional_metrics_preserves_existing_latency_fields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py deployment_scripts/ant/local_inference_breakdown.py deployment_scripts/ant/local_inference_tensorrt_breakdown.py
git commit -m "feat: wire optional metrics into ant benchmarks"
```

### Task 5: Verify the selected ANT benchmark surface end-to-end

**Files:**
- Modify: `deployment_scripts/ant/test_ant_cli.py`
- Modify: `deployment_scripts/ant/test_quality_metrics.py`
- Modify: `deployment_scripts/ant/test_system_metrics.py`
- Modify: `deployment_scripts/ant/test_local_inference_breakdown.py`
- Modify: `deployment_scripts/ant/test_local_inference_tensorrt.py`

**Step 1: Run the focused test suite**

Run: `python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py deployment_scripts/ant/test_profile_utils.py deployment_scripts/ant/test_quality_metrics.py deployment_scripts/ant/test_system_metrics.py -v`
Expected: PASS

**Step 2: Run a smoke CLI check**

Run: `python deployment_scripts/ant/local_inference_e2e.py --help`
Expected: PASS and output includes the new flags.

**Step 3: Record implementation summary**

Append a short summary to `/home/jetson/Desktop/project/codex.md` with the metric boundaries and verification commands used.

**Step 4: Commit**

```bash
git add docs/plans/2026-03-11-ant-offline-eval-design.md docs/plans/2026-03-11-ant-offline-eval-implementation.md deployment_scripts/ant
git commit -m "feat: add ant offline evaluation metrics"
```
