# TensorRT Ant Local Inference Scripts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add TensorRT local inference `e2e` and `breakdown` benchmark scripts under `deployment_scripts/ant` that mirror the existing PyTorch benchmark workflow and reporting format.

**Architecture:** Reuse the existing ant dataset-loading and transform-breakdown helpers, then switch the policy into TensorRT mode with `setup_tensorrt_engines()`. Keep `e2e` focused on end-to-end `policy.get_action()` latency, and implement `breakdown` by following the real TensorRT forward path in `deployment_scripts/trt_model_forward.py` so engine timings align with the deployed runtime rather than a synthetic approximation.

**Tech Stack:** Python, argparse, PyTorch CUDA events, TensorRT Python runtime wrappers, unittest, subprocess CLI verification

---

### Task 1: Add failing CLI and parser tests

**Files:**
- Modify: `deployment_scripts/ant/test_ant_cli.py`
- Create: `deployment_scripts/ant/test_local_inference_tensorrt.py`

**Step 1: Write the failing test**

```python
def test_local_inference_tensorrt_breakdown_help_succeeds(self):
    result = _run_help("local_inference_tensorrt_breakdown.py")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--trt-engine-path", result.stdout)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_tensorrt.py -v`
Expected: FAIL because the new TensorRT scripts do not exist yet.

**Step 3: Write minimal implementation**

Create the new TensorRT benchmark modules with parser builders that expose the TensorRT engine arguments and import cleanly without requiring TensorRT during `--help`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_tensorrt.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_tensorrt.py deployment_scripts/ant/local_inference_tensorrt_e2e.py deployment_scripts/ant/local_inference_tensorrt_breakdown.py
git commit -m "feat: add ant tensorrt local benchmark scripts"
```

### Task 2: Implement TensorRT E2E benchmark

**Files:**
- Create: `deployment_scripts/ant/local_inference_tensorrt_e2e.py`
- Test: `deployment_scripts/ant/test_ant_cli.py`
- Test: `deployment_scripts/ant/test_local_inference_tensorrt.py`

**Step 1: Write the failing test**

```python
def test_e2e_parser_exposes_engine_dtype_flags():
    parser = build_e2e_parser()
    args = parser.parse_args([])
    assert args.vit_dtype
    assert args.llm_dtype
    assert args.dit_dtype
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest deployment_scripts/ant/test_local_inference_tensorrt.py::TensorRTCliParserTest::test_e2e_parser_exposes_engine_dtype_flags -v`
Expected: FAIL because the parser function does not exist yet.

**Step 3: Write minimal implementation**

Implement the TensorRT runtime loader, enable `setup_tensorrt_engines()`, keep warmup/measure loops aligned with the existing PyTorch E2E script, and record TensorRT engine metadata in each latency record.

**Step 4: Run test to verify it passes**

Run: `python -m pytest deployment_scripts/ant/test_local_inference_tensorrt.py::TensorRTCliParserTest::test_e2e_parser_exposes_engine_dtype_flags -v`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/local_inference_tensorrt_e2e.py deployment_scripts/ant/test_local_inference_tensorrt.py
git commit -m "feat: add tensorrt e2e ant benchmark"
```

### Task 3: Implement TensorRT breakdown benchmark

**Files:**
- Create: `deployment_scripts/ant/local_inference_tensorrt_breakdown.py`
- Test: `deployment_scripts/ant/test_local_inference_tensorrt.py`

**Step 1: Write the failing test**

```python
def test_sum_metrics_only_counts_requested_names():
    total = _sum_metrics({"a": 1.0, "b": 2.0}, ("a",))
    assert total == 1.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest deployment_scripts/ant/test_local_inference_tensorrt.py::TensorRTBreakdownHelperTest::test_sum_metrics_only_counts_requested_names -v`
Expected: FAIL because the helper does not exist yet.

**Step 3: Write minimal implementation**

Implement the TensorRT breakdown path by reusing transform preprocessing from the existing ant script, timing each TensorRT engine call separately, and aggregating engine plus residual PyTorch tensor-work metrics into the same `LatencyRecord` format.

**Step 4: Run test to verify it passes**

Run: `python -m pytest deployment_scripts/ant/test_local_inference_tensorrt.py::TensorRTBreakdownHelperTest::test_sum_metrics_only_counts_requested_names -v`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/local_inference_tensorrt_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py
git commit -m "feat: add tensorrt breakdown ant benchmark"
```

### Task 4: Verify behavior and document operational risks

**Files:**
- Modify: `deployment_scripts/ant/test_ant_cli.py`
- Modify: `deployment_scripts/ant/test_local_inference_tensorrt.py`
- Modify: `/home/jetson/Desktop/project/codex.md`

**Step 1: Write the failing test**

```python
def test_breakdown_parser_defaults_engine_path_to_repo_engine_dir():
    args = build_breakdown_parser().parse_args([])
    assert args.trt_engine_path.endswith("gr00t_engine")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest deployment_scripts/ant/test_local_inference_tensorrt.py::TensorRTCliParserTest::test_breakdown_parser_defaults_engine_path_to_repo_engine_dir -v`
Expected: FAIL until the parser default is finalized.

**Step 3: Write minimal implementation**

Finalize parser defaults, run the targeted test suite, then append the implementation summary and operational caveats to `codex.md`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_tensorrt.py deployment_scripts/ant/test_local_inference_breakdown.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/plans/2026-03-11-ant-tensorrt-local-inference-plan.md deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_tensorrt.py /home/jetson/Desktop/project/codex.md
git commit -m "docs: record ant tensorrt benchmark implementation plan"
```
