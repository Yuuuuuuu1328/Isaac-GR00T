# ANT Inference Entrypoint Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate ANT local inference execution behind one shared entrypoint while preserving the four existing commands as compatibility wrappers.

**Architecture:** Add a shared `deployment_scripts/ant/local_inference.py` module that models the two independent axes explicitly, then convert the four existing scripts into thin wrappers that pin `backend`, `mode`, and legacy script metadata. Keep the current runtime helpers and timing logic intact, and only centralize parser composition and execution orchestration.

**Tech Stack:** Python, argparse, unittest, pytest, existing ANT profiling helpers

---

### Task 1: Add failing tests for the shared entrypoint parser and compatibility wrappers

**Files:**
- Modify: `deployment_scripts/ant/test_ant_cli.py`
- Modify: `deployment_scripts/ant/test_local_inference_tensorrt.py`
- Modify: `deployment_scripts/ant/test_local_inference_breakdown.py`

**Step 1: Write the failing test**

```python
def test_shared_parser_defaults_to_pytorch_e2e(self):
    args = build_shared_parser().parse_args([])
    self.assertEqual(args.backend, "pytorch")
    self.assertEqual(args.mode, "e2e")
```

```python
def test_legacy_wrapper_help_hides_backend_switch(self):
    result = _run_help("local_inference_e2e.py")
    self.assertNotIn("--backend", result.stdout)
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py -q`
Expected: FAIL because the shared entrypoint does not exist yet and wrappers are not forwarding.

**Step 3: Write minimal implementation**

Add parser tests and wrapper expectations first, without implementation.

**Step 4: Run test to verify it fails for the expected reason**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py -q`
Expected: FAIL due to missing shared parser or wrapper behavior.

**Step 5: Commit**

```bash
git add deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py
git commit -m "test: add ant shared entrypoint expectations"
```

### Task 2: Add the shared entrypoint module with parser composition and dispatch

**Files:**
- Create: `deployment_scripts/ant/local_inference.py`

**Step 1: Write the failing test**

```python
def test_run_mode_dispatches_to_pytorch_e2e():
    args = build_parser().parse_args([])
    records = run_with_args(args, script_name="local_inference_e2e.py")
    assert records is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_tensorrt.py -q`
Expected: FAIL because the shared entrypoint module does not exist.

**Step 3: Write minimal implementation**

Implement:

- shared parser helpers for common, PyTorch-only, and TensorRT-only options
- dispatch configuration for the four backend/mode combinations
- shared warmup and measurement loop
- script-name injection so legacy wrappers preserve metadata labels

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_tensorrt.py -q`
Expected: PASS for the new parser and dispatch expectations.

**Step 5: Commit**

```bash
git add deployment_scripts/ant/local_inference.py deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_tensorrt.py
git commit -m "feat: add shared ant local inference entrypoint"
```

### Task 3: Convert the four legacy scripts into thin compatibility wrappers

**Files:**
- Modify: `deployment_scripts/ant/local_inference_e2e.py`
- Modify: `deployment_scripts/ant/local_inference_breakdown.py`
- Modify: `deployment_scripts/ant/local_inference_tensorrt_e2e.py`
- Modify: `deployment_scripts/ant/local_inference_tensorrt_breakdown.py`

**Step 1: Write the failing test**

```python
def test_legacy_wrapper_build_parser_preserves_existing_defaults(self):
    args = build_e2e_parser().parse_args([])
    self.assertFalse(hasattr(args, "backend"))
```

```python
def test_breakdown_wrapper_default_engine_path_is_preserved(self):
    args = build_breakdown_parser().parse_args([])
    self.assertTrue(args.trt_engine_path.endswith("gr00t_engine"))
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py -q`
Expected: FAIL until wrappers expose the expected legacy parser views.

**Step 3: Write minimal implementation**

Replace duplicated orchestration in the four legacy files with:

- imports of the shared module
- compatibility `build_parser()` calls with fixed mode/backend
- `main()` forwarding with legacy script names
- retention of helper functions that current tests import directly

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/local_inference_e2e.py deployment_scripts/ant/local_inference_breakdown.py deployment_scripts/ant/local_inference_tensorrt_e2e.py deployment_scripts/ant/local_inference_tensorrt_breakdown.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py
git commit -m "refactor: route ant legacy entrypoints through shared runner"
```

### Task 4: Verify compatibility across the ANT surface

**Files:**
- Modify: `deployment_scripts/ant/test_ant_cli.py`

**Step 1: Run focused verification**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py deployment_scripts/ant/test_profile_utils.py deployment_scripts/ant/test_quality_metrics.py deployment_scripts/ant/test_system_metrics.py deployment_scripts/ant/test_torch_compile_utils.py -q`
Expected: PASS

**Step 2: Run CLI smoke checks**

Run: `python deployment_scripts/ant/local_inference.py --help`
Expected: PASS and includes `--backend` and `--mode`

Run: `python deployment_scripts/ant/local_inference_e2e.py --help`
Expected: PASS and does not expose `--backend`

Run: `python deployment_scripts/ant/local_inference_tensorrt_breakdown.py --help`
Expected: PASS and includes TensorRT dtype flags

**Step 3: Review diff for behavior preservation**

Check that:

- helper functions used by existing tests remain available
- TensorRT default engine paths are unchanged
- `meta["script"]` is still the legacy script name under wrapper execution

**Step 4: Commit**

```bash
git add docs/plans/2026-03-18-ant-inference-entrypoint-consolidation-design.md docs/plans/2026-03-18-ant-inference-entrypoint-consolidation-implementation.md deployment_scripts/ant
git commit -m "refactor: consolidate ant local inference entrypoints"
```
