# Ant Torch Compile Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional `--use-torch-compile` switch to the local PyTorch ANT inference scripts only.

**Architecture:** Introduce a shared helper under `deployment_scripts/ant` that exposes the parser flag and wires `torch.compile` to the exact PyTorch callables each script already uses. Keep TensorRT scripts, ONNX export, and TRT quantization/build behavior unchanged.

**Tech Stack:** Python, argparse, PyTorch `torch.compile`, unittest, pytest

---

### Task 1: Add failing parser tests for the new compile flag

**Files:**
- Modify: `deployment_scripts/ant/test_ant_cli.py`
- Modify: `deployment_scripts/ant/test_local_inference_breakdown.py`

**Step 1: Write the failing test**

```python
def test_parser_exposes_use_torch_compile_flag(self):
    args = build_parser().parse_args([])
    self.assertFalse(args.use_torch_compile)
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference_breakdown.py::LocalInferenceBreakdownTransformTest::test_parser_exposes_use_torch_compile_flag -q`
Expected: FAIL because the parser does not expose the flag yet.

**Step 3: Write minimal implementation**

Add `--use-torch-compile` to the two PyTorch ANT parsers.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference_breakdown.py::LocalInferenceBreakdownTransformTest::test_parser_exposes_use_torch_compile_flag -q`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/local_inference_e2e.py deployment_scripts/ant/local_inference_breakdown.py
git commit -m "feat: add torch compile flags to ant pytorch scripts"
```

### Task 2: Add failing tests for compile wiring helpers

**Files:**
- Create: `deployment_scripts/ant/torch_compile_utils.py`
- Create: `deployment_scripts/ant/test_torch_compile_utils.py`

**Step 1: Write the failing test**

```python
def test_enable_torch_compile_for_e2e_compiles_model_get_action():
    policy = SimpleNamespace(model=SimpleNamespace(get_action=lambda x: x))
    enable_torch_compile_for_e2e(policy)
    assert hasattr(policy.model, "_compiled_get_action")
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_torch_compile_utils.py -q`
Expected: FAIL because the helper module does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `add_torch_compile_arg()`
- `enable_torch_compile_for_e2e()`
- `enable_torch_compile_for_breakdown()`

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_torch_compile_utils.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/torch_compile_utils.py deployment_scripts/ant/test_torch_compile_utils.py
git commit -m "feat: add ant torch compile helpers"
```

### Task 3: Wire compile support into the two PyTorch scripts

**Files:**
- Modify: `deployment_scripts/ant/local_inference_e2e.py`
- Modify: `deployment_scripts/ant/local_inference_breakdown.py`

**Step 1: Write the failing test**

```python
def test_compile_flag_is_visible_in_help():
    result = _run_help("local_inference_e2e.py")
    self.assertIn("--use-torch-compile", result.stdout)
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ant_cli.py -q`
Expected: FAIL until help output includes the new flag.

**Step 3: Write minimal implementation**

Wire the helper into `_load_runtime()` or immediately after it so:

- E2E compiles `policy.model.get_action`
- breakdown stores compiled callable attributes and uses them in existing stage functions

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_torch_compile_utils.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/local_inference_e2e.py deployment_scripts/ant/local_inference_breakdown.py deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_torch_compile_utils.py
git commit -m "feat: wire torch compile into ant pytorch inference"
```

### Task 4: Verify the narrowed ANT surface

**Files:**
- Modify: `/home/jetson/Desktop/project/codex.md`

**Step 1: Run focused verification**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_torch_compile_utils.py deployment_scripts/ant/test_quality_metrics.py deployment_scripts/ant/test_system_metrics.py deployment_scripts/ant/test_profile_utils.py -q`
Expected: PASS

**Step 2: Run CLI smoke checks**

Run: `python deployment_scripts/ant/local_inference_e2e.py --help`
Expected: PASS and includes `--use-torch-compile`

Run: `python deployment_scripts/ant/local_inference_breakdown.py --help`
Expected: PASS and includes `--use-torch-compile`

**Step 3: Record implementation summary**

Append the compile-scope and verification result to `/home/jetson/Desktop/project/codex.md`.

**Step 4: Commit**

```bash
git add docs/plans/2026-03-12-ant-torch-compile-design.md docs/plans/2026-03-12-ant-torch-compile-implementation.md deployment_scripts/ant /home/jetson/Desktop/project/codex.md
git commit -m "feat: add optional torch compile to ant pytorch scripts"
```
