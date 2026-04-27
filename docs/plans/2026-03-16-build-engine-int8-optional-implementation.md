# Optional INT8 TensorRT Script Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional INT8 support to the TensorRT build and inference scripts while preserving existing defaults.

**Architecture:** Keep the current dtype-derived ONNX and engine naming convention. Expand accepted dtype values to include `int8`, then update docs and focused regression tests to cover the explicit `gr00t_onnx_int8` workflow.

**Tech Stack:** Bash, Python `argparse`, Python `unittest`, pytest

---

### Task 1: Add focused failing tests for INT8 support

**Files:**
- Modify: `deployment_scripts/test_build_engine.py`
- Modify: `deployment_scripts/ant/test_local_inference_tensorrt.py`

**Step 1: Write the failing test**

Add assertions that the build script mentions `int8` in dtype declarations and validation, and that TensorRT CLI parsers accept `--vit-dtype int8 --llm-dtype int8 --dit-dtype int8`.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/test_build_engine.py deployment_scripts/ant/test_local_inference_tensorrt.py -q`

Expected: FAIL because current dtype validation and parser choices do not include `int8`.

**Step 3: Write minimal implementation**

Do not implement in this task.

**Step 4: Run test to verify it passes**

Do not run in this task.

**Step 5: Commit**

Skip commit because the workspace is already dirty.

### Task 2: Implement optional INT8 support in build and inference entrypoints

**Files:**
- Modify: `deployment_scripts/build_engine.sh`
- Modify: `deployment_scripts/gr00t_inference.py`
- Modify: `deployment_scripts/ant/local_inference_tensorrt_e2e.py`
- Modify: `deployment_scripts/ant/local_inference_tensorrt_breakdown.py`
- Modify: `deployment_scripts/trt_model_forward.py`

**Step 1: Write the failing test**

Use Task 1 as the red state.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/test_build_engine.py deployment_scripts/ant/test_local_inference_tensorrt.py -q`

Expected: FAIL with missing `int8` support.

**Step 3: Write minimal implementation**

Expand dtype validation, parser `choices`, and help text to include `int8` while preserving all existing defaults.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/test_build_engine.py deployment_scripts/ant/test_local_inference_tensorrt.py -q`

Expected: PASS.

**Step 5: Commit**

Skip commit because the workspace is already dirty.

### Task 3: Document the explicit INT8 workflow

**Files:**
- Modify: `deployment_scripts/README.md`
- Modify: `deployment_scripts/orin/README.md`

**Step 1: Write the failing test**

No automated doc test required.

**Step 2: Run test to verify it fails**

Not applicable.

**Step 3: Write minimal implementation**

Add explicit `gr00t_onnx_int8` build and inference examples.

**Step 4: Run test to verify it passes**

Manually confirm the examples match the implemented CLI options and script behavior.

**Step 5: Commit**

Skip commit because the workspace is already dirty.
