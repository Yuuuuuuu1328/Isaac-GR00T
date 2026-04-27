# Aistudio TensorRT Breakdown Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep the Aistudio service request/response contract unchanged while switching the TensorRT `/predict` execution path in `scripts/aistudio_test.py` to the new-interaction breakdown flow.

**Architecture:** Extract a reusable TensorRT breakdown inference helper from `deployment_scripts/ant/local_inference_tensorrt_breakdown.py` that returns both postprocessed actions and metrics. Update `scripts/aistudio_test.py` to preserve the current external payload shape, perform any needed internal input reshaping, and call the helper only for TensorRT. Lock the behavior with unit tests before changing production code.

**Tech Stack:** Python, unittest/pytest, FastAPI adapter code, NumPy, TensorRT breakdown helpers

---

### Task 1: Add Aistudio service regression tests

**Files:**
- Create: `tests/test_aistudio_service.py`
- Test: `tests/test_aistudio_service.py`

**Step 1: Write the failing test**

Write tests that verify:
- TensorRT runtime keeps the incoming payload format unchanged and returns the same response envelope.
- TensorRT uses an injected breakdown runner with internally reshaped observation data.
- PyTorch continues to use `policy.get_action(...)` with the existing batched input layout.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_service.py -q`
Expected: FAIL because the runtime does not yet support a TensorRT-specific breakdown runner or separate internal observation conversion.

**Step 3: Write minimal implementation**

Add the smallest production changes needed to support the tests.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_service.py -q`
Expected: PASS

### Task 2: Extract reusable TensorRT breakdown action helper

**Files:**
- Modify: `deployment_scripts/ant/local_inference_tensorrt_breakdown.py`
- Test: `deployment_scripts/ant/test_local_inference_tensorrt.py`

**Step 1: Write the failing test**

Add a test proving the shared helper returns postprocessed actions and preserves the existing metrics used by `_run_single_breakdown(...)`.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference_tensorrt.py -q`
Expected: FAIL because the reusable helper does not exist yet.

**Step 3: Write minimal implementation**

Refactor the breakdown code so the existing profiling path reuses a helper that returns `(postprocessed_action, metrics)`.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference_tensorrt.py -q`
Expected: PASS

### Task 3: Wire TensorRT Aistudio runtime to the shared helper

**Files:**
- Modify: `scripts/aistudio_test.py`
- Test: `tests/test_aistudio_service.py`

**Step 1: Write the failing test**

Ensure the TensorRT runtime branch uses the shared breakdown helper and keeps the service response format untouched.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_service.py -q`
Expected: FAIL until the runtime is wired to the new helper.

**Step 3: Write minimal implementation**

Add internal observation conversion for the breakdown helper, inject the TensorRT action runner at runtime construction, and keep the existing output postprocessing and envelope.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_service.py -q`
Expected: PASS

### Task 4: Final verification

**Files:**
- Test: `tests/test_aistudio_service.py`
- Test: `deployment_scripts/ant/test_local_inference_tensorrt.py`

**Step 1: Run focused verification**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_service.py deployment_scripts/ant/test_local_inference_tensorrt.py -q`
Expected: PASS
