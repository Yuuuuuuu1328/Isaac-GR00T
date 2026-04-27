# Formal INT8 ONNX Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add first-class INT8 ONNX export support for ViT, LLM, and DiT in `deployment_scripts/export_onnx.py` so the exported artifacts are produced by the current script instead of relying on external or stale INT8 files.

**Architecture:** Extend the existing dtype-driven export pipeline instead of adding a second script. Introduce explicit INT8 quantization config selection for ViT, LLM, and DiT, plus a small ViT-specific ONNX dtype-alignment pass so TensorRT can parse the exported QDQ graph. Keep existing defaults unchanged and keep file naming as `vit_<dtype>.onnx`, `llm_<dtype>.onnx`, and `DiT_<dtype>.onnx`.

**Tech Stack:** Python, argparse, unittest, ONNX, ModelOpt quantization

---

### Task 1: Add focused failing tests for INT8 export support

**Files:**
- Create: `deployment_scripts/test_export_onnx.py`

**Step 1: Write the failing test**

Add focused tests that verify:
- `export_onnx.py` parser accepts `--vit-dtype int8 --llm-dtype int8 --dit-dtype int8`
- ViT quantization config selection supports `int8`
- LLM quantization config selection supports `int8`
- DiT quantization config selection supports `int8`

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/test_export_onnx.py -q`

Expected: FAIL because `export_onnx.py` currently only accepts `fp16/fp8` for ViT and DiT, and `fp16/nvfp4/fp8` for LLM.

**Step 3: Write minimal implementation**

Do not implement in this task.

**Step 4: Run test to verify it passes**

Do not run in this task.

**Step 5: Commit**

Skip commit because the workspace is already dirty.

### Task 2: Implement INT8 config selection and parser support

**Files:**
- Modify: `deployment_scripts/export_onnx.py`

**Step 1: Write the failing test**

Use Task 1 as the red state.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/test_export_onnx.py -q`

Expected: FAIL with parser choice or unsupported precision assertions.

**Step 3: Write minimal implementation**

Add helpers that choose quantization configs from ModelOpt:
- ViT: `fp8 -> mtq.FP8_DEFAULT_CFG`, `int8 -> mtq.INT8_DEFAULT_CFG`, `fp16 -> no quantization`
- LLM: `nvfp4 -> mtq.NVFP4_AWQ_FULL_CFG`, `fp8 -> mtq.FP8_DEFAULT_CFG`, `int8 -> mtq.INT8_DEFAULT_CFG`
- DiT: `fp8 -> mtq.FP8_DEFAULT_CFG`, `int8 -> mtq.INT8_DEFAULT_CFG`, `fp16 -> no quantization`

Update parser choices and precision assertions accordingly.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/test_export_onnx.py -q`

Expected: PASS.

**Step 5: Commit**

Skip commit because the workspace is already dirty.

### Task 3: Fix ViT INT8 ONNX dtype alignment

**Files:**
- Modify: `deployment_scripts/export_onnx.py`
- Modify: `deployment_scripts/test_export_onnx.py`

**Step 1: Write the failing test**

Add a focused unit test for a ViT INT8 ONNX post-process helper that rewrites LayerNorm scale/bias tensors to match float residual inputs after INT8 QDQ export.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/test_export_onnx.py -q`

Expected: FAIL because the helper does not exist yet.

**Step 3: Write minimal implementation**

Add a small ONNX graph rewrite helper used only for ViT INT8 export:
- Load the exported ONNX
- Find `LayerNormalization` nodes
- If the data input is `FLOAT` and weight/bias initializers are `FLOAT16`, rewrite those initializers to `FLOAT`
- Save the graph back in place

Keep the pass narrow so it only normalizes the known TensorRT-breaking mismatch.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/test_export_onnx.py -q`

Expected: PASS.

**Step 5: Commit**

Skip commit because the workspace is already dirty.

### Task 4: Run focused verification

**Files:**
- Modify: `deployment_scripts/export_onnx.py`
- Create: `deployment_scripts/test_export_onnx.py`

**Step 1: Write the failing test**

No additional test.

**Step 2: Run test to verify it fails**

Not applicable.

**Step 3: Write minimal implementation**

No additional code.

**Step 4: Run test to verify it passes**

Run:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/test_export_onnx.py -q`
- `python -m py_compile deployment_scripts/export_onnx.py`

Expected: both commands succeed.

**Step 5: Commit**

Skip commit because the workspace is already dirty.
