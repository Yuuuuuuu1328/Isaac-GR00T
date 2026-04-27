# build_engine ONNX Root Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow `deployment_scripts/build_engine.sh` to switch ONNX input roots through an `ONNX_ROOT` environment variable without breaking existing commands.

**Architecture:** Keep the change localized to the shell script by introducing one new environment variable, routing all ONNX inputs through it, and validating the selected paths before build execution. Add a focused regression test that locks the expected default and override path templates in place.

**Tech Stack:** Bash, Python `unittest`

---

### Task 1: Lock the expected ONNX root behavior with a failing test

**Files:**
- Create: `deployment_scripts/test_build_engine.py`
- Test: `deployment_scripts/test_build_engine.py`

**Step 1: Write the failing test**

```python
def test_build_script_uses_configurable_onnx_root():
    script = Path("deployment_scripts/build_engine.sh").read_text()
    assert "ONNX_ROOT=${ONNX_ROOT:-gr00t_onnx}" in script
    assert "--onnx=${ONNX_ROOT}/eagle2/vit_${VIT_DTYPE}.onnx" in script
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest deployment_scripts.test_build_engine -v`
Expected: FAIL because `build_engine.sh` still hardcodes `gr00t_onnx`.

### Task 2: Implement the minimal shell change

**Files:**
- Modify: `deployment_scripts/build_engine.sh`

**Step 1: Write minimal implementation**

- Define `ONNX_ROOT=${ONNX_ROOT:-gr00t_onnx}`.
- Replace hardcoded ONNX source paths with `${ONNX_ROOT}`.
- Print the active ONNX root.
- Validate each required ONNX file before invoking `trtexec`.

**Step 2: Run the focused test**

Run: `python -m unittest deployment_scripts.test_build_engine -v`
Expected: PASS.

### Task 3: Update usage docs

**Files:**
- Modify: `deployment_scripts/README.md`
- Modify: `deployment_scripts/orin/README.md`

**Step 1: Document the new override**

- Add one example showing `ONNX_ROOT=gr00t_onnx_fp8 ... bash deployment_scripts/build_engine.sh`.

**Step 2: Re-run focused verification**

Run: `python -m unittest deployment_scripts.test_build_engine -v`
Expected: PASS.
