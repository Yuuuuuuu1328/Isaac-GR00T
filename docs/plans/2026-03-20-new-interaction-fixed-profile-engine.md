# New Interaction Fixed-Profile Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a dedicated TensorRT build script for the `new_interaction_group` FP16 ONNX export that fixes the engine profile to `batch=1` and `sequence_length=283` without modifying the native `deployment_scripts/build_engine.sh`.

**Architecture:** Keep the existing generic builder untouched and add a second shell script that hardcodes the validated `new_interaction_group` workload defaults. Cover the new script with focused text-based tests so the fixed ONNX root, output directory, precision, and profile settings are locked in before building engines.

**Tech Stack:** Bash, TensorRT `trtexec`, Python `unittest`

---

### Task 1: Add regression coverage for the dedicated builder

**Files:**
- Create: `deployment_scripts/test_build_engine_new_interaction.py`
- Read: `deployment_scripts/build_engine.sh`

**Step 1: Write the failing test**

```python
import unittest
from pathlib import Path


class BuildEngineNewInteractionScriptTest(unittest.TestCase):
    def test_fixed_profile_defaults(self) -> None:
        script = Path("deployment_scripts/build_engine_new_interaction_group_fp16.sh").read_text()

        self.assertIn("ONNX_ROOT=${ONNX_ROOT:-/home/jetson/Desktop/project/Isaac-GR00T/gr00t_onnx_new_interaction_group_fp16}", script)
        self.assertIn("ENGINE_ROOT=${ENGINE_ROOT:-/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16_bs1_len283}", script)
        self.assertIn("VIT_DTYPE=${VIT_DTYPE:-fp16}", script)
        self.assertIn("LLM_DTYPE=${LLM_DTYPE:-fp16}", script)
        self.assertIn("DIT_DTYPE=${DIT_DTYPE:-fp16}", script)
        self.assertIn("MAX_BATCH=${MAX_BATCH:-1}", script)
        self.assertIn("MIN_LEN=${MIN_LEN:-283}", script)
        self.assertIn("OPT_LEN=${OPT_LEN:-283}", script)
        self.assertIn("MAX_LEN=${MAX_LEN:-283}", script)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest deployment_scripts.test_build_engine_new_interaction -v`
Expected: FAIL because the new script does not exist yet.

**Step 3: Write minimal implementation**

Create `deployment_scripts/build_engine_new_interaction_group_fp16.sh` by copying the generic build flow, swapping `gr00t_engine/` for `${ENGINE_ROOT}`, and pinning the validated fixed-shape FP16 defaults.

**Step 4: Run test to verify it passes**

Run: `python -m unittest deployment_scripts.test_build_engine_new_interaction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/test_build_engine_new_interaction.py deployment_scripts/build_engine_new_interaction_group_fp16.sh
git commit -m "feat: add fixed-profile new interaction engine builder"
```

### Task 2: Build and verify the dedicated engine set

**Files:**
- Create: `/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16_bs1_len283`
- Read: `/home/jetson/Desktop/project/Isaac-GR00T/gr00t_onnx_new_interaction_group_fp16`

**Step 1: Run the dedicated builder**

```bash
bash deployment_scripts/build_engine_new_interaction_group_fp16.sh
```

**Step 2: Verify outputs exist**

Run: `ls -lh /home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16_bs1_len283`
Expected: All seven `.engine` files plus build logs exist.

**Step 3: Smoke-check the builder configuration**

Run: `rg -n "MAX_BATCH|MIN_LEN|OPT_LEN|MAX_LEN|ENGINE_ROOT|ONNX_ROOT" deployment_scripts/build_engine_new_interaction_group_fp16.sh`
Expected: The script prints the fixed `batch=1`, `len=283`, ONNX root, and dedicated output directory defaults.

**Step 4: Commit**

```bash
git add docs/plans/2026-03-20-new-interaction-fixed-profile-engine.md deployment_scripts/build_engine_new_interaction_group_fp16.sh deployment_scripts/test_build_engine_new_interaction.py
git commit -m "feat: add new interaction fixed-profile engine build script"
```
