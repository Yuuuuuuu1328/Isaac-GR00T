# Local Inference New Interaction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a new `deployment_scripts/ant/local_inference_new_interaction.py` entrypoint that runs local `new_interaction_group` inference against the local `ossfs` `gr00t` package while preserving existing shared utils behavior.

**Architecture:** The new script will mirror the selector-style orchestration from `deployment_scripts/ant/local_inference.py`, but it will own its runtime-loading path. It will import the local `ossfs` `gr00t` package by prepending the workspace root to `sys.path`, resolve `data_config` from `DATA_CONFIG_MAP`, and reuse existing profiling/metrics/breakdown helpers from the current repo without modifying them.

**Tech Stack:** Python, argparse, existing deployment_scripts/ant profiling helpers, local `ossfs` `gr00t` package, unittest.

---

### Task 1: Add failing parser and dispatch tests

**Files:**
- Create: `deployment_scripts/ant/test_local_inference_new_interaction.py`
- Modify: `deployment_scripts/ant/test_ant_cli.py`

**Step 1: Write the failing test**

Add tests that assert:
- the new parser defaults match the requested new-interaction values;
- the script exposes only the intended selector/engine/data arguments;
- `run_with_args()` dispatches each backend/mode combination to the correct runner;
- `--help` for the new script succeeds.

**Step 2: Run test to verify it fails**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference_new_interaction.py deployment_scripts/ant/test_ant_cli.py -q
```

Expected:
- FAIL because the new script does not exist yet.

### Task 2: Implement the new ossfs-backed entrypoint

**Files:**
- Create: `deployment_scripts/ant/local_inference_new_interaction.py`

**Step 1: Add the path/bootstrap layer**

Prepend `/home/jetson/Desktop/project/ossfs/node_59823209/workspace` to `sys.path` before importing `gr00t`, while keeping the repo root on `sys.path` for local `deployment_scripts.ant` imports.

**Step 2: Implement narrow CLI defaults**

Expose only the used parameters, with defaults:
- `backend=tensorrt`
- `mode=e2e`
- `model-path=/home/jetson/Desktop/project/new_model/left_hand_v2_1223`
- `dataset-path=/home/jetson/Desktop/project/Isaac-GR00T/demo_data/new_interaction_group/Real_test_data_0305`
- `data-config=new_interaction_group`
- `embodiment-tag=new_embodiment`
- `denoising-steps=4`
- `video-backend=decord`
- `trt-engine-path=/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16`
- `vit-dtype=fp16`
- `llm-dtype=fp16`
- `dit-dtype=fp16`
- `sample-index=0`

**Step 3: Implement custom runtime loaders**

Create dedicated loaders for:
- PyTorch E2E / breakdown
- TensorRT E2E / breakdown

These loaders must:
- import `DATA_CONFIG_MAP` from the local `ossfs` `gr00t`;
- resolve `data_config` from the map directly;
- construct `Gr00tPolicy`;
- build `LeRobotSingleDataset`;
- set up TensorRT engines for TensorRT modes.

**Step 4: Reuse existing measurement helpers**

Reuse existing:
- `profile_utils`
- `quality_metrics`
- `system_metrics`
- `torch_compile_utils`
- `_run_single_breakdown` helpers

Do not modify shared utils; add any needed compatibility glue only inside the new script.

### Task 3: Verify green

**Files:**
- Test: `deployment_scripts/ant/test_local_inference_new_interaction.py`
- Test: `deployment_scripts/ant/test_ant_cli.py`

**Step 1: Run targeted tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference_new_interaction.py deployment_scripts/ant/test_ant_cli.py -q
```

Expected:
- PASS for the new tests and CLI smoke coverage.

**Step 2: Run an existing nearby regression check**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py -q
```

Expected:
- PASS, showing existing shared logic was not broken.
