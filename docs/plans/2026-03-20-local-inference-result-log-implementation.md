# Local Inference Result Log Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Persist one experiment-level JSONL record for every `local_inference.py` execution, including all parsed hyperparameters and experiment results.

**Architecture:** Keep the existing per-run latency JSONL export untouched and add a second always-on append-only experiment log. Build the payload inside `local_inference.py` after summary generation so the code has direct access to normalized args, summary, and per-run records.

**Tech Stack:** Python, `argparse`, `json`, `pathlib`, `unittest`

---

### Task 1: Add a failing regression test for experiment logging

**Files:**
- Create: `deployment_scripts/ant/test_local_inference.py`
- Modify: `deployment_scripts/ant/local_inference.py`
- Test: `deployment_scripts/ant/test_local_inference.py`

**Step 1: Write the failing test**

```python
def test_run_profile_loop_appends_one_experiment_result_per_invocation():
    ...
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference.py -q`
Expected: FAIL because experiment logging is not implemented.

**Step 3: Write minimal implementation**

```python
def _append_experiment_result(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference.py -q`
Expected: PASS

### Task 2: Wire the experiment log into the profile loop

**Files:**
- Modify: `deployment_scripts/ant/local_inference.py`
- Test: `deployment_scripts/ant/test_local_inference.py`

**Step 1: Update `_run_profile_loop()` to build one experiment-level payload**

```python
payload = {
    "args": ...,
    "summary": summary,
    "records": ...,
}
```

**Step 2: Append the payload to the fixed JSONL path**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_local_inference.py -q`
Expected: PASS

**Step 3: Run nearby regression coverage**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference.py -q`
Expected: PASS
