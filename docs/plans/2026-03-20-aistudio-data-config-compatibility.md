# Aistudio Data Config Compatibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `scripts/aistudio_s.py` start successfully when the local OSSFS `gr00t.experiment.data_config` module exposes `DATA_CONFIG_MAP` but not `load_data_config`.

**Architecture:** Keep the OSSFS runtime helper backward-compatible by treating `load_data_config` as optional, then update the Aistudio runtime builder to resolve the requested config from `DATA_CONFIG_MAP` directly. Cover both behaviors with focused regression tests so the startup path fails loudly only when the requested config name is absent.

**Tech Stack:** Python, `unittest`, `unittest.mock`

---

### Task 1: Add helper compatibility regression test

**Files:**
- Create: `deployment_scripts/ant/test_ossfs_gr00t_runtime.py`
- Modify: `deployment_scripts/ant/ossfs_gr00t_runtime.py`

**Step 1: Write the failing test**

```python
def test_import_local_gr00t_runtime_accepts_missing_load_data_config():
    ...
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ossfs_gr00t_runtime.py -q`
Expected: FAIL because `_import_local_gr00t_runtime()` accesses `data_config_mod.load_data_config`

**Step 3: Write minimal implementation**

```python
load_data_config = getattr(data_config_mod, "load_data_config", None)
```

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ossfs_gr00t_runtime.py -q`
Expected: PASS

### Task 2: Add Aistudio runtime regression test

**Files:**
- Modify: `tests/test_aistudio_service.py`
- Modify: `scripts/aistudio_s.py`

**Step 1: Write the failing test**

```python
def test_build_runtime_uses_data_config_map_when_load_data_config_is_absent():
    ...
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_service.py -q`
Expected: FAIL because `build_runtime()` expects `load_data_config`

**Step 3: Write minimal implementation**

```python
data_config = _resolve_data_config(ossfs_runtime.DATA_CONFIG_MAP, args.data_config)
```

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_service.py -q`
Expected: PASS

### Task 3: Run focused verification

**Files:**
- Verify: `deployment_scripts/ant/test_ossfs_gr00t_runtime.py`
- Verify: `tests/test_aistudio_service.py`

**Step 1: Run focused regression suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_ossfs_gr00t_runtime.py tests/test_aistudio_service.py -q`
Expected: PASS with `0` failures
