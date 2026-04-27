# New Interaction Group Config Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the missing single-arm built-in data configs needed to export the `new_embodiment` model from the current branch.

**Architecture:** Reuse the known-good `NewInteractionGroup` and `NewInteractionGroupFuture` definitions from the local worktree/ossfs copies and merge only those blocks into the current [`gr00t/experiment/data_config.py`](/home/jetson/Desktop/project/Isaac-GR00T/gr00t/experiment/data_config.py). Register both names in `DATA_CONFIG_MAP` so `load_data_config(...)` can resolve them without an external module.

**Tech Stack:** Python, existing GR00T data config classes, shell verification via `python -c`.

---

### Task 1: Merge the missing built-in configs

**Files:**
- Modify: `gr00t/experiment/data_config.py`
- Reference: `.worktrees/aistudio-fastapi/gr00t/experiment/data_config.py`
- Reference: `/home/jetson/Desktop/project/ossfs/node_59823209/workspace/gr00t/experiment/data_config.py`

**Step 1: Add the missing config classes**

Insert `NewInteractionGroup` and `NewInteractionGroupFuture` near the bottom of `gr00t/experiment/data_config.py`, preserving the local file style.

**Step 2: Register the config names**

Add `"new_interaction_group"` and `"new_interaction_group_future"` to `DATA_CONFIG_MAP`.

**Step 3: Run a direct load check**

Run:

```bash
python - <<'PY'
from gr00t.experiment.data_config import load_data_config
for name in ("new_interaction_group", "new_interaction_group_future"):
    cfg = load_data_config(name)
    print(name, type(cfg).__name__, sorted(cfg.modality_config().keys()))
PY
```

Expected:
- Both names resolve successfully.
- The first config exposes `video`, `state`, `action`, `language`.
- The future-view config exposes `video`, `future`, `state`, `action`, `language`.

**Step 4: Use the config for export/build**

After the load check passes, run the FP16 export with `--data-config new_interaction_group` and `--embodiment-tag new_embodiment`, then build engines from the new ONNX root.
