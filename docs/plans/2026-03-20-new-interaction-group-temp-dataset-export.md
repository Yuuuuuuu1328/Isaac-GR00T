# New Interaction Group Temp Dataset Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unblock and complete FP16 ONNX export for the `left_hand_v2_1223` `new_embodiment` model by supplying the missing LeRobot modality metadata through a temporary dataset overlay.

**Architecture:** Keep the original `demo_data/new_interaction_group/Real_test_data_0305` dataset immutable. Create a temporary dataset root in the workspace that symlinks the original data, videos, images, and metadata files, then add a minimal `meta/modality.json` matching the built-in `new_interaction_group` config. Use that overlay as the export dataset path.

**Tech Stack:** Python, GR00T dataset loader, ONNX export script, symlinked workspace files

---

### Task 1: Reproduce the loader failure

**Files:**
- Read: `gr00t/data/dataset.py`
- Read: `demo_data/new_interaction_group/Real_test_data_0305/meta/info.json`

**Step 1: Run the focused loader reproduction**

```bash
python - <<'PY'
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config

cfg = load_data_config("new_interaction_group")
LeRobotSingleDataset(
    dataset_path="demo_data/new_interaction_group/Real_test_data_0305",
    modality_configs=cfg.modality_config(),
    embodiment_tag="new_embodiment",
)
PY
```

Expected: Fail with `Please provide a meta/modality.json file`.

### Task 2: Create the temporary dataset overlay

**Files:**
- Create: `.tmp/new_interaction_group_export_dataset/Real_test_data_0305/meta/modality.json`

**Step 1: Create the temporary directory structure**

```bash
mkdir -p .tmp/new_interaction_group_export_dataset/Real_test_data_0305/meta
```

**Step 2: Symlink the existing dataset content**

```bash
ln -s /home/jetson/Desktop/project/Isaac-GR00T/demo_data/new_interaction_group/Real_test_data_0305/data .tmp/new_interaction_group_export_dataset/Real_test_data_0305/data
ln -s /home/jetson/Desktop/project/Isaac-GR00T/demo_data/new_interaction_group/Real_test_data_0305/videos .tmp/new_interaction_group_export_dataset/Real_test_data_0305/videos
ln -s /home/jetson/Desktop/project/Isaac-GR00T/demo_data/new_interaction_group/Real_test_data_0305/images .tmp/new_interaction_group_export_dataset/Real_test_data_0305/images
ln -s /home/jetson/Desktop/project/Isaac-GR00T/demo_data/new_interaction_group/Real_test_data_0305/meta/info.json .tmp/new_interaction_group_export_dataset/Real_test_data_0305/meta/info.json
ln -s /home/jetson/Desktop/project/Isaac-GR00T/demo_data/new_interaction_group/Real_test_data_0305/meta/tasks.jsonl .tmp/new_interaction_group_export_dataset/Real_test_data_0305/meta/tasks.jsonl
ln -s /home/jetson/Desktop/project/Isaac-GR00T/demo_data/new_interaction_group/Real_test_data_0305/meta/episodes.jsonl .tmp/new_interaction_group_export_dataset/Real_test_data_0305/meta/episodes.jsonl
ln -s /home/jetson/Desktop/project/Isaac-GR00T/demo_data/new_interaction_group/Real_test_data_0305/meta/episodes_stats.jsonl .tmp/new_interaction_group_export_dataset/Real_test_data_0305/meta/episodes_stats.jsonl
```

**Step 3: Write the minimal modality metadata**

```json
{
  "state": {
    "single_arm": {
      "start": 0,
      "end": 6
    }
  },
  "action": {
    "single_arm": {
      "start": 0,
      "end": 6
    }
  },
  "video": {
    "ego_view": {
      "original_key": "observation.images.main"
    }
  },
  "annotation": {
    "task_index": {
      "original_key": "task_index"
    }
  }
}
```

### Task 3: Verify the overlay loads

**Files:**
- Read: `.tmp/new_interaction_group_export_dataset/Real_test_data_0305/meta/modality.json`

**Step 1: Run the loader check against the overlay**

```bash
python - <<'PY'
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config

cfg = load_data_config("new_interaction_group")
dataset = LeRobotSingleDataset(
    dataset_path=".tmp/new_interaction_group_export_dataset/Real_test_data_0305",
    modality_configs=cfg.modality_config(),
    embodiment_tag="new_embodiment",
)
print(dataset.dataset_name, len(dataset))
PY
```

Expected: Dataset loads and prints a positive length.

### Task 4: Re-run FP16 ONNX export

**Files:**
- Write: `gr00t_onnx_new_interaction_group_fp16/`

**Step 1: Run export against the overlay dataset**

```bash
python deployment_scripts/export_onnx.py \
  --dataset-path /home/jetson/Desktop/project/Isaac-GR00T/.tmp/new_interaction_group_export_dataset/Real_test_data_0305 \
  --model-path /home/jetson/Desktop/project/new_model/left_hand_v2_1223 \
  --onnx-model-path /home/jetson/Desktop/project/Isaac-GR00T/gr00t_onnx_new_interaction_group_fp16 \
  --data-config new_interaction_group \
  --embodiment-tag new_embodiment \
  --llm-dtype fp16 \
  --vit-dtype fp16 \
  --dit-dtype fp16 \
  --denoising-steps 4
```

Expected: Export completes and writes ONNX artifacts under the requested output root.

### Task 5: Verify exported artifacts

**Files:**
- Read: `gr00t_onnx_new_interaction_group_fp16/`

**Step 1: List the output tree**

```bash
find gr00t_onnx_new_interaction_group_fp16 -maxdepth 3 -type f | sort
```

Expected: ONNX files exist under `eagle2/` and `action_head/`.
