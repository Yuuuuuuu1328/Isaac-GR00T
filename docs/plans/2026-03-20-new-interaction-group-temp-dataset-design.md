# New Interaction Group Temp Dataset Export Design

**Goal:** Export `/home/jetson/Desktop/project/new_model/left_hand_v2_1223` to FP16 ONNX at `/home/jetson/Desktop/project/Isaac-GR00T/gr00t_onnx_new_interaction_group_fp16` without modifying the original demo dataset.

**Problem:** `deployment_scripts/export_onnx.py` requires a LeRobot dataset root with `meta/modality.json`. The bundled `demo_data/new_interaction_group/Real_test_data_0305` dataset has usable parquet, video, task, and stats files, but it is missing `meta/modality.json`, so `LeRobotSingleDataset` aborts before export starts.

**Chosen Approach:** Build a temporary dataset overlay inside the workspace. Reuse the original dataset contents through symlinks, add only the missing `meta/modality.json`, and point `export_onnx.py` at that temporary root.

**Why This Approach:**
- It preserves the original dataset untouched.
- It fixes the exact loader precondition the export script enforces.
- It keeps the export inputs aligned with the `new_interaction_group` config already present in the repo.

**Dataset Mapping:**
- `video.ego_view` -> `observation.images.main`
- `state.single_arm` -> `observation.state`
- `action.single_arm` -> `action`
- `annotation.task_index` -> `task_index`

**Validation Plan:**
1. Reproduce the current loader failure against the original dataset root.
2. Create the temporary overlay and write a minimal valid `meta/modality.json`.
3. Verify `LeRobotSingleDataset` loads successfully from the temporary root.
4. Re-run FP16 ONNX export against the temporary root.
5. Verify the requested ONNX output directory contains exported artifacts.
