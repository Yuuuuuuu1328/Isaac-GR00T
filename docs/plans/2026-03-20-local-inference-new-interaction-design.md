# Local Inference New Interaction Design

**Goal:** Add a dedicated local inference entrypoint for the `new_interaction_group` / `new_embodiment` flow that always uses the local `ossfs` `gr00t` package without changing existing official scripts or shared utils behavior.

**Chosen approach:** Create a new script at [`deployment_scripts/ant/local_inference_new_interaction.py`](/home/jetson/Desktop/project/Isaac-GR00T/deployment_scripts/ant/local_inference_new_interaction.py) that hard-prepends `/home/jetson/Desktop/project/ossfs/node_59823209/workspace` to `sys.path`, exposes only the parameters this flow uses, and implements its own runtime loaders for PyTorch/TensorRT and E2E/breakdown modes. Existing profiling, summary, optional quality metrics, optional system metrics, and breakdown helpers will be reused as-is.

**Key constraints:**
- Do not change the behavior of existing shared utils or official entrypoints.
- Support both `backend={pytorch,tensorrt}` and `mode={e2e,breakdown}`.
- Default CLI values must match the user-specified `new_interaction_group` deployment settings.
- Do not rely on official `load_data_config()` because the local `ossfs` package does not expose it.

**Rejected alternatives:**
- Reusing official runtime loaders directly: not viable because they import `load_data_config()` from the official interface.
- Modifying official `local_inference.py` to add an `--use-ossfs-gr00t` switch: higher coupling and risk to current workflows.
- Narrow TensorRT-only script: too limited for future PyTorch comparisons.

