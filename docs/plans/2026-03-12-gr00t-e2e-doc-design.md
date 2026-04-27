# GR00T E2E Inference Documentation Design

**Goal**

Create a detailed markdown document that explains the real end-to-end GR00T inference path for the current default deployment setup, including raw inputs, transformed inputs, model internals, output reconstruction, repeated-request semantics, and practical acceleration opportunities.

**Primary Path**

The document should follow the actual default path used in this repository:

- Model: `GR00T-N1.5-3B`
- Data config: `fourier_gr1_arms_only`
- Single ego-view image input
- State horizon: `1`
- Action horizon: `16`
- Default action denoising steps: `4`

**Scope**

The document should explain:

1. What a "task" means versus what a single inference request means.
2. What the raw user/environment inputs look like for one request.
3. How the policy transform pipeline changes shapes and datatypes.
4. How the backbone and action head consume tensors and produce actions.
5. How repeated inference requests reuse task intent but recompute actions from new observations.
6. Which dimensions are fixed by the current config and which are configuration-dependent.
7. Which parts of the pipeline are promising acceleration targets.

**Structure**

The document should be organized into:

1. High-level overview
2. One task vs many inference requests
3. Raw input schema with a verified demo example
4. Preprocessing and transform pipeline with step-by-step shape changes
5. Policy/model execution path
6. Backbone details
7. Action-head denoising details
8. Output reconstruction
9. Repeated-request timeline
10. Configuration-dependent notes
11. Acceleration analysis

**Evidence Source**

The document should be grounded in:

- `gr00t/model/policy.py`
- `gr00t/model/gr00t_n1.py`
- `gr00t/model/backbone/eagle_backbone.py`
- `gr00t/model/action_head/flow_matching_action_head.py`
- `gr00t/model/transforms.py`
- `gr00t/experiment/data_config.py`
- `gr00t/data/transform/state_action.py`
- `gr00t/data/transform/concat.py`
- `deployment_scripts/ant/local_inference_breakdown.py`
- Verified sample shapes from `demo_data/robot_sim.PickNPlace`

**Non-Goals**

- Do not document training.
- Do not generalize the main narrative around every possible data config.
- Do not claim unavailable metrics or undocumented model behavior.
- Do not present speculative optimizations as guaranteed speedups.
