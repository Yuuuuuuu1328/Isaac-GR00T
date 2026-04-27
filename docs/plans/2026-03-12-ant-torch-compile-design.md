# Ant Torch Compile Design

**Date:** 2026-03-12

**Goal:** Add an optional `--use-torch-compile` flag to the local PyTorch ANT inference scripts without changing the main inference logic or affecting TensorRT export/build paths.

## Scope

This design only applies to:

- `deployment_scripts/ant/local_inference_e2e.py`
- `deployment_scripts/ant/local_inference_breakdown.py`

It does not apply to:

- TensorRT ANT scripts
- `deployment_scripts/export_onnx.py`
- TensorRT engine build or quantization flow

## Constraints

- `torch.compile` must remain optional and disabled by default.
- The implementation must stay local to `deployment_scripts/ant`.
- TensorRT paths must remain behaviorally unchanged.
- Export-to-ONNX behavior must not be coupled to `torch.compile`.

## Approach

Add a small shared helper under `deployment_scripts/ant` to:

- expose a common `--use-torch-compile` flag
- safely enable compile only when requested
- keep compile entry points aligned with how each script already runs

For `local_inference_e2e.py`, the least invasive point is `policy.model.get_action`, because `Gr00tPolicy.get_action()` already delegates to it after preprocessing.

For `local_inference_breakdown.py`, the script manually calls lower-level PyTorch subpaths instead of `policy.model.get_action()`. To preserve the current stage breakdown, compile should be applied only to the internal hot callables that the script already invokes, using dedicated compiled-callable attributes rather than replacing core modules globally.

## Runtime Behavior

### E2E

When `--use-torch-compile` is enabled:

- compile `policy.model.get_action`
- keep transforms, unapply, telemetry, and optional offline metrics unchanged

### Breakdown

When `--use-torch-compile` is enabled:

- compile callable subpaths used by `_run_backbone_breakdown()` and `_run_action_head_breakdown()`
- keep the same measured stages and metric names
- avoid rewriting the script into a single fused model call

## Error Handling

If `torch.compile` is unavailable or fails to initialize:

- fail fast with a clear runtime error
- do not silently ignore the flag

This makes benchmarking results unambiguous.

## Testing

Use TDD with:

- parser tests for `--use-torch-compile`
- helper tests that verify compile is wired to the expected callables
- CLI help tests for the new flag

No test should require actual compile speedup or GPU-specific timing improvements.
