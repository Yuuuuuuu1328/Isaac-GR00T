# Optional INT8 TensorRT Script Support Design

**Date:** 2026-03-16

## Goal

Add optional `int8` support to the TensorRT build and inference scripts without changing existing defaults or engine naming.

## Scope

- Extend `deployment_scripts/build_engine.sh` to accept `int8` for ViT, LLM, and DiT.
- Extend TensorRT CLI entrypoints to accept `int8`.
- Document the explicit `gr00t_onnx_int8` workflow.

## Non-Goals

- No default precision changes.
- No new build script.
- No engine filename changes.
- No runtime graph changes beyond dtype option acceptance.

## Design

The repository already derives ONNX and engine filenames from dtype tokens such as `vit_${VIT_DTYPE}` and `llm_${LLM_DTYPE}`. Because `gr00t_onnx_int8` already contains `vit_int8.onnx`, `llm_int8.onnx`, and `DiT_int8.onnx`, the lowest-risk design is to extend accepted dtype values and keep the existing naming convention intact.

## Validation

- Focused regression tests for `deployment_scripts/build_engine.sh`
- Focused parser tests for TensorRT CLI scripts
- Documentation examples updated to show explicit INT8 invocation

## Notes

The workspace contains unrelated local edits, so this design doc is recorded without creating a commit.
