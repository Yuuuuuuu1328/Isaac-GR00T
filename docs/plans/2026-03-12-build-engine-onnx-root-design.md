# build_engine ONNX Root Design

**Problem:** `deployment_scripts/build_engine.sh` hardcodes `gr00t_onnx/...`, but this repository already stores multiple ONNX roots such as `gr00t_onnx`, `gr00t_onnx_fp8`, and `gr00t_onnx_llm_nvfp4`.

**Approved Direction:** Add an `ONNX_ROOT` environment variable override while preserving the current default behavior.

## Design

- Add `ONNX_ROOT=${ONNX_ROOT:-gr00t_onnx}` near the existing dtype environment variables.
- Replace every hardcoded `gr00t_onnx/...` reference with `${ONNX_ROOT}/...`.
- Print the selected `ONNX_ROOT` in the script summary so the active source directory is visible in logs.
- Validate the expected ONNX files before invoking `trtexec`, and fail fast with a clear error if any required file is missing.

## Error Handling

- Missing ONNX files should produce a direct path-specific error and exit non-zero.
- The default no-override path must remain `gr00t_onnx` to avoid breaking existing commands.

## Testing

- Add a focused unit test that verifies the script text exposes `ONNX_ROOT` with the default `gr00t_onnx` value and uses `${ONNX_ROOT}` in engine build paths.
- Run the focused test after the script change.

