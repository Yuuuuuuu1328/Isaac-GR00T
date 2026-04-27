# Inference Raw Launcher Design

**Date:** 2026-03-19

**Goal:** Add a small launcher script that starts `scripts/inference_raw.py` in one step based on two short mode selectors: server/client and HTTP/ZMQ.

## Scope

This design applies to:

- `scripts/inference_raw.py`
- a new launcher script under `scripts/`
- a focused CLI test under `tests/`

This design does not change:

- inference behavior inside `scripts/inference_raw.py`
- model loading defaults beyond wrapping the user's existing commands
- the existing Python service/client entrypoints

## Problem

The current workflow requires manually retyping two different command families:

- role: server vs client
- transport: ZMQ vs HTTP

The user wants a single command wrapper that keeps the current defaults but reduces startup friction to a minimal selector interface.

## Constraints

- The launcher must accept the short selectors the user requested: `s` for server, `c` for client, `h` for HTTP, and `z` for ZMQ.
- The launcher should run only one process per invocation; it must not auto-launch both server and client.
- The launcher should preserve the user's current default ports, host values, model path, TensorRT flags, and dtype flags.
- The launcher should work from anywhere by switching into the repository root before invoking Python.
- The launcher should avoid depending on interactive shell initialization for `conda activate`.

## Approaches Considered

### Option 1: Fixed Bash wrapper with only two positional selectors

Pros:

- smallest implementation
- closest to the user's original ask

Cons:

- changing host, port, or model path later requires editing the file

### Option 2: Bash wrapper with selector pair plus a few optional overrides

Pros:

- keeps the simple `s/c` and `h/z` entrypoint
- still allows small runtime overrides such as `--host`, `--port`, and `--model-path`
- stays lightweight and shell-native

Cons:

- slightly more argument parsing logic

### Option 3: Python wrapper

Pros:

- stronger argument parsing
- easier to extend later

Cons:

- heavier than needed for a small launch surface
- duplicates the "simple shell command" use case

## Chosen Approach

Use Option 2.

Create `scripts/run_inference.sh` as a Bash launcher that accepts:

- `bash scripts/run_inference.sh s z`
- `bash scripts/run_inference.sh c z`
- `bash scripts/run_inference.sh s h`
- `bash scripts/run_inference.sh c h`

It will keep the current command defaults and allow a small set of overrides for practical reuse.

## Architecture

The launcher will:

1. validate the first two positional arguments
2. parse optional override flags
3. resolve the repository root from the script location
4. build the final `conda run -n gr00t-orin python scripts/inference_raw.py ...` command
5. `exec` that command so the launched process becomes the foreground process

Using `conda run` instead of `conda activate` avoids interactive-shell setup requirements and keeps behavior stable in a non-interactive script.

## Data Flow

Input:

- role selector: `s` or `c`
- transport selector: `h` or `z`
- optional overrides such as `--host`, `--port`, `--model-path`

Output:

- exactly one `python scripts/inference_raw.py ...` invocation with the matching flags

Mode mapping:

- `s z` -> `--server` on port `5555`
- `c z` -> `--client --host localhost --port 5555`
- `s h` -> `--server --http-server --host 0.0.0.0 --port 8000`
- `c h` -> `--client --http-server --host localhost --port 8000`

Server mode also preserves the current model and TensorRT defaults:

- `--model-path /home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B`
- `--embodiment-tag gr1`
- `--data-config fourier_gr1_arms_waist`
- `--denoising-steps 4`
- `--use-tensorrt`
- `--trt-engine-path gr00t_engine_fp16`
- `--vit-dtype fp16`
- `--llm-dtype fp16`
- `--dit-dtype fp16`

## Error Handling

- Invalid or missing selectors print usage and exit non-zero.
- Unknown optional flags print a clear error and usage.
- Missing `conda` prints a clear error before attempting launch.

## Testing

Use TDD with a focused CLI test file that:

- verifies help output
- verifies the launcher builds the expected command for `s z`
- verifies the launcher builds the expected command for `c h`
- verifies optional overrides replace the expected defaults

The tests will stub `conda` on `PATH` so the launcher can be verified without starting real inference processes.
