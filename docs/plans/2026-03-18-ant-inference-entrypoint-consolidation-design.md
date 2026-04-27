# ANT Inference Entrypoint Consolidation Design

**Date:** 2026-03-18

**Goal:** Consolidate the four ANT local inference entrypoints into one shared entrypoint controlled by two axes, while preserving the current runtime logic, CLI behavior, and profiling semantics.

## Scope

This design applies to:

- `deployment_scripts/ant/local_inference_e2e.py`
- `deployment_scripts/ant/local_inference_breakdown.py`
- `deployment_scripts/ant/local_inference_tensorrt_e2e.py`
- `deployment_scripts/ant/local_inference_tensorrt_breakdown.py`
- related tests under `deployment_scripts/ant`

This design does not change:

- PyTorch inference internals
- TensorRT engine setup logic
- stage timing definitions
- offline evaluation metrics
- JSONL schema beyond the script name reported in compatibility wrappers

## Problem

The current ANT local inference surface is split into four top-level scripts that represent the cartesian product of two independent choices:

- execution shape: E2E vs breakdown
- backend: PyTorch vs TensorRT

That duplication makes CLI maintenance and test coverage harder, even though each script follows the same high-level flow:

1. build parser
2. load runtime
3. optionally enable backend-specific setup
4. warm up
5. measure runs
6. collect optional system and quality metrics
7. summarize and optionally write JSONL

## Constraints

- The effective runtime logic for each of the four current modes must remain unchanged.
- Existing script filenames must keep working.
- Existing tests that import helper functions from the current modules should keep working or be updated with compatibility-preserving aliases.
- PyTorch-only flags such as `--use-torch-compile` must not appear on TensorRT wrappers.
- TensorRT-only flags such as engine path and dtype options must not appear on PyTorch wrappers.
- Reported `meta["script"]` values must stay aligned with the legacy script the user invoked when using compatibility wrappers.

## Approaches Considered

### Option 1: One shared core module plus thin compatibility wrappers

Create a new unified entrypoint that models the two axes explicitly and keep the four existing files as wrappers that forward to the shared implementation with fixed mode selections.

Pros:

- lowest migration risk
- preserves existing commands
- allows one place for parser composition and shared execution flow
- keeps future cleanup possible once callers migrate

Cons:

- leaves five files instead of literally one
- requires a small compatibility layer for per-script metadata

### Option 2: Replace four scripts with one file and remove old names

Move all logic into one script and delete the old entrypoints.

Pros:

- smallest visible surface long-term

Cons:

- breaks existing commands, docs, and habits
- increases coordination cost for users
- not compatible with the no-behavior-change requirement

### Option 3: Keep four files and only extract helpers

Factor out common utilities but retain four independent main functions.

Pros:

- lowest code movement risk

Cons:

- does not actually consolidate the entrypoint surface
- still duplicates parser assembly and orchestration

## Chosen Approach

Use Option 1.

Add a new shared entrypoint module under `deployment_scripts/ant` that exposes two explicit switches:

- `--backend {pytorch,tensorrt}`
- `--mode {e2e,breakdown}`

The shared module will own parser construction and main-loop orchestration. The four existing scripts will become thin wrappers that:

- call the shared module with fixed defaults
- preserve the old script-specific help surface
- preserve script-specific metadata labels

## Architecture

### Shared entrypoint

Create a new module, tentatively `deployment_scripts/ant/local_inference.py`, with:

- parser builders for common, PyTorch-only, and TensorRT-only arguments
- backend/mode dispatch to the existing runtime and breakdown helpers
- a single measurement loop parameterized by runtime strategy

### Runtime strategies

The shared module will dispatch to the existing behavior rather than re-implement it:

- PyTorch E2E reuses `local_inference_e2e` runtime loading and compile wiring
- PyTorch breakdown reuses `local_inference_breakdown` helpers
- TensorRT E2E reuses `local_inference_tensorrt_e2e` runtime loading
- TensorRT breakdown reuses `local_inference_tensorrt_breakdown` helpers

Where orchestration logic is duplicated today, move that orchestration into the shared module only if it does not change the called functions or measured regions.

### Compatibility wrappers

Each legacy script will:

- expose `build_parser()` for its legacy argument surface
- expose `main()` for direct execution
- forward to the shared module with fixed `backend`, `mode`, and `script_name`

This preserves imports used by current tests and keeps `python <legacy-script> --help` working.

## Data Flow

For all four combinations, the normalized flow remains:

1. parse CLI
2. load runtime objects for the selected backend
3. apply backend-specific setup
4. perform warmup runs
5. perform measured runs
6. collect optional system metrics
7. collect optional quality metrics
8. summarize and optionally emit JSONL

The only structural change is that step selection is driven by `backend` and `mode` instead of by script filename.

## Error Handling

- Invalid backend/mode combinations are prevented by parser choices.
- TensorRT mode retains the current CUDA and engine-path validation behavior.
- Wrapper scripts do not swallow exceptions; they propagate the same failures as today.
- Shared dispatch must fail fast if a required strategy hook is missing.

## Testing

Use TDD to cover:

- unified parser defaults and backend/mode overrides
- wrapper help output compatibility
- wrapper-to-shared dispatch behavior
- preservation of existing helper imports for current unit tests

Targeted regression coverage should include the current ANT CLI tests plus the PyTorch and TensorRT helper tests already in the tree.
