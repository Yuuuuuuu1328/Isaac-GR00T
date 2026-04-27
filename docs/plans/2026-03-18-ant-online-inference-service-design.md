# ANT Online Inference Service Design

**Date:** 2026-03-18

**Goal:** Add a dedicated ANT online inference service entrypoint that supports production-style real-time serving while preserving `deployment_scripts/ant/local_inference.py` as a local benchmark and analysis tool.

## Scope

This design applies to:

- `deployment_scripts/ant/online_inference_service.py` (new)
- supporting tests under `deployment_scripts/ant`
- reuse of existing runtime helpers from `deployment_scripts/ant` and `scripts/aistudio.py`

This design does not change:

- `deployment_scripts/ant/local_inference.py` benchmark responsibilities
- `breakdown` runtime paths as an online serving mode
- the existing `scripts/inference_service.py` entrypoint in the first phase

## Problem

The repository currently has two disconnected capabilities:

- `deployment_scripts/ant/local_inference.py` can run local ANT inference and profiling, but it has no server/client transport or production service surface.
- `scripts/inference_service.py` can serve GR00T over ZMQ and HTTP, but it only exposes the canonical `observation -> action` interface and does not include ANT-specific serving requirements or the business compatibility path in `scripts/aistudio.py`.

The real business input/output contract is defined by `scripts/aistudio.py`, which introduces:

- a strict `query` wrapper contract
- JPEG framebuffer decoding and preprocessing
- dual-model routing between left and right policies
- a custom result envelope `resultCode / errorMessage / resultMap`

The online service needs to support both:

- a standard production interface for canonical GR00T inference
- a strict Aistudio-compatible HTTP interface

## Constraints

- `deployment_scripts/ant/local_inference.py` must remain a local benchmark/offline analysis tool.
- Online serving should only expose `e2e` inference, not `breakdown`.
- Standard service mode must support both PyTorch and TensorRT backends.
- Aistudio compatibility mode must preserve the current `scripts/aistudio.py` semantics, including:
  - dual policies (`policy_left`, `policy_right`)
  - routing by `joint_angles[3]`
  - current fixed prompt behavior
  - return semantics matching `resultCode / errorMessage / resultMap`
- Aistudio compatibility mode must not be reduced to unpacked JSON; it must preserve the outer `query` field.
- Aistudio mode must support TensorRT, with a shared default engine directory and optional left/right overrides.
- Standard mode should provide a more complete production interface, but without breaking older clients that expect `/act`, `/health`, and the current ZMQ endpoint names.

## Approaches Considered

### Option 1: Extend `deployment_scripts/ant/local_inference.py`

Add service subcommands directly to the new ANT shared inference script.

Pros:

- single ANT entrypoint

Cons:

- mixes benchmark/profiling with online serving
- increases CLI complexity
- weak separation of concerns

### Option 2: Add a dedicated ANT online service entrypoint

Create a new service-focused entrypoint and leave `local_inference.py` unchanged as a benchmark tool.

Pros:

- clear responsibility split
- easier to test and operate
- matches the requested “new file” direction
- keeps future service growth isolated from profiling logic

Cons:

- adds another top-level ANT script

### Option 3: Extend `scripts/inference_service.py`

Fold ANT and Aistudio-specific logic into the existing generic service script.

Pros:

- reuses existing server/client surface

Cons:

- couples generic GR00T serving with ANT and Aistudio-specific business logic
- harder to maintain clean boundaries

## Chosen Approach

Use Option 2.

Add a new entrypoint:

- `deployment_scripts/ant/online_inference_service.py`

This new file will own the ANT online serving surface and will not replace `local_inference.py`.

## Service Modes

The new service entrypoint will expose two mutually exclusive modes:

### Standard mode

Purpose:

- canonical production inference service using native `observation -> action`

Capabilities:

- backend: `pytorch` or `tensorrt`
- transport: `http` or `zmq`

### Aistudio mode

Purpose:

- strict business compatibility for the contract implied by `scripts/aistudio.py`

Capabilities:

- backend: `pytorch` or `tensorrt`
- transport: `http` only

Important:

- `aistudio + zmq` is intentionally unsupported
- standard and Aistudio mode do not run in the same process

## CLI Shape

The new script should expose a single CLI with explicit axes:

- `--role server|client`
- `--service-mode standard|aistudio`
- `--transport http|zmq`
- `--backend pytorch|tensorrt`

Shared runtime configuration will include:

- model path
- data config
- embodiment tag
- host / port
- auth token
- TensorRT engine path settings

Aistudio mode additionally needs:

- left model path
- right model path
- default TensorRT engine path
- optional `left-trt-engine-path`
- optional `right-trt-engine-path`

## API Design

### Standard HTTP

Primary routes:

- `POST /v1/act`
- `GET /v1/health/live`
- `GET /v1/health/ready`
- `GET /v1/metadata`
- `GET /v1/latency/ping`
- `POST /v1/latency/echo`

Compatibility aliases:

- `POST /act`
- `GET /health`

### Standard ZMQ

Endpoints:

- `get_action`
- `get_modality_config`
- `get_server_info`
- `ping`
- `echo`
- `kill` only for local debug use

### Aistudio HTTP

Primary routes:

- `POST /v1/aistudio/predict`
- `GET /v1/aistudio/health`
- `GET /v1/aistudio/latency/ping`
- `POST /v1/aistudio/latency/echo`

The request must preserve the outer wrapper:

```json
{
  "query": "{\"joint_angles\": \"[...]\", ...}"
}
```

The response must preserve the current business envelope:

```json
{
  "resultCode": 0,
  "errorMessage": "ok",
  "resultMap": {
    "...": "..."
  }
}
```

## Internal Architecture

The implementation should separate transport, runtime setup, and business adaptation.

### 1. Service configuration

Centralize validation for supported combinations:

- standard + http + pytorch|tensorrt
- standard + zmq + pytorch|tensorrt
- aistudio + http + pytorch|tensorrt

Reject invalid combinations early:

- aistudio + zmq
- breakdown serving

### 2. Runtime factory

Provide runtime loaders for:

- standard single-policy serving
- Aistudio dual-policy serving

Reuse existing ANT loading helpers where possible to avoid duplicating model setup.

### 3. Standard adapter

Input:

- canonical `observation` payload

Output:

- action payload plus optional metadata

### 4. Aistudio adapter

Input:

- strict `query` wrapper

Processing:

- decode outer `query`
- parse list-like string fields
- decode JPEG framebuffer
- resize and RGB conversion
- build the expected batch
- select left or right policy from `joint_angles[3]`
- run inference
- apply the current action post-processing and result shaping

Output:

- strict `resultCode / errorMessage / resultMap`

### 5. Transport layer

HTTP:

- FastAPI-based routes
- auth, health, latency, and request timing

ZMQ:

- endpoint dispatch using existing service patterns in `gr00t.eval.service`
- only enabled in standard mode

## Latency Semantics

Latency support is part of the product surface, not just local debugging.

The service should expose:

- lightweight `ping` for minimal round-trip timing
- `echo` for payload-inclusive RTT timing
- request timing metadata for inference requests

The response metadata should include enough server-side timestamps to separate:

- client RTT
- server processing time
- approximate network and serialization overhead

## Error Handling

### Standard mode

HTTP:

- invalid request: `400`
- auth failure: `401`
- not ready: `503`
- internal inference failure: `500`

ZMQ:

- preserve current `{"error": ...}` style responses

### Aistudio mode

Preserve business semantics as much as possible:

- business-level failures should return a compatibility payload with non-zero `resultCode`
- protocol-level malformed requests may still return `400`

## Production Safeguards

- readiness only reports healthy after models and TensorRT engines are loaded
- debug artifacts like `cv2.imwrite("./output.jpg", ...)` must be removed from the production path
- request IDs should be preserved when present and generated when absent in standard mode
- destructive debug endpoints like `kill` should be disabled unless explicitly enabled

## Testing

Use TDD with coverage for:

- config validation and supported mode matrix
- standard HTTP and ZMQ interfaces
- compatibility aliases (`/act`, `/health`)
- Aistudio strict `query` parsing and response compatibility
- left/right routing behavior
- TensorRT path resolution for shared and overridden engine directories
- latency and metadata endpoints

The first phase should keep `scripts/inference_service.py` intact and validate the new entrypoint independently.
