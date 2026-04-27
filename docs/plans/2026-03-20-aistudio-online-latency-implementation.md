# Aistudio Online Latency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-request server latency logging and a minimal Aistudio client example without changing the service request or JSON response format.

**Architecture:** Extend `scripts/aistudio_test.py` so `/predict` measures route-level total processing time and pure inference time, appends per-request records to `deployment_scripts/ant/output/online_result.jsonl`, and returns the timing values through HTTP headers. Add a small client script that builds the current Aistudio payload format from user inputs, measures HTTP round-trip latency, and prints both client-observed and server-reported timings.

**Tech Stack:** Python, FastAPI, NumPy, requests, JSONL logging, unittest/pytest

---

### Task 1: Add failing service latency tests

**Files:**
- Modify: `tests/test_aistudio_service.py`
- Test: `tests/test_aistudio_service.py`

**Step 1: Write the failing test**

Add tests that prove:
- `AistudioRuntime.predict(...)` returns per-request timing metadata alongside the existing response.
- `/predict` adds latency response headers without changing the JSON body.
- The service appends one JSON object per request to `deployment_scripts/ant/output/online_result.jsonl`.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_service.py -q`
Expected: FAIL because the service does not yet expose or persist the latency data.

**Step 3: Write minimal implementation**

Implement the smallest set of helpers needed for timing capture, printing, response headers, and JSONL append.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_service.py -q`
Expected: PASS

### Task 2: Add failing minimal client tests

**Files:**
- Create: `tests/test_aistudio_client_example.py`
- Create: `scripts/aistudio_client_example.py`
- Test: `tests/test_aistudio_client_example.py`

**Step 1: Write the failing test**

Add tests that prove the client:
- builds the strict Aistudio request shape from command-line inputs,
- measures and prints `client_round_trip_ms`,
- reads and prints `X-Server-Total-Ms` and `X-Inference-Ms`.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_client_example.py -q`
Expected: FAIL because the script does not exist yet.

**Step 3: Write minimal implementation**

Create the smallest example client that uses `requests` and accepts explicit input data.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_client_example.py -q`
Expected: PASS

### Task 3: Final focused verification

**Files:**
- Test: `tests/test_aistudio_service.py`
- Test: `tests/test_aistudio_client_example.py`

**Step 1: Run focused verification**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_service.py tests/test_aistudio_client_example.py -q`
Expected: PASS
