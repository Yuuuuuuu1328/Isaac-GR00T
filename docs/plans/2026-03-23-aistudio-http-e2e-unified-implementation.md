# Aistudio HTTP E2E Unified Script Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new single script `scripts/aistudio_http_e2e.py` that unifies Aistudio server/client benchmarking with `--role server|client`, keeps request/response compatibility, uses HTTP FastAPI only, and records inference_raw-style latency metrics to JSONL (run-only + summary).

**Architecture:** Build one script with shared protocol helpers, runtime builder, FastAPI server path, and client benchmark path. The server preserves `/predict` payload/response shape and emits latency headers. The client sends compatible requests, computes total/inference/network/proportion latencies, prints the fixed average block, writes one JSONL run record per measured request, then appends one summary record.

**Tech Stack:** Python, FastAPI, uvicorn, NumPy, requests, OpenCV, torch, JSONL, unittest/pytest

---

### Task 1: Add failing tests for fixed defaults and payload compatibility

**Files:**
- Create: `tests/test_aistudio_http_e2e.py`
- Test: `tests/test_aistudio_http_e2e.py`

**Step 1: Write the failing test**

Add tests asserting:
- `build_parser().parse_args([])` returns fixed defaults:
  - `data_config="new_interaction_group"`
  - `embodiment_tag="new_embodiment"`
  - `response_action_horizon=14`
  - `task_prompt="Move to center the book in view. Do nothing if no book is present."`
  - fixed model/ossfs/trt/output paths
- `build_request_payload(...)` returns `{"query": "<json-string>"}` and inner fields match current protocol keys.

```python
def test_parser_defaults_are_fixed_for_new_device():
    module = _load_module()
    args = module.build_parser().parse_args([])
    assert args.data_config == "new_interaction_group"
    assert args.embodiment_tag == "new_embodiment"
    assert args.response_action_horizon == 14

def test_payload_keeps_aistudio_query_shape():
    module = _load_module()
    args = module.build_parser().parse_args(["--role", "client"])
    payload = module.build_request_payload(args, request_id="req-1", rng=module.np.random.default_rng(0))
    inner = json.loads(payload["query"])
    assert set(["joint_angles", "predicted_coords_2d", "history_angles", "framebuffer", "framebuffer_size", "device_id", "request_id"]).issubset(inner.keys())
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_http_e2e.py -q`  
Expected: FAIL because the new script does not exist yet.

**Step 3: Write minimal implementation**

Create `scripts/aistudio_http_e2e.py` with:
- constants for all confirmed defaults,
- parser with `--role`,
- payload helpers copied/adapted from `scripts/aistudio_c.py`.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_http_e2e.py::test_parser_defaults_are_fixed_for_new_device tests/test_aistudio_http_e2e.py::test_payload_keeps_aistudio_query_shape -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/aistudio_http_e2e.py tests/test_aistudio_http_e2e.py
git commit -m "test: add unified script default and payload compatibility coverage"
```

### Task 2: Add failing tests for server `/predict` response contract and latency headers

**Files:**
- Modify: `tests/test_aistudio_http_e2e.py`
- Modify: `scripts/aistudio_http_e2e.py`
- Test: `tests/test_aistudio_http_e2e.py`

**Step 1: Write the failing test**

Add tests asserting:
- `/predict` returns `resultCode/errorMessage/resultMap` with legacy field names.
- headers include `X-Server-Total-Ms`, `X-Pure-Inference-Ms`, `X-Inference-Ms`, `X-Server-Overhead-Ms`.
- inference timing wraps only `policy.get_action(...)` call.

```python
def test_predict_route_keeps_response_shape_and_latency_headers():
    module = _load_module()
    runtime = SimpleNamespace(
        ready=True,
        metadata={"backend": "tensorrt"},
        predict_with_timing=mock.Mock(return_value=(module.build_aistudio_response(result_map={"request_id": "req-1"}), {"pure_inference_ms": 10.0, "request_id": "req-1", "device_id": "dev-1", "backend": "tensorrt"})),
    )
    app = module.create_app(runtime=runtime)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/predict")
    with mock.patch.object(module.time, "perf_counter_ns", side_effect=[0, 25_000_000]):
        response = endpoint({"query": "{}"})
    assert response.headers["X-Server-Total-Ms"] == "25.0"
    assert response.headers["X-Pure-Inference-Ms"] == "10.0"
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_http_e2e.py::test_predict_route_keeps_response_shape_and_latency_headers -q`  
Expected: FAIL until server route and headers are implemented.

**Step 3: Write minimal implementation**

Implement in `scripts/aistudio_http_e2e.py`:
- `parse_aistudio_query`,
- `build_aistudio_response` / `build_aistudio_error_response`,
- `AistudioRuntime.predict_with_timing`,
- `create_app` route and latency headers.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_http_e2e.py::test_predict_route_keeps_response_shape_and_latency_headers -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/aistudio_http_e2e.py tests/test_aistudio_http_e2e.py
git commit -m "feat: add unified aistudio server route with latency headers"
```

### Task 3: Add failing tests for inference_raw-style client latency summary output

**Files:**
- Modify: `tests/test_aistudio_http_e2e.py`
- Modify: `scripts/aistudio_http_e2e.py`
- Test: `tests/test_aistudio_http_e2e.py`

**Step 1: Write the failing test**

Add a test that runs `main(["--role", "client", ...])` with mocked HTTP responses and checks final stdout contains exactly the required four average lines:
- `Average total latency`
- `Average inference latency`
- `Average network latency`
- `Average inference proportion`

```python
def test_client_prints_required_average_latency_block():
    module = _load_module()
    stdout = StringIO()
    with (
        mock.patch.object(module, "send_request", side_effect=[(_resp(40.0, 10.0), 40.0)] * 15),
        mock.patch.object(module, "build_request_payload", side_effect=lambda args, request_id, rng: {"query": request_id}),
        redirect_stdout(stdout),
    ):
        module.main(["--role", "client", "--warmup-runs", "5", "--measure-runs", "10"])
    text = stdout.getvalue()
    assert "=== Average Latency ===" in text
    assert "Average total latency:" in text
    assert "Average inference latency:" in text
    assert "Average network latency:" in text
    assert "Average inference proportion:" in text
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_http_e2e.py::test_client_prints_required_average_latency_block -q`  
Expected: FAIL until client benchmark path is implemented.

**Step 3: Write minimal implementation**

Implement:
- warmup loop (no JSONL writes),
- run loop metrics aggregation,
- fixed average summary print block with matching labels/precision.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_http_e2e.py::test_client_prints_required_average_latency_block -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/aistudio_http_e2e.py tests/test_aistudio_http_e2e.py
git commit -m "feat: add unified client benchmark latency summary output"
```

### Task 4: Add failing tests for JSONL run-only records and summary append

**Files:**
- Modify: `tests/test_aistudio_http_e2e.py`
- Modify: `scripts/aistudio_http_e2e.py`
- Test: `tests/test_aistudio_http_e2e.py`

**Step 1: Write the failing test**

Add tests asserting:
- warmup phase does not write JSONL,
- exactly `measure_runs` `record_type="run"` entries are written,
- one final `record_type="summary"` entry is appended,
- each run record contains `config`, `request_meta`, `latency`, `server_headers`, `response`.

```python
def test_client_jsonl_writes_run_only_plus_summary(tmp_path):
    module = _load_module()
    out = tmp_path / "result.jsonl"
    with mock.patch.object(module, "send_request", side_effect=[(_resp(40.0, 10.0), 40.0)] * 15):
        module.main(["--role", "client", "--output-jsonl", str(out), "--warmup-runs", "5", "--measure-runs", "10"])
    lines = [json.loads(x) for x in out.read_text(encoding="ascii").splitlines()]
    run_lines = [x for x in lines if x["record_type"] == "run"]
    summary_lines = [x for x in lines if x["record_type"] == "summary"]
    assert len(run_lines) == 10
    assert len(summary_lines) == 1
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_http_e2e.py::test_client_jsonl_writes_run_only_plus_summary -q`  
Expected: FAIL until JSONL schema and writing are implemented.

**Step 3: Write minimal implementation**

Implement:
- `_append_jsonl_record(...)`,
- per-run record writer,
- terminal summary record writer.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_http_e2e.py::test_client_jsonl_writes_run_only_plus_summary -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/aistudio_http_e2e.py tests/test_aistudio_http_e2e.py
git commit -m "feat: write run-only latency records and final summary to jsonl"
```

### Task 5: Implement runtime builder with ossfs workspace override and TensorRT e2e support

**Files:**
- Modify: `scripts/aistudio_http_e2e.py`
- Modify: `tests/test_aistudio_http_e2e.py`
- Test: `tests/test_aistudio_http_e2e.py`

**Step 1: Write the failing test**

Add tests for:
- custom `--ossfs-workspace` is applied before importing `gr00t`,
- backend `tensorrt` calls `setup_tensorrt_engines` with configured dtypes and default engine path,
- backend `pytorch` skips TensorRT setup.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_http_e2e.py -k "ossfs or tensorrt" -q`  
Expected: FAIL until runtime wiring is implemented.

**Step 3: Write minimal implementation**

Implement runtime builder:
- load local ossfs runtime module helpers,
- resolve data config and instantiate `Gr00tPolicy`,
- apply TensorRT setup only for `tensorrt` backend.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_http_e2e.py -k "ossfs or tensorrt" -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/aistudio_http_e2e.py tests/test_aistudio_http_e2e.py
git commit -m "feat: add unified runtime builder for ossfs and backend selection"
```

### Task 6: Final verification and migration notes

**Files:**
- Create: `scripts/aistudio_http_e2e.py`
- Create: `tests/test_aistudio_http_e2e.py`
- Create: `docs/plans/2026-03-23-aistudio-http-e2e-unified-design.md`
- Create: `docs/plans/2026-03-23-aistudio-http-e2e-unified-implementation.md`

**Step 1: Run full focused test suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_http_e2e.py tests/test_aistudio_service.py tests/test_aistudio_client_example.py -q`  
Expected: PASS

**Step 2: Run script syntax check**

Run: `python -m py_compile scripts/aistudio_http_e2e.py`  
Expected: PASS

**Step 3: Optional manual smoke (local)**

Run server:
```bash
python scripts/aistudio_http_e2e.py --role server --backend tensorrt
```

Run client:
```bash
python scripts/aistudio_http_e2e.py --role client --backend tensorrt --measure-runs 10
```

Expected:
- `/predict` responds with compatible JSON body.
- client prints the required average latency block.
- JSONL contains 10 run records + 1 summary record.

**Step 4: Final commit**

```bash
git add scripts/aistudio_http_e2e.py tests/test_aistudio_http_e2e.py docs/plans/2026-03-23-aistudio-http-e2e-unified-design.md docs/plans/2026-03-23-aistudio-http-e2e-unified-implementation.md
git commit -m "feat: add unified aistudio http e2e server/client script with latency jsonl"
```
