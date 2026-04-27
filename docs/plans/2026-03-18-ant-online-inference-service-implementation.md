# ANT Online Inference Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a dedicated ANT online inference service entrypoint with standard production serving and strict Aistudio-compatible HTTP serving.

**Architecture:** Create `deployment_scripts/ant/online_inference_service.py` as a service-only entrypoint that separates runtime loading, request adaptation, and transport handling. Standard mode serves canonical `observation -> action` over HTTP and ZMQ with compatibility aliases, while Aistudio mode serves strict `query`-wrapped HTTP requests using dual-model routing and supports both PyTorch and TensorRT.

**Tech Stack:** Python, argparse, FastAPI, uvicorn, requests/httpx, zmq, unittest, existing GR00T/ANT runtime helpers

---

### Task 1: Add failing tests for service config validation and parser defaults

**Files:**
- Create: `deployment_scripts/ant/test_online_inference_service.py`

**Step 1: Write the failing test**

```python
def test_default_parser_uses_standard_http_pytorch_server():
    args = build_parser().parse_args(["--role", "server"])
    assert args.service_mode == "standard"
    assert args.transport == "http"
    assert args.backend == "pytorch"
```

```python
def test_aistudio_mode_rejects_zmq_transport():
    args = build_parser().parse_args(
        ["--role", "server", "--service-mode", "aistudio", "--transport", "zmq"]
    )
    with pytest.raises(ValueError):
        validate_args(args)
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_online_inference_service.py -q`
Expected: FAIL because the online service module and parser do not exist yet.

**Step 3: Write minimal implementation**

Create the test file first and import placeholders from the future service module.

**Step 4: Run test to verify it fails for the expected reason**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_online_inference_service.py -q`
Expected: FAIL with import or missing symbol errors.

**Step 5: Commit**

```bash
git add deployment_scripts/ant/test_online_inference_service.py
git commit -m "test: add ant online service config expectations"
```

### Task 2: Add the new service entrypoint skeleton and configuration validation

**Files:**
- Create: `deployment_scripts/ant/online_inference_service.py`
- Test: `deployment_scripts/ant/test_online_inference_service.py`

**Step 1: Write the failing test**

```python
def test_validate_args_allows_standard_zmq_tensorrt():
    args = build_parser().parse_args(
        ["--role", "server", "--service-mode", "standard", "--transport", "zmq", "--backend", "tensorrt"]
    )
    validate_args(args)
```

```python
def test_aistudio_engine_paths_default_to_shared_directory():
    args = build_parser().parse_args(["--role", "server", "--service-mode", "aistudio"])
    resolved = resolve_aistudio_engine_paths(args)
    assert resolved.left == resolved.right
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_online_inference_service.py -q`
Expected: FAIL until the parser and validation helpers exist.

**Step 3: Write minimal implementation**

Implement:

- parser construction
- service-mode/transport/backend validation
- shared versus left/right TensorRT engine path resolution

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_online_inference_service.py -q`
Expected: PASS for config validation tests.

**Step 5: Commit**

```bash
git add deployment_scripts/ant/online_inference_service.py deployment_scripts/ant/test_online_inference_service.py
git commit -m "feat: add ant online service config layer"
```

### Task 3: Add failing tests for the Aistudio compatibility adapter

**Files:**
- Modify: `deployment_scripts/ant/test_online_inference_service.py`

**Step 1: Write the failing test**

```python
def test_parse_aistudio_query_accepts_stringified_json():
    payload = {"query": json.dumps({"joint_angles": "[1,2,3,4,5]", "framebuffer": [1, 2], "framebuffer_size": 2})}
    parsed = parse_aistudio_query(payload)
    assert parsed["joint_angles"] == [1, 2, 3, 4, 5]
```

```python
def test_route_aistudio_policy_uses_left_model_when_joint_angle_is_negative():
    router = AistudioPolicyRouter(left_policy="left", right_policy="right")
    assert router.select_policy([-1, 0, 0, -0.1, 0]) == "left"
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_online_inference_service.py -q`
Expected: FAIL because the Aistudio adapter helpers do not exist yet.

**Step 3: Write minimal implementation**

Implement:

- strict `query` parsing
- list-like field normalization
- left/right routing helper
- result envelope helper

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_online_inference_service.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/online_inference_service.py deployment_scripts/ant/test_online_inference_service.py
git commit -m "feat: add ant aistudio compatibility adapter"
```

### Task 4: Add failing tests for HTTP routes and compatibility aliases

**Files:**
- Modify: `deployment_scripts/ant/test_online_inference_service.py`

**Step 1: Write the failing test**

```python
def test_standard_http_registers_v1_and_compat_routes():
    app = create_http_app(fake_runtime)
    routes = {route.path for route in app.routes}
    assert "/v1/act" in routes
    assert "/act" in routes
    assert "/health" in routes
```

```python
def test_aistudio_http_registers_predict_route_only():
    app = create_http_app(fake_runtime)
    routes = {route.path for route in app.routes}
    assert "/v1/aistudio/predict" in routes
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_online_inference_service.py -q`
Expected: FAIL until the FastAPI route factory exists.

**Step 3: Write minimal implementation**

Implement:

- standard HTTP routes
- compatibility aliases
- Aistudio HTTP routes
- latency endpoints and health endpoints

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_online_inference_service.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/online_inference_service.py deployment_scripts/ant/test_online_inference_service.py
git commit -m "feat: add ant online service http surface"
```

### Task 5: Add failing tests for ZMQ standard mode and runtime dispatch

**Files:**
- Modify: `deployment_scripts/ant/test_online_inference_service.py`

**Step 1: Write the failing test**

```python
def test_standard_zmq_registers_get_server_info_and_echo():
    server = create_standard_zmq_server(fake_runtime, host="*", port=5555)
    assert "get_server_info" in server._endpoints
    assert "echo" in server._endpoints
```

```python
def test_main_rejects_aistudio_zmq_combination():
    with pytest.raises(ValueError):
        main(["--role", "server", "--service-mode", "aistudio", "--transport", "zmq"])
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_online_inference_service.py -q`
Expected: FAIL until the ZMQ factory and main dispatch exist.

**Step 3: Write minimal implementation**

Implement:

- standard ZMQ server creation
- runtime dispatch by mode and backend
- top-level main routing for server/client execution

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_online_inference_service.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add deployment_scripts/ant/online_inference_service.py deployment_scripts/ant/test_online_inference_service.py
git commit -m "feat: add ant online service zmq and dispatch"
```

### Task 6: Verify the ANT online service surface

**Files:**
- Modify: `deployment_scripts/ant/test_ant_cli.py`
- Modify: `deployment_scripts/ant/test_online_inference_service.py`

**Step 1: Add or update CLI smoke tests**

Ensure `python deployment_scripts/ant/online_inference_service.py --help` exposes:

- role
- service mode
- transport
- backend
- Aistudio TensorRT path options

**Step 2: Run focused verification**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment_scripts/ant/test_online_inference_service.py deployment_scripts/ant/test_ant_cli.py deployment_scripts/ant/test_local_inference_breakdown.py deployment_scripts/ant/test_local_inference_tensorrt.py deployment_scripts/ant/test_profile_utils.py deployment_scripts/ant/test_quality_metrics.py deployment_scripts/ant/test_system_metrics.py deployment_scripts/ant/test_torch_compile_utils.py -q`
Expected: PASS

**Step 3: Run CLI smoke checks**

Run: `python deployment_scripts/ant/online_inference_service.py --help`
Expected: PASS

Run: `python deployment_scripts/ant/online_inference_service.py --role server --service-mode standard --transport http --help`
Expected: PASS

Run: `python deployment_scripts/ant/online_inference_service.py --role server --service-mode aistudio --transport http --help`
Expected: PASS

**Step 4: Review diff for scope safety**

Check that:

- `local_inference.py` remains benchmark-only
- `scripts/inference_service.py` is untouched in phase one
- Aistudio mode still requires the outer `query` field
- standard mode still offers compatibility aliases

**Step 5: Commit**

```bash
git add docs/plans/2026-03-18-ant-online-inference-service-design.md docs/plans/2026-03-18-ant-online-inference-service-implementation.md deployment_scripts/ant
git commit -m "feat: add ant online inference service"
```
