# Inference Raw Launcher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a one-command launcher for `scripts/inference_raw.py` that selects server/client and HTTP/ZMQ modes from short CLI selectors while preserving the user's existing defaults.

**Architecture:** Create a Bash wrapper at `scripts/run_inference.sh` that validates two selector arguments, applies a small set of optional overrides, resolves the repository root from the script location, and execs `conda run -n gr00t-orin python scripts/inference_raw.py ...`. Add focused CLI tests that stub `conda` and verify the generated command line without starting real inference workloads.

**Tech Stack:** Bash, Python, pytest, subprocess, pathlib

---

### Task 1: Add failing launcher CLI tests

**Files:**
- Create: `tests/test_run_inference_launcher.py`

**Step 1: Write the failing test**

```python
def test_server_zmq_invokes_expected_defaults():
    result = _run_launcher("s", "z")
    assert result.returncode == 0
    assert "--server" in _stdout_args(result)
```

```python
def test_client_http_invokes_expected_defaults():
    result = _run_launcher("c", "h")
    assert result.returncode == 0
    assert "--http-server" in _stdout_args(result)
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_run_inference_launcher.py -q`
Expected: FAIL because `scripts/run_inference.sh` does not exist yet.

**Step 3: Write minimal implementation**

Do not implement yet. Keep only the failing test in place.

**Step 4: Run test to verify it fails for the expected reason**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_run_inference_launcher.py -q`
Expected: FAIL with missing launcher script or non-zero launcher invocation.

**Step 5: Commit**

```bash
git add tests/test_run_inference_launcher.py
git commit -m "test: add inference launcher cli expectations"
```

### Task 2: Implement the launcher script

**Files:**
- Create: `scripts/run_inference.sh`

**Step 1: Write the failing test**

```python
def test_server_http_override_replaces_host_port_and_model_path():
    result = _run_launcher(
        "s",
        "h",
        "--host",
        "127.0.0.1",
        "--port",
        "9000",
        "--model-path",
        "/tmp/model",
    )
    args = _stdout_args(result)
    assert "127.0.0.1" in args
    assert "9000" in args
    assert "/tmp/model" in args
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_run_inference_launcher.py -q`
Expected: FAIL because the override handling does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- help output and selector validation
- default host and port mapping for ZMQ and HTTP
- server-only defaults for model, TensorRT, and dtypes
- optional overrides for `--host`, `--port`, `--model-path`, `--denoising-steps`, `--trt-engine-path`, and `--env-name`
- repository-root resolution and `exec conda run ...`

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_run_inference_launcher.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/run_inference.sh tests/test_run_inference_launcher.py
git commit -m "feat: add inference raw launcher"
```

### Task 3: Verify launcher behavior and docs

**Files:**
- Create: `docs/plans/2026-03-19-inference-raw-launcher-design.md`
- Create: `docs/plans/2026-03-19-inference-raw-launcher-implementation.md`

**Step 1: Run focused verification**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_run_inference_launcher.py -q`
Expected: PASS

**Step 2: Run CLI smoke checks**

Run: `bash scripts/run_inference.sh --help`
Expected: PASS and prints the four selector combinations

Run: `bash scripts/run_inference.sh s z`
Expected: attempts to run `scripts/inference_raw.py` with server ZMQ defaults

Run: `bash scripts/run_inference.sh c h`
Expected: attempts to run `scripts/inference_raw.py` with client HTTP defaults

**Step 3: Review diff for scope control**

Check that:

- only the new launcher, focused tests, and plan docs were added
- `scripts/inference_raw.py` behavior is unchanged
- the launcher does not auto-start background processes

**Step 4: Commit**

```bash
git add docs/plans/2026-03-19-inference-raw-launcher-design.md docs/plans/2026-03-19-inference-raw-launcher-implementation.md scripts/run_inference.sh tests/test_run_inference_launcher.py
git commit -m "docs: add inference raw launcher design and plan"
```
