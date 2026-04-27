# Aistudio Client Random Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `scripts/aistudio_client_example.py` run with zero arguments by generating random compatible test inputs and executing a default 5-warmup/10-measure benchmark loop.

**Architecture:** Replace required input arguments with optional overrides and deterministic random defaults driven by a seed. Split request generation, request-id derivation, and benchmark execution into small helpers so tests can lock zero-arg defaults, payload shape, and benchmark loop counts independently.

**Tech Stack:** Python, NumPy, requests, argparse, unittest/pytest

---

### Task 1: Add failing client benchmark tests

**Files:**
- Modify: `tests/test_aistudio_client_example.py`
- Test: `tests/test_aistudio_client_example.py`

**Step 1: Write the failing test**

Add tests that prove:
- parser defaults allow zero-arg execution,
- missing input overrides trigger random request generation,
- `main()` performs `5` warmup runs and `10` measured runs by default and prints summary lines.

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_client_example.py -q`
Expected: FAIL because the client still requires explicit input arguments and only performs one request.

**Step 3: Write minimal implementation**

Add only the helpers and loop logic required to satisfy the tests.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_client_example.py -q`
Expected: PASS

### Task 2: Run final focused verification

**Files:**
- Test: `tests/test_aistudio_client_example.py`

**Step 1: Run focused verification**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_aistudio_client_example.py -q`
Expected: PASS
