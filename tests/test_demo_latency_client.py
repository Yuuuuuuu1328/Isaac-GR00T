import importlib.util
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "demo_c.py"
    if not script_path.exists():
        raise AssertionError(f"missing script: {script_path}")

    spec = importlib.util.spec_from_file_location("scripts.demo_c", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load module spec: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DemoClientLatencyTest(unittest.TestCase):
    def test_main_prints_per_run_and_average_latency_summary(self):
        module = _load_module()

        class _Response:
            status_code = 200
            headers = {
                "X-Server-Total-Ms": "90.0",
                "X-Pure-Inference-Ms": "30.0",
                "X-Inference-Ms": "30.0",
            }

            def raise_for_status(self):
                return None

            def json(self):
                return {"resultCode": 0, "resultMap": {"request_id": "req-ok"}}

        call_index = {"value": 0}

        def _send_request(*, host, port, payload, timeout_ms):
            del host, port, payload, timeout_ms
            index = call_index["value"]
            call_index["value"] += 1
            if index == 0:
                return _Response(), 50.0  # warmup
            if index == 1:
                return _Response(), 120.0
            return _Response(), 80.0

        stdout = StringIO()
        with (
            mock.patch.object(module, "build_request_payload", return_value={"query": "payload"}),
            mock.patch.object(module, "send_request", side_effect=_send_request),
            redirect_stdout(stdout),
        ):
            exit_code = module.main(["--warmup-runs", "1", "--measure-runs", "2"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("=== Run 1 ===", output)
        self.assertIn("Total latency: 120.00000 ms", output)
        self.assertIn("Inference latency: 30.00000 ms", output)
        self.assertIn("Network latency: 90.00000 ms", output)
        self.assertIn("Inference proportion: 25.00%", output)
        self.assertIn("=== Average Latency ===", output)
        self.assertIn("Average total latency: 100.00000 ms", output)
        self.assertIn("Average inference latency: 30.00000 ms", output)
        self.assertIn("Average network latency: 70.00000 ms", output)
        self.assertIn("Average inference proportion: 30.00%", output)

    def test_failed_runs_are_excluded_from_average(self):
        module = _load_module()

        class _FailedResponse:
            status_code = 200
            headers = {
                "X-Server-Total-Ms": "88.0",
                "X-Pure-Inference-Ms": "44.0",
            }

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "resultCode": 1,
                    "errorMessage": "runner crash",
                    "resultMap": {"request_id": "req-failed"},
                }

        class _SuccessResponse:
            status_code = 200
            headers = {
                "X-Server-Total-Ms": "88.0",
                "X-Pure-Inference-Ms": "22.0",
            }

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "resultCode": 0,
                    "resultMap": {"request_id": "req-ok"},
                }

        call_index = {"value": 0}

        def _send_request(*, host, port, payload, timeout_ms):
            del host, port, payload, timeout_ms
            index = call_index["value"]
            call_index["value"] += 1
            if index == 0:
                return _FailedResponse(), 100.0
            return _SuccessResponse(), 80.0

        stdout = StringIO()
        with (
            mock.patch.object(module, "build_request_payload", return_value={"query": "payload"}),
            mock.patch.object(module, "send_request", side_effect=_send_request),
            redirect_stdout(stdout),
        ):
            exit_code = module.main(["--warmup-runs", "0", "--measure-runs", "2"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("[run 1/2] request_id=req-failed resultCode=1", output)
        self.assertIn("Average total latency: 80.00000 ms", output)
        self.assertIn("Average inference latency: 22.00000 ms", output)
        self.assertIn("Average network latency: 58.00000 ms", output)
        self.assertIn("Average inference proportion: 27.50%", output)


if __name__ == "__main__":
    unittest.main()
