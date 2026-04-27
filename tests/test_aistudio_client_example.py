import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "aistudio_c.py"
    if not script_path.exists():
        raise AssertionError(f"missing script: {script_path}")

    spec = importlib.util.spec_from_file_location("scripts.aistudio_c", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load module spec: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AistudioClientExampleTest(unittest.TestCase):
    def test_parser_defaults_support_zero_arg_random_benchmark(self):
        module = _load_module()
        args = module.build_parser().parse_args([])

        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8000)
        self.assertEqual(args.timeout_ms, 15000)
        self.assertEqual(args.warmup_runs, 5)
        self.assertEqual(args.measure_runs, 10)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.device_id, "dev-random")
        self.assertEqual(args.request_id, "req-0001")
        self.assertEqual(args.image_path, "")
        self.assertEqual(args.joint_angles, "")

    def test_build_request_payload_generates_random_defaults_when_inputs_missing(self):
        module = _load_module()
        args = module.build_parser().parse_args([])

        with (
            mock.patch.object(module, "generate_random_rgb_image", return_value="image"),
            mock.patch.object(module, "encode_rgb_to_framebuffer", return_value=([1, 2, 3], 3)),
        ):
            payload = module.build_request_payload(
                args,
                request_id="req-1",
                rng=module.np.random.default_rng(0),
            )

        inner = json.loads(payload["query"])
        joint_angles = module.parse_literal_list(inner["joint_angles"])
        predicted_coords_2d = module.parse_literal_list(inner["predicted_coords_2d"])
        history_angles = module.parse_literal_list(inner["history_angles"])
        self.assertEqual(len(joint_angles), 5)
        self.assertEqual(len(predicted_coords_2d), 2)
        self.assertEqual(len(history_angles), 1)
        self.assertEqual(len(history_angles[0]), 5)
        self.assertEqual(inner["framebuffer"], [1, 2, 3])
        self.assertEqual(inner["framebuffer_size"], 3)
        self.assertEqual(inner["device_id"], "dev-random")
        self.assertEqual(inner["request_id"], "req-1")

    def test_main_runs_default_warmup_and_measure_benchmark(self):
        module = _load_module()

        class _Response:
            headers = {
                "X-Server-Total-Ms": "25.0",
                "X-Pure-Inference-Ms": "7.89",
                "X-Inference-Ms": "7.89",
                "X-Server-Overhead-Ms": "17.11",
            }

            def raise_for_status(self):
                return None

            def json(self):
                return {"resultCode": 0, "resultMap": {"request_id": "req-1"}}

        send_calls = []

        def _send_request(*, host, port, payload, timeout_ms):
            send_calls.append((host, port, payload, timeout_ms))
            return _Response(), 40.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "online_result.jsonl"
            stdout = StringIO()
            with (
                mock.patch.object(module, "_DEFAULT_ONLINE_RESULT_JSONL", output_path),
                mock.patch.object(
                    module,
                    "build_request_payload",
                    side_effect=lambda args, request_id, rng: {"query": request_id},
                ),
                mock.patch.object(module, "send_request", side_effect=_send_request),
                redirect_stdout(stdout),
            ):
                exit_code = module.main([])

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(send_calls), 15)
            self.assertEqual(send_calls[0][2]["query"], "req-0001-warmup-00")
            self.assertEqual(send_calls[4][2]["query"], "req-0001-warmup-04")
            self.assertEqual(send_calls[5][2]["query"], "req-0001-run-00")
            self.assertEqual(send_calls[-1][2]["query"], "req-0001-run-09")
            output = stdout.getvalue()
            self.assertIn("[run 1/10]", output)
            self.assertIn("client_round_trip_ms=40.0000", output)
            self.assertIn("pure_inference_ms=7.8900", output)
            self.assertIn("network_ms=32.1100", output)
            self.assertIn("=== Average Latency ===", output)
            self.assertIn("Average total latency: 40.000 ms", output)
            self.assertIn("Average inference latency: 7.890 ms", output)
            self.assertIn("Average network latency: 32.110 ms", output)
            self.assertIn("Average inference proportion: 19.72%", output)

            records = [json.loads(line) for line in output_path.read_text(encoding="ascii").splitlines()]
            self.assertEqual(len(records), 15)
            self.assertEqual(records[0]["meta"]["phase"], "warmup")
            self.assertEqual(records[0]["meta"]["request_id"], "req-1")
            self.assertEqual(records[5]["meta"]["phase"], "run")
            self.assertEqual(records[5]["meta"]["result_code"], 0)
            self.assertEqual(records[5]["response"]["resultCode"], 0)
            self.assertEqual(records[5]["metrics"]["client_round_trip_ms"], 40.0)


if __name__ == "__main__":
    unittest.main()
