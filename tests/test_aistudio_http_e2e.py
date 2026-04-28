import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from starlette.websockets import WebSocketDisconnect
from types import SimpleNamespace
from unittest import mock


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "aistudio_http_e2e.py"
    if not script_path.exists():
        raise AssertionError(f"missing script: {script_path}")

    fake_trt_module = types.ModuleType("deployment_scripts.trt_model_forward")
    fake_trt_module.setup_tensorrt_engines = mock.Mock()

    fake_ossfs_helper = types.ModuleType("deployment_scripts.ant.ossfs_gr00t_runtime")
    fake_ossfs_helper._OSSFS_WORKSPACE = Path(
        "/home/jetson/Desktop/project/ossfs/node_59823209/workspace"
    )
    fake_ossfs_helper._ensure_ossfs_gr00t_on_path = lambda: fake_ossfs_helper._OSSFS_WORKSPACE
    fake_ossfs_helper._import_local_gr00t_runtime = mock.Mock()

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)

    module_name = f"scripts.aistudio_http_e2e_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load module spec: {script_path}")

    with mock.patch.dict(
        sys.modules,
        {
            "deployment_scripts.trt_model_forward": fake_trt_module,
            "deployment_scripts.ant.ossfs_gr00t_runtime": fake_ossfs_helper,
            "torch": fake_torch,
        },
        clear=False,
    ):
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


class ParserAndPayloadTest(unittest.TestCase):
    def test_parser_defaults_are_fixed_for_new_device(self):
        module = _load_module()
        args = module.build_parser().parse_args([])

        self.assertEqual(args.role, "server")
        self.assertEqual(args.data_config, "new_interaction_group")
        self.assertEqual(args.embodiment_tag, "new_embodiment")
        self.assertEqual(args.response_action_horizon, 14)
        self.assertEqual(
            args.task_prompt,
            "Move to center the book in view. Do nothing if no book is present.",
        )
        self.assertEqual(
            args.model_path, "/home/jetson/Desktop/project/new_model/left_hand_v2_1223"
        )
        self.assertEqual(
            args.ossfs_workspace, "/home/jetson/Desktop/project/ossfs/node_59823209/workspace"
        )
        self.assertEqual(
            args.trt_engine_path,
            "/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16",
        )
        self.assertEqual(args.ws_path, "/ws/predict")
        self.assertEqual(args.output_jsonl, str(module.DEFAULT_OUTPUT_JSONL))
        self.assertEqual(args.backend, "tensorrt")

    def test_payload_keeps_aistudio_query_shape(self):
        module = _load_module()
        args = module.build_parser().parse_args(["--role", "client"])
        with (
            mock.patch.object(module, "generate_random_rgb_image", return_value="image"),
            mock.patch.object(module, "encode_rgb_to_framebuffer", return_value=([1, 2, 3], 3)),
        ):
            payload = module.build_request_payload(
                args, request_id="req-1", rng=module.np.random.default_rng(0)
            )

        self.assertEqual(set(payload.keys()), {"query"})
        inner = json.loads(payload["query"])
        self.assertEqual(
            set(inner.keys()),
            {
                "joint_angles",
                "predicted_coords_2d",
                "history_angles",
                "framebuffer",
                "framebuffer_size",
                "device_id",
                "request_id",
            },
        )
        self.assertEqual(inner["request_id"], "req-1")
        self.assertEqual(inner["framebuffer"], [1, 2, 3])
        self.assertEqual(inner["framebuffer_size"], 3)


class ServerSurfaceTest(unittest.TestCase):
    def test_websocket_route_dependency_marks_websocket_param(self):
        module = _load_module()
        runtime = SimpleNamespace(
            ready=True,
            metadata={},
            predict_with_timing=mock.Mock(return_value=(module.build_aistudio_response(result_map={}), {})),
        )
        app = module.create_app(runtime=runtime)
        ws_route = next(
            route for route in app.router.routes if getattr(route, "path", "") == "/ws/predict"
        )

        self.assertEqual(ws_route.dependant.websocket_param_name, "websocket")
        self.assertEqual([param.name for param in ws_route.dependant.query_params], [])

    def test_websocket_predict_route_keeps_response_shape_and_latency_fields(self):
        module = _load_module()
        runtime = SimpleNamespace(
            ready=True,
            metadata={"backend": "tensorrt"},
            predict_with_timing=mock.Mock(
                return_value=(
                    module.build_aistudio_response(
                        result_map={
                            "action_sequence": "[[1,2,3,4,5,6]]",
                            "joint_angles": "[1,2,3,4,5]",
                            "predicted_coords_2d": "[0.1,0.2]",
                            "history_angles": "[]",
                            "is_success": True,
                            "error": "",
                            "key_infos": "key_infos",
                            "device_id": "dev-1",
                            "request_id": "req-1",
                        }
                    ),
                    {
                        "preprocess_ms": 5.0,
                        "get_action_ms": 10.0,
                        "model_forward_ms": 7.0,
                        "transform_ms": 2.5,
                        "postprocess_ms": 0.1,
                    },
                )
            ),
        )
        app = module.create_app(runtime=runtime)
        self.assertFalse(any(route.path == "/predict" for route in app.routes))
        ws_predict_endpoint = next(
            route.endpoint for route in app.router.routes if getattr(route, "path", "") == "/ws/predict"
        )

        class _FakeWebSocket:
            def __init__(self):
                self.accepted = False
                self.messages = [json.dumps({"query": "{}"})]
                self.sent_json = []

            async def accept(self):
                self.accepted = True

            async def receive_text(self):
                if not self.messages:
                    raise WebSocketDisconnect()
                return self.messages.pop(0)

            async def send_json(self, payload):
                self.sent_json.append(payload)

        ws = _FakeWebSocket()
        with mock.patch.object(module.time, "perf_counter_ns", side_effect=[0, 25_000_000]):
            asyncio.run(ws_predict_endpoint(ws))

        self.assertTrue(ws.accepted)
        self.assertEqual(len(ws.sent_json), 1)
        envelope = ws.sent_json[0]
        body = envelope["result"]
        self.assertEqual(body["resultCode"], 0)
        self.assertEqual(body["resultMap"]["request_id"], "req-1")
        latency = envelope["latency"]
        self.assertEqual(latency["server_total_ms"], 25.0)
        self.assertEqual(latency["get_action_ms"], 10.0)
        self.assertEqual(latency["model_forward_ms"], 7.0)
        self.assertEqual(latency["transform_ms"], 2.5)
        self.assertEqual(latency["preprocess_ms"], 5.0)
        self.assertEqual(latency["postprocess_ms"], 0.1)
        self.assertIn("server_overhead_ms", latency)


class ClientLatencyAndJsonlTest(unittest.TestCase):
    @staticmethod
    def _fake_response(request_id: str):
        return SimpleNamespace(
            status_code=200,
            server_latency={
                "server_total_ms": 30.0,
                "preprocess_ms": 12.0,
                "get_action_ms": 15.0,
                "model_forward_ms": 8.0,
                "transform_ms": 5.0,
                "postprocess_ms": 0.5,
                "server_overhead_ms": 2.5,
            },
            json=lambda: {"resultCode": 0, "errorMessage": "ok", "resultMap": {"request_id": request_id}},
        )

    def test_client_prints_required_average_latency_block_and_writes_run_plus_summary(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "result.jsonl"
            stdout = StringIO()

            def _send_request(*, host, port, ws_path, payload, timeout_ms):
                del host, port, timeout_ms
                self.assertEqual(ws_path, "/ws/predict")
                request_id = payload["query"]
                return self._fake_response(request_id), 40.0

            with (
                mock.patch.object(
                    module,
                    "build_request_payload",
                    side_effect=lambda args, request_id, rng: {"query": request_id},
                ),
                mock.patch.object(module, "send_request", side_effect=_send_request),
                redirect_stdout(stdout),
            ):
                exit_code = module.main(
                    [
                        "--role",
                        "client",
                        "--warmup-runs",
                        "5",
                        "--measure-runs",
                        "10",
                        "--output-jsonl",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("=== Average Latency (10 runs) ===", output)
            self.assertIn("Client round-trip:", output)
            self.assertIn("Network RTT:", output)
            self.assertIn("Server total:", output)
            self.assertIn("Model fwd:", output)
            self.assertIn("Transforms:", output)
            self.assertIn("GPU inference ratio:", output)

            records = [json.loads(line) for line in output_path.read_text(encoding="ascii").splitlines()]
            run_records = [record for record in records if record["record_type"] == "run"]
            summary_records = [record for record in records if record["record_type"] == "summary"]
            self.assertEqual(len(run_records), 10)
            self.assertEqual(len(summary_records), 1)
            self.assertEqual(run_records[0]["request_meta"]["phase"], "run")
            lat = run_records[0]["latency"]
            self.assertEqual(lat["client_total_ms"], 40.0)
            self.assertEqual(lat["server_total_ms"], 30.0)
            self.assertEqual(lat["model_forward_ms"], 8.0)
            self.assertEqual(lat["network_rtt_ms"], 10.0)
            self.assertEqual(summary_records[0]["run_count"], 10)

    def test_client_fails_fast_when_websocket_dependency_is_missing(self):
        module = _load_module()
        with mock.patch.object(
            module,
            "send_request",
            side_effect=RuntimeError("websocket-client is required to run client benchmark"),
        ):
            with self.assertRaisesRegex(RuntimeError, "websocket-client is required"):
                module.main(
                    [
                        "--role",
                        "client",
                        "--warmup-runs",
                        "1",
                        "--measure-runs",
                        "1",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
