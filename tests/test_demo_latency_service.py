import importlib.util
import json
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "demo_s.py"
    if not script_path.exists():
        raise AssertionError(f"missing script: {script_path}")

    module_name = f"scripts.demo_s_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load module spec: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class DemoServerLatencyTest(unittest.TestCase):
    def _build_payload(self) -> dict:
        return {
            "query": json.dumps(
                {
                    "joint_angles": [1, 2, 3, 4, 5],
                    "predicted_coords_2d": [0.1, 0.2],
                    "history_angles": [[1, 2, 3, 4, 5]],
                    "framebuffer": [1, 2, 3],
                    "framebuffer_size": 3,
                    "device_id": "dev-1",
                    "request_id": "req-1",
                }
            )
        }

    def test_postprocess_actions_accepts_single_batch_dimension(self):
        module = _load_module()
        previous_actions = np.array(
            [
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                ]
            ],
            dtype=np.float32,
        )
        robot_state = np.array([1, 2, 3, 4, 5, 0], dtype=np.float32)

        action_string, is_success, _ = module.postprocess_actions(
            previous_actions,
            robot_state,
            response_action_horizon=2,
        )

        self.assertTrue(is_success)
        self.assertEqual(
            json.loads(action_string),
            [
                [1.1, 2.2, 3.3, 4.4, 5.5, 0.6],
                [1.6, 2.7, 3.8, 4.9, 6.0, 1.1],
            ],
        )

    def test_predict_with_timing_prefers_runner_breakdown_e2e_metric(self):
        module = _load_module()
        runtime = module.AistudioRuntime(
            policy=SimpleNamespace(get_action=mock.Mock(side_effect=AssertionError("unexpected call"))),
            task_prompt="task prompt",
            response_action_horizon=1,
            action_runner=lambda _observation: (
                np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32),
                {"e2e_total_ms": 42.5},
            ),
            metadata={"backend": "tensorrt"},
        )

        with (
            mock.patch.object(
                module,
                "decode_framebuffer",
                return_value=np.zeros((640, 480, 3), dtype=np.uint8),
            ),
            mock.patch.object(module, "measure_inference_latency_ms", side_effect=lambda fn: (fn(), 8.0)),
        ):
            response, timing = runtime.predict_with_timing(self._build_payload())

        self.assertEqual(response["resultCode"], 0)
        self.assertEqual(timing["pure_inference_ms"], 42.5)
        self.assertEqual(timing["inference_ms"], 42.5)
        self.assertEqual(timing["request_id"], "req-1")
        self.assertEqual(timing["device_id"], "dev-1")

    def test_create_app_excludes_failed_runs_from_average_metrics(self):
        module = _load_module()
        runtime = SimpleNamespace(
            ready=True,
            metadata={"backend": "tensorrt"},
            predict_with_timing=mock.Mock(
                side_effect=[
                    (
                        module.build_aistudio_response(
                            result_code=1,
                            error_message="bad request",
                            result_map={"request_id": "req-failed", "device_id": "dev-1"},
                        ),
                        {
                            "request_id": "req-failed",
                            "device_id": "dev-1",
                            "backend": "tensorrt",
                            "pure_inference_ms": 7.0,
                            "inference_ms": 7.0,
                        },
                    ),
                    (
                        module.build_aistudio_response(
                            result_map={"request_id": "req-ok", "device_id": "dev-1"},
                        ),
                        {
                            "request_id": "req-ok",
                            "device_id": "dev-1",
                            "backend": "tensorrt",
                            "pure_inference_ms": 10.0,
                            "inference_ms": 10.0,
                        },
                    ),
                ]
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "online_result.jsonl"
            app = module.create_app(runtime=runtime, online_result_path=output_path)
            predict_endpoint = next(route.endpoint for route in app.routes if route.path == "/predict")

            with mock.patch.object(
                module.time,
                "perf_counter_ns",
                side_effect=[0, 100_000_000, 100_000_000, 120_000_000],
            ):
                failed_response = predict_endpoint({"query": "{}"})
                success_response = predict_endpoint({"query": "{}"})

            self.assertEqual(failed_response.status_code, 200)
            self.assertEqual(success_response.status_code, 200)
            self.assertEqual(failed_response.headers["X-Server-Total-Ms"], "100.0")
            self.assertEqual(success_response.headers["X-Server-Total-Ms"], "20.0")

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first_record = json.loads(lines[0])
            second_record = json.loads(lines[1])
            self.assertEqual(first_record["metrics"]["sample_count"], 0)
            self.assertEqual(second_record["metrics"]["sample_count"], 1)
            self.assertEqual(second_record["metrics"]["average_total_latency_ms"], 20.0)
            self.assertEqual(second_record["metrics"]["average_inference_latency_ms"], 10.0)
            self.assertEqual(second_record["metrics"]["average_network_latency_ms"], 10.0)

    def test_build_runtime_tensorrt_converts_string_prompt_to_array_for_breakdown(self):
        module = _load_module()
        captured = {}

        def _run_breakdown_inference(policy, raw_obs, unsqueeze_dict_values, *, compute_dtype=None):
            del policy, unsqueeze_dict_values, compute_dtype
            captured["raw_obs"] = raw_obs
            return {"action.single_arm": np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)}, {}

        fake_policy = SimpleNamespace()
        fake_data_config = SimpleNamespace(
            modality_config=lambda: {"cfg": "value"},
            transform=lambda: "transform",
        )
        fake_ossfs_runtime = SimpleNamespace(
            DATA_CONFIG_MAP={"new_interaction_group": fake_data_config},
            Gr00tPolicy=mock.Mock(return_value=fake_policy),
            unsqueeze_dict_values="unsqueeze_dict_values",
            COMPUTE_DTYPE="fp16",
        )
        fake_ossfs_helper_module = types.SimpleNamespace(
            _import_local_gr00t_runtime=lambda: fake_ossfs_runtime
        )
        fake_trt_module = types.SimpleNamespace(setup_tensorrt_engines=mock.Mock())
        fake_breakdown_module = types.SimpleNamespace(run_breakdown_inference=_run_breakdown_inference)
        args = SimpleNamespace(
            backend="tensorrt",
            model_path="/tmp/model",
            trt_engine_path="/tmp/engine",
            data_config="new_interaction_group",
            embodiment_tag="new_embodiment",
            task_prompt="task prompt",
            response_action_horizon=14,
            denoising_steps=4,
            vit_dtype="fp16",
            llm_dtype="fp16",
            dit_dtype="fp16",
        )

        with mock.patch.dict(
            sys.modules,
            {
                "deployment_scripts.ant.ossfs_gr00t_runtime": fake_ossfs_helper_module,
                "deployment_scripts.trt_model_forward": fake_trt_module,
                "deployment_scripts.ant.local_inference_tensorrt_breakdown": fake_breakdown_module,
            },
            clear=False,
        ):
            runtime = module.build_runtime(args)
            runtime.action_runner(
                {
                    "video.ego_view": np.zeros((640, 480, 3), dtype=np.uint8),
                    "annotation.task_index": "task prompt",
                    "state.single_arm": np.zeros((6,), dtype=np.float32),
                }
            )

        self.assertIn("raw_obs", captured)
        self.assertIsInstance(captured["raw_obs"]["annotation.task_index"], np.ndarray)


if __name__ == "__main__":
    unittest.main()
