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
from types import SimpleNamespace
from unittest import mock

import numpy as np


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "aistudio_s.py"
    if not script_path.exists():
        raise AssertionError(f"missing script: {script_path}")

    fake_trt_module = types.ModuleType("deployment_scripts.trt_model_forward")
    fake_trt_module.setup_tensorrt_engines = mock.Mock()

    fake_gr00t = types.ModuleType("gr00t")
    fake_gr00t_experiment = types.ModuleType("gr00t.experiment")
    fake_gr00t_model = types.ModuleType("gr00t.model")

    fake_data_config = types.ModuleType("gr00t.experiment.data_config")
    fake_data_config.DATA_CONFIG_MAP = {}

    fake_policy_module = types.ModuleType("gr00t.model.policy")
    fake_policy_module.Gr00tPolicy = mock.Mock()
    fake_policy_module.unsqueeze_dict_values = lambda value: value

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)

    module_name = f"scripts.aistudio_test_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load module spec: {script_path}")

    with mock.patch.dict(
        sys.modules,
        {
            "deployment_scripts.trt_model_forward": fake_trt_module,
            "gr00t": fake_gr00t,
            "gr00t.experiment": fake_gr00t_experiment,
            "gr00t.experiment.data_config": fake_data_config,
            "gr00t.model": fake_gr00t_model,
            "gr00t.model.policy": fake_policy_module,
            "torch": fake_torch,
        },
        clear=False,
    ):
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


class AistudioRuntimeTest(unittest.TestCase):
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

    def test_runtime_predict_uses_internal_tensorrt_runner_without_changing_response_shape(self):
        module = _load_module()
        policy = SimpleNamespace(get_action=mock.Mock(side_effect=AssertionError("unexpected call")))
        captured = {}

        def _runner(observation: dict) -> np.ndarray:
            captured["observation"] = observation
            return np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)

        runtime = module.AistudioRuntime(
            policy=policy,
            task_prompt="task prompt",
            response_action_horizon=1,
            action_runner=_runner,
        )

        with mock.patch.object(
            module,
            "decode_framebuffer",
            return_value=np.zeros((640, 480, 3), dtype=np.uint8),
        ):
            response = runtime.predict(self._build_payload())

        self.assertEqual(captured["observation"]["video.ego_view"].shape, (640, 480, 3))
        self.assertEqual(captured["observation"]["state.single_arm"].shape, (6,))
        self.assertEqual(captured["observation"]["annotation.task_index"], "task prompt")
        self.assertEqual(response["resultCode"], 0)
        self.assertEqual(response["errorMessage"], "ok")
        self.assertEqual(response["resultMap"]["request_id"], "req-1")
        self.assertEqual(response["resultMap"]["device_id"], "dev-1")
        self.assertEqual(response["resultMap"]["joint_angles"], json.dumps([1, 2, 3, 4, 5]))
        self.assertEqual(response["resultMap"]["predicted_coords_2d"], json.dumps([0.1, 0.2]))
        self.assertEqual(response["resultMap"]["history_angles"], json.dumps([[1, 2, 3, 4, 5]]))
        self.assertEqual(
            json.loads(response["resultMap"]["action_sequence"]),
            [[1.1, 2.2, 3.3, 4.4, 5.5, 0.6]],
        )
        policy.get_action.assert_not_called()

    def test_runtime_predict_without_runner_keeps_existing_batched_policy_input(self):
        module = _load_module()
        policy = SimpleNamespace(
            get_action=mock.Mock(
                return_value={"action.single_arm": np.zeros((1, 6), dtype=np.float32)}
            )
        )
        runtime = module.AistudioRuntime(
            policy=policy,
            task_prompt="task prompt",
            response_action_horizon=1,
        )

        with mock.patch.object(
            module,
            "decode_framebuffer",
            return_value=np.zeros((640, 480, 3), dtype=np.uint8),
        ):
            response = runtime.predict(self._build_payload())

        batch = policy.get_action.call_args.args[0]
        self.assertEqual(batch["video.ego_view"].shape, (1, 640, 480, 3))
        self.assertEqual(batch["state.single_arm"].shape, (1, 6))
        self.assertEqual(batch["annotation.task_index"], ["task prompt"])
        self.assertEqual(response["resultCode"], 0)

    def test_runtime_predict_with_timing_uses_direct_inference_elapsed_time_like_inference_raw(self):
        module = _load_module()
        policy = SimpleNamespace(get_action=mock.Mock(side_effect=AssertionError("unexpected call")))

        runtime = module.AistudioRuntime(
            policy=policy,
            task_prompt="task prompt",
            response_action_horizon=1,
            action_runner=lambda observation: (
                np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32),
                {
                    "backbone_total_ms": 5.67,
                    "action_head_total_ms": 6.67,
                    "e2e_total_ms": 20.0,
                },
            ),
        )

        with (
            mock.patch.object(
                module,
                "decode_framebuffer",
                return_value=np.zeros((640, 480, 3), dtype=np.uint8),
            ),
            mock.patch.object(module, "measure_preferred_latency_ms", side_effect=lambda fn: (fn(), 6.5)),
        ):
            response, timing = runtime.predict_with_timing(self._build_payload())

        self.assertEqual(response["resultCode"], 0)
        self.assertEqual(timing["pure_inference_ms"], 6.5)
        self.assertEqual(timing["inference_ms"], 6.5)
        self.assertEqual(timing["request_id"], "req-1")
        self.assertEqual(timing["device_id"], "dev-1")

    def test_runtime_predict_with_timing_falls_back_to_wall_time_when_cuda_measurement_is_zero(self):
        module = _load_module()
        policy = SimpleNamespace(get_action=mock.Mock(side_effect=AssertionError("unexpected call")))

        runtime = module.AistudioRuntime(
            policy=policy,
            task_prompt="task prompt",
            response_action_horizon=1,
            action_runner=lambda observation: np.array(
                [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
                dtype=np.float32,
            ),
        )

        with (
            mock.patch.object(
                module,
                "decode_framebuffer",
                return_value=np.zeros((640, 480, 3), dtype=np.uint8),
            ),
            mock.patch.object(module, "measure_cuda_time_ms", side_effect=lambda fn: (fn(), 0.0)),
            mock.patch.object(module.time, "perf_counter_ns", side_effect=[0, 0, 6_500_000]),
        ):
            response, timing = runtime.predict_with_timing(self._build_payload())

        self.assertEqual(response["resultCode"], 0)
        self.assertEqual(timing["pure_inference_ms"], 6.5)
        self.assertEqual(timing["inference_ms"], 6.5)

    def test_runtime_predict_with_timing_uses_stage_wall_time_when_runner_raises(self):
        module = _load_module()
        policy = SimpleNamespace(get_action=mock.Mock(side_effect=AssertionError("unexpected call")))

        def _runner(_observation: dict) -> np.ndarray:
            raise RuntimeError("runner crash")

        runtime = module.AistudioRuntime(
            policy=policy,
            task_prompt="task prompt",
            response_action_horizon=1,
            action_runner=_runner,
        )

        with (
            mock.patch.object(
                module,
                "decode_framebuffer",
                return_value=np.zeros((640, 480, 3), dtype=np.uint8),
            ),
            mock.patch.object(module.time, "perf_counter_ns", side_effect=[0, 0, 6_500_000]),
            mock.patch.object(module, "measure_cuda_time_ms", side_effect=lambda fn: (fn(), 0.0)),
        ):
            response, timing = runtime.predict_with_timing(self._build_payload())

        self.assertEqual(response["resultCode"], 1)
        self.assertEqual(response["errorMessage"], "runner crash")
        self.assertEqual(timing["pure_inference_ms"], 6.5)
        self.assertEqual(timing["inference_ms"], 6.5)


class BuildRuntimeTest(unittest.TestCase):
    def test_build_runtime_uses_data_config_map_when_load_data_config_is_absent(self):
        module = _load_module()
        fake_policy = SimpleNamespace()
        fake_data_config = SimpleNamespace(
            modality_config=lambda: {"cfg": "value"},
            transform=lambda: "transform",
        )
        fake_breakdown_module = types.ModuleType(
            "deployment_scripts.ant.local_inference_tensorrt_breakdown"
        )
        helper = mock.Mock(
            return_value=(
                {"action.single_arm": np.array([[0.5, 0.4, 0.3, 0.2, 0.1, 0.0]], dtype=np.float32)},
                {"e2e_total_ms": 1.0},
            )
        )
        fake_breakdown_module.run_breakdown_inference = helper
        fake_ossfs_runtime = SimpleNamespace(
            DATA_CONFIG_MAP={"new_interaction_group": fake_data_config},
            Gr00tPolicy=mock.Mock(return_value=fake_policy),
            COMPUTE_DTYPE="ossfs_compute_dtype",
            unsqueeze_dict_values="ossfs_unsqueeze",
        )
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

        with (
            mock.patch.object(module, "_import_local_gr00t_runtime", return_value=fake_ossfs_runtime),
            mock.patch.object(module, "setup_tensorrt_engines") as setup_tensorrt_engines,
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.dict(
                sys.modules,
                {
                    "deployment_scripts.ant.local_inference_tensorrt_breakdown": fake_breakdown_module
                },
                clear=False,
            ),
        ):
            runtime = module.build_runtime(args)
            action = runtime.action_runner(
                {
                    "video.ego_view": np.zeros((640, 480, 3), dtype=np.uint8),
                    "annotation.task_index": "task prompt",
                    "state.single_arm": np.zeros((6,), dtype=np.float32),
                }
            )

        setup_tensorrt_engines.assert_called_once()
        np.testing.assert_allclose(action, [[0.5, 0.4, 0.3, 0.2, 0.1, 0.0]])
        fake_ossfs_runtime.Gr00tPolicy.assert_called_once()
        self.assertEqual(
            fake_ossfs_runtime.Gr00tPolicy.call_args.kwargs["modality_config"],
            {"cfg": "value"},
        )
        self.assertEqual(
            fake_ossfs_runtime.Gr00tPolicy.call_args.kwargs["modality_transform"],
            "transform",
        )
        helper.assert_called_once_with(
            fake_policy,
            {
                "video.ego_view": mock.ANY,
                "annotation.task_index": "task prompt",
                "state.single_arm": mock.ANY,
            },
            "ossfs_unsqueeze",
            compute_dtype="ossfs_compute_dtype",
        )


class AppLatencySurfaceTest(unittest.TestCase):
    def test_predict_route_adds_latency_headers_and_prints_average_latency_summary(self):
        module = _load_module()

        runtime = SimpleNamespace(
            ready=True,
            metadata={"backend": "tensorrt"},
            predict_with_timing=mock.Mock(
                side_effect=[
                    (
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
                            "request_id": "req-1",
                            "device_id": "dev-1",
                            "backend": "tensorrt",
                            "pure_inference_ms": 10.0,
                            "inference_ms": 10.0,
                        },
                    ),
                    (
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
                                "request_id": "req-2",
                            }
                        ),
                        {
                            "request_id": "req-2",
                            "device_id": "dev-1",
                            "backend": "tensorrt",
                            "pure_inference_ms": 20.0,
                            "inference_ms": 20.0,
                        },
                    ),
                ]
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "online_result.jsonl"
            app = module.create_app(runtime=runtime, online_result_path=output_path)
            predict_endpoint = next(route.endpoint for route in app.routes if route.path == "/predict")
            stdout = StringIO()
            with (
                mock.patch.object(
                    module.time,
                    "perf_counter_ns",
                    side_effect=[0, 25_000_000, 50_000_000, 80_000_000],
                ),
                redirect_stdout(stdout),
            ):
                response = predict_endpoint({"query": "{}"})
                second_response = predict_endpoint({"query": "{}"})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(second_response.status_code, 200)
            body = json.loads(response.body)
            self.assertEqual(body["resultMap"]["request_id"], "req-1")
            self.assertEqual(response.headers["X-Server-Total-Ms"], "25.0")
            self.assertEqual(response.headers["X-Pure-Inference-Ms"], "10.0")
            self.assertEqual(response.headers["X-Inference-Ms"], "10.0")
            self.assertEqual(response.headers["X-Server-Overhead-Ms"], "15.0")
            self.assertEqual(second_response.headers["X-Server-Total-Ms"], "30.0")
            self.assertEqual(second_response.headers["X-Pure-Inference-Ms"], "20.0")
            self.assertEqual(second_response.headers["X-Server-Overhead-Ms"], "10.0")
            self.assertIn("=== Average Latency ===", stdout.getvalue())
            self.assertIn("Average total latency: 27.500 ms", stdout.getvalue())
            self.assertIn("Average inference latency: 15.000 ms", stdout.getvalue())
            self.assertIn("Average network latency: 12.500 ms", stdout.getvalue())
            self.assertIn("Average inference proportion: 54.55%", stdout.getvalue())
            self.assertNotIn("[aistudio]", stdout.getvalue())

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first_record = json.loads(lines[0])
            second_record = json.loads(lines[1])
            self.assertEqual(first_record["meta"]["request_id"], "req-1")
            self.assertEqual(first_record["meta"]["device_id"], "dev-1")
            self.assertEqual(first_record["meta"]["backend"], "tensorrt")
            self.assertEqual(first_record["meta"]["result_code"], 0)
            self.assertEqual(first_record["response"]["resultCode"], 0)
            self.assertEqual(first_record["response"]["errorMessage"], "ok")
            self.assertEqual(first_record["response"]["resultMap"]["request_id"], "req-1")
            self.assertEqual(first_record["metrics"]["average_total_latency_ms"], 25.0)
            self.assertEqual(first_record["metrics"]["average_inference_latency_ms"], 10.0)
            self.assertEqual(first_record["metrics"]["average_network_latency_ms"], 15.0)
            self.assertEqual(first_record["metrics"]["average_inference_proportion_pct"], 40.0)
            self.assertEqual(second_record["meta"]["request_id"], "req-2")
            self.assertEqual(second_record["metrics"]["average_total_latency_ms"], 27.5)
            self.assertEqual(second_record["metrics"]["average_inference_latency_ms"], 15.0)
            self.assertEqual(second_record["metrics"]["average_network_latency_ms"], 12.5)
            self.assertEqual(second_record["metrics"]["average_inference_proportion_pct"], 54.55)


if __name__ == "__main__":
    unittest.main()
