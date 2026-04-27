import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from deployment_scripts.ant.online_inference_service import (
    AistudioPolicyRouter,
    AistudioServiceAdapter,
    AistudioModelPaths,
    StandardServiceAdapter,
    main,
    build_aistudio_batch,
    build_aistudio_response,
    build_parser,
    create_http_app,
    parse_aistudio_query,
    postprocess_aistudio_actions,
    register_standard_zmq_endpoints,
    resolve_aistudio_engine_paths,
    resolve_aistudio_model_paths,
    validate_args,
)


class OnlineInferenceServiceConfigTest(unittest.TestCase):
    def test_default_parser_uses_standard_http_pytorch_server(self):
        args = build_parser().parse_args(["--role", "server"])
        self.assertEqual(args.service_mode, "standard")
        self.assertEqual(args.transport, "http")
        self.assertEqual(args.backend, "pytorch")

    def test_validate_args_allows_standard_zmq_tensorrt(self):
        args = build_parser().parse_args(
            [
                "--role",
                "server",
                "--service-mode",
                "standard",
                "--transport",
                "zmq",
                "--backend",
                "tensorrt",
            ]
        )

        validated = validate_args(args)

        self.assertIs(validated, args)

    def test_aistudio_mode_rejects_zmq_transport(self):
        args = build_parser().parse_args(
            ["--role", "server", "--service-mode", "aistudio", "--transport", "zmq"]
        )

        with self.assertRaises(ValueError):
            validate_args(args)

    def test_aistudio_engine_paths_default_to_shared_directory(self):
        args = build_parser().parse_args(["--role", "server", "--service-mode", "aistudio"])
        resolved = resolve_aistudio_engine_paths(args)

        self.assertEqual(resolved.left, resolved.right)
        self.assertTrue(resolved.left.endswith("gr00t_engine_fp16"))


class AistudioCompatibilityHelperTest(unittest.TestCase):
    def test_parse_aistudio_query_accepts_stringified_json(self):
        payload = {
            "query": (
                '{"joint_angles":"[1,2,3,4,5]",'
                '"predicted_coords_2d":"[0.1,0.2]",'
                '"history_angles":"[[1,2,3,4,5]]",'
                '"framebuffer":[1,2,3],'
                '"framebuffer_size":3,'
                '"device_id":"dev-1",'
                '"request_id":"req-1"}'
            )
        }

        parsed = parse_aistudio_query(payload)

        self.assertEqual(parsed["joint_angles"], [1, 2, 3, 4, 5])
        self.assertEqual(parsed["predicted_coords_2d"], [0.1, 0.2])
        self.assertEqual(parsed["history_angles"], [[1, 2, 3, 4, 5]])
        self.assertEqual(parsed["framebuffer"], [1, 2, 3])
        self.assertEqual(parsed["framebuffer_size"], 3)

    def test_policy_router_uses_left_model_when_joint_angle_is_negative(self):
        router = AistudioPolicyRouter(left_policy="left", right_policy="right")
        selected = router.select_policy([0.0, 0.0, 0.0, -0.1, 0.0])
        self.assertEqual(selected, "left")

    def test_build_aistudio_response_preserves_result_envelope(self):
        response = build_aistudio_response(
            result_map={"action_sequence": "[[1,2,3]]", "request_id": "req-1"},
            result_code=0,
            error_message="ok",
        )

        self.assertEqual(response["resultCode"], 0)
        self.assertEqual(response["errorMessage"], "ok")
        self.assertEqual(response["resultMap"]["request_id"], "req-1")

    def test_aistudio_model_paths_default_to_left_and_right_subdirectories(self):
        args = build_parser().parse_args(
            ["--role", "server", "--service-mode", "aistudio", "--model-path", "/tmp/models"]
        )

        resolved = resolve_aistudio_model_paths(args)

        self.assertEqual(
            resolved,
            AistudioModelPaths(
                left="/tmp/models/left_hand_v2_1223",
                right="/tmp/models/right_hand_v2_1223",
            ),
        )

    def test_build_aistudio_batch_preserves_prompt_and_state_shape(self):
        frame_rgb = np.zeros((640, 480, 3), dtype=np.uint8)
        batch, robot_state = build_aistudio_batch(
            frame_rgb=frame_rgb,
            joint_angles=[1, 2, 3, 4, 5],
        )

        self.assertEqual(batch["video.ego_view"].shape, (1, 640, 480, 3))
        self.assertEqual(batch["state.single_arm"].shape, (1, 6))
        self.assertEqual(batch["annotation.task_index"][0], "Move to center the book in view. Do nothing if no book is present.")
        self.assertEqual(robot_state.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 0.0])

    def test_postprocess_aistudio_actions_returns_json_sequence(self):
        previous_actions = np.array(
            [
                [1.11119, 2.0, 3.0, 4.0, 5.0, 6.0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        robot_state = np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)

        action_string, is_success = postprocess_aistudio_actions(previous_actions, robot_state)

        self.assertTrue(is_success)
        self.assertEqual(
            action_string,
            '[[11.1112, 22.0, 33.0, 44.0, 55.0, 66.0], [11.6112, 22.5, 33.5, 44.5, 55.5, 66.5]]',
        )


class TransportSurfaceTest(unittest.TestCase):
    def test_standard_http_registers_v1_and_compat_routes(self):
        adapter = SimpleNamespace(
            predict=lambda observation: {"action": observation},
            metadata=lambda: {"service_mode": "standard"},
            ping=lambda: {"status": "ok"},
            echo=lambda payload: payload,
        )

        app = create_http_app(service_mode="standard", adapter=adapter)
        route_paths = {route.path for route in app.routes}

        self.assertIn("/v1/act", route_paths)
        self.assertIn("/act", route_paths)
        self.assertIn("/health", route_paths)
        self.assertIn("/v1/latency/ping", route_paths)
        self.assertIn("/v1/latency/echo", route_paths)

    def test_aistudio_http_registers_predict_route_only(self):
        adapter = SimpleNamespace(
            predict=lambda payload: build_aistudio_response(result_map={"request_id": "req-1"})
        )

        app = create_http_app(service_mode="aistudio", adapter=adapter)
        route_paths = {route.path for route in app.routes}

        self.assertIn("/v1/aistudio/predict", route_paths)
        self.assertIn("/v1/aistudio/health", route_paths)
        self.assertIn("/v1/aistudio/latency/ping", route_paths)
        self.assertIn("/v1/aistudio/latency/echo", route_paths)
        self.assertNotIn("/v1/act", route_paths)

    def test_register_standard_zmq_endpoints_adds_required_names(self):
        adapter = SimpleNamespace(
            predict=lambda observation: {"action": observation},
            metadata=lambda: {"service_mode": "standard"},
            ping=lambda: {"status": "ok"},
            echo=lambda payload: payload,
            modality_config=lambda: {"action": "cfg"},
        )
        server = SimpleNamespace(_endpoints={}, register_endpoint=lambda name, handler, requires_input=True: server._endpoints.setdefault(name, (handler, requires_input)))

        register_standard_zmq_endpoints(server, adapter)

        self.assertIn("get_action", server._endpoints)
        self.assertIn("get_modality_config", server._endpoints)
        self.assertIn("get_server_info", server._endpoints)
        self.assertIn("echo", server._endpoints)


class ServiceAdapterTest(unittest.TestCase):
    def test_standard_service_adapter_calls_policy_get_action(self):
        policy = SimpleNamespace(
            get_action=mock.Mock(return_value={"action.single_arm": np.array([[1, 2, 3]])}),
            get_modality_config=mock.Mock(return_value={"action": "cfg"}),
        )
        adapter = StandardServiceAdapter(
            policy=policy,
            metadata={"service_mode": "standard", "backend": "pytorch"},
        )

        action = adapter.predict({"state.single_arm": np.array([[1, 2, 3]])})

        self.assertIn("action.single_arm", action)
        policy.get_action.assert_called_once()

    def test_aistudio_service_adapter_routes_and_formats_response(self):
        left_policy = SimpleNamespace(
            get_action=mock.Mock(
                return_value={
                    "action.single_arm": np.array(
                        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
                        dtype=np.float32,
                    )
                }
            )
        )
        right_policy = SimpleNamespace(get_action=mock.Mock())
        adapter = AistudioServiceAdapter(left_policy=left_policy, right_policy=right_policy)
        payload = {
            "query": (
                '{"joint_angles":"[1,2,3,-4,5]",'
                '"predicted_coords_2d":"[0.1,0.2]",'
                '"history_angles":"[[1,2,3,4,5]]",'
                '"framebuffer":[1,2,3],'
                '"framebuffer_size":3,'
                '"device_id":"dev-1",'
                '"request_id":"req-1"}'
            )
        }

        with mock.patch(
            "deployment_scripts.ant.online_inference_service.decode_aistudio_framebuffer",
            return_value=np.zeros((640, 480, 3), dtype=np.uint8),
        ):
            response = adapter.predict(payload)

        self.assertEqual(response["resultCode"], 0)
        self.assertEqual(response["errorMessage"], "ok")
        self.assertEqual(response["resultMap"]["request_id"], "req-1")
        self.assertEqual(
            response["resultMap"]["action_sequence"],
            '[[2.0, 3.0, 4.0, -3.0, 6.0, 1.0], [2.5, 3.5, 4.5, -2.5, 6.5, 1.5]]',
        )
        left_policy.get_action.assert_called_once()
        right_policy.get_action.assert_not_called()


class MainDispatchTest(unittest.TestCase):
    def test_main_rejects_aistudio_zmq_combination(self):
        with self.assertRaises(ValueError):
            main(["--role", "server", "--service-mode", "aistudio", "--transport", "zmq"])

    def test_main_standard_http_server_builds_http_app_and_runs_it(self):
        with (
            mock.patch(
                "deployment_scripts.ant.online_inference_service.build_service_adapter",
                return_value="adapter",
            ) as build_adapter,
            mock.patch(
                "deployment_scripts.ant.online_inference_service.create_http_app",
                return_value="app",
            ) as create_app,
            mock.patch("deployment_scripts.ant.online_inference_service.run_http_server") as run_http,
        ):
            exit_code = main(["--role", "server"])

        self.assertEqual(exit_code, 0)
        build_adapter.assert_called_once()
        create_app.assert_called_once_with(service_mode="standard", adapter="adapter", api_token=None)
        run_http.assert_called_once_with("app", host="0.0.0.0", port=8000)

    def test_main_standard_zmq_server_builds_zmq_server_and_runs_it(self):
        with (
            mock.patch(
                "deployment_scripts.ant.online_inference_service.build_service_adapter",
                return_value="adapter",
            ) as build_adapter,
            mock.patch(
                "deployment_scripts.ant.online_inference_service.create_standard_zmq_server",
                return_value="server",
            ) as create_zmq,
            mock.patch("deployment_scripts.ant.online_inference_service.run_zmq_server") as run_zmq,
        ):
            exit_code = main(["--role", "server", "--transport", "zmq"])

        self.assertEqual(exit_code, 0)
        build_adapter.assert_called_once()
        create_zmq.assert_called_once_with("adapter", host="0.0.0.0", port=8000, api_token=None)
        run_zmq.assert_called_once_with("server")

    def test_main_aistudio_http_server_builds_http_app_and_runs_it(self):
        with (
            mock.patch(
                "deployment_scripts.ant.online_inference_service.build_service_adapter",
                return_value="adapter",
            ) as build_adapter,
            mock.patch(
                "deployment_scripts.ant.online_inference_service.create_http_app",
                return_value="app",
            ) as create_app,
            mock.patch("deployment_scripts.ant.online_inference_service.run_http_server") as run_http,
        ):
            exit_code = main(["--role", "server", "--service-mode", "aistudio"])

        self.assertEqual(exit_code, 0)
        build_adapter.assert_called_once()
        create_app.assert_called_once_with(service_mode="aistudio", adapter="adapter", api_token=None)
        run_http.assert_called_once_with("app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    unittest.main()
