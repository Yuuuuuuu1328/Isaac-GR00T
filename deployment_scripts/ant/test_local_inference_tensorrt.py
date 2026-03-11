import unittest
from types import SimpleNamespace
from unittest import mock

import deployment_scripts.ant.local_inference_tensorrt_breakdown as tensorrt_breakdown
from deployment_scripts.ant.local_inference_tensorrt_breakdown import (
    _run_single_breakdown,
    _sum_metrics,
    build_parser as build_breakdown_parser,
)
from deployment_scripts.ant.local_inference_tensorrt_e2e import build_parser as build_e2e_parser


class TensorRTCliParserTest(unittest.TestCase):
    def test_e2e_parser_exposes_engine_dtype_flags(self):
        args = build_e2e_parser().parse_args([])
        self.assertEqual(args.vit_dtype, "fp8")
        self.assertEqual(args.llm_dtype, "nvfp4")
        self.assertEqual(args.dit_dtype, "fp8")

    def test_breakdown_parser_defaults_engine_path_to_repo_engine_dir(self):
        args = build_breakdown_parser().parse_args([])
        self.assertTrue(args.trt_engine_path.endswith("gr00t_engine"))


class TensorRTBreakdownHelperTest(unittest.TestCase):
    def test_sum_metrics_only_counts_requested_names(self):
        total = _sum_metrics({"a": 1.0, "b": 2.0}, ("a",))
        self.assertEqual(total, 1.0)

    def test_run_single_breakdown_uses_policy_inference_context(self):
        state = {"inference_mode": False, "autocast": False}

        class _TrackingContext:
            def __init__(self, flag_name):
                self.flag_name = flag_name

            def __enter__(self):
                state[self.flag_name] = True
                return self

            def __exit__(self, exc_type, exc, tb):
                state[self.flag_name] = False
                return False

        def _assert_model_context_active(*args, **kwargs):
            del args, kwargs
            assert state["inference_mode"]
            assert state["autocast"]

        policy = SimpleNamespace(
            model=SimpleNamespace(
                prepare_input=lambda normalized_input: ("backbone_inputs", "action_inputs"),
                backbone=object(),
                action_head=object(),
            ),
            unapply_transforms=lambda action: action,
        )

        with (
            mock.patch.object(
                tensorrt_breakdown, "_run_transform_breakdown", return_value=({"obs": 1}, {})
            ),
            mock.patch.object(
                tensorrt_breakdown,
                "_run_backbone_breakdown",
                side_effect=lambda backbone, inputs: (
                    _assert_model_context_active(backbone, inputs),
                    ("backbone_output", {"backbone_total_ms": 2.0}),
                )[1],
            ),
            mock.patch.object(
                tensorrt_breakdown,
                "_run_action_head_breakdown",
                side_effect=lambda action_head, backbone_output, action_inputs: (
                    _assert_model_context_active(action_head, backbone_output, action_inputs),
                    ({"action_pred": mock.Mock(float=lambda: mock.Mock(cpu=lambda: "cpu_action"))}, {"action_head_total_ms": 3.0}),
                )[1],
            ),
            mock.patch.object(
                tensorrt_breakdown, "measure_cuda_time_ms", side_effect=lambda fn: (fn(), 0.1)
            ),
            mock.patch.object(
                tensorrt_breakdown, "measure_wall_time_ms", side_effect=lambda fn: (fn(), 0.2)
            ),
            mock.patch(
                "torch.inference_mode", side_effect=lambda: _TrackingContext("inference_mode")
            ),
            mock.patch(
                "torch.autocast", side_effect=lambda *args, **kwargs: _TrackingContext("autocast")
            ),
        ):
            metrics = _run_single_breakdown(policy, {"raw": 1}, lambda value: value)

        self.assertEqual(metrics["prepare_input_ms"], 0.1)
        self.assertEqual(metrics["backbone_total_ms"], 2.0)
        self.assertEqual(metrics["action_head_total_ms"], 3.0)


if __name__ == "__main__":
    unittest.main()
