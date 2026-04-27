import unittest
from collections import defaultdict

import numpy as np
import torch

from deployment_scripts.ant.local_inference_breakdown import (
    _merge_optional_metrics,
    _apply_gr00t_transform_with_breakdown,
    build_parser,
)
from deployment_scripts.ant.local_inference import build_parser as build_shared_parser


class _FakeTokenizer:
    init_kwargs = {}

    def __call__(self, text_list, **kwargs):
        return {
            "input_ids": torch.tensor([[11, 12]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True):
        return "prompt"

    def process_vision_info(self, conversation):
        return [object()], []

    def _merge_kwargs(self, *args, **kwargs):
        return {"videos_kwargs": {}, "text_kwargs": {}}

    def replace_media_placeholder(
        self,
        sample,
        image_inputs,
        video_inputs,
        timestamps_list,
        fps_list,
        **kwargs,
    ):
        del sample, video_inputs, timestamps_list, fps_list, kwargs
        return "prompt", torch.ones((1, 3, 4, 4)), torch.tensor([[4, 4]]), len(image_inputs), 0


class _FakeTransform:
    def __init__(self):
        self.eagle_processor = _FakeProcessor()

    def check_keys_and_batch_size(self, data):
        return True, data["video"].shape[0]

    def _prepare_video(self, elem):
        del elem
        return np.zeros((1, 1, 3, 2, 2), dtype=np.uint8)

    def _prepare_language(self, elem):
        del elem
        return "pick up cube"

    def _prepare_state(self, elem):
        del elem
        state = np.zeros((1, 2), dtype=np.float32)
        state_mask = np.ones((1, 2), dtype=bool)
        return state, state_mask, 1

    def get_embodiment_tag(self):
        return 7


class LocalInferenceBreakdownTransformTest(unittest.TestCase):
    def test_shared_parser_defaults_to_pytorch_e2e(self):
        args = build_shared_parser().parse_args([])
        self.assertEqual(args.backend, "pytorch")
        self.assertEqual(args.mode, "e2e")

    def test_parser_exposes_offline_eval_flags(self):
        args = build_parser().parse_args([])
        self.assertFalse(args.measure_system)
        self.assertFalse(args.measure_open_loop)
        self.assertFalse(args.measure_smoothness)
        self.assertFalse(args.measure_proxy_success)
        self.assertFalse(args.use_torch_compile)
        self.assertEqual(args.open_loop_trajs, 1)
        self.assertEqual(args.open_loop_steps, 150)

    def test_legacy_breakdown_parser_does_not_expose_selector_flags(self):
        args = build_parser().parse_args([])
        self.assertFalse(hasattr(args, "backend"))
        self.assertFalse(hasattr(args, "mode"))

    def test_apply_gr00t_transform_preserves_eagle_prefix(self):
        transform = _FakeTransform()
        metrics = defaultdict(float)
        batched_data = {
            "video": np.zeros((1, 1, 1, 2, 2, 3), dtype=np.uint8),
        }

        result = _apply_gr00t_transform_with_breakdown(transform, batched_data, metrics)

        self.assertIn("eagle_pixel_values", result)
        self.assertIn("eagle_image_sizes", result)
        self.assertIn("eagle_input_ids", result)
        self.assertIn("eagle_attention_mask", result)
        self.assertNotIn("pixel_values", result)
        self.assertEqual(result["embodiment_id"].tolist(), [7])

    def test_merge_optional_metrics_preserves_existing_latency_fields(self):
        merged = _merge_optional_metrics(
            {"e2e_total_ms": 11.0},
            {"open_loop_rmse": 0.7},
            {"smoothness_pullback_ratio": 0.1},
            {"tegrastats_gr3d_freq_pct_max": 92.0},
        )

        self.assertEqual(merged["e2e_total_ms"], 11.0)
        self.assertEqual(merged["open_loop_rmse"], 0.7)
        self.assertEqual(merged["smoothness_pullback_ratio"], 0.1)
        self.assertEqual(merged["tegrastats_gr3d_freq_pct_max"], 92.0)


if __name__ == "__main__":
    unittest.main()
