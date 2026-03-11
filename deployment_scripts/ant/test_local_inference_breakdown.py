import unittest
from collections import defaultdict

import numpy as np
import torch

from deployment_scripts.ant.local_inference_breakdown import (
    _apply_gr00t_transform_with_breakdown,
)


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


if __name__ == "__main__":
    unittest.main()
