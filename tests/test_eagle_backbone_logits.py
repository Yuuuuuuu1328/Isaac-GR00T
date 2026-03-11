from types import SimpleNamespace

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.backbone.eagle_backbone import EagleBackbone
from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import (
    Eagle2_5_VLForConditionalGeneration,
)


class _FakeLanguageModel:
    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(vocab_size=8)

    def get_input_embeddings(self):
        return lambda input_ids: torch.zeros(*input_ids.shape, 4)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch, _, _ = kwargs["inputs_embeds"].shape
        kept = kwargs["logits_to_keep"]
        seq = kept if isinstance(kept, int) else 1
        return SimpleNamespace(
            logits=torch.zeros(batch, seq, self.config.vocab_size),
            past_key_values=None,
            hidden_states=("hidden",),
            attentions=None,
        )


def test_eagle_wrapper_forwards_logits_to_keep_to_language_model():
    model = object.__new__(Eagle2_5_VLForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(use_return_dict=True)
    model.language_model = _FakeLanguageModel()
    model.extract_feature = lambda pixel_values: torch.ones(1, 4)
    model.image_token_index = 99

    Eagle2_5_VLForConditionalGeneration.forward(
        model,
        pixel_values=torch.zeros(1, 1),
        input_ids=torch.tensor([[99, 1]]),
        attention_mask=torch.ones(1, 2),
        output_hidden_states=True,
        logits_to_keep=1,
        return_dict=True,
    )

    assert model.language_model.calls[0]["logits_to_keep"] == 1


def test_eagle_backbone_limits_logits_by_default():
    backbone = object.__new__(EagleBackbone)
    nn.Module.__init__(backbone)
    backbone.select_layer = -1
    backbone.eagle_linear = nn.Identity()

    calls = []

    class _FakeEagleModel:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(hidden_states=[torch.ones(1, 2, 4)])

    backbone.eagle_model = _FakeEagleModel()

    features, mask = EagleBackbone.forward_eagle(
        backbone,
        BatchFeature(
            data={
                "eagle_pixel_values": torch.zeros(1, 1),
                "eagle_attention_mask": torch.ones(1, 2),
                "eagle_image_sizes": torch.tensor([[1, 1]]),
            }
        ),
    )

    assert calls[0]["logits_to_keep"] == 1
    assert "image_sizes" not in calls[0]
    assert features.shape == (1, 2, 4)
    assert torch.equal(mask, torch.ones(1, 2))
