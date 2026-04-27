import argparse
import unittest
from types import SimpleNamespace
from unittest import mock

from deployment_scripts.ant.torch_compile_utils import (
    add_torch_compile_arg,
    enable_torch_compile_for_breakdown,
    enable_torch_compile_for_e2e,
)


class TorchCompileUtilsTest(unittest.TestCase):
    def test_add_torch_compile_arg_defaults_to_false(self):
        parser = argparse.ArgumentParser()
        add_torch_compile_arg(parser)

        args = parser.parse_args([])

        self.assertFalse(args.use_torch_compile)

    def test_enable_torch_compile_for_e2e_compiles_model_get_action(self):
        policy = SimpleNamespace(model=SimpleNamespace(get_action=lambda payload: payload))

        with mock.patch("torch.compile", side_effect=lambda fn, **kwargs: ("compiled", fn, kwargs)):
            enable_torch_compile_for_e2e(policy, enabled=True)

        self.assertEqual(policy.model.get_action[0], "compiled")
        self.assertEqual(policy.model.get_action[2]["mode"], "reduce-overhead")

    def test_enable_torch_compile_for_breakdown_stores_compiled_callables(self):
        language_model = SimpleNamespace(
            get_input_embeddings=lambda: None,
            config=SimpleNamespace(vocab_size=8),
        )
        policy = SimpleNamespace(
            model=SimpleNamespace(
                backbone=SimpleNamespace(
                    eagle_model=SimpleNamespace(
                        extract_feature=lambda payload: payload,
                        language_model=language_model,
                    )
                ),
                action_head=SimpleNamespace(
                    process_backbone_output=lambda payload: payload,
                    state_encoder=lambda state, embodiment: (state, embodiment),
                    action_encoder=lambda actions, timesteps, embodiment: (actions, timesteps, embodiment),
                    model=lambda **kwargs: kwargs,
                    action_decoder=lambda payload, embodiment: (payload, embodiment),
                ),
            )
        )

        with mock.patch("torch.compile", side_effect=lambda fn, **kwargs: ("compiled", fn, kwargs)) as compile_mock:
            enable_torch_compile_for_breakdown(policy, enabled=True)

        self.assertEqual(policy.model.backbone.eagle_model._compiled_extract_feature[0], "compiled")
        self.assertEqual(policy.model.backbone.eagle_model._compiled_language_model[0], "compiled")
        self.assertEqual(policy.model.action_head._compiled_process_backbone_output[0], "compiled")
        self.assertEqual(policy.model.action_head._compiled_state_encoder[0], "compiled")
        self.assertEqual(policy.model.action_head._compiled_action_encoder[0], "compiled")
        self.assertEqual(policy.model.action_head._compiled_model[0], "compiled")
        self.assertEqual(policy.model.action_head._compiled_action_decoder[0], "compiled")
        for call in compile_mock.call_args_list:
            self.assertEqual(call.kwargs["mode"], "reduce-overhead")
            self.assertEqual(call.kwargs["options"], {"triton.cudagraphs": False})


if __name__ == "__main__":
    unittest.main()
