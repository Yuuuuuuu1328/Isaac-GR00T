from __future__ import annotations

import argparse


def add_torch_compile_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--use-torch-compile", action="store_true")
    return parser


def _compile_callable(fn, *, mode: str = "reduce-overhead", options: dict[str, object] | None = None):
    import torch

    if not hasattr(torch, "compile"):
        raise RuntimeError("当前 PyTorch 版本不支持 torch.compile")
    return torch.compile(fn, mode=mode, options=options)


def enable_torch_compile_for_e2e(policy, enabled: bool, *, mode: str = "reduce-overhead") -> None:
    if not enabled:
        return
    policy.model.get_action = _compile_callable(policy.model.get_action, mode=mode)


def enable_torch_compile_for_breakdown(
    policy,
    enabled: bool,
    *,
    mode: str = "reduce-overhead",
) -> None:
    if not enabled:
        return

    backbone = policy.model.backbone
    eagle_model = backbone.eagle_model
    action_head = policy.model.action_head
    breakdown_compile_options = {"triton.cudagraphs": False}

    eagle_model._compiled_extract_feature = _compile_callable(
        eagle_model.extract_feature,
        mode=mode,
        options=breakdown_compile_options,
    )
    eagle_model._compiled_language_model = _compile_callable(
        eagle_model.language_model,
        mode=mode,
        options=breakdown_compile_options,
    )
    action_head._compiled_process_backbone_output = _compile_callable(
        action_head.process_backbone_output,
        mode=mode,
        options=breakdown_compile_options,
    )
    action_head._compiled_state_encoder = _compile_callable(
        action_head.state_encoder,
        mode=mode,
        options=breakdown_compile_options,
    )
    action_head._compiled_action_encoder = _compile_callable(
        action_head.action_encoder,
        mode=mode,
        options=breakdown_compile_options,
    )
    action_head._compiled_model = _compile_callable(
        action_head.model,
        mode=mode,
        options=breakdown_compile_options,
    )
    action_head._compiled_action_decoder = _compile_callable(
        action_head.action_decoder,
        mode=mode,
        options=breakdown_compile_options,
    )
