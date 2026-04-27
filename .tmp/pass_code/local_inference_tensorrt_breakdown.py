from __future__ import annotations

import argparse
import copy
import socket
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _ensure_repo_root_on_path()

from deployment_scripts.ant.local_inference_breakdown import _run_transform_breakdown  # noqa: E402
from deployment_scripts.ant.local_inference_tensorrt_e2e import _load_runtime, _resolve_repo_path  # noqa: E402
from deployment_scripts.ant.quality_metrics import (  # noqa: E402
    add_optional_eval_args,
    collect_optional_metrics,
    merge_optional_metrics as _merge_optional_metrics,
)
from deployment_scripts.ant.profile_utils import (  # noqa: E402
    LatencyRecord,
    format_summary,
    measure_cuda_time_ms,
    measure_wall_time_ms,
    summarize_latency_records,
    write_jsonl,
)
from deployment_scripts.ant.system_metrics import (  # noqa: E402
    TegrastatsSampler,
    collect_torch_peak_memory_metrics,
    reset_torch_peak_memory_stats,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 GR00T TensorRT 细粒度阶段时延测试脚本")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B",
    )
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--data-config", type=str, default="fourier_gr1_arms_only")
    parser.add_argument("--embodiment-tag", type=str, default="gr1")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--video-backend", type=str, choices=["decord", "torchcodec"], default="decord")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measure-runs", type=int, default=30)
    parser.add_argument("--trt-engine-path", type=str, default=str(_REPO_ROOT / "gr00t_engine"))
    parser.add_argument("--vit-dtype", type=str, choices=["fp16", "fp8", "int8"], default="fp16")
    parser.add_argument(
        "--llm-dtype", type=str, choices=["fp16", "nvfp4", "fp8", "int8"], default="fp16"
    )
    parser.add_argument("--dit-dtype", type=str, choices=["fp16", "fp8", "int8"], default="fp16")
    parser.add_argument("--output-jsonl", type=str, default="")
    return add_optional_eval_args(parser)


def _sum_metrics(metrics: dict[str, float], metric_names: tuple[str, ...]) -> float:
    return round(sum(metrics.get(name, 0.0) for name in metric_names), 4)


def _run_backbone_breakdown(backbone, backbone_inputs) -> tuple[dict[str, Any], dict[str, float]]:
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    metrics: dict[str, float] = defaultdict(float)
    eagle_input = {
        key.removeprefix("eagle_"): value
        for key, value in backbone_inputs.items()
        if key.startswith("eagle_")
    }
    eagle_input.pop("image_sizes", None)

    backbone.set_frozen_modules_to_eval_mode()
    batch_size = eagle_input["pixel_values"].shape[0]
    position_ids = torch.arange(backbone.num_patches, device="cuda").expand((batch_size, -1))
    pixel_values = eagle_input["pixel_values"]
    if pixel_values.dtype != torch.float16:
        pixel_values = pixel_values.to(torch.float16)

    if pixel_values.shape[0] > 8:
        raise ValueError(
            "TensorRT backbone 要求 pixel_values batch_size <= 8；如需更大 batch，请重建 engine"
        )

    def _run_vit_engine():
        backbone.vit_engine.set_runtime_tensor_shape("pixel_values", pixel_values.shape)
        backbone.vit_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)
        return backbone.vit_engine(pixel_values, position_ids)["vit_embeds"]

    vit_embeds, vit_engine_ms = measure_cuda_time_ms(_run_vit_engine)
    metrics["backbone_vit_engine_ms"] += vit_engine_ms

    def _postprocess_vit():
        current = vit_embeds.view(1, -1, vit_embeds.shape[-1])
        if backbone.eagle_model.use_pixel_shuffle:
            height = width = int(current.shape[1] ** 0.5)
            current = current.reshape(current.shape[0], height, width, -1)
            current = backbone.pixel_shuffle(current, scale_factor=backbone.downsample_ratio)
            current = current.reshape(current.shape[0], -1, current.shape[-1])
        return backbone.eagle_model.mlp1(current)

    vit_embeds, vit_postprocess_ms = measure_cuda_time_ms(_postprocess_vit)
    metrics["backbone_vit_postprocess_ms"] += vit_postprocess_ms

    def _build_input_embeddings():
        input_embeds = backbone.embedding_layer(eagle_input["input_ids"])
        if input_embeds.dtype != torch.float16:
            input_embeds = input_embeds.to(torch.float16)
        return input_embeds

    input_embeds, input_embedding_ms = measure_cuda_time_ms(_build_input_embeddings)
    metrics["backbone_input_embedding_ms"] += input_embedding_ms

    if vit_embeds.dtype != torch.float16:
        vit_embeds = vit_embeds.to(torch.float16)

    def _fuse_image_tokens():
        batch, token_count, hidden = input_embeds.shape
        flat_embeds = input_embeds.reshape(batch * token_count, hidden)
        input_ids_flat = eagle_input["input_ids"].reshape(batch * token_count)
        selected = input_ids_flat == backbone.image_token_index
        fused_vit = vit_embeds.reshape(-1, hidden)
        if selected.sum() > fused_vit.shape[0]:
            fused_vit = fused_vit[: selected.sum()]
        flat_embeds[selected] = flat_embeds[selected] * 0.0 + fused_vit
        return flat_embeds.reshape(batch, token_count, hidden)

    input_embeds, token_fusion_ms = measure_cuda_time_ms(_fuse_image_tokens)
    metrics["backbone_token_fusion_ms"] += token_fusion_ms

    attention_mask = eagle_input["attention_mask"]

    def _run_llm_engine():
        backbone.llm_engine.set_runtime_tensor_shape("inputs_embeds", input_embeds.shape)
        backbone.llm_engine.set_runtime_tensor_shape("attention_mask", attention_mask.shape)
        return backbone.llm_engine(input_embeds, attention_mask)["embeddings"]

    embeddings, llm_engine_ms = measure_cuda_time_ms(_run_llm_engine)
    metrics["backbone_llm_engine_ms"] += llm_engine_ms
    metrics["backbone_total_ms"] = _sum_metrics(
        metrics,
        (
            "backbone_vit_engine_ms",
            "backbone_vit_postprocess_ms",
            "backbone_input_embedding_ms",
            "backbone_token_fusion_ms",
            "backbone_llm_engine_ms",
        ),
    )
    return (
        BatchFeature(
            data={
                "backbone_features": embeddings,
                "backbone_attention_mask": attention_mask,
            }
        ),
        metrics,
    )


def _run_action_head_breakdown(action_head, backbone_output, action_input) -> tuple[dict[str, Any], dict[str, float]]:
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    metrics: dict[str, float] = defaultdict(float)

    def _run_features_process():
        features = backbone_output.backbone_features
        if features.dtype != torch.float16:
            features = features.to(torch.float16)
        action_head.vlln_vl_self_attention_engine.set_runtime_tensor_shape(
            "backbone_features", features.shape
        )
        return action_head.vlln_vl_self_attention_engine(features)["output"]

    vl_embs, features_process_ms = measure_cuda_time_ms(_run_features_process)
    metrics["action_head_features_process_ms"] += features_process_ms

    embodiment_id = action_input.embodiment_id
    if embodiment_id.dtype != torch.int64:
        embodiment_id = embodiment_id.to(torch.int64)
    state = action_input.state
    if state.dtype != torch.float16:
        state = state.to(torch.float16)

    def _run_state_encoder():
        action_head.state_encoder_engine.set_runtime_tensor_shape("state", state.shape)
        action_head.state_encoder_engine.set_runtime_tensor_shape(
            "embodiment_id", embodiment_id.shape
        )
        return action_head.state_encoder_engine(state, embodiment_id)["output"]

    state_features, state_encoder_ms = measure_cuda_time_ms(_run_state_encoder)
    metrics["action_head_state_encoder_ms"] += state_encoder_ms

    batch_size = vl_embs.shape[0]
    device = vl_embs.device
    if hasattr(action_head, "init_actions"):
        actions = action_head.init_actions.expand((batch_size, -1, -1))
    else:
        actions = torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

    num_steps = action_head.num_inference_timesteps
    dt = 1.0 / num_steps
    for step_index in range(num_steps):
        t_cont = step_index / float(num_steps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)

        def _run_action_encoder():
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_head.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
            action_head.action_encoder_engine.set_runtime_tensor_shape(
                "timesteps_tensor", timesteps_tensor.shape
            )
            action_head.action_encoder_engine.set_runtime_tensor_shape(
                "embodiment_id", embodiment_id.shape
            )
            encoded = action_head.action_encoder_engine(actions, timesteps_tensor, embodiment_id)[
                "output"
            ]
            return encoded, timesteps_tensor

        (action_features, timesteps_tensor), action_encoder_ms = measure_cuda_time_ms(
            _run_action_encoder
        )
        metrics["action_head_action_encoder_ms"] += action_encoder_ms

        if action_head.config.add_pos_embed:
            def _add_pos_embedding():
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
                return action_features + pos_embs

            action_features, pos_embed_ms = measure_cuda_time_ms(_add_pos_embedding)
            metrics["action_head_pos_embed_ms"] += pos_embed_ms

        def _concat_action_inputs():
            future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.cat((state_features, future_tokens, action_features), dim=1).to(torch.float16)

        sa_embs, concat_ms = measure_cuda_time_ms(_concat_action_inputs)
        metrics["action_head_concat_ms"] += concat_ms

        def _run_dit():
            action_head.DiT_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
            action_head.DiT_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
            action_head.DiT_engine.set_runtime_tensor_shape("timesteps_tensor", timesteps_tensor.shape)
            return action_head.DiT_engine(sa_embs, vl_embs, timesteps_tensor)["output"]

        model_output, dit_ms = measure_cuda_time_ms(_run_dit)
        metrics["action_head_dit_block_ms"] += dit_ms

        def _run_action_decoder():
            action_head.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
            action_head.action_decoder_engine.set_runtime_tensor_shape(
                "embodiment_id", embodiment_id.shape
            )
            return action_head.action_decoder_engine(model_output, embodiment_id)["output"]

        pred, action_decoder_ms = measure_cuda_time_ms(_run_action_decoder)
        metrics["action_head_action_decoder_ms"] += action_decoder_ms

        def _update_actions():
            pred_velocity = pred[:, -action_head.action_horizon :]
            return actions + dt * pred_velocity

        actions, euler_update_ms = measure_cuda_time_ms(_update_actions)
        metrics["action_head_euler_update_ms"] += euler_update_ms

    metrics["action_head_total_ms"] = _sum_metrics(
        metrics,
        (
            "action_head_features_process_ms",
            "action_head_state_encoder_ms",
            "action_head_action_encoder_ms",
            "action_head_pos_embed_ms",
            "action_head_concat_ms",
            "action_head_dit_block_ms",
            "action_head_action_decoder_ms",
            "action_head_euler_update_ms",
        ),
    )
    return BatchFeature(data={"action_pred": actions}), metrics


def run_breakdown_inference(
    policy,
    raw_obs: dict[str, Any],
    unsqueeze_dict_values,
    *,
    compute_dtype=None,
) -> tuple[dict[str, Any], dict[str, float]]:
    import contextlib
    import torch

    if compute_dtype is None:
        from gr00t.model.policy import COMPUTE_DTYPE as compute_dtype

    metrics: dict[str, float] = defaultdict(float)
    normalized_input, transform_metrics = _run_transform_breakdown(
        policy,
        raw_obs,
        unsqueeze_dict_values,
    )
    metrics.update(transform_metrics)

    def _prepare_model_input():
        return policy.model.prepare_input(normalized_input)

    (backbone_inputs, action_inputs), prepare_input_ms = measure_cuda_time_ms(_prepare_model_input)
    metrics["prepare_input_ms"] = prepare_input_ms

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=compute_dtype)
        if torch.cuda.is_available() and compute_dtype is not None
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        backbone_output, backbone_metrics = _run_backbone_breakdown(
            policy.model.backbone, backbone_inputs
        )
        metrics.update(backbone_metrics)
        normalized_action, action_head_metrics = _run_action_head_breakdown(
            policy.model.action_head,
            backbone_output,
            action_inputs,
        )
        metrics.update(action_head_metrics)

    def _postprocess():
        return policy.unapply_transforms({"action": normalized_action["action_pred"].float().cpu()})

    postprocessed_action, postprocess_ms = measure_wall_time_ms(_postprocess)
    metrics["postprocess_ms"] = postprocess_ms
    metrics["e2e_total_ms"] = _sum_metrics(
        metrics,
        (
            "transform_total_ms",
            "prepare_input_ms",
            "backbone_total_ms",
            "action_head_total_ms",
            "postprocess_ms",
        ),
    )
    return postprocessed_action, metrics


def _run_single_breakdown(
    policy,
    raw_obs: dict[str, Any],
    unsqueeze_dict_values,
    *,
    compute_dtype=None,
) -> dict[str, float]:
    _, metrics = run_breakdown_inference(
        policy,
        raw_obs,
        unsqueeze_dict_values,
        compute_dtype=compute_dtype,
    )
    return metrics


def main() -> int:
    from deployment_scripts.ant.local_inference import run_legacy_main

    return run_legacy_main(
        build_parser_fn=build_parser,
        backend="tensorrt",
        mode="breakdown",
        script_name="local_inference_tensorrt_breakdown.py",
    )


if __name__ == "__main__":
    raise SystemExit(main())
