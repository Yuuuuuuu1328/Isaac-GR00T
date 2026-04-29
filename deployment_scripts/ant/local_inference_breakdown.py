from __future__ import annotations

import argparse
import copy
import os
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
from deployment_scripts.ant.torch_compile_utils import (  # noqa: E402
    add_torch_compile_arg,
    enable_torch_compile_for_breakdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 GR00T 细粒度阶段时延测试脚本")
    parser.add_argument("--model-path", type=str, default="/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--data-config", type=str, default="fourier_gr1_arms_only")
    parser.add_argument("--embodiment-tag", type=str, default="gr1")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--video-backend", type=str, choices=["decord", "torchcodec"], default="decord")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measure-runs", type=int, default=30)
    parser.add_argument("--output-jsonl", type=str, default="")
    return add_torch_compile_arg(add_optional_eval_args(parser))


def _load_runtime(args: argparse.Namespace):
    import torch
    import gr00t
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import COMPUTE_DTYPE, Gr00tPolicy, unsqueeze_dict_values

    repo_path = os.path.dirname(os.path.dirname(gr00t.__file__))
    dataset_path = args.dataset_path or os.path.join(repo_path, "demo_data/robot_sim.PickNPlace")

    data_config = load_data_config(args.data_config)
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=policy.modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )
    return policy, dataset, COMPUTE_DTYPE, unsqueeze_dict_values


def _prepare_batched_observation(raw_obs: dict[str, Any], unsqueeze_dict_values) -> dict[str, Any]:
    import numpy as np

    obs_copy = copy.deepcopy(raw_obs)
    obs_copy = unsqueeze_dict_values(obs_copy)
    for key, value in obs_copy.items():
        if not isinstance(value, np.ndarray):
            obs_copy[key] = np.array(value)
    return obs_copy


def _categorize_transform(transform) -> str:
    apply_to = getattr(transform, "apply_to", [])
    if all(key.startswith("video.") for key in apply_to):
        return "transform_image_preprocess_ms"
    if all(key.startswith("state.") for key in apply_to):
        return "transform_state_ms"
    if all(key.startswith("action.") for key in apply_to):
        return "transform_action_ms"
    return "transform_misc_ms"


def _apply_concat_transform_with_breakdown(transform, data: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    import numpy as np
    import torch

    grouped_keys: dict[str, list[str]] = {}
    for key in list(data.keys()):
        try:
            modality, _ = key.split(".")
        except ValueError:
            modality = "language" if "annotation" in key else "others"
        grouped_keys.setdefault(modality, []).append(key)

    if "video" in grouped_keys:
        def _concat_video():
            unsqueezed_videos = []
            for video_key in transform.video_concat_order:
                video_data = data.pop(video_key)
                unsqueezed_videos.append(np.expand_dims(video_data, axis=-4))
            data["video"] = np.concatenate(unsqueezed_videos, axis=-4)
            return data

        _, elapsed_ms = measure_wall_time_ms(_concat_video)
        metrics["transform_image_preprocess_ms"] += elapsed_ms

    if "state" in grouped_keys and transform.state_concat_order is not None:
        def _concat_state():
            data["state"] = torch.cat([data.pop(key) for key in transform.state_concat_order], dim=-1)
            return data

        _, elapsed_ms = measure_wall_time_ms(_concat_state)
        metrics["transform_state_ms"] += elapsed_ms

    if "action" in grouped_keys and transform.action_concat_order is not None:
        def _concat_action():
            data["action"] = torch.cat([data.pop(key) for key in transform.action_concat_order], dim=-1)
            return data

        _, elapsed_ms = measure_wall_time_ms(_concat_action)
        metrics["transform_action_ms"] += elapsed_ms

    return data


def _build_processor_output(processor, text_list: list[str], image_inputs: list[Any]) -> tuple[BatchFeature, float, float]:
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    from gr00t.model.backbone.eagle2_hg_model.processing_eagle2_5_vl import Eagle2_5_VLProcessorKwargs

    output_kwargs = processor._merge_kwargs(  # noqa: SLF001
        Eagle2_5_VLProcessorKwargs,
        tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
        return_tensors="pt",
        padding=True,
    )
    timestamps_batch = output_kwargs["videos_kwargs"].pop("timestamps", None)
    fps_batch = output_kwargs["videos_kwargs"].pop("fps", None)

    def _replace_media():
        pixel_values_list = []
        image_sizes_list = []
        new_sample_list = []
        image_start_idx = 0
        video_start_idx = 0

        for sample in text_list:
            timestamps_list = timestamps_batch[video_start_idx:] if timestamps_batch is not None else None
            fps_list = fps_batch[video_start_idx:] if fps_batch is not None else None
            (
                new_sample,
                pixel_values,
                image_sizes,
                num_images,
                num_videos,
            ) = processor.replace_media_placeholder(
                sample,
                image_inputs[image_start_idx:],
                [],
                timestamps_list,
                fps_list,
                **output_kwargs,
            )
            new_sample_list.append(new_sample)
            if pixel_values is not None:
                pixel_values_list.append(pixel_values)
                image_sizes_list.append(image_sizes)
            image_start_idx += num_images
            video_start_idx += num_videos
        return new_sample_list, pixel_values_list, image_sizes_list, output_kwargs

    (new_sample_list, pixel_values_list, image_sizes_list, output_kwargs), image_preprocess_ms = measure_wall_time_ms(_replace_media)

    def _tokenize_text():
        return processor.tokenizer(new_sample_list, **output_kwargs["text_kwargs"])

    text_inputs, text_tokenization_ms = measure_wall_time_ms(_tokenize_text)
    image_batch = {}
    if pixel_values_list:
        image_batch = {
            "pixel_values": torch.cat(pixel_values_list),
            "image_sizes": torch.cat(image_sizes_list),
        }
    return BatchFeature(data={**text_inputs, **image_batch}), image_preprocess_ms, text_tokenization_ms


def _prefix_eagle_batch_keys(eagle_batch: BatchFeature) -> dict[str, Any]:
    return {f"eagle_{key}": value for key, value in dict(eagle_batch).items()}


def _apply_gr00t_transform_with_breakdown(transform, data: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    import numpy as np
    import tree
    import torch
    from PIL import Image
    from einops import rearrange

    is_batched, batch_size = transform.check_keys_and_batch_size(data)
    if not is_batched:
        raise ValueError("细分脚本要求 batched 输入")

    processed_samples = []
    text_lists: list[str] = []
    image_inputs_all: list[Any] = []
    for sample_index in range(batch_size):
        elem = tree.map_structure(lambda x: x[sample_index], data)

        images, image_prepare_ms = measure_wall_time_ms(lambda: transform._prepare_video(elem).astype(np.uint8))
        metrics["transform_image_preprocess_ms"] += image_prepare_ms

        language, language_ms = measure_wall_time_ms(lambda: transform._prepare_language(elem))
        metrics["transform_text_tokenization_ms"] += language_ms

        (state, state_mask, _), state_ms = measure_wall_time_ms(lambda: transform._prepare_state(elem))
        metrics["transform_state_ms"] += state_ms

        def _build_conversation():
            np_images = rearrange(images, "v t c h w -> (t v) c h w")
            eagle_images = [Image.fromarray(np.transpose(frame, (1, 2, 0))) for frame in np_images]
            eagle_conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image} for image in eagle_images]
                    + [{"type": "text", "text": language}],
                }
            ]
            return eagle_conversation

        eagle_conversation, conversation_ms = measure_wall_time_ms(_build_conversation)
        metrics["transform_image_preprocess_ms"] += conversation_ms

        text_list, chat_template_ms = measure_wall_time_ms(
            lambda: [
                transform.eagle_processor.apply_chat_template(
                    eagle_conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ]
        )
        metrics["transform_text_tokenization_ms"] += chat_template_ms

        (image_inputs, _video_inputs), process_vision_info_ms = measure_wall_time_ms(
            lambda: transform.eagle_processor.process_vision_info(eagle_conversation)
        )
        metrics["transform_image_preprocess_ms"] += process_vision_info_ms

        text_lists.extend(text_list)
        image_inputs_all.extend(image_inputs)
        processed_samples.append(
            {
                "state": state,
                "state_mask": state_mask,
                "embodiment_id": transform.get_embodiment_tag(),
            }
        )

    eagle_batch, replace_media_ms, tokenizer_ms = _build_processor_output(
        transform.eagle_processor,
        text_lists,
        image_inputs_all,
    )
    metrics["transform_image_preprocess_ms"] += replace_media_ms
    metrics["transform_text_tokenization_ms"] += tokenizer_ms

    state = torch.from_numpy(np.stack([sample["state"] for sample in processed_samples]))
    state_mask = torch.from_numpy(np.stack([sample["state_mask"] for sample in processed_samples]))
    embodiment_id = torch.from_numpy(np.stack([sample["embodiment_id"] for sample in processed_samples]))

    return {
        **_prefix_eagle_batch_keys(eagle_batch),
        "state": state,
        "state_mask": state_mask,
        "embodiment_id": embodiment_id,
    }


def _run_transform_breakdown(policy, raw_obs: dict[str, Any], unsqueeze_dict_values) -> tuple[dict[str, Any], dict[str, float]]:
    from gr00t.data.transform.concat import ConcatTransform
    from gr00t.model.transforms import GR00TTransform

    data = _prepare_batched_observation(raw_obs, unsqueeze_dict_values)
    metrics: dict[str, float] = defaultdict(float)

    for transform in policy.modality_transform.transforms:
        if isinstance(transform, ConcatTransform):
            data = _apply_concat_transform_with_breakdown(transform, data, metrics)
            continue
        if isinstance(transform, GR00TTransform):
            data = _apply_gr00t_transform_with_breakdown(transform, data, metrics)
            continue

        category = _categorize_transform(transform)
        data, elapsed_ms = measure_wall_time_ms(lambda current=data, t=transform: t(current))
        metrics[category] += elapsed_ms

    metrics["transform_total_ms"] = round(
        metrics["transform_state_ms"]
        + metrics["transform_action_ms"]
        + metrics["transform_image_preprocess_ms"]
        + metrics["transform_text_tokenization_ms"]
        + metrics["transform_misc_ms"],
        4,
    )
    return data, metrics


def _run_backbone_breakdown(backbone, backbone_inputs: BatchFeature) -> tuple[BatchFeature, dict[str, float]]:
    from transformers.feature_extraction_utils import BatchFeature

    metrics: dict[str, float] = defaultdict(float)

    eagle_input = {
        key.removeprefix("eagle_"): value
        for key, value in backbone_inputs.items()
        if key.startswith("eagle_")
    }
    eagle_input.pop("image_sizes", None)

    # 这里按 Eagle 模型真实结构拆成视觉塔和语言模型两段。
    extract_feature = getattr(
        backbone.eagle_model,
        "_compiled_extract_feature",
        backbone.eagle_model.extract_feature,
    )
    vit_embeds, vision_ms = measure_cuda_time_ms(lambda: extract_feature(eagle_input["pixel_values"]))
    metrics["backbone_vision_encoder_ms"] += vision_ms

    def _run_llm():
        input_ids = eagle_input["input_ids"]
        attention_mask = eagle_input["attention_mask"]
        input_embeds = backbone.eagle_model.language_model.get_input_embeddings()(input_ids)

        batch_size, token_count, hidden_size = input_embeds.shape
        flat_embeds = input_embeds.reshape(batch_size * token_count, hidden_size)
        flat_ids = input_ids.reshape(batch_size * token_count)
        selected = flat_ids == backbone.eagle_model.image_token_index
        fused_vit = vit_embeds.reshape(-1, hidden_size)
        if selected.sum() > fused_vit.shape[0]:
            raise ValueError("视觉 token 数量不足，无法完成 Eagle 输入融合")
        flat_embeds[selected] = fused_vit[: selected.sum()]
        input_embeds = flat_embeds.reshape(batch_size, token_count, hidden_size)
        language_model = getattr(
            backbone.eagle_model,
            "_compiled_language_model",
            backbone.eagle_model.language_model,
        )
        outputs = language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,
        )
        hidden_states = outputs.hidden_states[backbone.select_layer]
        return backbone.eagle_linear(hidden_states), attention_mask

    (eagle_features, eagle_mask), llm_ms = measure_cuda_time_ms(_run_llm)
    metrics["backbone_llm_ms"] += llm_ms
    metrics["backbone_total_ms"] = round(
        metrics["backbone_vision_encoder_ms"] + metrics["backbone_llm_ms"], 4
    )
    return BatchFeature(
        data={"backbone_features": eagle_features, "backbone_attention_mask": eagle_mask}
    ), metrics


def _run_action_head_breakdown(action_head, backbone_output: BatchFeature, action_input: BatchFeature) -> tuple[BatchFeature, dict[str, float]]:
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    metrics: dict[str, float] = defaultdict(float)

    with torch.no_grad():
        process_backbone_output = getattr(
            action_head,
            "_compiled_process_backbone_output",
            action_head.process_backbone_output,
        )
        processed_backbone_output, features_process_ms = measure_cuda_time_ms(
            lambda: process_backbone_output(backbone_output)
        )
        metrics["action_head_features_process_ms"] += features_process_ms

        vl_embs = processed_backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        state_encoder = getattr(action_head, "_compiled_state_encoder", action_head.state_encoder)
        state_features, state_encoder_ms = measure_cuda_time_ms(
            lambda: state_encoder(action_input.state, embodiment_id)
        )
        metrics["action_head_state_encoder_ms"] += state_encoder_ms

        batch_size = vl_embs.shape[0]
        device = vl_embs.device
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
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)

            action_encoder = getattr(
                action_head,
                "_compiled_action_encoder",
                action_head.action_encoder,
            )
            action_features, action_encoder_ms = measure_cuda_time_ms(
                lambda tt=timesteps_tensor, current_actions=actions: action_encoder(
                    current_actions,
                    tt,
                    embodiment_id,
                )
            )
            metrics["action_head_action_encoder_ms"] += action_encoder_ms

            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            dit_model = getattr(action_head, "_compiled_model", action_head.model)
            model_output, dit_ms = measure_cuda_time_ms(
                lambda states=sa_embs, tt=timesteps_tensor: dit_model(
                    hidden_states=states,
                    encoder_hidden_states=vl_embs,
                    timestep=tt,
                )
            )
            metrics["action_head_dit_block_ms"] += dit_ms

            action_decoder = getattr(
                action_head,
                "_compiled_action_decoder",
                action_head.action_decoder,
            )
            pred, action_decoder_ms = measure_cuda_time_ms(
                lambda mo=model_output: action_decoder(mo, embodiment_id)
            )
            metrics["action_head_action_decoder_ms"] += action_decoder_ms
            pred_velocity = pred[:, -action_head.action_horizon :]
            actions = actions + dt * pred_velocity

    metrics["action_head_total_ms"] = round(
        metrics["action_head_features_process_ms"]
        + metrics["action_head_state_encoder_ms"]
        + metrics["action_head_action_encoder_ms"]
        + metrics["action_head_dit_block_ms"]
        + metrics["action_head_action_decoder_ms"],
        4,
    )
    return BatchFeature(data={"action_pred": actions}), metrics


def _run_single_breakdown(policy, raw_obs: dict[str, Any], compute_dtype, unsqueeze_dict_values) -> dict[str, float]:
    import contextlib
    import torch

    metrics: dict[str, float] = defaultdict(float)

    normalized_input, transform_metrics = _run_transform_breakdown(policy, raw_obs, unsqueeze_dict_values)
    metrics.update(transform_metrics)

    def _prepare_model_input():
        return policy.model.prepare_input(normalized_input)

    (backbone_inputs, action_inputs), prepare_input_ms = measure_cuda_time_ms(_prepare_model_input)
    metrics["prepare_input_ms"] = prepare_input_ms

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=compute_dtype)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        backbone_output, backbone_metrics = _run_backbone_breakdown(policy.model.backbone, backbone_inputs)
        metrics.update(backbone_metrics)
        normalized_action, action_head_metrics = _run_action_head_breakdown(
            policy.model.action_head,
            backbone_output,
            action_inputs,
        )
        metrics.update(action_head_metrics)

    def _postprocess():
        return policy.unapply_transforms({"action": normalized_action["action_pred"].float().cpu()})

    _, postprocess_ms = measure_wall_time_ms(_postprocess)
    metrics["postprocess_ms"] = postprocess_ms
    metrics["e2e_total_ms"] = round(
        metrics["transform_total_ms"]
        + metrics["prepare_input_ms"]
        + metrics["backbone_total_ms"]
        + metrics["action_head_total_ms"]
        + metrics["postprocess_ms"],
        4,
    )
    return metrics


def main() -> int:
    from deployment_scripts.ant.local_inference import run_legacy_main

    return run_legacy_main(
        build_parser_fn=build_parser,
        backend="pytorch",
        mode="breakdown",
        script_name="local_inference_breakdown.py",
    )


if __name__ == "__main__":
    raise SystemExit(main())
