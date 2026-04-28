# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
import time
from collections import Counter
from typing import Dict, Optional

import modelopt.torch.quantization as mtq
import numpy as np
import onnx
import torch
import torch.utils.checkpoint as cp
from onnx import TensorProto, numpy_helper
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionEmbeddings,
    SiglipVisionTransformer,
)

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH, EagleBackbone
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values


def no_batch_collate_fn(batch):
    """Collate function that returns the first item without adding batch dimension."""
    return batch[0]


def _get_vit_quant_cfg(precision: str):
    if precision == "fp16":
        return None
    if precision == "fp8":
        quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    elif precision == "int8":
        quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    else:
        raise ValueError(f"Unsupported ViT precision: {precision}")

    # Keep convs in higher precision to preserve the existing export behavior.
    quant_cfg["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}
    return quant_cfg


def _get_dit_quant_cfg(precision: str):
    if precision == "fp16":
        return None
    if precision == "fp8":
        quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        quant_cfg["quant_cfg"]["*[qkv]_bmm_quantizer"] = {"num_bits": (4, 3), "axis": None}
        quant_cfg["quant_cfg"]["*softmax_quantizer"] = {"num_bits": (4, 3), "axis": None}
        return quant_cfg
    if precision == "int8":
        quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
        # Disable BMM and softmax quantizers — INT8 precision is too coarse for these ops,
        # causing numerical instability in attention scores.
        quant_cfg["quant_cfg"]["*[qkv]_bmm_quantizer"] = {"enable": False}
        quant_cfg["quant_cfg"]["*softmax_quantizer"] = {"enable": False}
        return quant_cfg
    raise ValueError(f"Unsupported DiT precision: {precision}")


def _get_llm_quant_cfg(precision: str, full_layer_quant: bool, int8_algo: str = "smoothquant"):
    if precision == "fp16":
        return None
    if precision == "nvfp4":
        quant_cfg = copy.deepcopy(mtq.NVFP4_AWQ_FULL_CFG)
        if not full_layer_quant:
            print("Using selective layer quantization (disabling down_proj and o_proj)")
            quant_cfg["quant_cfg"][
                "eagle_model.language_model.model.layers.*.mlp.down_proj.*_quantizer"
            ] = {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
                "enable": False,
            }
            quant_cfg["quant_cfg"][
                "eagle_model.language_model.model.layers.*.self_attn.o_proj.*_quantizer"
            ] = {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
                "enable": False,
            }
        else:
            print("Using full layer quantization (all layers enabled)")
        return quant_cfg
    if precision == "fp8":
        return copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    if precision == "int8":
        if int8_algo == "smoothquant":
            # SmoothQuant migrates quantization difficulty from activations to weights,
            # significantly improving INT8 accuracy for transformer models (Orin NX target).
            if hasattr(mtq, "INT8_SMOOTHQUANT_CFG"):
                quant_cfg = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
            else:
                quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
                quant_cfg["algorithm"] = {"method": "smoothquant", "alpha": 0.5}
            print("Using INT8 SmoothQuant quantization for LLM")
        else:
            quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
            print("Using INT8 max (min/max) quantization for LLM")
        if not full_layer_quant:
            print("Using selective layer quantization (disabling down_proj and o_proj)")
            quant_cfg["quant_cfg"][
                "eagle_model.language_model.model.layers.*.mlp.down_proj.*_quantizer"
            ] = {"enable": False}
            quant_cfg["quant_cfg"][
                "eagle_model.language_model.model.layers.*.self_attn.o_proj.*_quantizer"
            ] = {"enable": False}
        else:
            print("Using full layer quantization (all layers enabled)")
        return quant_cfg
    raise ValueError(f"Unsupported LLM precision: {precision}")


def _rewrite_layernorm_initializers_for_float_inputs(onnx_path):
    onnx_model = onnx.load(os.fspath(onnx_path), load_external_data=True)
    initializer_map = {init.name: init for init in onnx_model.graph.initializer}

    for node in onnx_model.graph.node:
        if node.op_type != "LayerNormalization" or len(node.input) < 3:
            continue

        for initializer_name in node.input[1:3]:
            initializer = initializer_map.get(initializer_name)
            if initializer is None or initializer.data_type != TensorProto.FLOAT16:
                continue
            tensor = numpy_helper.to_array(initializer).astype(np.float32)
            initializer.CopyFrom(numpy_helper.from_array(tensor, name=initializer_name))

    onnx.save_model(onnx_model, os.fspath(onnx_path))


def _configure_llm_modelopt_onnx_quantizers(model, precision: str, tag: str = "LLM"):
    """
    Configure ModelOpt quantizers before torch.onnx.export for the LLM path only.

    Why LLM-only?
      - ViT and DiT already exported valid Q/DQ graphs in the original pipeline.
      - Applying this configuration globally can force ViT/DiT activations through
        Float-side Q/DQ and inflate ONNX files dramatically.
      - The observed failure was specific to the LLM path: LLM INT8 previously did
        not preserve Q/DQ into ONNX/TensorRT.

    Export convention:
      - activation/input quantizer: dynamic
      - weight quantizer: static
      - INT8 input Q/DQ high precision dtype: Float, safest for mixed FP16/FP32 graphs
      - NVFP4 keeps Half, matching the original NVIDIA export path
      - FP8 does not force _trt_high_precision_dtype
    """
    if precision not in ["int8", "fp8", "nvfp4"]:
        return

    try:
        from modelopt.torch.quantization.utils import is_quantized_linear
    except Exception as e:
        print(f"[{tag}] Warning: cannot import is_quantized_linear: {e}")
        return

    n_linear = 0
    n_qlinear = 0
    n_input_q = 0
    n_weight_q = 0

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        n_linear += 1

        if not is_quantized_linear(module):
            continue

        n_qlinear += 1

        if hasattr(module, "input_quantizer"):
            module.input_quantizer._onnx_quantizer_type = "dynamic"

            if precision == "int8":
                # LLM export can contain Float-side Q/DQ. ModelOpt requires either
                # the dtype to match the ONNX input type or the Q/DQ to be Float.
                module.input_quantizer._trt_high_precision_dtype = "Float"
            elif precision == "nvfp4":
                # Preserve original NVIDIA NVFP4 behavior.
                module.input_quantizer._trt_high_precision_dtype = "Half"
            # For FP8, leave _trt_high_precision_dtype untouched.

            n_input_q += 1

        if hasattr(module, "weight_quantizer"):
            module.weight_quantizer._onnx_quantizer_type = "static"
            n_weight_q += 1

    print(
        f"[{tag}] configured LLM ONNX quantizers for {precision}: "
        f"quantized_linears={n_qlinear}/{n_linear}, "
        f"input_quantizers={n_input_q}, weight_quantizers={n_weight_q}"
    )

    if precision == "int8" and n_qlinear == 0:
        print(f"[{tag}] WARNING: no quantized Linear layers found for INT8 LLM export.")


def _inspect_torch_quantization(model, tag: str = "model"):
    """
    Inspect whether ModelOpt quantization was actually injected into the PyTorch model.
    This check happens before ONNX export.
    """
    print(f"\n===== Inspect PyTorch quantization: {tag} =====")

    try:
        from modelopt.torch.quantization.utils import is_quantized_linear
    except Exception as e:
        print(f"[{tag}] Warning: cannot import is_quantized_linear: {e}")
        return

    total_linears = 0
    quantized_linears = 0
    total_quantizers = 0
    enabled_quantizers = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            total_linears += 1
            if is_quantized_linear(module):
                quantized_linears += 1
                print(f"[{tag}] QLinear: {name}")

        cls_name = module.__class__.__name__
        if "Quantizer" in cls_name:
            total_quantizers += 1

            enabled = None
            if hasattr(module, "is_enabled"):
                try:
                    enabled = module.is_enabled()
                except Exception:
                    enabled = getattr(module, "_enabled", None)
            else:
                enabled = getattr(module, "_enabled", None)

            if enabled:
                enabled_quantizers += 1

            print(f"[{tag}] Quantizer: {name}, class={cls_name}, enabled={enabled}")

    print(f"[{tag}] total_linears={total_linears}")
    print(f"[{tag}] quantized_linears={quantized_linears}")
    print(f"[{tag}] total_quantizers={total_quantizers}")
    print(f"[{tag}] enabled_quantizers={enabled_quantizers}")

    if total_quantizers == 0:
        print(f"[{tag}] ERROR: no quantizers found in PyTorch model.")
    if quantized_linears == 0:
        print(f"[{tag}] ERROR: no quantized Linear modules found.")


def _inspect_onnx_qdq(onnx_path: str, tag: str = "onnx"):
    """
    Inspect whether exported ONNX contains Q/DQ nodes.
    This is the most important check for TensorRT explicit quantization.
    """
    print(f"\n===== Inspect ONNX Q/DQ: {tag} =====")

    if not os.path.exists(onnx_path):
        print(f"[{tag}] ERROR: ONNX file does not exist: {onnx_path}")
        return

    try:
        onnx_model = onnx.load(onnx_path, load_external_data=False)
    except Exception as e:
        print(f"[{tag}] ERROR: failed to load ONNX model: {e}")
        return

    op_counts = Counter(node.op_type for node in onnx_model.graph.node)

    for op, cnt in op_counts.most_common(30):
        print(f"[{tag}] {op}: {cnt}")

    q = op_counts.get("QuantizeLinear", 0)
    dq = op_counts.get("DequantizeLinear", 0)
    matmul = op_counts.get("MatMul", 0)
    gemm = op_counts.get("Gemm", 0)

    print(f"[{tag}] QuantizeLinear={q}, DequantizeLinear={dq}")
    print(f"[{tag}] MatMul={matmul}, Gemm={gemm}")

    if q == 0 and dq == 0:
        print(f"[{tag}] WARNING: ONNX has no Q/DQ nodes. Quantization may not be exported.")
    else:
        print(f"[{tag}] OK: ONNX contains Q/DQ nodes.")


def _inspect_onnx_initializer_dtypes(onnx_path: str, tag: str = "onnx"):
    """
    Inspect ONNX initializer data types. Useful for checking whether weights are
    still mostly FP16 or have been converted to INT8/UINT8.
    """
    print(f"\n===== Inspect ONNX initializer dtypes: {tag} =====")

    if not os.path.exists(onnx_path):
        print(f"[{tag}] ERROR: ONNX file does not exist: {onnx_path}")
        return

    try:
        onnx_model = onnx.load(onnx_path, load_external_data=False)
    except Exception as e:
        print(f"[{tag}] ERROR: failed to load ONNX model: {e}")
        return

    dtype_counts = Counter(init.data_type for init in onnx_model.graph.initializer)

    dtype_name = {
        TensorProto.FLOAT: "FLOAT32",
        TensorProto.FLOAT16: "FLOAT16",
        TensorProto.BFLOAT16: "BFLOAT16",
        TensorProto.INT8: "INT8",
        TensorProto.UINT8: "UINT8",
        TensorProto.INT32: "INT32",
        TensorProto.INT64: "INT64",
    }

    for dtype, cnt in dtype_counts.items():
        print(f"[{tag}] {dtype_name.get(dtype, dtype)}: {cnt}")

    int8_count = dtype_counts.get(TensorProto.INT8, 0) + dtype_counts.get(TensorProto.UINT8, 0)
    fp16_count = dtype_counts.get(TensorProto.FLOAT16, 0)
    fp32_count = dtype_counts.get(TensorProto.FLOAT, 0)

    print(f"[{tag}] INT8/UINT8 initializers={int8_count}")
    print(f"[{tag}] FP16 initializers={fp16_count}")
    print(f"[{tag}] FP32 initializers={fp32_count}")


def _print_onnx_file_size(onnx_path: str, tag: str = "onnx"):
    """
    Print ONNX file size. If external data is used, this only prints the main
    .onnx file size, not the external tensor data file.
    """
    if not os.path.exists(onnx_path):
        print(f"[{tag}] ONNX file does not exist: {onnx_path}")
        return

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"[{tag}] ONNX file size: {size_mb:.2f} MB, path={onnx_path}")


def _resolve_calibration_policy(
    policy,
    model_path: str,
    embodiment_tag: str,
    denoising_steps: int,
    data_config: str,
    device: str = "cuda",
):
    if policy is not None:
        return policy

    data_config_obj = load_data_config(data_config)
    modality_config = data_config_obj.modality_config()
    modality_transform = data_config_obj.transform()
    return Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=denoising_steps,
        device=device,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Run Groot Inference")
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the dataset",
        default=os.path.join(os.getcwd(), "demo_data/robot_sim.PickNPlace"),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model",
        default="nvidia/GR00T-N1.5-3B",
    )

    parser.add_argument(
        "--onnx-model-path",
        type=str,
        help="Path where the ONNX model will be stored",
        default=os.path.join(os.getcwd(), "gr00t_onnx"),
    )

    parser.add_argument(
        "--data-config",
        type=str,
        help="The name of the data config to use (e.g. fourier_gr1_arms_only, fourier_gr1_arms_waist, unitree_g1, etc.) or a path to a custom data config file",
        default="fourier_gr1_arms_only",
    )

    parser.add_argument(
        "--embodiment-tag",
        type=str,
        help="The embodiment tag for the model (e.g. gr1, g1, so100, etc.)",
        default="gr1",
    )

    parser.add_argument(
        "--llm-dtype",
        type=str,
        choices=["fp16", "nvfp4", "fp8", "int8"],
        help="Data type for LLM export (fp16, nvfp4, fp8, or int8 for quantization)",
        default="nvfp4",
    )

    parser.add_argument(
        "--vit-dtype",
        type=str,
        choices=["fp16", "fp8", "int8"],
        help="Data type for ViT export (fp16, fp8, or int8 for quantization)",
        default="fp8",
    )

    parser.add_argument(
        "--dit-dtype",
        type=str,
        choices=["fp16", "fp8", "int8"],
        help="Data type for DiT (Diffusion Transformer) export (fp16, fp8, or int8 for quantization)",
        default="fp8",
    )

    parser.add_argument(
        "--calib-dataset-path",
        type=str,
        help="Path to the LeRobot dataset for calibration (if different from dataset_path)",
        default=None,
    )

    parser.add_argument(
        "--calib-size",
        type=int,
        help="Number of calibration samples to use",
        default=32,
    )

    parser.add_argument(
        "--denoising-steps",
        type=int,
        help="Number of denoising steps for diffusion model inference",
        default=4,
    )

    parser.add_argument(
        "--video-backend",
        type=str,
        choices=["decord", "torchcodec", "opencv", "torchvision_av"],
        help="Video backend to use for loading video frames",
        default="decord",
    )

    parser.add_argument(
        "--full-layer-quant",
        action="store_true",
        help="Enable full layer quantization for LLM (default: False, which disables down_proj and o_proj layers). Applies to both nvfp4 and int8.",
    )

    parser.add_argument(
        "--int8-algo",
        type=str,
        choices=["smoothquant", "max"],
        help="INT8 quantization algorithm for LLM: smoothquant (recommended for Orin NX, better accuracy) or max (basic min/max calibration)",
        default="smoothquant",
    )

    return parser


class ViTCalibrationDataset(Dataset):
    """
    A dataset that uses LeRobotSingleDataset for ViT calibration data.
    This provides realistic calibration data for the vision transformer.
    """

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict,
        embodiment_tag: str,
        policy: Gr00tPolicy,
        calib_size: int = 100,
        video_backend: str = "decord",
    ):
        self.calib_size = calib_size
        self.policy = policy

        self.lerobot_dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
        )

        self.dataset_size = len(self.lerobot_dataset)
        print(f"ViT Dataset size: {self.dataset_size}")
        self.calib_size = min(calib_size, self.dataset_size)

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        data = self.lerobot_dataset[idx]
        processed_data = self._process_vit_data(data)
        return processed_data

    def _process_vit_data(self, data):
        """
        Process LeRobot data to extract pixel_values and position_ids for ViT calibration.
        """
        try:
            is_batch = self.policy._check_state_is_batched(data)
            if not is_batch:
                data = unsqueeze_dict_values(data)

            transformed_data = self.policy.apply_transforms(data)

            if "eagle_pixel_values" in transformed_data:
                pixel_values = transformed_data["eagle_pixel_values"]
                batch_size = pixel_values.shape[0]
                num_patches = (
                    self.policy.model.backbone.eagle_model.vision_model.vision_model.embeddings.num_patches
                )
                position_ids = torch.arange(
                    num_patches, dtype=torch.long, device=pixel_values.device
                ).expand((batch_size, -1))
                return {
                    "pixel_values": pixel_values,
                    "position_ids": position_ids,
                }
            else:
                raise RuntimeError(
                    "eagle data not found in transformed_data. This indicates an issue with apply_transforms()."
                )
        except Exception as e:
            print(f"Warning: ViT data processing failed: {e}, using dummy data")
            raise RuntimeError(f"apply_transforms() failed: {e}")


class LLMCalibrationDataset(Dataset):
    """
    A dataset that uses LeRobotSingleDataset for LLM calibration data.
    This provides more realistic calibration data compared to random data.
    Uses apply_transforms() for consistent data processing.
    """

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict,
        embodiment_tag: str,
        policy: Gr00tPolicy,
        calib_size: int = 100,
        video_backend: str = "decord",
        enable_comparison: bool = False,
    ):
        self.calib_size = calib_size
        self.policy = policy
        self.enable_comparison = enable_comparison

        self.lerobot_dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
        )

        self.dataset_size = len(self.lerobot_dataset)
        print(f"Dataset size: {self.dataset_size}")
        self.calib_size = min(calib_size, self.dataset_size)

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        data = self.lerobot_dataset[idx]
        processed_data = self._process_llm_data(data)
        return processed_data

    def _process_llm_data(self, data):
        """
        Process LLM data to extract input_ids, vit_embeds, and attention_mask
        for LLM calibration using apply_transforms() for consistent data processing.
        """
        try:
            is_batch = self.policy._check_state_is_batched(data)
            if not is_batch:
                data = unsqueeze_dict_values(data)

            transformed_data = self.policy.apply_transforms(data)

            if "eagle_input_ids" in transformed_data and "eagle_pixel_values" in transformed_data:
                pixel_values = transformed_data["eagle_pixel_values"]
                input_ids = transformed_data["eagle_input_ids"]
                attention_mask = transformed_data["eagle_attention_mask"]

                try:
                    with torch.no_grad():
                        pixel_values = pixel_values.to("cuda")
                        if self.policy.model.backbone.eagle_model.select_layer == -1:
                            vit_embeds = self.policy.model.backbone.eagle_model.vision_model(
                                pixel_values=pixel_values,
                                output_hidden_states=False,
                                return_dict=True,
                            )
                            if hasattr(vit_embeds, "last_hidden_state"):
                                vit_embeds = vit_embeds.last_hidden_state
                        else:
                            vit_embeds = self.policy.model.backbone.eagle_model.vision_model(
                                pixel_values=pixel_values,
                                output_hidden_states=True,
                                return_dict=True,
                            ).hidden_states[self.policy.model.backbone.eagle_model.select_layer]

                except Exception as e:
                    raise RuntimeError(f"Policy eagle model failed: {e}")

                vit_embeds = vit_embeds.view(1, -1, vit_embeds.shape[-1])

                if self.policy.model.backbone.eagle_model.use_pixel_shuffle:
                    h = w = int(vit_embeds.shape[1] ** 0.5)
                    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
                    vit_embeds = self.policy.model.backbone.pixel_shuffle(
                        vit_embeds, scale_factor=self.policy.model.backbone.downsample_ratio
                    )
                    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

                if (
                    self.policy.model.backbone.eagle_model.mlp_checkpoint
                    and vit_embeds.requires_grad
                ):
                    vit_embeds = cp.checkpoint(
                        self.policy.model.backbone.eagle_model.mlp1, vit_embeds
                    )
                else:
                    vit_embeds = self.policy.model.backbone.eagle_model.mlp1(vit_embeds)

                input_ids = input_ids.to(next(self.policy.model.parameters()).device)

                input_embeds = (
                    self.policy.model.backbone.eagle_model.language_model.get_input_embeddings()(
                        input_ids
                    )
                )

                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)

                input_ids_flat = input_ids.reshape(B * N)
                selected = (
                    input_ids_flat == self.policy.model.backbone.eagle_model.image_token_index
                )
                try:
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                        -1, C
                    )
                except Exception as e:
                    vit_embeds = vit_embeds.reshape(-1, C)
                    print(
                        f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                        f"vit_embeds.shape={vit_embeds.shape}"
                    )
                    n_token = selected.sum()
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

                input_embeds = input_embeds.reshape(B, N, C)
                input_embeds = input_embeds.to(torch.float16)

                return {
                    "inputs_embeds": input_embeds,
                    "attention_mask": attention_mask,
                }
            else:
                raise RuntimeError(
                    "eagle data not found in transformed_data. This indicates an issue with apply_transforms()."
                )

        except Exception as e:
            raise RuntimeError(f"apply_transforms() failed: {e}")


class DiTCalibrationDataset(Dataset):
    """
    A dataset that uses LeRobotSingleDataset for DiT calibration data.
    This provides realistic calibration data for the diffusion transformer.
    """

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict,
        embodiment_tag: str,
        policy: Gr00tPolicy,
        calib_size: int = 100,
        video_backend: str = "decord",
    ):
        self.calib_size = calib_size
        self.policy = policy

        self.lerobot_dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
        )

        self.dataset_size = len(self.lerobot_dataset)
        print(f"DiT Dataset size: {self.dataset_size}")
        self.calib_size = min(calib_size, self.dataset_size)

        self.input_embedding_dim = policy.model.action_head.config.input_embedding_dim
        self.backbone_embedding_dim = policy.model.action_head.config.backbone_embedding_dim
        self.action_horizon = policy.model.action_head.config.action_horizon
        self.num_target_vision_tokens = policy.model.action_head.config.num_target_vision_tokens

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        data = self.lerobot_dataset[idx]
        processed_data = self._process_dit_data(data)
        return processed_data

    def _process_dit_data(self, data):
        """
        Process LeRobot data to prepare inputs for DiT calibration.
        Returns the necessary inputs for running the denoising loop.
        """
        try:
            is_batch = self.policy._check_state_is_batched(data)
            if not is_batch:
                data = unsqueeze_dict_values(data)

            transformed_data = self.policy.apply_transforms(data)

            backbone_inputs, action_inputs = self.policy.model.prepare_input(transformed_data)
            backbone_outputs = self.policy.model.backbone(backbone_inputs)

            action_head_dtype = next(self.policy.model.action_head.parameters()).dtype
            for key in backbone_outputs:
                if isinstance(backbone_outputs[key], torch.Tensor) and backbone_outputs[key].is_floating_point():
                    backbone_outputs[key] = backbone_outputs[key].to(action_head_dtype)

            backbone_output = self.policy.model.action_head.process_backbone_output(backbone_outputs)

            vl_embs = backbone_output["backbone_features"]
            embodiment_id = action_inputs["embodiment_id"]

            state_features = self.policy.model.action_head.state_encoder(
                action_inputs["state"], embodiment_id
            )

            batch_size = vl_embs.shape[0]
            device = vl_embs.device
            actions = torch.randn(
                size=(
                    batch_size,
                    self.policy.model.action_head.config.action_horizon,
                    self.policy.model.action_head.config.action_dim,
                ),
                dtype=vl_embs.dtype,
                device=device,
            )

            return {
                "vl_embs": vl_embs,
                "state_features": state_features,
                "actions": actions,
                "embodiment_id": embodiment_id,
            }
        except Exception as e:
            raise RuntimeError(f"DiT data processing failed: {e}")


def _quantize_model(model, calib_dataloader, quant_cfg):
    """
    The calibration loop for the model can be setup using the modelopt API.
    """

    def calibrate_loop(model):
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            if idx % 10 == 0:
                print(f"Calibrating batch {idx}...")
            data = {k: v.to(next(model.parameters()).device) for k, v in data.items()}
            model(**data)

    print("Starting quantization...")
    start_time = time.time()
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization finishes in {end_time - start_time}s.")

    return model


def _quantize_dit_model(model, calib_dataloader, quant_cfg, action_head):
    """
    Custom calibration loop for DiT model that runs the full denoising process.
    DiT requires multiple forward passes (typically 4 steps) for proper calibration.
    """
    dit_dtype = next(model.parameters()).dtype
    action_head.to(dit_dtype)

    def calibrate_loop(model):
        """Run the denoising loop for DiT calibration."""
        model_dtype = next(model.parameters()).dtype
        for idx, data in enumerate(calib_dataloader):
            if idx % 10 == 0:
                print(f"Calibrating DiT batch {idx}...")

            device = next(model.parameters()).device
            vl_embs = data["vl_embs"].to(device=device, dtype=model_dtype)
            state_features = data["state_features"].to(device=device, dtype=model_dtype)
            actions = data["actions"].to(device=device, dtype=model_dtype)
            embodiment_id = data["embodiment_id"].to(device)

            batch_size = vl_embs.shape[0]
            num_steps = action_head.num_inference_timesteps
            dt = 1.0 / num_steps

            for t in range(num_steps):
                t_cont = t / float(num_steps)
                t_discretized = int(t_cont * action_head.num_timestep_buckets)

                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = action_head.action_encoder(
                    actions, timesteps_tensor, embodiment_id
                ).to(model_dtype)

                if action_head.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1], dtype=torch.long, device=device
                    )
                    pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs.to(model_dtype)

                future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(
                    batch_size, -1, -1
                ).to(model_dtype)
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

                _ = model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )

                model_output = action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )

                pred = action_head.action_decoder(model_output, embodiment_id)
                pred_velocity = pred[:, -action_head.config.action_horizon :]

                actions = (actions + dt * pred_velocity).to(model_dtype)

    print("Starting DiT quantization with multi-step denoising...")
    start_time = time.time()
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"DiT quantization finishes in {end_time - start_time}s.")

    return model


def quantize_vit(
    model,
    precision="fp8",
    calib_size=10,
    batch_size=1,
    dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    video_backend="decord",
    policy=None,
    compare_accuracy=True,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
):
    if mtq is None:
        raise ImportError("modelopt is required for quantization")

    assert precision in ["fp8", "fp16", "int8"], (
        f"Only fp8, int8, and fp16 are supported for ViT. You passed: {precision}."
    )

    quant_cfg = _get_vit_quant_cfg(precision)

    if dataset_path is None or modality_configs is None or policy is None:
        raise ValueError(
            "ViT quantization requires valid dataset_path, modality_configs, and policy."
        )

    print(f"Using LeRobot dataset for ViT calibration: {dataset_path}")
    calibration_policy = _resolve_calibration_policy(
        policy=policy,
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
        data_config=data_config,
        device="cuda",
    )
    dataset = ViTCalibrationDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
        policy=calibration_policy,
        calib_size=calib_size,
        video_backend=video_backend,
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=no_batch_collate_fn)

    if quant_cfg is not None:
        quantized_model = _quantize_model(model, data_loader, quant_cfg)
        mtq.print_quant_summary(quantized_model)
        return quantized_model
    else:
        print("No quantization applied to ViT model")
        return model


def quantize_dit(
    model,
    precision="fp8",
    calib_size=10,
    batch_size=1,
    dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    video_backend="decord",
    policy=None,
    compare_accuracy=True,
    attention_mask=None,
    input_state=None,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
):
    if mtq is None:
        raise ImportError("modelopt is required for quantization")

    assert precision in ["fp8", "fp16", "int8"], (
        f"Only fp8, int8, and fp16 are supported for DiT. You passed: {precision}."
    )

    quant_cfg = _get_dit_quant_cfg(precision)

    if dataset_path is None or modality_configs is None or policy is None:
        raise ValueError(
            "DiT quantization requires valid dataset_path, modality_configs, and policy."
        )

    print(f"Using LeRobot dataset for DiT calibration: {dataset_path}")
    calibration_policy = _resolve_calibration_policy(
        policy=policy,
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
        data_config=data_config,
        device="cuda",
    )
    dataset = DiTCalibrationDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
        policy=calibration_policy,
        calib_size=calib_size,
        video_backend=video_backend,
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=no_batch_collate_fn)

    if quant_cfg is not None:
        quantized_model = _quantize_dit_model(
            model, data_loader, quant_cfg, calibration_policy.model.action_head
        )
        mtq.print_quant_summary(quantized_model)
        return quantized_model
    else:
        print("No quantization applied to DiT model")
        return model


def compare_model_accuracy(original_model, quantized_model, test_data, device="cuda"):
    """
    Compare accuracy between original and quantized models.
    """
    print("\n🔍 COMPARING MODEL ACCURACY BEFORE AND AFTER QUANTIZATION...")

    original_model.eval()
    quantized_model.eval()

    differences = []
    max_diffs = []

    with torch.no_grad():
        for i, data in enumerate(test_data):
            if i >= 5:
                break

            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

            try:
                original_output = original_model(**data)
                quantized_output = quantized_model(**data)

                if isinstance(original_output, torch.Tensor) and isinstance(quantized_output, torch.Tensor):
                    diff = torch.abs(original_output - quantized_output)
                    mean_diff = torch.mean(diff).item()
                    max_diff = torch.max(diff).item()
                    orig_flat = original_output.flatten()
                    quant_flat = quantized_output.flatten()
                    cos_sim = torch.nn.functional.cosine_similarity(
                        orig_flat.unsqueeze(0), quant_flat.unsqueeze(0)
                    ).item()
                    print(f"Cosine similarity: {cos_sim:.6f}")

                    differences.append(mean_diff)
                    max_diffs.append(max_diff)
                    print(f"Sample {i+1}: Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")
                    max_diff_pos = (diff == torch.max(diff)).nonzero(as_tuple=True)
                    if len(max_diff_pos[0]) > 0:
                        idx = tuple(pos[0].item() for pos in max_diff_pos)
                        print(f"Position of max diff: {idx}")
                        print(
                            f"Original value at this position: {original_output[idx] if isinstance(original_output, torch.Tensor) else original_output.logits[idx]}"
                        )

                elif hasattr(original_output, "logits") and hasattr(quantized_output, "logits"):
                    diff = torch.abs(original_output.logits - quantized_output.logits)
                    mean_diff = torch.mean(diff).item()
                    max_diff = torch.max(diff).item()

                    differences.append(mean_diff)
                    max_diffs.append(max_diff)

                    print(f"Sample {i+1}: Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")

            except Exception as e:
                print(f"Error comparing sample {i+1}: {e}")
                continue

    if differences:
        avg_mean_diff = sum(differences) / len(differences)
        avg_max_diff = sum(max_diffs) / len(max_diffs)
        overall_max_diff = max(max_diffs)

        print("\n📊 QUANTIZATION ACCURACY SUMMARY:")
        print(f"Average mean difference: {avg_mean_diff:.6f}")
        print(f"Average max difference: {avg_max_diff:.6f}")
        print(f"Overall max difference: {overall_max_diff:.6f}")

        if avg_mean_diff < 0.01:
            print("✅ Quantization accuracy is excellent (< 0.01)")
        elif avg_mean_diff < 0.1:
            print("✅ Quantization accuracy is good (< 0.1)")
        elif avg_mean_diff < 0.5:
            print("⚠️  Quantization accuracy is acceptable (< 0.5)")
        else:
            print("❌ Quantization accuracy is poor (> 0.5)")

        return {
            "avg_mean_diff": avg_mean_diff,
            "avg_max_diff": avg_max_diff,
            "overall_max_diff": overall_max_diff,
            "num_samples": len(differences),
        }
    else:
        print("❌ No valid comparisons could be made")
        return None


def quantize_llm(
    model,
    precision,
    calib_size=10,
    batch_size=1,
    dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    video_backend="decord",
    policy=None,
    compare_accuracy=True,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
    full_layer_quant=False,
    int8_algo="smoothquant",
):
    if mtq is None:
        raise ImportError("modelopt is required for quantization")

    assert precision in ["nvfp4", "fp8", "int8"], (
        f"Only nvfp4 (W4A4), fp8, and int8 are supported. You passed an unsupported precision: {precision}."
    )

    quant_cfg = _get_llm_quant_cfg(precision, full_layer_quant, int8_algo)

    print(f"Quantization configuration: {quant_cfg}")

    if dataset_path is None or modality_configs is None:
        raise ValueError("LLM quantization requires valid dataset_path and modality_configs.")

    print(f"Using LeRobot dataset for calibration: {dataset_path}")
    calibration_policy = _resolve_calibration_policy(
        policy=policy,
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
        data_config=data_config,
        device="cuda",
    )
    dataset = LLMCalibrationDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
        policy=calibration_policy,
        calib_size=calib_size,
        video_backend=video_backend,
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=no_batch_collate_fn)

    original_model = None
    if compare_accuracy:
        print("📋 Creating backup of original model for accuracy comparison...")
        original_model = copy.deepcopy(model)
        original_model.eval()

    quantized_model = _quantize_model(model, data_loader, quant_cfg)
    mtq.print_quant_summary(quantized_model)

    if compare_accuracy and original_model is not None:
        print("\n🔍 Starting accuracy comparison...")
        try:
            test_data = DataLoader(
                dataset, batch_size=1, shuffle=False, collate_fn=no_batch_collate_fn
            )
            comparison_results = compare_model_accuracy(original_model, quantized_model, test_data)

            if comparison_results:
                print("\n📈 FINAL QUANTIZATION RESULTS:")
                print("✅ Quantization completed successfully!")
                print(f"📊 Accuracy comparison: {comparison_results['num_samples']} samples tested")
                print(f"📊 Average mean difference: {comparison_results['avg_mean_diff']:.6f}")
                print(f"📊 Overall max difference: {comparison_results['overall_max_diff']:.6f}")
            else:
                print("⚠️  Accuracy comparison failed, but quantization completed")

        except Exception as e:
            print(f"⚠️  Accuracy comparison failed: {e}")
            print("✅ Quantization completed successfully despite comparison failure")

    return quantized_model


def get_input_info(policy, observations):
    is_batch = policy._check_state_is_batched(observations)
    if not is_batch:
        observations = unsqueeze_dict_values(observations)

    normalized_input = unsqueeze_dict_values
    normalized_input = policy.apply_transforms(observations)

    return normalized_input["eagle_attention_mask"], normalized_input["state"]


def export_eagle2_vit(
    vision_model,
    output_dir,
    vit_dtype="fp16",
    calib_dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    calib_size=10,
    policy=None,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
    video_backend="decord",
):
    class SiglipVisionEmbeddingsOpt(SiglipVisionEmbeddings):
        def __init__(self, config):
            super().__init__(config)

        def forward(
            self,
            pixel_values: torch.FloatTensor,
            position_ids: torch.LongTensor,
            interpolate_pos_encoding=False,
        ) -> torch.Tensor:
            _, _, height, width = pixel_values.shape
            target_dtype = self.patch_embedding.weight.dtype
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            if interpolate_pos_encoding:
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings

    class SiglipVisionTransformerOpt(SiglipVisionTransformer):
        def __init__(self, config: SiglipVisionConfig):
            config._attn_implementation = "eager"
            super().__init__(config)
            self.embeddings = SiglipVisionEmbeddingsOpt(config)

        def forward(
            self,
            pixel_values,
            position_ids,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: Optional[bool] = False,
        ):
            output_attentions = (
                output_attentions if output_attentions is not None else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )

            hidden_states = self.embeddings(
                pixel_values,
                position_ids=position_ids,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            last_hidden_state = encoder_outputs.last_hidden_state
            last_hidden_state = self.post_layernorm(last_hidden_state)

            return last_hidden_state

    model = SiglipVisionTransformerOpt(vision_model.config).to(torch.float16)
    model.load_state_dict(vision_model.state_dict())
    model.eval().cuda()

    if vit_dtype in ["fp8", "int8"]:
        print(f"Quantizing Eagle2 ViT to {vit_dtype}")
        model = quantize_vit(
            model,
            precision=vit_dtype,
            calib_size=calib_size,
            dataset_path=calib_dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            policy=policy,
            denoising_steps=denoising_steps,
            data_config=data_config,
            model_path=model_path,
            video_backend=video_backend,
        )

    num_video_views = 1
    if modality_configs is not None and "video" in modality_configs:
        num_video_views = len(modality_configs["video"].modality_keys)
        print(f"Number of video views detected from modality config: {num_video_views}")
    else:
        print(f"Using default number of video views: {num_video_views}")

    pixel_values = torch.randn(
        (
            num_video_views,
            model.config.num_channels,
            model.config.image_size,
            model.config.image_size,
        ),
        dtype=torch.float16,
        device="cuda",
    )
    position_ids = torch.arange(model.embeddings.num_patches, device="cuda").expand(
        (num_video_views, -1)
    )

    os.makedirs(output_dir, exist_ok=True)
    vit_onnx_path = f"{output_dir}/eagle2/vit_{vit_dtype}.onnx"

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (pixel_values, position_ids),
            vit_onnx_path,
            input_names=["pixel_values", "position_ids"],
            output_names=["vit_embeds"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "position_ids": {0: "batch_size"},
                "vit_embeds": {0: "batch_size"},
            },
        )

    if vit_dtype == "int8":
        _rewrite_layernorm_initializers_for_float_inputs(vit_onnx_path)

    if vit_dtype in ["fp8", "int8"]:
        _print_onnx_file_size(vit_onnx_path, tag=f"Eagle2 ViT {vit_dtype}")
        _inspect_onnx_qdq(vit_onnx_path, tag=f"Eagle2 ViT {vit_dtype}")
        _inspect_onnx_initializer_dtypes(vit_onnx_path, tag=f"Eagle2 ViT {vit_dtype}")


def export_eagle2_llm(
    backbone_model,
    backbone_config,
    output_dir,
    attention_mask,
    llm_dtype="fp16",
    calib_dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    calib_size=10,
    policy=None,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
    video_backend="decord",
    full_layer_quant=False,
    int8_algo="smoothquant",
):
    class EagleBackboneOpt(EagleBackbone):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
            config._attn_implementation = "eager"

            assert config.text_config.architectures[0] == "Qwen3ForCausalLM"
            self.eagle_model.language_model = Qwen3ForCausalLM(config.text_config)

            while len(self.eagle_model.language_model.model.layers) > kwargs["select_layer"]:
                self.eagle_model.language_model.model.layers.pop(-1)

        def forward(self, inputs_embeds, attention_mask):
            outputs = self.eagle_model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            eagle_features = outputs.hidden_states[self.select_layer]

            eagle_features = self.eagle_linear(eagle_features)
            return eagle_features

    model = EagleBackboneOpt(**backbone_config).to(torch.float16)
    model.load_state_dict(backbone_model.state_dict())
    model.eval().cuda()

    if llm_dtype in ["nvfp4", "fp8", "int8"]:
        print(f"Quantizing Eagle2 LLM to {llm_dtype}")

        model = quantize_llm(
            model,
            llm_dtype,
            calib_size=calib_size,
            dataset_path=calib_dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            policy=policy,
            denoising_steps=denoising_steps,
            data_config=data_config,
            model_path=model_path,
            video_backend=video_backend,
            full_layer_quant=full_layer_quant,
            compare_accuracy=False,
            int8_algo=int8_algo,
        )

        _inspect_torch_quantization(model, tag=f"Eagle2 LLM {llm_dtype}")
        _configure_llm_modelopt_onnx_quantizers(
            model,
            precision=llm_dtype,
            tag=f"Eagle2 LLM {llm_dtype}",
        )

        if llm_dtype == "nvfp4":
            from modelopt.torch.quantization.utils import is_quantized_linear

            for module in model.modules():
                assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)

    inputs_embeds = torch.randn(
        (
            1,
            attention_mask.shape[1],
            model.eagle_model.language_model.config.hidden_size,
        ),
        dtype=torch.float16,
    ).cuda()
    attention_mask = torch.ones((1, attention_mask.shape[1]), dtype=torch.int64).cuda()

    llm_dtype_suffix = (
        f"{llm_dtype}_full"
        if (llm_dtype in ("nvfp4", "int8") and full_layer_quant)
        else llm_dtype
    )
    onnx_path = f"{output_dir}/eagle2/llm_{llm_dtype_suffix}.onnx"
    onnx_dir = os.path.dirname(onnx_path)
    os.makedirs(onnx_dir, exist_ok=True)

    start_time = time.time()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (inputs_embeds, attention_mask),
            onnx_path,
            input_names=["inputs_embeds", "attention_mask"],
            output_names=["embeddings"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size", 1: "sequence_length"},
            },
        )
    end_time = time.time()
    print(
        f"Eagle2 LLM ONNX Export from torch completed in {end_time - start_time}s. ONNX file is saved to {onnx_path}."
    )

    if llm_dtype in ["nvfp4", "fp8", "int8"]:
        _print_onnx_file_size(onnx_path, tag=f"Eagle2 LLM {llm_dtype} before postprocess")
        _inspect_onnx_qdq(onnx_path, tag=f"Eagle2 LLM {llm_dtype} before postprocess")
        _inspect_onnx_initializer_dtypes(
            onnx_path,
            tag=f"Eagle2 LLM {llm_dtype} before postprocess",
        )

    if llm_dtype == "nvfp4":
        print("Converting nvfp4 ONNX model to 2dq")
        import onnx
        from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq

        onnx_model = onnx.load(onnx_path, load_external_data=True)
        onnx_model = fp4qdq_to_2dq(onnx_model, verbose=True)

        onnx.save_model(
            onnx_model,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"llm_{llm_dtype}.onnx_data",
            convert_attribute=True,
        )

        _print_onnx_file_size(onnx_path, tag=f"Eagle2 LLM {llm_dtype} after fp4qdq_to_2dq")
        _inspect_onnx_qdq(onnx_path, tag=f"Eagle2 LLM {llm_dtype} after fp4qdq_to_2dq")
        _inspect_onnx_initializer_dtypes(
            onnx_path,
            tag=f"Eagle2 LLM {llm_dtype} after fp4qdq_to_2dq",
        )


class VLLN_VLSelfAttention(torch.nn.Module):
    def __init__(self, vlln, vl_self_attention):
        super().__init__()
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, backbone_features):
        x = self.vlln(backbone_features)
        x = self.vl_self_attention(x)
        return x


def export_action_head(
    policy,
    ONNX_export_path,
    input_state,
    attention_mask,
    dit_dtype="fp16",
    calib_dataset_path=None,
    modality_configs=None,
    embodiment_tag="gr1",
    calib_size=10,
    denoising_steps=4,
    data_config="fourier_gr1_arms_only",
    model_path="nvidia/GR00T-N1.5-3B",
    video_backend="decord",
):
    """
    Export the action head models to ONNX format with optional DiT quantization.
    """
    process_backbone_model = (
        VLLN_VLSelfAttention(
            policy.model.action_head.vlln, policy.model.action_head.vl_self_attention
        )
        .to(torch.float16)
        .cuda()
    )
    backbone_features = torch.randn(
        (1, attention_mask.shape[1], policy.model.action_head.config.backbone_embedding_dim),
        dtype=torch.float16,
    ).cuda()

    torch.onnx.export(
        process_backbone_model,
        (backbone_features),
        os.path.join(ONNX_export_path, "action_head/vlln_vl_self_attention.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["backbone_features"],
        output_names=["output"],
        dynamic_axes={
            "backbone_features": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        },
    )

    state_encoder = policy.model.action_head.state_encoder.to(torch.float16)

    state_tensor = torch.randn(
        (1, input_state.shape[1], input_state.shape[2]), dtype=torch.float16
    ).cuda()
    embodiment_id_tensor = torch.ones((1), dtype=torch.int64).cuda()

    torch.onnx.export(
        state_encoder,
        (state_tensor, embodiment_id_tensor),
        os.path.join(ONNX_export_path, "action_head/state_encoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["state", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    action_encoder = policy.model.action_head.action_encoder.to(torch.float16)
    actions_tensor = torch.randn(
        (
            1,
            policy.model.action_head.config.action_horizon,
            policy.model.action_head.config.action_dim,
        ),
        dtype=torch.float16,
    ).cuda()
    timesteps_tensor = torch.ones((1), dtype=torch.int64).cuda()

    torch.onnx.export(
        action_encoder,
        (actions_tensor, timesteps_tensor, embodiment_id_tensor),
        os.path.join(ONNX_export_path, "action_head/action_encoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["actions", "timesteps_tensor", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "actions": {0: "batch_size"},
            "timesteps_tensor": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    dit_model_dtype = torch.float16
    DiT = policy.model.action_head.model.to(dit_model_dtype).cuda()

    if dit_dtype in ["fp8", "int8"]:
        print(f"Quantizing DiT to {dit_dtype}")
        dataset_path_for_calib = (
            calib_dataset_path if calib_dataset_path is not None else "dummy_path"
        )
        DiT = quantize_dit(
            DiT,
            precision=dit_dtype,
            calib_size=calib_size,
            dataset_path=dataset_path_for_calib,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            policy=policy,
            attention_mask=attention_mask,
            input_state=input_state,
            denoising_steps=denoising_steps,
            data_config=data_config,
            model_path=model_path,
            video_backend=video_backend,
        )

    sa_embs_tensor = torch.randn(
        (
            1,
            input_state.shape[1]
            + policy.model.action_head.config.action_horizon
            + policy.model.action_head.config.num_target_vision_tokens,
            policy.model.action_head.config.input_embedding_dim,
        ),
        dtype=dit_model_dtype,
    ).cuda()
    vl_embs_tensor = torch.randn(
        (1, attention_mask.shape[1], policy.model.action_head.config.backbone_embedding_dim),
        dtype=dit_model_dtype,
    ).cuda()

    onnx_path = os.path.join(ONNX_export_path, f"action_head/DiT_{dit_dtype}.onnx")
    torch.onnx.export(
        DiT,
        (sa_embs_tensor, vl_embs_tensor, timesteps_tensor),
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_names=["sa_embs", "vl_embs", "timesteps_tensor"],
        output_names=["output"],
        dynamic_axes={
            "sa_embs": {0: "batch_size"},
            "vl_embs": {0: "batch_size", 1: "sequence_length"},
            "timesteps_tensor": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"DiT ONNX exported to {onnx_path}")

    if dit_dtype in ["fp8", "int8"]:
        _print_onnx_file_size(onnx_path, tag=f"DiT {dit_dtype}")
        _inspect_onnx_qdq(onnx_path, tag=f"DiT {dit_dtype}")
        _inspect_onnx_initializer_dtypes(onnx_path, tag=f"DiT {dit_dtype}")

    action_decoder = policy.model.action_head.action_decoder.to(torch.float16)
    model_output_tensor = torch.randn(
        (
            1,
            input_state.shape[1]
            + policy.model.action_head.config.action_horizon
            + policy.model.action_head.config.num_target_vision_tokens,
            policy.model.action_head.config.hidden_size,
        ),
        dtype=torch.float16,
    ).cuda()
    torch.onnx.export(
        action_decoder,
        (model_output_tensor, embodiment_id_tensor),
        os.path.join(ONNX_export_path, "action_head/action_decoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["model_output", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "model_output": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


def run_groot_inference(
    dataset_path: str,
    model_path: str,
    onnx_model_path: str,
    data_config: str = "fourier_gr1_arms_only",
    embodiment_tag: str = "gr1",
    denoising_steps: int = 4,
    device: str = "cuda",
    llm_dtype: str = "fp16",
    vit_dtype: str = "fp16",
    dit_dtype: str = "fp16",
    calib_dataset_path: str = None,
    calib_size: int = 32,
    video_backend: str = "decord",
    full_layer_quant: bool = False,
    int8_algo: str = "smoothquant",
) -> Dict[str, float]:

    data_config_obj = load_data_config(data_config)
    modality_config = data_config_obj.modality_config()
    modality_transform = data_config_obj.transform()

    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=denoising_steps,
        device=device,
    )
    modality_config = policy.modality_config

    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        video_backend_kwargs=None,
        transforms=None,
    )

    step_data = dataset[0]
    predicted_action = policy.get_action(step_data)

    attention_mask, state = get_input_info(policy, step_data)

    os.makedirs(onnx_model_path, exist_ok=True)
    os.makedirs(os.path.join(onnx_model_path, "eagle2"), exist_ok=True)
    os.makedirs(os.path.join(onnx_model_path, "action_head"), exist_ok=True)

    export_eagle2_vit(
        policy.model.backbone.eagle_model.vision_model.vision_model,
        onnx_model_path,
        vit_dtype=vit_dtype,
        calib_dataset_path=calib_dataset_path or dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        calib_size=calib_size,
        policy=policy,
        denoising_steps=denoising_steps,
        data_config=data_config,
        model_path=model_path,
        video_backend=video_backend,
    )
    export_eagle2_llm(
        policy.model.backbone,
        policy.model.config.backbone_cfg,
        onnx_model_path,
        attention_mask,
        llm_dtype,
        calib_dataset_path=calib_dataset_path or dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        calib_size=calib_size,
        policy=policy,
        denoising_steps=denoising_steps,
        data_config=data_config,
        model_path=model_path,
        video_backend=video_backend,
        full_layer_quant=full_layer_quant,
        int8_algo=int8_algo,
    )
    export_action_head(
        policy,
        onnx_model_path,
        state,
        attention_mask,
        dit_dtype=dit_dtype,
        calib_dataset_path=calib_dataset_path or dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        calib_size=calib_size,
        denoising_steps=denoising_steps,
        data_config=data_config,
        model_path=model_path,
        video_backend=video_backend,
    )

    return predicted_action


if __name__ == "__main__":
    # Make sure you have logged in to huggingface using `huggingface-cli login` with your nvidia email.
    parser = build_parser()
    args = parser.parse_args()

    print(f"Dataset path: {args.dataset_path}")
    print(f"Model path: {args.model_path}")
    print(f"ONNX model path: {args.onnx_model_path}")
    print(f"Data config: {args.data_config}")
    print(f"Embodiment tag: {args.embodiment_tag}")
    print(f"Denoising steps: {args.denoising_steps}")
    print(f"Video backend: {args.video_backend}")
    print(f"LLM data type: {args.llm_dtype}")
    print(f"ViT data type: {args.vit_dtype}")
    print(f"DiT data type: {args.dit_dtype}")
    print(f"Calibration dataset path: {args.calib_dataset_path or args.dataset_path}")
    print(f"Calibration size: {args.calib_size}")
    print(f"Full layer quantization: {args.full_layer_quant}")
    print(f"INT8 algorithm: {args.int8_algo}")

    predicted_action = run_groot_inference(
        args.dataset_path,
        args.model_path,
        args.onnx_model_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        llm_dtype=args.llm_dtype,
        vit_dtype=args.vit_dtype,
        dit_dtype=args.dit_dtype,
        calib_dataset_path=args.calib_dataset_path,
        calib_size=args.calib_size,
        video_backend=args.video_backend,
        full_layer_quant=args.full_layer_quant,
        int8_algo=args.int8_algo,
    )

    for key, value in predicted_action.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
