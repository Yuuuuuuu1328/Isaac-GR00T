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

"""
Build TensorRT engines from ONNX models using the Python TensorRT API.
Replaces the shell script build_engine.sh / build_engine_new_interaction_group_int8.sh
so that trtexec binary is not required.

Usage:
    python deployment_scripts/build_engine.py \
        --onnx-root gr00t_onnx_int8_new \
        --engine-root gr00t_engine_int8_new \
        --vit-dtype int8 --llm-dtype int8 --dit-dtype int8

    python deployment_scripts/build_engine.py \
        --onnx-root gr00t_onnx \
        --engine-root gr00t_engine \
        --vit-dtype fp8 --llm-dtype nvfp4 --dit-dtype fp8
"""

import argparse
import os
import sys
import time

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def parse_shape(shape_str: str) -> tuple[str, list[int]]:
    """Parse 'name:1x49x1536' into ('name', [1, 49, 1536])."""
    name, dims = shape_str.split(":")
    return name, [int(d) for d in dims.split("x")]


def build_engine(
    onnx_path: str,
    engine_path: str,
    min_shapes: dict[str, list[int]],
    opt_shapes: dict[str, list[int]],
    max_shapes: dict[str, list[int]],
    strongly_typed: bool = True,
    use_cuda_graph: bool = True,
):
    """Build a TensorRT engine from an ONNX model with dynamic shape profiles."""
    print(f"\n{'='*60}")
    print(f"Building: {os.path.basename(engine_path)}")
    print(f"  ONNX: {onnx_path}")
    print(f"  Engine: {engine_path}")
    print(f"{'='*60}")

    if not os.path.isfile(onnx_path):
        print(f"  ERROR: ONNX file not found: {onnx_path}")
        return False

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if strongly_typed:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"  Parsing ONNX model...")
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f"  Parse error {i}: {parser.get_error(i)}")
        return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GB

    if use_cuda_graph and not strongly_typed:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    profile = builder.create_optimization_profile()
    for name in min_shapes:
        profile.set_shape(name, min_shapes[name], opt_shapes[name], max_shapes[name])
        print(f"  Profile '{name}': min={min_shapes[name]} opt={opt_shapes[name]} max={max_shapes[name]}")

    config.add_optimization_profile(profile)

    print(f"  Building engine (this may take a while)...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print(f"  ERROR: Engine build failed!")
        return False

    elapsed = time.time() - t0
    engine_bytes = bytes(serialized_engine)
    print(f"  Engine built in {elapsed:.1f}s, size: {len(engine_bytes) / 1024 / 1024:.1f} MB")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"  Saved: {engine_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX (no trtexec needed)")
    parser.add_argument("--onnx-root", type=str, default="gr00t_onnx", help="ONNX model root directory")
    parser.add_argument("--engine-root", type=str, default="gr00t_engine", help="Output engine directory")
    parser.add_argument("--vit-dtype", type=str, default="fp8", choices=["fp16", "fp8", "int8"])
    parser.add_argument("--llm-dtype", type=str, default="nvfp4", choices=["fp16", "nvfp4", "nvfp4_full", "fp8", "int8"])
    parser.add_argument("--dit-dtype", type=str, default="fp8", choices=["fp16", "fp8", "int8"])
    parser.add_argument("--full-layer-quant", action="store_true", help="Match export_onnx_now.py --full-layer-quant: use llm_{dtype}_full.onnx filename")
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--video-views", type=int, default=1, choices=[1, 2])
    parser.add_argument("--only", type=str, default="", help="Comma-separated list of model names to build (e.g. vit_int8,llm_int8). Empty means all.")
    args = parser.parse_args()

    if args.video_views == 2:
        min_len, opt_len, max_len = 80, 568, 600
    else:
        min_len, opt_len, max_len = 80, 296, 300

    max_batch = args.max_batch
    llm_max_batch = 1 if args.llm_dtype.startswith("nvfp4") else max_batch

    llm_suffix = (
        f"{args.llm_dtype}_full"
        if (args.llm_dtype in ("nvfp4", "int8") and args.full_layer_quant)
        else args.llm_dtype
    )

    onnx_root = args.onnx_root
    engine_root = args.engine_root

    print(f"TensorRT Engine Builder (Python API)")
    print(f"  TensorRT version: {trt.__version__}")
    print(f"  ONNX root: {onnx_root}")
    print(f"  Engine root: {engine_root}")
    print(f"  ViT: {args.vit_dtype}, LLM: {args.llm_dtype}, DiT: {args.dit_dtype}")
    print(f"  Video views: {args.video_views}")
    print(f"  Max batch: {max_batch}, LLM max batch: {llm_max_batch}")
    print(f"  Sequence lengths: min={min_len}, opt={opt_len}, max={max_len}")

    os.makedirs(engine_root, exist_ok=True)

    models = [
        {
            "name": "vlln_vl_self_attention",
            "onnx": f"{onnx_root}/action_head/vlln_vl_self_attention.onnx",
            "engine": f"{engine_root}/vlln_vl_self_attention.engine",
            "min": {"backbone_features": [1, min_len, 2048]},
            "opt": {"backbone_features": [1, opt_len, 2048]},
            "max": {"backbone_features": [max_batch, max_len, 2048]},
        },
        {
            "name": f"DiT_{args.dit_dtype}",
            "onnx": f"{onnx_root}/action_head/DiT_{args.dit_dtype}.onnx",
            "engine": f"{engine_root}/DiT_{args.dit_dtype}.engine",
            "min": {"sa_embs": [1, 49, 1536], "vl_embs": [1, min_len, 2048], "timesteps_tensor": [1]},
            "opt": {"sa_embs": [1, 49, 1536], "vl_embs": [1, opt_len, 2048], "timesteps_tensor": [1]},
            "max": {"sa_embs": [max_batch, 49, 1536], "vl_embs": [max_batch, max_len, 2048], "timesteps_tensor": [max_batch]},
        },
        {
            "name": "state_encoder",
            "onnx": f"{onnx_root}/action_head/state_encoder.onnx",
            "engine": f"{engine_root}/state_encoder.engine",
            "min": {"state": [1, 1, 64], "embodiment_id": [1]},
            "opt": {"state": [1, 1, 64], "embodiment_id": [1]},
            "max": {"state": [max_batch, 1, 64], "embodiment_id": [max_batch]},
        },
        {
            "name": "action_encoder",
            "onnx": f"{onnx_root}/action_head/action_encoder.onnx",
            "engine": f"{engine_root}/action_encoder.engine",
            "min": {"actions": [1, 16, 32], "timesteps_tensor": [1], "embodiment_id": [1]},
            "opt": {"actions": [1, 16, 32], "timesteps_tensor": [1], "embodiment_id": [1]},
            "max": {"actions": [max_batch, 16, 32], "timesteps_tensor": [max_batch], "embodiment_id": [max_batch]},
        },
        {
            "name": "action_decoder",
            "onnx": f"{onnx_root}/action_head/action_decoder.onnx",
            "engine": f"{engine_root}/action_decoder.engine",
            "min": {"model_output": [1, 49, 1024], "embodiment_id": [1]},
            "opt": {"model_output": [1, 49, 1024], "embodiment_id": [1]},
            "max": {"model_output": [max_batch, 49, 1024], "embodiment_id": [max_batch]},
        },
        {
            "name": f"vit_{args.vit_dtype}",
            "onnx": f"{onnx_root}/eagle2/vit_{args.vit_dtype}.onnx",
            "engine": f"{engine_root}/vit_{args.vit_dtype}.engine",
            "min": {"pixel_values": [1, 3, 224, 224], "position_ids": [1, 256]},
            "opt": {"pixel_values": [args.video_views, 3, 224, 224], "position_ids": [args.video_views, 256]},
            "max": {"pixel_values": [max_batch, 3, 224, 224], "position_ids": [max_batch, 256]},
        },
        {
            "name": f"llm_{llm_suffix}",
            "onnx": f"{onnx_root}/eagle2/llm_{llm_suffix}.onnx",
            "engine": f"{engine_root}/llm_{llm_suffix}.engine",
            "min": {"inputs_embeds": [1, min_len, 2048], "attention_mask": [1, min_len]},
            "opt": {"inputs_embeds": [1, opt_len, 2048], "attention_mask": [1, opt_len]},
            "max": {"inputs_embeds": [llm_max_batch, max_len, 2048], "attention_mask": [llm_max_batch, max_len]},
        },
    ]

    only_set = set(s.strip() for s in args.only.split(",") if s.strip()) if args.only else set()

    results = []
    for model in models:
        if only_set and model["name"] not in only_set:
            continue
        success = build_engine(
            onnx_path=model["onnx"],
            engine_path=model["engine"],
            min_shapes=model["min"],
            opt_shapes=model["opt"],
            max_shapes=model["max"],
        )
        results.append((model["name"], success))

    print(f"\n{'='*60}")
    print("Build Summary:")
    print(f"{'='*60}")
    all_ok = True
    for name, success in results:
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")
        if not success:
            all_ok = False

    print(f"\nEngines saved in: {engine_root}/")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
