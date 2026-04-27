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
GR00T Inference Service

This script provides both ZMQ and HTTP server/client implementations for deploying GR00T models.
The HTTP server exposes a REST API for easy integration with web applications and other services.

1. Default is zmq server.

Run server: python scripts/inference_service.py --server
Run client: python scripts/inference_service.py --client

2. Run as Http Server:

Dependencies for `http_server` mode:
    => Server (runs GR00T model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

HTTP Server Usage:
    python scripts/inference_service.py --server --http-server --port 8000

HTTP Client Usage (assuming a server running on 0.0.0.0:8000):
    python scripts/inference_service.py --client --http-server --host 0.0.0.0 --port 8000

You can use bore to forward the port to your client: `159.223.171.199` is bore.pub.
    bore local 8000 --to 159.223.171.199

3. TensorRT Support:

For accelerated inference using TensorRT, first build the TensorRT engines using the deployment scripts,
then run the server with the --use-tensorrt flag:

TensorRT Server Usage:
    python scripts/inference_service.py --server --use-tensorrt --trt-engine-path gr00t_engine

TensorRT HTTP Server Usage:
    python scripts/inference_service.py --server --http-server --use-tensorrt --trt-engine-path gr00t_engine --port 8000

Note: TensorRT engines must be built before running with --use-tensorrt flag.
See deployment_scripts/README.md for instructions on building TensorRT engines.

4. Local Inference Mode:
Run local inference with latency measurement:
    python scripts/inference_service.py --local-inference
    # With TensorRT acceleration
    python scripts/inference_service.py --local-inference --use-tensorrt --trt-engine-path gr00t_engine
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = "/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """The embodiment tag for the model."""

    data_config: str = "fourier_gr1_arms_waist"
    """
    The name of the data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.

    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    port: int = 5555
    """The port number for the server."""

    host: str = "localhost"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    local_inference: bool = False
    """Whether to run local inference (no server/client) and measure latency."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    api_token: str = None
    """API token for authentication. If not provided, authentication is disabled."""

    http_server: bool = False
    """Whether to run it as HTTP server. Default is ZMQ server."""

    use_tensorrt: bool = False
    """Whether to use TensorRT for inference. Requires TensorRT engines to be built."""

    trt_engine_path: str = "gr00t_engine_fp16"
    """Path to the TensorRT engine directory. Only used when use_tensorrt is True."""

    vit_dtype: Literal["fp16", "fp8"] = "fp16"
    """ViT model dtype (fp16, fp8). Only used when use_tensorrt is True."""

    llm_dtype: Literal["fp16", "nvfp4", "fp8"] = "fp16"
    """LLM model dtype (fp16, nvfp4, fp8). Only used when use_tensorrt is True."""

    dit_dtype: Literal["fp16", "fp8"] = "fp16"
    """DiT model dtype (fp16, fp8). Only used when use_tensorrt is True."""

    warmup_runs: int = 5
    """Number of warmup runs before measuring latency (to exclude initialization overhead)."""

    num_runs: int = 10
    """Number of inference runs to measure average latency."""


#####################################################################################


def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example ZMQ client call to the server.
    """
    # Original ZMQ client mode
    # Create a policy wrapper
    policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)

    print("Available modality config available:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())

    time_start = time.time()
    action = policy_client.get_action(obs)
    print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
    return action


def _example_http_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example HTTP client call to the server.
    """
    import json_numpy

    json_numpy.patch()
    import requests

    # Send request to HTTP server
    print("Testing HTTP server...")

    time_start = time.time()
    response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
    print(f"Total time taken to get action from HTTP server: {time.time() - time_start} seconds")

    if response.status_code == 200:
        action = response.json()
        return action
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


def run_local_inference(args: ArgsConfig):
    """
    Run local inference with detailed latency measurement.
    """
    # Load data config and create policy
    print(f"Loading data config: {args.data_config}")
    data_config = load_data_config(args.data_config)
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # Create policy instance
    print(f"Initializing GR00T policy from: {args.model_path}")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    # Setup TensorRT if requested
    if args.use_tensorrt:
        print(f"\nSetting up TensorRT engines from: {args.trt_engine_path}")
        print(f"  ViT dtype: {args.vit_dtype}")
        print(f"  LLM dtype: {args.llm_dtype}")
        print(f"  DiT dtype: {args.dit_dtype}")
        from deployment_scripts.trt_model_forward import setup_tensorrt_engines

        setup_tensorrt_engines(
            policy, args.trt_engine_path, args.vit_dtype, args.llm_dtype, args.dit_dtype
        )
        print("TensorRT engines loaded successfully!\n")

    # Create test observation
    obs = {
        "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "state.left_arm": np.random.rand(1, 7),
        "state.right_arm": np.random.rand(1, 7),
        "state.left_hand": np.random.rand(1, 6),
        "state.right_hand": np.random.rand(1, 6),
        "state.waist": np.random.rand(1, 3),
        "annotation.human.action.task_description": ["do your thing!"],
    }

    # Warmup runs (to exclude initialization overhead)
    print(f"Running {args.warmup_runs} warmup iterations...")
    for _ in range(args.warmup_runs):
        _ = policy.get_action(obs)

    # Measure inference latency
    print(f"\nRunning {args.num_runs} inference iterations to measure latency...")
    latencies = []
    
    for i in range(args.num_runs):
        time_start = time.perf_counter()  # Use high-precision timer
        action = policy.get_action(obs)
        time_end = time.perf_counter()
        
        latency = (time_end - time_start) * 1000  # Convert to milliseconds
        latencies.append(latency)
        print(f"Iteration {i+1}: {latency:.2f} ms")

    # Calculate statistics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    # Print results
    print("\n" + "="*50)
    print("Inference Latency Results (ms)")
    print("="*50)
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Standard deviation: {std_latency:.2f} ms")
    print(f"Minimum latency: {min_latency:.2f} ms")
    print(f"Maximum latency: {max_latency:.2f} ms")
    print("="*50)

    # Print action shapes
    print("\nAction Output Shapes:")
    for key, value in action.items():
        print(f"  {key}: {np.array(value).shape}")

    return action, latencies


def main(args: ArgsConfig):
    if args.server:
        # Original server logic
        data_config = load_data_config(args.data_config)
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )

        if args.use_tensorrt:
            print(f"Setting up TensorRT engines from: {args.trt_engine_path}")
            print(f"  ViT dtype: {args.vit_dtype}")
            print(f"  LLM dtype: {args.llm_dtype}")
            print(f"  DiT dtype: {args.dit_dtype}")
            from deployment_scripts.trt_model_forward import setup_tensorrt_engines

            setup_tensorrt_engines(
                policy, args.trt_engine_path, args.vit_dtype, args.llm_dtype, args.dit_dtype
            )
            print("TensorRT engines loaded successfully!")

        if args.http_server:
            from gr00t.eval.http_server import HTTPInferenceServer  # noqa: F401

            server = HTTPInferenceServer(
                policy, port=args.port, host=args.host, api_token=args.api_token
            )
            server.run()
        else:
            server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
            server.run()

    elif args.client:
        # Original client logic
        obs = {
            "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }

        if args.http_server:
            action = _example_http_client_call(obs, args.host, args.port, args.api_token)
        else:
            action = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")
    
    elif args.local_inference:
        # New local inference mode
        run_local_inference(args)
        
    else:
        raise ValueError("Please specify either --server, --client, or --local-inference")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
