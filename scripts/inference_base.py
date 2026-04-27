import time
from dataclasses import dataclass
from typing import Literal, Dict, Any

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
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    data_config: str = "fourier_gr1_arms_waist"
    port: int = 5555
    host: str = "localhost"
    server: bool = False
    client: bool = False
    denoising_steps: int = 4
    api_token: str = None
    http_server: bool = False
    use_tensorrt: bool = False
    trt_engine_path: str = "gr00t_engine_fp16"
    vit_dtype: Literal["fp16", "fp8"] = "fp16"
    llm_dtype: Literal["fp16", "nvfp4", "fp8"] = "fp16"
    dit_dtype: Literal["fp16", "fp8"] = "fp16"


class TimedRobotInferenceServer(RobotInferenceServer):
    """Extended inference server with latency measurement."""
    
    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Override get_action method to measure pure inference latency."""
        infer_start = time.time()
        action = super().get_action(obs)
        infer_latency = time.time() - infer_start
        
        # Ensure proper inference latency is being captured
        if infer_latency == 0.0:
            print("Warning: Inference latency is 0.0, check the inference process.")
        
        action["__infer_latency__"] = infer_latency
        print(f"Inference latency: {infer_latency * 1000:.5f} ms")
        return action


class TimedHTTPInferenceServer:
    """HTTP inference server with latency measurement."""
    
    def __init__(self, policy, port: int, host: str = "0.0.0.0", api_token: str = None):
        self.policy = policy
        self.port = port
        self.host = host
        self.api_token = api_token
        
    def run(self):
        import uvicorn
        from fastapi import FastAPI, HTTPException, Depends, Header
        from fastapi.responses import JSONResponse
        import json_numpy
        
        json_numpy.patch()
        app = FastAPI(title="GR00T Inference Server", version="1.0")
        
        def authenticate(x_api_token: str = Header(None)):
            if self.api_token and x_api_token != self.api_token:
                raise HTTPException(status_code=401, detail="Invalid API token")
            return True
        
        @app.post("/act")
        async def act(request: Dict[str, Any], authenticated: bool = Depends(authenticate if self.api_token else lambda: True)):
            try:
                obs = request.get("observation", {})
                infer_start = time.time()
                action = self.policy.get_action(obs)
                infer_latency = time.time() - infer_start
                
                # Ensure proper inference latency is being captured
                if infer_latency == 0.0:
                    print("Warning: Inference latency is 0.0, check the inference process.")
                
                print(f"Inference latency: {infer_latency * 1000:.5f} ms")
                return JSONResponse({**action, "__infer_latency__": infer_latency})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        uvicorn.run(app, host=self.host, port=self.port)


def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str, warmup_runs: int = 5, actual_runs: int = 10):
    policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)

    print("Available modality config:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())

    print(f"Performing {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        _ = policy_client.get_action(obs)

    print(f"Performing {actual_runs} actual inference runs...")
    total_latency_sum = 0
    infer_latency_sum = 0
    network_latency_sum = 0

    for i in range(actual_runs):
        total_start = time.time()
        action = policy_client.get_action(obs)
        total_latency = (time.time() - total_start) * 1000

        infer_latency = action.pop("__infer_latency__", 0.0) * 1000
        network_latency = total_latency - infer_latency

        total_latency_sum += total_latency
        infer_latency_sum += infer_latency
        network_latency_sum += network_latency

        print(f"\n=== Run {i+1} ===")
        print(f"📊 Total latency: {total_latency:.5f} ms")
        print(f"⚡ Inference latency: {infer_latency:.5f} ms")
        print(f"🌐 Network latency: {network_latency:.5f} ms")
        print(f"📈 Inference proportion: {(infer_latency/total_latency)*100:.2f}%")

    print("\n=== Average Latency ===")
    print(f"📊 Average total latency: {total_latency_sum / actual_runs:.5f} ms")
    print(f"⚡ Average inference latency: {infer_latency_sum / actual_runs:.5f} ms")
    print(f"🌐 Average network latency: {network_latency_sum / actual_runs:.5f} ms")
    print(f"📈 Average inference proportion: {(infer_latency_sum / total_latency_sum) * 100:.2f}%")

    return action, {
        "total_latency": total_latency_sum / actual_runs,
        "infer_latency": infer_latency_sum / actual_runs,
        "network_latency": network_latency_sum / actual_runs
    }


def _example_http_client_call(obs: dict, host: str, port: int, api_token: str, warmup_runs: int = 5, actual_runs: int = 10):
    import json_numpy
    json_numpy.patch()
    import requests

    print("Testing HTTP server...")

    headers = {}
    if api_token:
        headers["X-API-Token"] = api_token
    
    print(f"Performing {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        requests.post(f"http://{host}:{port}/act", json={"observation": obs}, headers=headers)

    print(f"Performing {actual_runs} actual inference runs...")
    total_latency_sum = 0
    infer_latency_sum = 0
    network_latency_sum = 0

    for i in range(actual_runs):
        total_start = time.time()
        response = requests.post(f"http://{host}:{port}/act", json={"observation": obs}, headers=headers)
        total_latency = (time.time() - total_start) * 1000

        if response.status_code == 200:
            action = response.json()

            infer_latency = action.pop("__infer_latency__", 0.0) * 1000
            network_latency = total_latency - infer_latency

            total_latency_sum += total_latency
            infer_latency_sum += infer_latency
            network_latency_sum += network_latency

            print(f"\n=== Run {i+1} ===")
            print(f"📊 Total latency: {total_latency:.3f} ms")
            print(f"⚡ Inference latency: {infer_latency:.3f} ms")
            print(f"🌐 Network latency: {network_latency:.3f} ms")
            print(f"📈 Inference proportion: {(infer_latency/total_latency)*100:.2f}%")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    print("\n=== Average Latency ===")
    print(f"📊 Average total latency: {total_latency_sum / actual_runs:.3f} ms")
    print(f"⚡ Average inference latency: {infer_latency_sum / actual_runs:.3f} ms")
    print(f"🌐 Average network latency: {network_latency_sum / actual_runs:.3f} ms")
    print(f"📈 Average inference proportion: {(infer_latency_sum / total_latency_sum) * 100:.2f}%")

    return action, {
        "total_latency": total_latency_sum / actual_runs,
        "infer_latency": infer_latency_sum / actual_runs,
        "network_latency": network_latency_sum / actual_runs
    }


def main(args: ArgsConfig):
    if args.server:
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

        print(f"Starting server on {args.host}:{args.port} with latency measurement...")
        if args.http_server:
            server = TimedHTTPInferenceServer(
                policy, port=args.port, host=args.host, api_token=args.api_token
            )
            server.run()
        else:
            server = TimedRobotInferenceServer(policy, port=args.port, api_token=args.api_token)
            server.run()

    elif args.client:
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
            action, latency_data = _example_http_client_call(obs, args.host, args.port, args.api_token)
        else:
            action, latency_data = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

        #print("\n=== Action Output ===")
        #for key, value in action.items():
        #    if isinstance(value, (np.ndarray, list)):
        #        print(f"Action: {key}: {np.array(value).shape}")
        #    else:
        #        print(f"Action: {key}: {value}")

        return latency_data
    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
