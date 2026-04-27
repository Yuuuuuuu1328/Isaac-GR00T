# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA Isaac GR00T N1.5 — an open foundation model for generalized humanoid robot reasoning and skills. The model uses a frozen Vision-Language Model (Eagle 2.5) backbone with a flow-matching diffusion transformer action head to predict robot actions from visual observations and language instructions.

## Common Commands

### Install
```bash
conda create -n gr00t python=3.10 && conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```

### Format and Lint
```bash
make format          # runs isort + black
make run-checks      # runs isort --check, black --check, ruff check, pytest
ruff check . --fix   # autofix lint issues
```

### Test
```bash
pytest -v --color=yes tests/              # all tests
pytest -v tests/test_dataset.py           # single test file
```

### Key Scripts
```bash
python scripts/gr00t_finetune.py --dataset-path <path> --num-gpus 1
python scripts/inference_service.py --server --model-path nvidia/GR00T-N1.5-3B
python scripts/eval_policy.py --plot --model_path nvidia/GR00T-N1.5-3B
```

## Code Style

- **black**: line-length 100
- **isort**: black profile, multi_line_output 3
- **ruff**: line-length 115, target py310; F401 ignored in `__init__.py`
- All source files must have an SPDX Apache-2.0 license header
- CI runs on Python 3.10 (style, lint, tests)

## Architecture

### Model Pipeline (`gr00t/model/`)

The model (`GR00T_N1_5` in `gr00t_n1.py`) is a HuggingFace `PreTrainedModel` with two components:

1. **Backbone** (`backbone/`): Eagle 2.5 VLM — processes images + language into `backbone_features` (batch, n, hidden_size). Frozen during both pretraining and finetuning.
2. **Action Head** (`action_head/`): `FlowmatchingActionHead` — a diffusion transformer that takes backbone features and predicts actions (batch, time, action_dim). Uses flow matching loss + FLARE objective. This is the trainable component.

`Gr00tPolicy` (`policy.py`) is the main inference wrapper — handles model loading (HuggingFace Hub or local), transforms, and exposes `get_action(observations)`.

### Data Pipeline (`gr00t/data/`)

- `LeRobotSingleDataset` / `LeRobotMixtureDataset` (`dataset.py`): Load LeRobot-format datasets with 4 modalities: video, state, action, language.
- `ModalityConfig`: Defines which keys and time indices each modality uses.
- `EmbodimentTag` (`embodiment_tags.py`): Maps robot types (GR1, OXE_DROID, AGIBOT_GENIE1, NEW_EMBODIMENT) to projector indices in the action expert module.
- `transform/`: Composable data transforms — video resize/crop/jitter, state/action normalization, sin/cos encoding.

### Training (`gr00t/experiment/`)

- `BaseDataConfig` (`data_config.py`): Abstract config defining modality keys, observation/action indices, and transform pipeline. Concrete subclasses exist for each embodiment (fourier_gr1, so100, unitree_g1, etc.).
- `TrainRunner` (`runner.py`): Main training coordinator.
- Finetuning options: full model, LoRA (via `peft`), or selective component tuning (tune_llm, tune_visual, tune_projector).

### Inference & Deployment (`gr00t/eval/`)

- `RobotInferenceServer/Client` (`robot.py`): ZMQ-based real-time inference.
- `http_server.py`: FastAPI REST API alternative.
- Supports PyTorch and TensorRT backends.
- `deployment_scripts/`: ONNX export, TensorRT engine building, Jetson Orin/Thor deployment.

### Adding a New Embodiment

1. Use `EmbodimentTag.NEW_EMBODIMENT` (projector index 31).
2. Create a `BaseDataConfig` subclass defining your robot's modality keys and normalization stats.
3. Finetune with `scripts/gr00t_finetune.py`. See `getting_started/3_1_new_embodiment_finetuning.ipynb`.

## Dependencies

Core: torch 2.5.1, transformers 4.51.3, diffusers 0.30.2, peft 0.17.0. Video decoding uses `decord` on Linux, `eva-decord` on macOS. Install extras: `[base]` (standard), `[dev]` (linting/testing), `[orin]`/`[thor]` (Jetson), `[deploy]` (TensorRT).
