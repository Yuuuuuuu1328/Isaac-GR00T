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

#!/bin/bash
set -euo pipefail

echo "Building fixed-profile TensorRT engines for new_interaction_group."
echo "Validated runtime target:"
echo "  MAX_BATCH=1"
echo "  MIN_LEN=OPT_LEN=MAX_LEN=283"
echo "  Precision: fp16 / fp16 / fp16"

export PATH=/usr/src/tensorrt/bin:$PATH

ONNX_ROOT=${ONNX_ROOT:-/home/jetson/Desktop/project/Isaac-GR00T/gr00t_onnx_new_interaction_group_fp16}
ENGINE_ROOT=${ENGINE_ROOT:-/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16_bs1_len283}
VIDEO_VIEWS=${VIDEO_VIEWS:-1}
VIT_DTYPE=${VIT_DTYPE:-fp16}
LLM_DTYPE=${LLM_DTYPE:-fp16}
DIT_DTYPE=${DIT_DTYPE:-fp16}
MAX_BATCH=${MAX_BATCH:-1}
MIN_LEN=${MIN_LEN:-283}
OPT_LEN=${OPT_LEN:-283}
MAX_LEN=${MAX_LEN:-283}

echo "Building TensorRT engines with the following configurations:"
echo "  ViT: ${VIT_DTYPE}"
echo "  LLM: ${LLM_DTYPE}"
echo "  DiT: ${DIT_DTYPE}"
echo "  ONNX_ROOT: ${ONNX_ROOT}"
echo "  ENGINE_ROOT: ${ENGINE_ROOT}"
echo "  Video Views: ${VIDEO_VIEWS}"
echo "  MAX_BATCH: ${MAX_BATCH}"
echo "  MIN_LEN: ${MIN_LEN}"
echo "  OPT_LEN: ${OPT_LEN}"
echo "  MAX_LEN: ${MAX_LEN}"

if [[ ! "$VIT_DTYPE" =~ ^(fp16)$ ]]; then
    echo "Error: VIT_DTYPE must be 'fp16', got '${VIT_DTYPE}'"
    exit 1
fi

if [[ ! "$LLM_DTYPE" =~ ^(fp16)$ ]]; then
    echo "Error: LLM_DTYPE must be 'fp16', got '${LLM_DTYPE}'"
    exit 1
fi

if [[ ! "$DIT_DTYPE" =~ ^(fp16)$ ]]; then
    echo "Error: DIT_DTYPE must be 'fp16', got '${DIT_DTYPE}'"
    exit 1
fi

if [[ "$VIDEO_VIEWS" != "1" ]]; then
    echo "Error: VIDEO_VIEWS must be '1' for the fixed new_interaction_group profile, got '${VIDEO_VIEWS}'"
    exit 1
fi

if [[ "$MAX_BATCH" != "1" ]]; then
    echo "Error: MAX_BATCH must be '1' for the fixed new_interaction_group profile, got '${MAX_BATCH}'"
    exit 1
fi

if [[ "$MIN_LEN" != "283" || "$OPT_LEN" != "283" || "$MAX_LEN" != "283" ]]; then
    echo "Error: MIN_LEN, OPT_LEN, and MAX_LEN must all be '283' for the fixed profile."
    exit 1
fi

required_onnx_files=(
    "${ONNX_ROOT}/action_head/vlln_vl_self_attention.onnx"
    "${ONNX_ROOT}/action_head/DiT_${DIT_DTYPE}.onnx"
    "${ONNX_ROOT}/action_head/state_encoder.onnx"
    "${ONNX_ROOT}/action_head/action_encoder.onnx"
    "${ONNX_ROOT}/action_head/action_decoder.onnx"
    "${ONNX_ROOT}/eagle2/vit_${VIT_DTYPE}.onnx"
    "${ONNX_ROOT}/eagle2/llm_${LLM_DTYPE}.onnx"
)

for onnx_file in "${required_onnx_files[@]}"; do
    if [ ! -f "${onnx_file}" ]; then
        echo "Error: required ONNX file not found: ${onnx_file}"
        exit 1
    fi
done

if [ ! -e /usr/src/tensorrt/bin/trtexec ]; then
    echo "The file /usr/src/tensorrt/bin/trtexec does not exist. Please install tensorrt"
    exit 1
fi

mkdir -p "${ENGINE_ROOT}"

echo "------------Building vlln_vl_self_attention Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=${ONNX_ROOT}/action_head/vlln_vl_self_attention.onnx --saveEngine=${ENGINE_ROOT}/vlln_vl_self_attention.engine --minShapes=backbone_features:1x${MIN_LEN}x2048 --optShapes=backbone_features:1x${OPT_LEN}x2048 --maxShapes=backbone_features:${MAX_BATCH}x${MAX_LEN}x2048 > ${ENGINE_ROOT}/vlln_vl_self_attention.log 2>&1

echo "------------Building DiT Model (${DIT_DTYPE})--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=${ONNX_ROOT}/action_head/DiT_${DIT_DTYPE}.onnx --saveEngine=${ENGINE_ROOT}/DiT_${DIT_DTYPE}.engine --minShapes=sa_embs:1x49x1536,vl_embs:1x${MIN_LEN}x2048,timesteps_tensor:1 --optShapes=sa_embs:1x49x1536,vl_embs:1x${OPT_LEN}x2048,timesteps_tensor:1 --maxShapes=sa_embs:${MAX_BATCH}x49x1536,vl_embs:${MAX_BATCH}x${MAX_LEN}x2048,timesteps_tensor:${MAX_BATCH} > ${ENGINE_ROOT}/DiT_${DIT_DTYPE}.log 2>&1

echo "------------Building State Encoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=${ONNX_ROOT}/action_head/state_encoder.onnx --saveEngine=${ENGINE_ROOT}/state_encoder.engine --minShapes=state:1x1x64,embodiment_id:1 --optShapes=state:1x1x64,embodiment_id:1 --maxShapes=state:${MAX_BATCH}x1x64,embodiment_id:${MAX_BATCH} > ${ENGINE_ROOT}/state_encoder.log 2>&1

echo "------------Building Action Encoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=${ONNX_ROOT}/action_head/action_encoder.onnx --saveEngine=${ENGINE_ROOT}/action_encoder.engine --minShapes=actions:1x16x32,timesteps_tensor:1,embodiment_id:1 --optShapes=actions:1x16x32,timesteps_tensor:1,embodiment_id:1 --maxShapes=actions:${MAX_BATCH}x16x32,timesteps_tensor:${MAX_BATCH},embodiment_id:${MAX_BATCH} > ${ENGINE_ROOT}/action_encoder.log 2>&1

echo "------------Building Action Decoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=${ONNX_ROOT}/action_head/action_decoder.onnx --saveEngine=${ENGINE_ROOT}/action_decoder.engine --minShapes=model_output:1x49x1024,embodiment_id:1 --optShapes=model_output:1x49x1024,embodiment_id:1 --maxShapes=model_output:${MAX_BATCH}x49x1024,embodiment_id:${MAX_BATCH} > ${ENGINE_ROOT}/action_decoder.log 2>&1

echo "------------Building VLM-ViT (${VIT_DTYPE})--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers --onnx=${ONNX_ROOT}/eagle2/vit_${VIT_DTYPE}.onnx --saveEngine=${ENGINE_ROOT}/vit_${VIT_DTYPE}.engine --minShapes=pixel_values:1x3x224x224,position_ids:1x256 --optShapes=pixel_values:${VIDEO_VIEWS}x3x224x224,position_ids:${VIDEO_VIEWS}x256 --maxShapes=pixel_values:${MAX_BATCH}x3x224x224,position_ids:${MAX_BATCH}x256 > ${ENGINE_ROOT}/vit_${VIT_DTYPE}.log 2>&1

echo "------------Building VLM-LLM (${LLM_DTYPE})--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
    --onnx=${ONNX_ROOT}/eagle2/llm_${LLM_DTYPE}.onnx \
    --saveEngine=${ENGINE_ROOT}/llm_${LLM_DTYPE}.engine \
    --minShapes=inputs_embeds:1x${MIN_LEN}x2048,attention_mask:1x${MIN_LEN} \
    --optShapes=inputs_embeds:1x${OPT_LEN}x2048,attention_mask:1x${OPT_LEN} \
    --maxShapes=inputs_embeds:${MAX_BATCH}x${MAX_LEN}x2048,attention_mask:${MAX_BATCH}x${MAX_LEN} \
    > ${ENGINE_ROOT}/llm_${LLM_DTYPE}.log 2>&1

echo ""
echo "============================================================"
echo "Fixed-profile TensorRT engine build complete!"
echo "============================================================"
echo "Engines saved in: ${ENGINE_ROOT}"
echo "Build logs saved in: ${ENGINE_ROOT}/*.log"
echo "============================================================"
