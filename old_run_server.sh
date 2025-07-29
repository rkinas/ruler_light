#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME

if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_name> $1 <benchmark_name> $2 <gpus> $3"
    exit 1
fi


# Root Directories
GPUS=${3} # GPU size for tensor_parallel.
MODEL_DIR="models" # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.



# Model and Tokenizer
source config_models.sh
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

echo "Using model: ${MODEL_NAME} at ${MODEL_PATH} with template type: ${MODEL_TEMPLATE_TYPE} and framework: ${MODEL_FRAMEWORK}"

export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}
export NCCL_P2P_DISABLE=1


# Benchmark and Tasks
source config_tasks.sh
BENCHMARK=${2}
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi


# Start server (you may want to run in other container.)
if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
    python pred/serve_vllm.py \
        --model=${MODEL_PATH} \
        --tensor-parallel-size=${GPUS} \
        --dtype bfloat16 #--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":8192}' --max-model-len 32768 #\
        # --disable-custom-all-reduce 

elif [ "$MODEL_FRAMEWORK" == "trtllm" ]; then
    python pred/serve_trt.py \
        --model_path=${MODEL_PATH} 

elif [ "$MODEL_FRAMEWORK" == "sglang" ]; then
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp ${GPUS} \
        --port 5000 \
        --enable-flashinfer 
    # use sglang/test/killall_sglang.sh to kill sglang server if it hangs
fi
