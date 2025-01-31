#!/bin/bash

# Define input and output directories
INPUT_DIR="./mis_test/mis_easy.json"
OUTPUT_DIR="./responses/mis_easy"

# Define the models to test with the corresponding GPU setup (our setting is A100-80G)
declare -A MODELS_GPU
MODELS_GPU=(
    ["qwen2_vl_7b"]="0"                # Model using 1 GPU (GPU 0)
    ["qwen2_vl_72b"]="0,1"             # Model using 2 GPUs (GPU 0 and 1)
    ["internvl2_5_8b"]="0"             # Model using 1 GPU (GPU 0)
    ["internvl2_5_78b"]="0,1,2,3"      # Model using 4 GPUs (GPU 0, 1, 2, 3)
    ["phi3_5_v"]="0"                  # Model using 1 GPU (GPU 0)
    ["idefics3_8b"]="0"                # Model using 1 GPU (GPU 0)
    ["llava_ov_72b_chat_hf"]="0,1"     # Model using 2 GPUs (GPU 0 and 1)
)

# Loop through each model and run the Python script
for MODEL in "${!MODELS_GPU[@]}"; do
    GPU_IDS="${MODELS_GPU[$MODEL]}"
    
    # Set the CUDA_VISIBLE_DEVICES environment variable for multi-GPU usage
    export CUDA_VISIBLE_DEVICES=$GPU_IDS

    echo "Running model: $MODEL with GPUs: $GPU_IDS"
    python ./evaluation//inf_vllm.py --model_name "$MODEL" --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"
    echo "Completed model: $MODEL"
    echo "--------------------------------"
done
