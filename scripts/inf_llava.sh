#!/bin/bash

# Define input and output directories
INPUT_DIR="./mis_test/mis_easy.json"
OUTPUT_DIR="./responses/mis_easy/llava_next_interleave.jsonl"

# Define the model name
MODEL_NAME="lmms-lab/llava-next-interleave-qwen-7b"

# Check if the model name is valid
echo "Testing model: $MODEL_NAME"

# Run the Python script for generating responses using llava-next-interleave model
python test_llava_next_interleave.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"
