#!/bin/bash

INPUT_DIR="./mis_test/mis_easy.json"
OUTPUT_DIR="./responses/mis_easy/llava_next_interleave.jsonl"

MODEL_NAME="lmms-lab/llava-next-interleave-qwen-7b"

python ./evaluation/inference_llava.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --model_name "$MODEL_NAME"
