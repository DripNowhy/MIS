#!/bin/bash

# 设置输入和输出目录（相对路径）
INPUT_DIR="./mis_test/mis_easy.json"
OUTPUT_DIR="./responses/mis_easy/llava_next_interleave.jsonl"

# 设置模型名称
MODEL_NAME="lmms-lab/llava-next-interleave-qwen-7b"

# 运行 Python 脚本来生成响应
python ./evaluation/inference_llava.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --model_name "$MODEL_NAME"
