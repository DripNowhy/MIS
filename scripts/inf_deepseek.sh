#!/bin/bash

# 设置输入和输出目录（相对路径）
INPUT_DIR="./mis_test/mis_easy.json"
OUTPUT_DIR="./responses/mis_easy/deepseek_vl2.jsonl"

# 设置要使用的 GPU，GPU 0 和 GPU 1
export CUDA_VISIBLE_DEVICES="0,1"

# 运行 Python 脚本来生成响应
python ./evaluation/inference_deepseek.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"
