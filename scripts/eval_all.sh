#!/bin/bash

# 设置输入和输出目录
INPUT_DIR="../responses/mis_easy"
OUTPUT_DIR="../eval_result"

# 运行 Python 脚本来评估所有的 .jsonl 文件
python ../evaluation/gpt_eval.py --input_path "$INPUT_DIR" --output_path "$OUTPUT_DIR"
