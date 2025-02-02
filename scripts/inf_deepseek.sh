#!/bin/bash

INPUT_DIR="./mis_test/mis_easy.json"
OUTPUT_DIR="./responses/mis_easy/deepseek_vl2.jsonl"

export CUDA_VISIBLE_DEVICES="0,1"

python ./evaluation/inference_deepseek.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"
