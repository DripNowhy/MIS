#!/bin/bash

INPUT_DIR="./responses/mis_easy"
OUTPUT_DIR="./eval_result"

python ./evaluation/gpt_eval.py --input_path "$INPUT_DIR" --output_path "$OUTPUT_DIR"
