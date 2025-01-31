import argparse
import json
import os
from PIL import Image

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

from qwen_vl_utils import process_vision_info

def load_model(model_name):
    if 'qwen2' in model_name.lower():
        if '72b' in model_name.lower():
            llm = LLM(
                model=model_name,
                max_model_len=4096,
                max_num_seqs=1,
                tensor_parallel_size=2, # Recommended for 72B using 2 A100-80G GPUs
                limit_mm_per_prompt={"image": 2}
            )
            processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=1280*28*28)
            return llm, processor
        elif '7b' in model_name.lower():
            llm = LLM(
                model=model_name,
                max_model_len=4096,
                max_num_seqs=5,
                limit_mm_per_prompt={"image": 2}
            )
            processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=1280*28*28)
            return llm, processor
    elif 'internvl' in model_name.lower():
        if '78b' in model_name.lower():
            llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=4, # Recommended for 78B using 4 A100-80G GPUs
            max_model_len=8192,
            limit_mm_per_prompt={"image": 2},
            mm_processor_kwargs={"max_dynamic_patch": 4},
            gpu_memory_utilization=0.9
        )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            return llm, tokenizer
        elif '8b' in model_name.lower():
            llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=8192,
            limit_mm_per_prompt={"image": 2},
            mm_processor_kwargs={"max_dynamic_patch": 4},
            gpu_memory_utilization=0.9
        )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            return llm, tokenizer
    elif 'phi' in model_name.lower():
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 2},
            mm_processor_kwargs={"max_dynamic_patch": 4},
            gpu_memory_utilization=0.9
        )
        return llm, None
    elif 'idefics' in model_name.lower():
        llm = LLM(
            model=model_name,
            max_model_len=8192,
            max_num_seqs=16,
            # enforce_eager=True,
            limit_mm_per_prompt={"image": 2},
            dtype=torch.bfloat16,
            # if you are running out of memory, you can reduce the "longest_edge".
            # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
            mm_processor_kwargs={
                "size": {
                    "longest_edge": 4 * 364
                },
            },
        )
        return llm, None
    elif 'llava' in model_name.lower():
        if '72b' in model_name.lower():
            llm = LLM(model=model_name,
                max_model_len=16384,
                tensor_parallel_size=4,
                limit_mm_per_prompt={"image": 2})
            stop_token_ids = None
            return llm, stop_token_ids
    else:
        raise ValueError(f"Model {model_name} not supported in vLLM")
    
def prepare_inputs(model_name, processor, query, image1, image2):
    # for Qwen2-VL series
    if 'qwen' in model_name.lower():
        images = [image1, image2]
        placeholders = [{"type": "image", "image": img} for img in images]
        messages = [
            {
            "role": "user",
            "content": [
                *placeholders,
                {
                    "type": "text",
                    "text": query
                },
            ],
        }]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        stop_token_ids = None
        return prompt, image_inputs, stop_token_ids
    # for InternVL2.5 series
    elif 'internvl' in model_name.lower():
        image1 = Image.open(image1)
        image2 = Image.open(image2)
        images = [image1, image2]
        placeholders = "\n".join(f"Image-{i}: <image>\n"
                                for i, _ in enumerate(images, start=1))
        messages = [{'role': 'user', 'content': f"{placeholders}\n{query}"}]

        # Preparation for inference
        prompt = processor.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [processor.convert_tokens_to_ids(i) for i in stop_tokens]
        return prompt, images, stop_token_ids
    # for Phi series
    elif 'phi' in model_name.lower():
        image1 = Image.open(image1)
        image2 = Image.open(image2)
        images = [image1, image2]
        placeholders = "\n".join(f"<|image_{i}|>"
                                    for i, _ in enumerate(images, start=1))
        prompt = f"<|user|>\n{placeholders}\n{query}<|end|>\n<|assistant|>\n"
        stop_token_ids = None
        return prompt, images, stop_token_ids
    # for Idefics series
    elif 'idefics' in model_name.lower():
        image1 = Image.open(image1)
        image2 = Image.open(image2)
        images = [image1, image2]

        placeholders = "\n".join(f"Image-{i}: <image>\n"
                                for i, _ in enumerate(images, start=1))
        prompt = f"<|begin_of_text|>User:{placeholders}\n{query}<end_of_utterance>\nAssistant:"
        stop_token_ids = None
        return prompt, images, stop_token_ids
    # for LLava series
    elif 'llava' in model_name.lower():
        if '72b' in model_name:
            image1 = Image.open(image1)
            image2 = Image.open(image2)
            images = [image1, image2]
            placeholders = "\n".join(f"Image-{i}: <image>\n"
                                    for i, _ in enumerate(images, start=1))
            prompt = f"<|user|>\n{placeholders}\n{query}<|end|>\n<|assistant|>\n"
            stop_token_ids = None
            return prompt, images, stop_token_ids
    else:
        raise ValueError(f"Model {model_name} not supported in vLLM")
    
def generate(model, model_name, processor, query, image1, image2, device="cuda"):
    text_inputs, image_inputs, stop_token_ids = prepare_inputs(model_name, processor, query, image1, image2)
    # Sampling parameters
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=1024,
                                     stop_token_ids=stop_token_ids)
    outputs = model.generate(
        {"prompt": text_inputs,
        "multi_modal_data":{
            "image": image_inputs
        }},
        sampling_params=sampling_params,
    )
    output_text = outputs[0].outputs[0].text
    return output_text

def main(args):
    model_path = {
        'qwen2_vl_7b': 'Qwen/Qwen2-VL-7B-Instruct',
        'qwen2_vl_72b': 'Qwen/Qwen2-VL-72B-Instruct',
        'internvl2_5_8b': 'OpenGVLab/InternVL2_5-8B',
        'internvl2_5_78b': 'OpenGVLab/InternVL2_5-78B',
        'phi3_5_v': 'microsoft/Phi-3.5-vision-instruct',
        'idefics3_8b': 'HuggingFaceM4/Idefics3-8B-Llama3',
        'llava_ov_72b_chat_hf': 'llava-hf/llava-onevision-qwen2-72b-ov-hf'
    }

    output_path = os.path.join(args.output_dir, f"{args.model_name}.jsonl")
    # Load VLMs
    model_name = model_path[args.model_name]
    model, processor = load_model(model_name)

    with open(args.input_dir, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # write the results to a jsonl file
    with open(output_path, 'a', encoding='utf-8') as output_file:
        for item in tqdm(data):

            query = item['question']
            image1 = os.path.join('mis_test',item['image_path1'])
            image2 = os.path.join('mis_test',item['image_path2'])

            response = generate(model, model_name, processor, query, image1, image2)
            print(f"Generated Response: {response}")

            item['response'] = response
            json.dump(item, output_file, ensure_ascii=False)
            output_file.write('\n')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=['qwen2_vl_7b', 'qwen2_vl_72b', 'internvl2_5_8b', 'internvl2_5_78b', 'phi3_5_v', 'idefics3_8b', 'llava_ov_72b_chat_hf'], help='Model name to use for inference')
    parser.add_argument('--input_dir', default="./mis_test/mis_hard.json", help='Path to input json file')
    parser.add_argument('--output_dir', default="./responses/mis_hard", help='Path to output json file')
    args = parser.parse_args()
    main(args)