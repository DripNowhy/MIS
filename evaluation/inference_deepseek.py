import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

from tqdm import tqdm
import json
import os
import argparse

def split_model(model_name):
    device_map = {}
    model_splits = {
        'deepseek-ai/deepseek-vl2': [15, 15], # 2 GPU for 27b
    }
    num_layers_per_gpu = model_splits[model_name]
    num_layers =  sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f'language.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision'] = 0
    device_map['projector'] = 0
    device_map['image_newline'] = 0
    device_map['view_seperator'] = 0
    device_map['language.model.embed_tokens'] = 0
    device_map['language.model.norm'] = 0
    device_map['language.lm_head'] = 0
    device_map[f'language.model.layers.{num_layers - 1}'] = 0
    return device_map


def main(args):
    model_path = "deepseek-ai/deepseek-vl2"
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    device_map = split_model(model_path)

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.bfloat16, device_map=device_map).eval()
    
    with open(args.input_dir, 'r') as f:
        data = json.load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.output_dir, 'a') as output_file:
        for item in tqdm(data):
            try:
                question = item['question']
                image1 = os.path.join('mis_test',item['image_path1'])
                image2 = os.path.join('mis_test',item['image_path2'])

                conversation = [
                    {
                        "role": "<|User|>",
                        "content": "<image>\n<image>\n" + question,
                        "images": [
                            image1,
                            image2
                        ],
                    },
                    {"role": "<|Assistant|>", "content": ""}
                ]

                # load images and prepare for inputs
                pil_images = load_pil_images(conversation)
                prepare_inputs = vl_chat_processor(
                    conversations=conversation,
                    images=pil_images,
                    force_batchify=True,
                    system_prompt=""
                ).to(vl_gpt.device)

                # run image encoder to get the image embeddings
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                # run the model to get the response
                outputs = vl_gpt.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True
                )

                response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
                print(response)
                item['response'] = response
                json.dump(item, output_file, ensure_ascii=False)
                output_file.write('\n')  # Write each result on a new line

            except Exception as e:
                print(e)
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from two images and a question using the llava-next-interleave model.")
    
    parser.add_argument('--input_dir', default="./mis_test/mis_easy.json", help='Path to input json file')
    parser.add_argument('--output_dir', default="./responses/mis_easy/deepseek_vl2.jsonl", help='Path to output json file')
    
    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)