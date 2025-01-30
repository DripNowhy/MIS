from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import json
import argparse
from tqdm import tqdm
from PIL import Image

import copy
import torch

import warnings

def load_llava(model_id):

    warnings.filterwarnings("ignore")
    pretrained = model_id
    model_name = "llava_qwen"
    device = "cuda"  # 明确指定使用 cuda:0
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name)  # Add any other thing you want to pass in llava_model_args

    model.eval()


    model.to(device)
    return model, tokenizer, image_processor

def generate(model, tokenizer, image_processor, qs, image1, image2, device="cuda"):
    
    image1 = Image.open(image1)
    image2 = Image.open(image2)
    images = [image1, image2]

    image_tensor = process_images(images, image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN  + "\n" + qs
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size for image in images]


    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=1024,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    return text_outputs

def main(args):
    # Load the model and processor
    model_name = "lmms-lab/llava-next-interleave-qwen-7b"
    model, tokenizer, image_processor = load_llava(model_name)

    with open(args.input_dir, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)
    # Open the output JSONL file for appending the results
    with open(args.output_dir, 'a', encoding='utf-8') as output_file:
        for item in tqdm(data):

            qs = item['question']
            image1 = item['image_path1']
            image2 = item['image_path2']

            # Generate a response
            response = generate(model, tokenizer, image_processor, qs, image1, image2)

            # Create a dictionary to store question and generated response
            item['response'] = response

            # Append the result to the JSONL file
            json.dump(item, output_file, ensure_ascii=False)
            output_file.write('\n')  # Write each result on a new line

            print(f"Generated Response: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from two images and a question using the llava-next-interleave model.")
    
    parser.add_argument('--input_dir', default="./mis_test/mis_easy.json", help='Path to input json file')
    parser.add_argument('--output_dir', default="./responses/mis_easy/llava_next_interleave.jsonl", help='Path to output json file')
    
    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)
