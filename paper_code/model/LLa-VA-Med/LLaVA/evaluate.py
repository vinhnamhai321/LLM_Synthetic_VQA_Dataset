import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Load questions
    question_file = os.path.expanduser(args.question_file)
    if question_file.endswith(".json"):
        with open(question_file, "r") as f:
            questions = json.load(f)
    else:
        questions = [json.loads(q) for q in open(question_file, "r")]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        idx = line.get("question_id", line.get("id"))
        image_file = line.get("image", None)
        
        # Handle LLaVA conversational format vs standard VQA format
        if "conversations" in line:
            # Assumes the first human message contains the prompt
            qs = next(c["value"] for c in line["conversations"] if c["from"] == "human")
        else:
            qs = line["text"]
            
        cur_prompt = qs
        
        # ensure <image> token format
        if image_file is not None:
            if DEFAULT_IMAGE_TOKEN not in qs:
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        if image_file is not None:
            if isinstance(image_file, list):
                images = [Image.open(os.path.join(args.image_folder, img)).convert('RGB') for img in image_file]
                image_tensors = process_images(images, image_processor, model.config)
                image_sizes = [img.size for img in images]
                if type(image_tensors) is list:
                    image_tensors = [img.to(model.device, dtype=torch.float16) for img in image_tensors]
                else:
                    image_tensors = image_tensors.to(model.device, dtype=torch.float16)
            else:
                image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
                image_tensors = process_images([image], image_processor, model.config)
                image_sizes = [image.size]
                if type(image_tensors) is list:
                    image_tensors = [img.to(model.device, dtype=torch.float16) for img in image_tensors]
                else:
                    image_tensors = image_tensors.to(model.device, dtype=torch.float16)
        else:
            image_tensors = None
            image_sizes = None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained LoRA checkpoint or base model")
    parser.add_argument("--model-base", type=str, default=None, help="Path to base model (e.g., microsoft/llava-med-v1.5-mistral-7b)")
    parser.add_argument("--image-folder", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--question-file", type=str, required=True, help="Path to the JSON or JSONL file with prompts")
    parser.add_argument("--answers-file", type=str, required=True, help="Path where outputs should be saved (.jsonl)")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation template format")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    eval_model(args)
