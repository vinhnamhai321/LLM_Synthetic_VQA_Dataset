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

from bert_score import score

from PIL import Image
import math


def eval_model(args):
    # Model
    # disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_base = args.model_base
    model_name = args.model_name
    bf16 = args.bf16
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    use_flash_attn = args.use_flash_attn
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, use_flash_attn=use_flash_attn, bf16=bf16)

    print(f"== Model type")
    print(type(model).__mro__)
    # --- VQA-RAD ---
    # Load evaluation data
    with open(args.question_file,"r") as f:
        vqa_rad_data = json.load(f)
    
    # Initialize lists to store BERTScore metrics
    P_list, R_list, F1_list = [], [], []
    ans_list = []
    print("=== LOAD IMAGE ===")
    for line in tqdm(vqa_rad_data):
        
        idx = line["qid"]
        qs = line["question"]
        reference = line["answer"]
            
        cur_prompt = qs
        
        image_file = line["image_name"]
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
                    image_tensors = [img.to(model.device, dtype=compute_dtype) for img in image_tensors]
                else:
                    image_tensors = image_tensors.to(model.device, dtype=compute_dtype)
            else:
                image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
                if image is None:
                    print("Can not load image properly")
                    print(f"Image_path: {os.path.join(args.image_folder, image_file)}")
                image_tensors = process_images([image], image_processor, model.config)
                image_sizes = [image.size]
                if type(image_tensors) is list:
                    image_tensors = [img.to(model.device, dtype=compute_dtype) for img in image_tensors]
                else:
                    image_tensors = image_tensors.to(model.device, dtype=compute_dtype)
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

        print(f"predicted_text: {outputs}")
        print(f"reference_ans: {reference}")

        # Calculate BERTScore
        # P, R, F1 = score(
        #     cands=outputs,
        #     refs=reference,
        #     lang="en",
        #     model_type="roberta-large",
        #     num_layers=24,
        #     device="cuda",
        #     rescale_with_baseline=True
        # )
        # P_list.append(P.item())
        # R_list.append(R.item())
        # F1_list.append(F1.item())
        ans_id = shortuuid.uuid()
        
        ans_list.append({"question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "reference": str(reference),
                        "answer_id": ans_id,
                        "model_id": model_name,
                        "metadata": {
                            "images": image_file
                        }})

    with open(args.answers_file, "w") as f:
        json.dump(ans_list, f, indent=2)
    # print("BERTScore")
    # print(f"P_score: {P_list.mean()}, R_score: {R_list.mean()}, F1_score: {F1_list.mean()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained LoRA checkpoint or base model")
    parser.add_argument("--model-base", type=str, required=True, help="Path to base model (e.g., microsoft/llava-med-v1.5-mistral-7b)")
    parser.add_argument("--model-name", type=str, required=True, help="Remember to add prefix lora, e.g. lora-llava-med")
    parser.add_argument("--use-flash-attn", action="store_true", help="Enable flash attention")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16")
    parser.add_argument("--image-folder", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--question-file", type=str, required=True, help="Path to the JSON or JSONL file with prompts")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation template format")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    eval_model(args)
