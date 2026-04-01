import argparse
import os
import json
import torch
import shortuuid
import re
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms
from accelerate import Accelerator
from einops import repeat
from PIL import Image
from tqdm import tqdm
from typing import Optional

import sys
sys.path.append('..')
from src.utils import FlamingoProcessor
from scripts.demo_utils import clean_generation

image_paths =  [
    'synpic50962.jpg',
    'synpic52767.jpg',
    'synpic30324.jpg',
    'synpic21044.jpg',
    'synpic54802.jpg',
    'synpic57813.jpg',
    'synpic47964.jpg'
]
# ---------------------------------------------------------------------------
# Few-shot context: (image_path, question, answer) triples used as prefix for
# every query.  Edit this list or externalise it to a JSON file as needed.
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {"image": "synpic50962.jpg", "question": "What is/are the structure near/in the middle of the brain?",      "answer": "pons"},
    {"image": "synpic52767.jpg", "question": "Is there evidence of a right apical pneumothorax on this chest x-ray?", "answer": "yes"},
    {"image": "synpic30324.jpg", "question": "Is/Are there air in the patient's peritoneal cavity?",            "answer": "no"},
    {"image": "synpic21044.jpg", "question": "Does the heart appear enlarged?",                                  "answer": "yes"},
    {"image": "synpic54802.jpg", "question": "What side are the infarcts located?",                             "answer": "bilateral"},
    {"image": "synpic57813.jpg", "question": "Which image modality is this?",                                   "answer": "mr flair"},
    {"image": "synpic47964.jpg", "question": "Where is the largest mass located in the cerebellum?",                                   "answer": "right"}
]

prompt = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question. <image>Question: What is/are the structure near/in the middle of the brain? Answer: pons.<|endofchunk|><image>Question: Is there evidence of a right apical pneumothorax on this chest x-ray? Answer: yes.<|endofchunk|><image>Question: Is/Are there air in the patient's peritoneal cavity? Answer: no.<|endofchunk|><image>Question: Does the heart appear enlarged? Answer: yes.<|endofchunk|><image>Question: What side are the infarcts located? Answer: bilateral.<|endofchunk|><image>Question: Which image modality is this? Answer: mr flair.<|endofchunk|><image>Question: Where is the largest mass located in the cerebellum? Answer:"


SYSTEM_PREFIX = (
    "You are a helpful medical assistant. You are being provided with images, "
    "a question about the image and an answer. Follow the examples and answer "
    "the last question. "
)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Inference Script for Med-Flamingo")

    # --- NEW batch arguments ---
    parser.add_argument("--input-directory", required=True,
                        help="Root directory containing the query chest X-ray images. "
                             "The 'images' field in the question file is resolved relative to this path.")
    parser.add_argument("--question-file", required=True,
                        help="Path to a JSONL file. Each line must be a JSON object with fields: "
                             "question_id, prompt, reference, images (relative image path).")
    parser.add_argument("--output-file", required=True,
                        help="Path to the JSONL file where inference results will be written.")

    # --- few-shot configuration ---
    parser.add_argument("--fewshot-directory", default=".",
                        help="Root directory for resolving few-shot image paths defined in "
                             "FEW_SHOT_EXAMPLES (default: current working directory).")
    parser.add_argument("--fewshot-file", default=None,
                        help="Optional path to a JSON file that overrides the built-in "
                             "FEW_SHOT_EXAMPLES list. Each entry needs: image, question, answer.")

    # --- model / runtime arguments ---
    parser.add_argument("--llama-path", default="../decapoda-research-llama-7B-hf",
                        help="Local path to the LLaMA-7B (v1) model.")
    parser.add_argument("--model-id", default="med-flamingo",
                        help="Model identifier written into every output record.")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="Maximum number of new tokens to generate per answer.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_questions(question_file: str) -> list[dict]:
    """Read a question file (JSON array or JSONL) with entries of the form:
        {"question_id": ..., "text": ..., "answer": ..., "image": ...}
    Trailing commas are tolerated.
    """
    def _strip_trailing_commas(text: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", text)

    with open(question_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Wrap bare {...},{...} into a valid JSON array if not already
    if not raw.startswith("["):
        raw = f"[{raw}]"

    try:
        entries = json.loads(_strip_trailing_commas(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse '{question_file}': {exc}") from exc

    questions = []
    for idx, entry in enumerate(entries):
        for field in ("question_id", "text", "answer", "image"):
            if field not in entry:
                raise KeyError(f"Entry {idx} is missing required field '{field}'.")
        questions.append({
            "question_id": entry["question_id"],
            "text":      entry["text"],
            "answer":   entry["answer"],
            "image":      entry["image"],
        })

    return questions

def load_fewshot_examples(fewshot_file: Optional[str]) -> list[dict]:
    """Return few-shot examples from an override file or the module-level constant."""
    if fewshot_file is None:
        return FEW_SHOT_EXAMPLES
    with open(fewshot_file, "r", encoding="utf-8") as f:
        examples = json.load(f)
    for i, ex in enumerate(examples):
        for field in ("image", "question", "answer"):
            if field not in ex:
                raise KeyError(f"Few-shot entry {i} is missing required field '{field}'")
    return examples


def build_prompt_and_images(
    fewshot_examples: list[dict],
    fewshot_dir: str,
    query_image_path: str,
    query_question: str,
) -> tuple[str, list[Image.Image]]:
    """Construct the full few-shot prompt string and the ordered image list.

    The image list must match the order of <image> tokens in the prompt:
        [few_shot_img_1, ..., few_shot_img_N, query_img]
    """
    images: list[Image.Image] = []
    prompt = SYSTEM_PREFIX

    # --- few-shot prefix ---
    for ex in fewshot_examples:
        abs_path = os.path.join(fewshot_dir, ex["image"])
        images.append(Image.open(abs_path).convert("RGB"))
        prompt += (
            f"<image>Question: {ex['question']} "
            f"Answer: {ex['answer']}.<|endofchunk|>"
        )

    # --- query image + open-ended question ---
    images.append(Image.open(query_image_path).convert("RGB"))
    prompt += f"<image>Question: {query_question} Answer:"

    return prompt, images


def run_inference(
    model,
    processor: FlamingoProcessor,
    device,
    fewshot_examples: list[dict],
    fewshot_dir: str,
    query_image_path: str,
    query_question: str,
    max_new_tokens: int,
) -> str:
    """Build the multimodal prompt for one query and return the model's answer."""
    prompt, all_images = build_prompt_and_images(
        fewshot_examples, fewshot_dir, query_image_path, query_question
    )

    # Preprocess
    pixels = processor.preprocess_images(all_images)
    pixels = repeat(pixels, "N c h w -> b N T c h w", b=1, T=1)
    tokenized_data = processor.encode_text(prompt)

    # Generate
    generated_ids = model.generate(
        vision_x=pixels.to(device),
        lang_x=tokenized_data["input_ids"].to(device),
        attention_mask=tokenized_data["attention_mask"].to(device),
        max_new_tokens=max_new_tokens,
    )

    response = processor.tokenizer.decode(generated_ids[0])
    return clean_generation(response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------ model
    accelerator = Accelerator()
    device = accelerator.device

    if not os.path.exists(args.llama_path):
        raise ValueError(
            f"LLaMA model not found at '{args.llama_path}'. "
            "Please check the --llama-path argument or the README."
        )

    print("Loading model …")
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=args.llama_path,
        tokenizer_path=args.llama_path,
        cross_attn_every_n_layers=4,
    )

    checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
    print(f"Downloaded Med-Flamingo checkpoint to {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

    processor = FlamingoProcessor(tokenizer, image_processor)
    model = accelerator.prepare(model)
    model.eval()
    print("Model initialisation finished.\n")

    # ------------------------------------------------------------------ data
    fewshot_examples = load_fewshot_examples(args.fewshot_file)
    print(f"Using {len(fewshot_examples)} few-shot examples.")

    questions = load_questions(args.question_file)
    print(f"Loaded {len(questions)} questions from {args.question_file}\n")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ loop
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for entry in tqdm(questions, desc="Running inference"):
            relative_image_path = entry["image"]
            absolute_image_path = os.path.join(args.input_directory, relative_image_path)

            if not os.path.exists(absolute_image_path):
                print(f"[WARNING] Image not found, skipping: {absolute_image_path}")
                continue

            try:
                generated_text = run_inference(
                    model=model,
                    processor=processor,
                    device=device,
                    fewshot_examples=fewshot_examples,
                    fewshot_dir=args.fewshot_directory,
                    query_image_path=absolute_image_path,
                    query_question=entry["text"],
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Failed on question_id={entry['question_id']}: {exc}")
                continue

            record = {
                "question_id": entry["question_id"],
                "prompt":      entry["text"],
                "text":        generated_text,
                "reference":   entry["answer"],
                "answer_id":   shortuuid.uuid(),
                "model_id":    args.model_id,
                "images":      relative_image_path,   # keep original relative path
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()   # write-through: partial results survive a crash

    print(f"\nDone. Results written to {args.output_file}")


if __name__ == "__main__":
    main()