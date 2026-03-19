import torch
import json

from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer
from huggingface_hub import hf_hub_download
from PIL import Image

def load_image_from_directory(directory: str):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    images = []
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(valid_extensions):
            file_path = os.path.join(directory,file_name)
            try:
                image = Image.open(file_path)
                images.append(image)
            except Exception as e:
                print(f"Skip image {file_name}: {e}")
    return images
            
# Load image
directory = ""
images = load_image_from_directory(directory)

sample_image = Image.open("abnormal-chest-x-ray.jpg")

model_path = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",   # requires FA2
    device_map="auto"                          # multi-GPU ready
)

# Example inference
# 1. Load chat_template
tokenizer = AutoTokenizer.from_pretrained(model_path)
template_path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
with open(template_path) as f:
    tokenizer.chat_template = json.load(f)["chat_template"]
# 2. Prepare prompts
questions = ["What is the main finding in this chest X-ray?","What is the main finding in this chest X-ray?"]
messages = [
    [{
        "role":"user",
        "content": [
            {"type":"image"},
            {"type":"text",
            "text":questions[i]}
        ]
    } for i,_ in enumerate(images)
    ]
]
message = [
    {
        "role":"user",
        "content":[
            {"type":"image"},
            {"type":"text",
            "text":questions[0]
            }
        ]
    }
]

tokenized_messages = [tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_tensors="pt") for msg in messages]

tokenzied_message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, return_tensors="pt")
# prompts = tokenized_messages
prompts = [tokenizer.decode(tokenized_messages[i]['input_ids']) for i,_ in enumerate(tokenized_messages)]
prompt = tokenzied_message
# 3. Prepare batched inputs with padding
processor = AutoProcessor.from_pretrained(model_path)
processor.tokenizer.padding_side = "left"
inputs = processor(images=[sample_image],
                   text=prompt,
                   return_tensors="pt").to(model.device,torch.bfloat16)

# 4. Inference
with torch.inference_mode():
    outs = model.generate(**inputs, max_new_tokens=256)
    

print("=== Response ===")
input_shape = inputs["input_ids"].shape
print(f"input_shape: {input_shape}")
input_len = input_shape[1]
print(f"input_len: {input_len}")
print(processor.decode(outs[0][input_len:],skip_speical_tokens=True))


### CLAUDE
# import json
# from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer
# from huggingface_hub import hf_hub_download
# import torch
# from PIL import Image

# image = Image.open("normal-chest-x-ray.png")
# model_path = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

# # Load model
# try:
#     model = LlavaForConditionalGeneration.from_pretrained(
#         model_path, torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2", device_map="auto"
#     )
# except Exception:
#     model = LlavaForConditionalGeneration.from_pretrained(
#         model_path, torch_dtype=torch.bfloat16, device_map="auto"
#     )

# # Load tokenizer and inject chat_template from repo
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# template_path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
# with open(template_path) as f:
#     tokenizer.chat_template = json.load(f)["chat_template"]

# # Build prompt — tokenize=False avoids the lossy encode/decode round-trip
# messages = [{
#     "role": "user",
#     "content": [
#         {"type": "image"},
#         {"type": "text", "text": "What is the main finding in this chest X-ray?"}
#     ]
# }]
# prompt = tokenizer.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )

# # Prepare inputs via processor
# processor = AutoProcessor.from_pretrained(model_path)
# inputs = processor(
#     images=[image], text=prompt, return_tensors="pt"
# ).to(model.device, torch.bfloat16)

# # Inference
# with torch.inference_mode():
#     out = model.generate(**inputs, max_new_tokens=256)

# # Decode only new tokens
# input_len = inputs["input_ids"].shape[1]
# print("=== Response ===")
# print(processor.decode(out[0][input_len:], skip_special_tokens=True))