# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("microsoft/llava-med-v1.5-mistral-7b", dtype="auto")
print("Load model sucessfully.")