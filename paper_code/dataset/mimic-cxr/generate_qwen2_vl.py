"""
Complete setup for using Qwen2-VL locally for Medical XAI VQA task
This replaces OpenAI API with local Qwen model
"""

import pandas as pd
import ast
import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os

# =============================================================================
# STEP 1: Install Required Packages (Run once)
# =============================================================================
"""
pip install transformers>=4.37.0
pip install accelerate
pip install pillow
pip install qwen-vl-utils
pip install torch torchvision  # If not already installed
"""

# =============================================================================
# STEP 2: Load Qwen2-VL Model (Choose based on your GPU memory)
# =============================================================================

def load_qwen_model(model_size="7B"):
    """
    Load Qwen2-VL model
    
    Model options:
    - "2B": Qwen/Qwen2-VL-2B-Instruct (Smallest, ~8GB VRAM)
    - "7B": Qwen/Qwen2-VL-7B-Instruct (Medium, ~20GB VRAM) 
    - "72B": Qwen/Qwen2-VL-72B-Instruct (Largest, requires multiple GPUs)
    """
    
    model_names = {
        "2B": "Qwen/Qwen2-VL-2B-Instruct",
        "7B": "Qwen/Qwen2-VL-7B-Instruct",
        "72B": "Qwen/Qwen2-VL-72B-Instruct"
    }
    
    model_name = model_names.get(model_size, model_names["2B"])
    
    print(f"Loading {model_name}...")
    print("This may take several minutes on first run (downloading ~4-15GB)...")
    
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        device_map="auto",  # Automatically use available GPU
        trust_remote_code=True
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    print(f"✓ Model loaded successfully on {model.device}")
    return model, processor


# =============================================================================
# STEP 3: Generate VQA Triplet with Qwen2-VL
# =============================================================================

def generate_vqa_triplet_qwen(image_path, text_content, model, processor):
    """
    Generate Question, Answer, and Explanation using Qwen2-VL
    
    Args:
        image_path: Path to chest X-ray image
        text_content: Radiology report text
        model: Loaded Qwen2-VL model
        processor: Loaded Qwen processor
    
    Returns:
        dict: {"question": str, "answer": str, "explanation": str}
    """
    try:
        # Check if text is empty
        if not text_content or len(text_content) < 5:
            print("Empty report text found.")
            return None
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        # Create prompt for Qwen
        prompt = f"""You are a medical AI assistant analyzing chest X-ray images and radiology reports.

Based on the provided chest X-ray image and radiology report, generate:
1. A clinical question about the image
2. A brief answer (yes/no/unknown or short phrase)
3. A detailed medical explanation without directly referencing the report

Radiology Report:
{text_content}

Return your response in JSON format:
{{
    "question": "...",
    "answer": "...",
    "explanation": "..."
}}"""

        # Prepare messages with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ],
            }
        ]
        
        # Prepare for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"DEBUG - Qwen Response: {output_text}")
        
        # Parse JSON response
        # Try to extract JSON from markdown code blocks if present
        if "```json" in output_text:
            json_start = output_text.find("```json") + 7
            json_end = output_text.find("```", json_start)
            output_text = output_text[json_start:json_end].strip()
        elif "```" in output_text:
            json_start = output_text.find("```") + 3
            json_end = output_text.find("```", json_start)
            output_text = output_text[json_start:json_end].strip()
        
        return json.loads(output_text)
        
    except Exception as e:
        print(f"Error generating VQA: {e}")
        return None


# =============================================================================
# STEP 4: Main Processing Pipeline
# =============================================================================

def main():
    # Load Qwen model (choose size based on your GPU)
    # For most servers with ~16-24GB VRAM, use "7B"
    # For limited GPU memory, use "2B"
    model, processor = load_qwen_model(model_size="2B")  # Change to "7B" if you have GPU
    
    # Load dataset
    try:
        df = pd.read_csv('mimic_cxr_aug_train.csv')
        print(f"Loaded CSV with {len(df)} rows.")
    except FileNotFoundError:
        print("Error: mimic_cxr_aug_train.csv not found!")
        return
    
    final_vqa_dataset = []
    
    for idx, row in df.iterrows():
        try:
            # Convert strings to lists
            images = ast.literal_eval(row['image'])
            reports = ast.literal_eval(row['text'])
            
            print(f"\nRow {idx}: Found {len(images)} images and {len(reports)} reports.")
            
            for i, img_path in enumerate(images):
                # Map report to image
                matching_report = reports[i] if i < len(reports) else reports[0]
                
                vqa_data = generate_vqa_triplet_qwen(
                    img_path, 
                    matching_report, 
                    model, 
                    processor
                )
                
                if vqa_data:
                    vqa_data['image_path'] = img_path
                    vqa_data['subject_id'] = row['subject_id']
                    final_vqa_dataset.append(vqa_data)
                    print(f"  [SUCCESS] Processed: {img_path}")
                else:
                    print(f"  [FAILED] No data generated for: {img_path}")
        
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
        
        if idx >= 2:  # Test with 3 rows first
            break
    
    # Save results
    if final_vqa_dataset:
        with open('mimic_xai_vqa_qwen.json', 'w') as f:
            json.dump(final_vqa_dataset, f, indent=4)
        print(f"\nDone! Saved {len(final_vqa_dataset)} entries to mimic_xai_vqa_qwen.json")
    else:
        print("No data was collected.")


if __name__ == "__main__":
    main()