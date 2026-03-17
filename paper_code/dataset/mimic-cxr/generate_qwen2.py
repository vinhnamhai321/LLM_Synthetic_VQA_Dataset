"""
Text-only VQA generation using Qwen2 with IMAGE PATHS included
WITH RESUME CAPABILITY and FILE-BASED INTERRUPT + AUTO-SAVE
"""
import pandas as pd
import ast
import json
import torch
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

AUTO_SAVE_INTERVAL = 5  
STOP_FILE = "STOP.txt"  
 
def load_checkpoint(checkpoint_file='checkpoint.json'):
    """Load processing checkpoint"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"✓ Loaded checkpoint: Resume from row {checkpoint['last_row_idx']}, report {checkpoint['last_report_idx']}")
        return checkpoint
    return {'last_row_idx': -1, 'last_report_idx': -1, 'processed_data': []}

def save_checkpoint(checkpoint_file, row_idx, report_idx, processed_data):
    """Save current progress to checkpoint"""
    checkpoint = {
        'last_row_idx': row_idx,
        'last_report_idx': report_idx,
        'processed_data': processed_data
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=4)

def save_final_results(processed_data, model_size, output_file=None):
    """Save final results to JSON file"""
    if output_file is None:
        output_file = f'mimic_xai_vqa_text_only_{model_size}.json'
    
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)
    
    print(f"\n💾 Saved {len(processed_data)} entries to {output_file}")
    return output_file

def check_stop_signal():
    """Check if user created STOP.txt file"""
    return os.path.exists(STOP_FILE)

def create_stop_instructions():
    """Show user how to stop gracefully"""
    print(f"\n{'='*70}")
    print(f"⚠️  TO STOP SAFELY: Create a file named '{STOP_FILE}' in this directory")
    print(f"   Example (Windows): type nul > {STOP_FILE}")
    print(f"   Example (Linux/Mac): touch {STOP_FILE}")
    print(f"   Or simply create an empty text file named '{STOP_FILE}'")
    print(f"{'='*70}\n")

# ============================================================================= 
# STEP 2: Load Qwen2 Text Model
# ============================================================================= 
def load_qwen_text_model(model_size="7B"):
    """
    Load Qwen2 text-only model
    Model options:
    - "1.5B": Qwen/Qwen2-1.5B-Instruct (Smallest, ~4GB VRAM)
    - "7B": Qwen/Qwen2-7B-Instruct (Medium, ~16GB VRAM)
    - "72B": Qwen/Qwen2-72B-Instruct (Largest, requires multiple GPUs)
    """
    model_names = {
        "1.5B": "Qwen/Qwen2-1.5B-Instruct",
        "7B": "Qwen/Qwen2-7B-Instruct",
        "72B": "Qwen/Qwen2-72B-Instruct"
    }
    
    model_name = model_names.get(model_size, model_names["1.5B"])
    
    print(f"Loading {model_name}...")
    print("This may take several minutes on first run...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"✓ Model loaded successfully on {model.device}")
    return model, tokenizer

# ============================================================================= 
# STEP 3: Generate VQA Triplet with Qwen2
# ============================================================================= 
def generate_vqa_triplet_text(text_content, model, tokenizer):
    """
    Generate Question, Answer, and Explanation from radiology report text only
    
    Args:
        text_content: Radiology report text
        model: Loaded Qwen2 model
        tokenizer: Loaded Qwen tokenizer
        
    Returns:
        dict: {"question": str, "answer": str, "explanation": str}
    """
    try:
        # Check if text is empty
        if not text_content or len(text_content) < 5:
            print("Empty report text found.")
            return None
        
        # Create prompt for Qwen
        prompt = f"""You are a medical AI assistant analyzing radiology reports.

Based on the provided radiology report, generate:
1. A clinical question about the findings
2. A brief answer (yes/no/unknown)
3. A detailed medical explanation

Radiology Report:
{text_content}

Return your response in JSON format:
{{
    "question": "...",
    "answer": "...",
    "explanation": "..."
}}"""
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a helpful medical AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode only the generated tokens (remove input)
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        output_text = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
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
    model_size = "7B"  # Change to "1.5B" if limited on VRAM
    checkpoint_file = f'checkpoint_{model_size}.json'
    output_file = f'mimic_xai_vqa_text_only_{model_size}.json'
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    processed_data = checkpoint['processed_data']
    start_row = checkpoint['last_row_idx'] + 1
    start_report = checkpoint['last_report_idx'] + 1
    
    print(f"\n{'='*60}")
    print(f"RESUMING FROM: Row {start_row}, Report {start_report}")
    print(f"Already processed: {len(processed_data)} entries")
    print(f"{'='*60}\n")
    
    # Show stop instructions
    create_stop_instructions()
    
    # Load Qwen text model
    model, tokenizer = load_qwen_text_model(model_size)
    
    # Load dataset
    try:
        df = pd.read_csv('mimic_cxr_aug_train.csv')
        print(f"✓ Loaded CSV with {len(df)} rows.\n")
    except FileNotFoundError:
        print("❌ Error: mimic_cxr_aug_train.csv not found!")
        return
    
    report_counter = 0  # Counter for auto-save
    
    try:
        for idx, row in df.iterrows():
            # Check for stop signal
            if check_stop_signal():
                print(f"\n🛑 STOP signal detected! Saving and exiting...")
                break
            
            # Skip already processed rows
            if idx < start_row:
                continue
            
            # Determine starting report index
            report_start_idx = start_report if idx == start_row else 0
            
            try:
                # Convert strings to lists
                reports = ast.literal_eval(row['text'])
                images = ast.literal_eval(row['image'])
                
                print(f"\n📄 Row {idx}: {len(reports)} reports, {len(images)} images | Subject: {row['subject_id']}")
                
                # Match reports with images
                for i, report_text in enumerate(reports):
                    # Check for stop signal
                    if check_stop_signal():
                        print(f"\n🛑 STOP signal detected!")
                        break
                    
                    # Skip already processed reports in current row
                    if idx == start_row and i < report_start_idx:
                        continue
                    
                    print(f"  [{i+1}/{len(reports)}] Processing...", end=" ")
                    
                    vqa_data = generate_vqa_triplet_text(
                        report_text,
                        model,
                        tokenizer
                    )
                    
                    if vqa_data:
                        vqa_data['row_index'] = idx
                        vqa_data['report_index'] = i
                        vqa_data['subject_id'] = row['subject_id']
                        vqa_data['original_text'] = report_text
                        vqa_data['image_path'] = images[i] if i < len(images) else None
                        
                        processed_data.append(vqa_data)
                        report_counter += 1
                        print(f"✅ SUCCESS (Total: {len(processed_data)})")
                    else:
                        print(f"❌ FAILED")
                    
                    # Save checkpoint after each report
                    save_checkpoint(checkpoint_file, idx, i, processed_data)
                    
                    # Auto-save to final JSON every N reports
                    if report_counter % AUTO_SAVE_INTERVAL == 0:
                        save_final_results(processed_data, model_size, output_file)
                        print(f"  💾 Auto-saved ({report_counter} reports processed)")
                
                # Check if loop was broken
                if check_stop_signal():
                    break
                    
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
                save_checkpoint(checkpoint_file, idx, -1, processed_data)
                continue
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        # ALWAYS save final results
        if processed_data:
            save_final_results(processed_data, model_size, output_file)
            print(f"\n✅ Processing complete!")
            print(f"   Total entries: {len(processed_data)}")
            print(f"   Output file: {output_file}")
            
            # Remove checkpoint if fully complete
            if not check_stop_signal() and idx >= len(df) - 1:
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                    print(f"   Checkpoint removed (processing complete)")
            else:
                print(f"   Checkpoint saved (resume with: python script.py)")
            
            # Remove stop file if it exists
            if os.path.exists(STOP_FILE):
                os.remove(STOP_FILE)
                print(f"   Stop file removed")
        else:
            print("\n⚠️  No data was collected.")

if __name__ == "__main__":
    main()