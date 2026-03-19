import pandas as pd
import ast
import json
import openai
import re

# 1. Setup API
openai.api_key = "" # Use a fresh key!

def generate_vqa_triplet(text_content):
    try:
        # Check if text is empty
        if not text_content or len(text_content) < 5:
            print("Empty report text found.")
            return None

        prompt = f"""
        Extract one clinical Question, a short Answer, and a detailed Explanation (XAI) 
        based on this radiology report. The answer should be brief, just ("yes", "no", or "unknown") if applicable.
        The reasoning exxplanation do not mention the report directly, but explain the medical rationale.
        Report: {text_content}
        Return JSON format:
        {{
            "question": "...",
            "answer": "...",
            "explanation": "..."
        }}
        """
        # Note: Ensure you are using the latest openai library version (v1.0.0+)
        response = openai.chat.completions.create(
            model="gpt-4o-mini", # mini is cheaper/faster for thesis work
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )

        raw_output = response.choices[0].message.content
        print(f"DEBUG - AI Response: {raw_output}") # ADD THIS LINE
        
        return json.loads(raw_output)
    except Exception as e:
        print(f"API Error: {e}") # This will tell us if it's a billing/key issue
        return None

# 2. Load Dataset
try:
    df = pd.read_csv('mimic_cxr_aug_train.csv')
    print(f"Loaded CSV with {len(df)} rows.")
except FileNotFoundError:
    print("Error: mimic_cxr_aug_train.csv not found!")
    exit()

final_vqa_dataset = []

for idx, row in df.iterrows():
    try:
        # Convert strings to lists
        images = ast.literal_eval(row['image'])
        reports = ast.literal_eval(row['text'])
        
        print(f"Row {idx}: Found {len(images)} images and {len(reports)} reports.")

        for i, img_path in enumerate(images):
            # Map report to image: using index matching if study_id logic is complex
            matching_report = reports[i] if i < len(reports) else reports[0]

            vqa_data = generate_vqa_triplet(matching_report)
            
            if vqa_data:
                vqa_data['image_path'] = img_path
                vqa_data['subject_id'] = row['subject_id']
                final_vqa_dataset.append(vqa_data)
                print(f"  [SUCCESS] Processed: {img_path}")
            else:
                print(f"  [FAILED] No data generated for: {img_path}")

    except Exception as e:
        print(f"Error processing row {idx}: {e}")

    if idx >= 1: break # Test with 3 rows

# 4. Save result
if final_vqa_dataset:
    with open('mimic_xai_vqa_final.json', 'w') as f:
        json.dump(final_vqa_dataset, f, indent=4)
    print(f"Done! Saved {len(final_vqa_dataset)} entries to JSON.")
else:
    print("No data was collected. JSON file will be empty.")