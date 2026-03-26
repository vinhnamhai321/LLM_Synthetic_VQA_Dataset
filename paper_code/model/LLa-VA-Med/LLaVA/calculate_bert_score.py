import json
import os
from collections import defaultdict
from bert_score import score

def calculate_and_print_bert_score(json_path):
    # 1. Read answers.jsonl
    with open(json_path, 'r', encoding='utf-8') as f:
        # Check if it's a JSON array or JSON lines
        content = f.read().strip()
        if content.startswith('['):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split('\n') if line.strip()]
            
    # Group by model_id
    model_data = defaultdict(lambda: {'candidates': [], 'references': [], 'items': []})
    
    for item in data:
        model_id = item.get('model_id', 'unknown')
        text = str(item.get('text', '')).strip()
        reference = str(item.get('reference', '')).strip()
        
        model_data[model_id]['candidates'].append(text)
        model_data[model_id]['references'].append(reference)
        model_data[model_id]['items'].append(item)
    
    error_logs = []

    for model_id, data_dict in model_data.items():
        candidates = data_dict['candidates']
        references = data_dict['references']
        items = data_dict['items']
        
        # 2. Use score function from bert_score
        P, R, F1 = score(
            cands=candidates,
            refs=references,
            lang="en",
            model_type="roberta-large",
            num_layers=24,
            rescale_with_baseline=True,
            idf=False,
        )
        
        # Log negative scores or empty strings
        for i, (cand, ref, p_val, r_val, f1_val) in enumerate(zip(candidates, references, P, R, F1)):
            issue = None
            if not cand or not ref:
                issue = 'Empty candidate or reference'
            elif p_val.item() < 0 or r_val.item() < 0 or f1_val.item() < 0:
                issue = 'Negative BERTScore'
                
            if issue:
                error_logs.append({
                    'model_id': model_id,
                    'question_id': items[i].get('question_id'),
                    'candidate': cand,
                    'reference': ref,
                    'issue': issue,
                    'scores': {'P': p_val.item(), 'R': r_val.item(), 'F1': f1_val.item()}
                })

        # 3. Print the table to show the evaluation result
        p_mean = P.mean().item()
        r_mean = R.mean().item()
        f1_mean = F1.mean().item()
        
        print(f"{'Model ID':<20} | {'Precision (P)':<15} | {'Recall (R)':<15} | {'F1-Score':<15}")
        print("-" * 75)
        print(f"{model_id:<20} | {p_mean:<15.4f} | {r_mean:<15.4f} | {f1_mean:<15.4f}")

    if error_logs:
        error_log_path = os.path.join(os.path.dirname(json_path), "error_logs.json")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump(error_logs, f, indent=4)
        print(f"\nSaved {len(error_logs)} error logs to {error_log_path}")

if __name__ == "__main__":
    # Ensure correct path to answers.jsonl
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "answers.jsonl")
    
    calculate_and_print_bert_score(json_path)