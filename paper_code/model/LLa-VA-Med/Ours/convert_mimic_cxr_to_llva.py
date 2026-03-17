import os
import json
from collections import defaultdict


def merge_mimic_cxr(input_path: str,output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)

    merged = []

    for entry in data:
        key = (entry["row_index"], entry["report_index"], entry["subject_id"])

        # Check only the LAST merged entry instead of searching the whole dict
        if merged and (merged[-1]["row_index"], merged[-1]["report_index"], merged[-1]["subject_id"]) == key:
            merged[-1]["original_text"].append(entry["original_text"])
            merged[-1]["question"].append(entry["question"])
            merged[-1]["rationale"].append(entry["rationale"])
            merged[-1]["answer"].append(entry["answer"])
        else:
            merged.append({
                "row_index":     entry["row_index"],
                "report_index":  entry["report_index"],
                "subject_id":    entry["subject_id"],
                "image_path":    entry["image_path"],
                "original_text": [entry["original_text"]],
                "question":      [entry["question"]],
                "rationale":     [entry["rationale"]],
                "answer":        [entry["answer"]],
            })

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=4)

    print(f"Original entries : {len(data)}")
    print(f"Merged entries   : {len(merged)}")
    return merged

def merge_image_paths(input_path: str,output_path: str):
    with open(input_path,"r") as file:
        data = json.load(file)
        
    for entry in data:
        entry["image_path"] = list(set(entry["image_path"]))
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

        

def convert_to_llava_format(data_path,output_path,dataset_directory):
    llava_data = []
    with open(data_path,"r") as f:
        data = json.load(f)
    for i, item in enumerate(data):
        image_path = item["image_path"] if isinstance(item["image_path"],list) else [item["image_path"]]
        image_tokens = "\n".join(["<image>"] * len(image_path))
        llava_entry = {
            "id": f"sample_{i:05d}",
            "image": [dataset_directory+img for img in item["image_path"]],
            "conversations": [
                {
                    "from": "human",
                    "value": f"{image_tokens}\n{item['question']}"
                },
                {
                    "from": "gpt",
                    "value": f"{item['rationale']} {item['answer']}"
                }
            ]
        }
        llava_data.append(llava_entry)
    
    with open(output_path,"w") as f:
        json.dump(llava_data,f, indent=2)             
        
if __name__ == "__main__":
    data_path = "./data_v1.json"
    output_path = "./data_v2.json"
    dataset_directory = "/home/jovyan/network-volume/nnthao16/paper_code/LLM-Synth-Data/deepseek/official_data_iccv_final/"
    convert_to_llava_format(data_path,output_path,dataset_directory)
    
     # Merge duplicate entries
#     merge_mimic_cxr(data_path,output_path)

    # Merge duplicate image_paths
#     merge_image_paths(data_path,output_path)