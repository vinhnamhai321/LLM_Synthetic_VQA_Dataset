import json

def split_data_on_patient_level(input_path: str, output_path: str):
    with open(input_path,"r") as f:
        data = json.load(f)
        
    patients = set()
    patient_level_entries = []
    for entry in data:
        patient_id = entry["image"][0].split("/files/")[-1].split("/")[1]
        # Check if patient_id exists
        if patient_id in patients:
                pat
        else:
    
    
if __name__ == "__main__":
    input_path = "./dataset/data_v2.json"
    output_path = "./dataset/data_v3.json"
    split_data_on_patient_level(input_path,output_path)