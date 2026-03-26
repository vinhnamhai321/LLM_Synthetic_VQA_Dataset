import json
import argparse
import itertools

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for evaluate.py")
    parser.add_argument("--questions", type=str, required=True, help="Path to question_file.txt (one question per line)")
    parser.add_argument("--images", type=str, required=True, help="Path to image_file.txt (one image per line, or comma-separated list of images)")
    parser.add_argument("--output", type=str, required=True, help="Path to output .jsonl file")
    parser.add_argument("--mode", type=str, choices=["zip", "all_pairs"], default="zip", 
                        help="Pairing mode: 'zip' for line-by-line pairing, 'all_pairs' to pair every image with every question.")
    
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as qf:
        questions = [line.strip() for line in qf if line.strip()]
        
    with open(args.images, "r", encoding="utf-8") as imf:
        images = [line.strip() for line in imf if line.strip()]

    if args.mode == "zip" and len(questions) != len(images):
        print(f"Warning: You have {len(questions)} questions but {len(images)} images.")
        print("Using 'zip' mode will only generate pairs up to the shortest list length.")

    # Determine pairing iterable
    if args.mode == "zip":
        pairs = zip(questions, images)
    else:
        # Cartesian product: every image is evaluated on every question
        pairs = itertools.product(questions, images)

    count = 0
    with open(args.output, "w", encoding="utf-8") as outf:
        for idx, (q, img_entry) in enumerate(pairs):
            
            # Sub-parsing: if image txt contains comma-separated lists for multi-image prompts
            if ',' in img_entry:
                img_data = [img.strip() for img in img_entry.split(',')]
            else:
                img_data = img_entry
                
            record = {
                "question_id": idx,
                "text": q,
                "answer": a,
                "image": img_data
            }
            
            outf.write(json.dumps(record) + "\n")
            count += 1
            
    print(f"Successfully generated {count} evaluation entries and saved to {args.output}")

if __name__ == "__main__":
    main()
