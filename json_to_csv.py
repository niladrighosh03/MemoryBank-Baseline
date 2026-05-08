import json
import csv
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "output", "inference_results.json")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "output", "inference_results.csv")

def main():
    parser = argparse.ArgumentParser(description="Convert inference JSON output to CSV")
    parser.add_argument("--input_json", type=str, default=INPUT_JSON,
                        help="Path to inference_results.json")
    parser.add_argument("--output_csv", type=str, default=OUTPUT_CSV,
                        help="Path to output CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        print(f"Error: Could not find input file: {args.input_json}")
        print("Make sure the inference pipeline has finished running.")
        return

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print(f"Loading JSON from {args.input_json} ...")
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prepare CSV headers
    headers = [
        "Persona Id",
        "conversation id",
        "date",
        "turn index",
        "user query",
        "ground response",
        "generated response"
    ]

    print(f"Writing to CSV: {args.output_csv} ...")
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        row_count = 0
        
        # Iterate over the nested structure
        # data format: {"P_001": [{"conversation_id": 9, "date": "2024-...", "turns": [{"turn": 1, ...}, ...]}, ...], ...}
        for persona_id, conv_list in data.items():
            for conv in conv_list:
                conv_id = conv.get("conversation_id", "")
                date = conv.get("date", "")
                
                # Iterate through each turn in the conversation
                for turn in conv.get("turns", []):
                    turn_idx = turn.get("turn_index", "")
                    user_query = turn.get("user_query", "")
                    ground_response = turn.get("ground_truth_response", "")
                    generated_response = turn.get("generated_response", "")
                    
                    writer.writerow([
                        persona_id,
                        conv_id,
                        date,
                        turn_idx,
                        user_query.strip(),
                        ground_response.strip(),
                        generated_response.strip()
                    ])
                    row_count += 1
                    
    print(f"Done! Extracted {row_count} total turns into {args.output_csv}.")

if __name__ == "__main__":
    main()
