import json
import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "output", "inference_results.json")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "output", "inference_results.csv")

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Could not find input file: {INPUT_JSON}")
        print("Make sure the inference pipeline has finished running.")
        return

    print(f"Loading JSON from {INPUT_JSON} ...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
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

    print(f"Writing to CSV: {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
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
                    
    print(f"Done! Extracted {row_count} total turns into {OUTPUT_CSV}.")

if __name__ == "__main__":
    main()
