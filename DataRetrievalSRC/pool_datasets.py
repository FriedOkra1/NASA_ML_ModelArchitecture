utf-8import os
import json
PROCESSED_DIR = "../Data/temp_processed"
COMBINED_DIR = "../Data/combined"
OUTPUT_FILE = os.path.join(COMBINED_DIR, "master_eva_dataset.jsonl")
def main():
    if not os.path.exists(COMBINED_DIR):
        os.makedirs(COMBINED_DIR)
    all_records = []
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(PROCESSED_DIR, filename)
            print(f"Pooling {filename}...")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for idx, record in enumerate(data):
                record["id"] = f"{record['mission']}_{idx:05d}"
                all_records.append(record)
    print(f"Total pooled records: {len(all_records)}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    print(f"Master dataset saved to {OUTPUT_FILE}")
if __name__ == "__main__":
    main()