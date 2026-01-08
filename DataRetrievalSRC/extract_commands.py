utf-8import json
import re
import os
INPUT_FILE = "../Data/combined/master_eva_dataset.jsonl"
OUTPUT_DIR = "../Data"
PATTERNS = {
    : [
        ,
        ,
        ,
    ],
    : [
        ,
        ,
        , 
    ],
    : [
        ,
        ,
    ],
    : [
        ,
        ,
    ],
    : [
        ,
        ,
    ]
}
CATEGORIES = [
    ,
    ,
    ,
    ,
    ,
]
def classify_text(text, speaker):
    text_lower = text.lower()
    if any(re.search(p, text_lower) for p in PATTERNS["System_Config"]):
        if re.search(r"\b(switch|select|set|turn)\b", text_lower):
            return "System_Config"
        if re.search(r"\b(circuit breaker|valve|antenna)\b", text_lower):
             return "System_Config"
    if any(re.search(p, text_lower) for p in PATTERNS["Navigation"]):
        if re.search(r"\b(go|proceed|head|climb|step)\b", text_lower):
            return "Navigation"
        if re.search(r"\b(yaw|pitch|roll)\b", text_lower):
            return "Navigation"
    if any(re.search(p, text_lower) for p in PATTERNS["Verification"]):
         if re.search(r"\b(verify|check|read|report)\b", text_lower):
             return "Verification"
         if re.search(r"\b(pressure|volts|status)\b", text_lower):
             return "Verification"
    if any(re.search(p, text_lower) for p in PATTERNS["Observation"]):
        if re.search(r"\b(see|look|looks like)\b", text_lower):
            return "Observation"
    if any(re.search(p, text_lower) for p in PATTERNS["Coordination"]):
         return "Coordination"
    return "Chatter"
def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Output directory {OUTPUT_DIR} does not exist.")
        return
    file_handles = {}
    for cat in CATEGORIES:
        filename = "chatter_logs.jsonl" if cat == "Chatter" else                   "system_commands.jsonl" if cat == "System_Config" else                   "navigation_commands.jsonl" if cat == "Navigation" else                   "observation_reports.jsonl" if cat == "Observation" else                   "verification_checks.jsonl" if cat == "Verification" else                   "coordination_events.jsonl"
        path = os.path.join(OUTPUT_DIR, filename)
        file_handles[cat] = open(path, "w", encoding="utf-8")
    stats = {cat: 0 for cat in CATEGORIES}
    total_processed = 0
    print(f"Processing {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            total_processed += 1
            record = json.loads(line)
            text = record.get("text", "")
            speaker = record.get("speaker", "")
            clean_text = re.sub(r"\(.*?\)", "", text).strip()
            clean_text = clean_text.replace("...", " ")
            clean_text = re.sub(r"\s+", " ", clean_text)
            category = classify_text(clean_text, speaker)
            output_record = {
                : clean_text,
                : category,
                : record
            }
            file_handles[category].write(json.dumps(output_record) + "\n")
            stats[category] += 1
    for handle in file_handles.values():
        handle.close()
    print("\n--- Processing Complete ---")
    print(f"Total Records: {total_processed}")
    print("\nDistribution:")
    for cat, count in stats.items():
        print(f"  {cat}: {count} ({count/total_processed*100:.1f}%)")
if __name__ == "__main__":
    main()