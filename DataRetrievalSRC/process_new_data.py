utf-8import os
import json
import re
from bs4 import BeautifulSoup
RAW_DIR = "../Data/temp_raw"
PROCESSED_DIR = "../Data/temp_processed"
def parse_transcript_file(filepath, mission_name):
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    extracted_data = []
    for b_tag in soup.find_all("b"):
        timestamp_text = b_tag.get_text(strip=True)
        if not re.match(r"^\d{2,3}:\d{2}:\d{2}", timestamp_text):
            continue
        full_line_text = ""
        current = b_tag.next_sibling
        while current:
            if current.name == "b" and re.match(r"^\d{2,3}:", current.get_text(strip=True)):
                break
            if current.name == "p" or current.name == "br":
                pass
            if isinstance(current, str):
                full_line_text += current
            elif current.name == "a":
                 full_line_text += current.get_text()
            elif current.name == "i":
                text = current.get_text()
                if not (text.strip().startswith("[") and text.strip().endswith("]")):
                    full_line_text += text
            else:
                 full_line_text += current.get_text()
            current = current.next_sibling
        full_line_text = full_line_text.strip()
        match = re.match(r"^([A-Za-z\s]+):(.*)", full_line_text)
        if match:
            speaker = match.group(1).strip()
            text = match.group(2).strip()
            text = re.sub(r"\[.*?\]", "", text).strip()
            if text:
                extracted_data.append({
                    : mission_name,
                    : timestamp_text,
                    : speaker,
                    : text,
                    : filename
                })
    return extracted_data
def stitch_segments(data):
    stitched = []
    i = 0
    while i < len(data):
        current = data[i]
        if current["text"].endswith("..."):
            lookahead = 1
            merged = False
            while i + lookahead < len(data) and lookahead < 5:
                candidate = data[i + lookahead]
                if candidate["speaker"] == current["speaker"] and candidate["text"].startswith("..."):
                    start_text = current["text"][:-3]
                    end_text = candidate["text"][3:]
                    current["text"] = (start_text + " " + end_text).strip()
                    data[i + lookahead]["SKIP"] = True
                    merged = True
                else:
                    if not candidate.get("SKIP"):
                         break
                lookahead += 1
        if not current.get("SKIP"):
            stitched.append(current)
        i += 1
    return stitched
def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    for mission_dir in os.listdir(RAW_DIR):
        mission_path = os.path.join(RAW_DIR, mission_dir)
        if not os.path.isdir(mission_path):
            continue
        print(f"Processing {mission_dir}...")
        all_mission_data = []
        for file in os.listdir(mission_path):
            if file.endswith(".html"):
                filepath = os.path.join(mission_path, file)
                try:
                    data = parse_transcript_file(filepath, mission_dir)
                    all_mission_data.extend(data)
                except Exception as e:
                    print(f"  Error parsing {file}: {e}")
        all_mission_data.sort(key=lambda x: x["timestamp"])
        final_data = stitch_segments(all_mission_data)
        output_file = os.path.join(PROCESSED_DIR, f"{mission_dir}_processed.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2)
        print(f"  Saved {len(final_data)} records to {output_file}")
if __name__ == "__main__":
    main()