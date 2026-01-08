utf-8import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin
BASE_ALSJ_URL = "https://history.nasa.gov/alsj/"
WAYBACK_PREFIX = "https://web.archive.org/web/20230601000000/" 
MISSIONS = {
    : {
        : "a11/a11.html",
        : "Apollo",
        : "Surface"
    },
    : {
        : "a12/a12.html", 
        : "Apollo",
        : "Surface"
    },
    : {
        : "a13/a13.html", 
        : "Apollo",
        : "Flight" 
    },
    : {
        : "a14/a14.html",
        : "Apollo",
        : "Surface"
    },
    : {
        : "a15/a15.html",
        : "Apollo",
        : "Surface"
    },
    : {
        : "a16/a16.html",
        : "Apollo",
        : "Surface"
    },
    : {
        : "a17/a17.html",
        : "Apollo",
        : "Surface"
    },
}
OUTPUT_DIR = "../Data/temp_raw"
def fetch_content(url, use_wayback=True):
    target_url = url
    if use_wayback and not url.startswith("https://web.archive.org"):
        target_url = WAYBACK_PREFIX + url
    print(f"Fetching: {target_url}")
    headers = {
        : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(target_url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 429: 
            print(f"Rate limited (429). Sleeping for 30s...")
            time.sleep(30)
            return None
        else:
            print(f"Failed to fetch {target_url} (Status: {response.status_code})")
            return None
    except Exception as e:
        print(f"Error fetching {target_url}: {e}")
        return None
def find_transcript_links(html_content, base_url):
    soup = BeautifulSoup(html_content, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)
        text = a.get_text().lower()
        if href.endswith(".html") and ("eva" in text or "surface" in text or "landing" in text or "deploy" in text):
             if "index" not in href and "copyright" not in href:
                links.append((full_url, text))
    return links
def save_raw_file(content, filename, mission_name):
    mission_dir = os.path.join(OUTPUT_DIR, mission_name)
    if not os.path.exists(mission_dir):
        os.makedirs(mission_dir)
    filepath = os.path.join(mission_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {filepath}")
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for mission, config in MISSIONS.items():
        mission_dir = os.path.join(OUTPUT_DIR, mission)
        if os.path.exists(mission_dir) and len(os.listdir(mission_dir)) > 0:
            if mission in ["Apollo11", "Apollo12", "Apollo14"]:
                 print(f"\n--- Skipping {mission} (Already Downloaded) ---")
                 continue
        print(f"\n--- Processing {mission} ---")
        full_index_url = BASE_ALSJ_URL + config["index_url"]
        index_html = fetch_content(full_index_url, use_wayback=True)
        time.sleep(10)
        if not index_html:
            print(f"Skipping {mission} due to index fetch failure.")
            continue
        links = find_transcript_links(index_html, full_index_url)
        print(f"Found {len(links)} potential transcript segments.")
        for link_url, link_text in links:
            original_filename = link_url.split("/")[-1]
            print(f"  Downloading segment: {link_text.strip()} -> {original_filename}")
            use_wb = True
            if link_url.startswith("https://web.archive.org"):
                use_wb = False 
            segment_html = fetch_content(link_url, use_wayback=use_wb) 
            if segment_html:
                save_raw_file(segment_html, original_filename, mission)
                print("  Sleeping 10s...")
                time.sleep(10) 
            else:
                print("  Failed to download segment. Sleeping 10s before next...")
                time.sleep(10)
if __name__ == "__main__":
    main()