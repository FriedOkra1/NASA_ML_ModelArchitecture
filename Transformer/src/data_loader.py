import pandas as pd
import glob
import os
def load_data(data_dir=None):
    if data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        data_dir = os.path.join(project_root, 'Data')
    jsonl_files = glob.glob(os.path.join(data_dir, '*.jsonl'))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
    dfs = []
    for file in jsonl_files:
        try:
            df = pd.read_json(file, lines=True)
            if 'input_command' in df.columns and 'predicted_action' in df.columns:
                dfs.append(df[['input_command', 'predicted_action']])
            else:
                pass
        except Exception:
            pass
    if not dfs:
        raise ValueError("No valid data loaded from files.")
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.dropna(subset=['input_command', 'predicted_action'])
    combined_df = combined_df[combined_df['input_command'].str.strip() != '']
    return combined_df