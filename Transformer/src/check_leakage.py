utf-8import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_data
def check_leakage():
    print("Loading data...")
    df = load_data()
    total_rows = len(df)
    duplicates = df.duplicated(subset=['input_command', 'predicted_action'], keep='first')
    num_duplicates = duplicates.sum()
    print(f"\n--- Duplicate Analysis ---")
    print(f"Total Rows: {total_rows}")
    print(f"Duplicate Rows: {num_duplicates}")
    print(f"Percentage Duplicates: {num_duplicates / total_rows:.2%}")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['predicted_action'])
    train_set = set(train_df['input_command'])
    test_set = set(test_df['input_command'])
    overlap = train_set.intersection(test_set)
    print(f"\n--- Leakage Analysis ---")
    print(f"Unique Commands in Train: {len(train_set)}")
    print(f"Unique Commands in Test: {len(test_set)}")
    print(f"Overlapping Commands (Leakage): {len(overlap)}")
    if len(overlap) > 0:
        print(f"LEAKAGE CONFIRMED: {len(overlap)} commands appear in both sets.")
        print(f"Leakage Percentage of Test Set: {len(overlap) / len(test_set):.2%}")
    else:
        print("No leakage detected.")
if __name__ == "__main__":
    check_leakage()