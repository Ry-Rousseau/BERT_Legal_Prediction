import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_static_splits(input_file="data/casehold/raw/casehold.csv", output_dir="data/casehold"):
    print(f"Reading {input_file}...")
    # Load the raw CSV (assuming no headers based on your previous description,
    # or use header=0 if it has them. Adjust 'names' to match your schema)
    # Based on your previous message, columns are 0-12
    df = pd.read_csv(input_file, header = 0)

    # Drop rows with NaN labels
    original_len = len(df)
    df = df.dropna(subset=[df.columns[12]])
    dropped = original_len - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing labels ({dropped/original_len*100:.2f}%)")

    # 1. Isolate the Test Set (e.g., 10%)
    # Stratify by label to ensure class balance
    train_dev, test_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df.iloc[:, 12]
    )
    
    # 2. Isolate the Dev (Validation) Set (e.g., 10% of total, which is ~11% of remaining)
    train_df, dev_df = train_test_split(
        train_dev, test_size=0.1111, random_state=42, stratify=train_dev.iloc[:, 12]
    )
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to static files
    print(f"Saving splits to {output_dir}...")
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(output_dir, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print("--- Summary ---")
    print(f"Train: {len(train_df)} examples")
    print(f"Dev:   {len(dev_df)} examples (For tuning Alpha/Beta)")
    print(f"Test:  {len(test_df)} examples (NEVER TOUCH UNTIL FINAL PAPER)")

if __name__ == "__main__":
    create_static_splits()