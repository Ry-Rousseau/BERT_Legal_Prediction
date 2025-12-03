# prepare_legal_data.py (Simplified)
import pandas as pd
import numpy as np
import os
import unicodedata
from sklearn.model_selection import train_test_split

# ================= CONFIGURATION =================
DATA_PATH = "data/justice.csv" 
TEXT_COLUMN = "facts"
TARGET_COLUMN = "first_party_winner"
OUTPUT_DIR = "data/data_tsv/"

def preprocess_text(text):
    if text is None: return ""
    text = str(text).strip()
    text = unicodedata.normalize("NFKC", text)
    text = text.lower() 
    text = pd.Series([text]).str.replace(r'<[^<>]*>', '', regex=True)[0]
    return text

def save_to_tsv(df, filename):
    out_df = df.copy()
    out_df.reset_index(inplace=True, drop=True)
    out_df['index'] = out_df.index
    
    # RENAME to standard names
    out_df = out_df.rename(columns={TEXT_COLUMN: 'text', TARGET_COLUMN: 'label'})
    
    # SELECT only what we need: Index, Text, Label
    out_df = out_df[['index', 'text', 'label']]
    
    path = os.path.join(OUTPUT_DIR, filename)
    print(f"Saving {len(out_df)} rows to {path}...")
    out_df.to_csv(path, sep='\t', index=False)

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print("Loading and cleaning...")
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN], inplace=True)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_text)
    
    print("Splitting...")
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df[TARGET_COLUMN], random_state=42)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[TARGET_COLUMN], random_state=42)
    
    save_to_tsv(train_df, "train.tsv")
    save_to_tsv(dev_df, "dev.tsv")
    save_to_tsv(test_df, "test.tsv")
