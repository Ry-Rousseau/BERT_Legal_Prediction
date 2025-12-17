from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_function(examples, tokenizer):
    # CaseHOLD CSV format: column 1 = context, columns 2-6 = 5 holdings, column 12 = label
    # We need to repeat the context 5 times, once for each choice
    first_sentences = [[str(context)] * 5 for context in examples["1"]]

    # Grab the 5 candidate columns (columns 2-6)
    second_sentences = [
        [str(examples[str(i+2)][j]) for i in range(5)]
        for j in range(len(examples["1"]))
    ]

    # Flatten inputs for tokenization
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        padding="max_length",
        max_length=512 # increasing to 512 from 216 for cloud compute run
    )

    # Un-flatten: Group back into sets of 5
    return {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}

def get_dataloaders(tokenizer, data_path=None, train_file=None, dev_file=None, test_file=None, return_dict=False):
    """
    Load and process data for multiple choice task.

    Args:
        tokenizer: HuggingFace tokenizer
        data_path: Path to single CSV file (for backwards compatibility)
        train_file: Path to training split CSV
        dev_file: Path to dev/validation split CSV
        test_file: Path to test split CSV
        return_dict: If True, return dict with train/dev/test. If False, return only train (default)

    Returns:
        If return_dict=True: dict with 'train', 'dev', 'test' datasets
        If return_dict=False: train dataset only
    """
    from datasets import Value

    # Handle backwards compatibility: single file
    if data_path is not None:
        dataset = load_dataset("csv", data_files=data_path, encoding="utf-8")
        encoded_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        encoded_dataset = encoded_dataset.rename_column("12", "labels")

        # Convert string labels to integers, handling None/empty values
        def convert_label(example):
            label = example["labels"]
            # Handle None, empty string, or whitespace
            if label is None or (isinstance(label, str) and label.strip() == ""):
                raise ValueError(f"Found invalid/missing label")
            # Convert to float first (handles "3.0"), then to int
            try:
                return {"labels": int(float(label))}
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert label '{label}' to int: {e}")

        encoded_dataset = encoded_dataset.map(
            convert_label,
            desc="Converting labels to int"
        )
        encoded_dataset = encoded_dataset.cast_column("labels", Value("int64"))
        encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return encoded_dataset["train"]

    # Handle pre-split files
    data_files = {}
    if train_file:
        data_files["train"] = train_file
    if dev_file:
        data_files["dev"] = dev_file
    if test_file:
        data_files["test"] = test_file

    if not data_files:
        # Default to pre-split files
        data_files = {
            "train": "data/casehold/train.csv",
            "dev": "data/casehold/dev.csv",
            "test": "data/casehold/test.csv"
        }

    # Load all splits
    dataset = load_dataset("csv", data_files=data_files, encoding="utf-8")

    # Process each split
    encoded_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Rename and cast labels for all splits
    for split in encoded_dataset.keys():
        encoded_dataset[split] = encoded_dataset[split].rename_column("12", "labels")

        # Convert string labels to integers, handling None/empty values
        def convert_label(example):
            label = example["labels"]
            # Handle None, empty string, or whitespace
            if label is None or (isinstance(label, str) and label.strip() == ""):
                # Provide more context for debugging
                raise ValueError(
                    f"Found invalid/missing label in {split} split\n"
                    f"  Label value: {repr(label)}\n"
                    f"  Label type: {type(label)}\n"
                    f"  Example keys: {list(example.keys())[:10]}\n"
                    f"  First column value: {repr(str(example.get('0', 'N/A'))[:50])}"
                )
            # Convert to float first (handles "3.0"), then to int
            try:
                return {"labels": int(float(label))}
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert label '{label}' (type: {type(label)}) to int in {split} split: {e}\n"
                    f"  Example keys: {list(example.keys())[:10]}"
                )

        encoded_dataset[split] = encoded_dataset[split].map(
            convert_label,
            desc="Converting labels to int"
        )
        encoded_dataset[split] = encoded_dataset[split].cast_column("labels", Value("int64"))
        encoded_dataset[split].set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Return based on return_dict flag
    if return_dict:
        return encoded_dataset
    else:
        # Backwards compatibility: return only train
        return encoded_dataset["train"] if "train" in encoded_dataset else list(encoded_dataset.values())[0]

def test_data_loading():
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    print("\n=== Test 1: Loading pre-split files (default) ===")
    datasets = get_dataloaders(tokenizer, return_dict=True)

    print(f"Train: {len(datasets['train'])} examples")
    print(f"Dev:   {len(datasets['dev'])} examples")
    print(f"Test:  {len(datasets['test'])} examples")

    # --- INSPECTION ---
    print("\n=== Inspecting Train Split ===")
    example = datasets['train'][0]

    # Check 1: Keys
    expected_keys = {'input_ids', 'attention_mask', 'labels'}
    assert expected_keys.issubset(example.keys()), f"Missing keys! Found: {example.keys()}"

    # Check 2: Shapes
    # Input IDs should be [5, Seq_Len] (5 choices)
    input_ids = example['input_ids']
    print(f"Example Input Shape: {input_ids.shape}")

    if input_ids.shape[0] != 5:
        print("ERROR: Expected 5 choices per question.")
    else:
        print("Correct: Found 5 choices per question.")

    # Check 3: Labels
    print(f"Label: {example['labels']} (Should be an integer 0-4)")

    # Check 4: Decoding
    print("\n--- Decoding First Choice ---")
    print(tokenizer.decode(input_ids[0])[:200] + "...")

    print("\n=== Test 2: Backwards compatibility (single file) ===")
    train_only = get_dataloaders(tokenizer, data_path="data/casehold/train.csv")
    print(f"Loaded {len(train_only)} training examples")

    print("\n=== All tests passed! ===")

"""
USAGE EXAMPLES:

# Example 1: Load pre-split files (RECOMMENDED)
datasets = get_dataloaders(tokenizer, return_dict=True)
train_dataset = datasets['train']
eval_dataset = datasets['dev']
test_dataset = datasets['test']

# Example 2: Load only training data (backwards compatible)
train_dataset = get_dataloaders(tokenizer)

# Example 3: Specify custom split files
datasets = get_dataloaders(
    tokenizer,
    train_file="data/casehold/train.csv",
    dev_file="data/casehold/dev.csv",
    test_file="data/casehold/test.csv",
    return_dict=True
)

# Example 4: Old method - single file (backwards compatibility)
train_dataset = get_dataloaders(tokenizer, data_path="data/casehold/train.csv")
"""

if __name__ == "__main__":
    test_data_loading()