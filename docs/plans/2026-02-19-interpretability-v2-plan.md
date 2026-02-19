# Interpretability Dev v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `Interpretability_Dev_v2.ipynb` — a new notebook extending the v1 interpretability pipeline with error analysis, confidence metrics, inference timing, and PKD-Last attribution coverage, all run on the test set.

**Architecture:** Single notebook with `MODE = "mini"/"full"` toggle. Teacher loaded once; student models loaded/deleted one at a time. All results written to CSV incrementally for crash recovery. Reuses all v1 helper functions unchanged; adds prediction pass, error-group sampling, and 4 summary plots.

**Tech Stack:** PyTorch, HuggingFace Transformers, Captum (IntegratedGradients), pandas, matplotlib, seaborn, scikit-learn. Existing `data_loader.py`, `EvalxNLP` package.

**Design doc:** `docs/plans/2026-02-19-interpretability-v2-design.md`

---

## Exact Checkpoint Paths

These are verified from the eval CSVs — use them verbatim in the notebook.

```
Teacher:
  results/training_runs/fine_tuned_base_bert_legal_teacher/run_lr_1e-05/checkpoint-1325

Vanilla KD (best: α=0.7, T=20, epoch 3):
  results/training_runs/vanilla_kd_grid_search/vanilla_L6_A0p7_T20/checkpoint-3975

PKD-Skip best per size:
  L6 B1000 epoch3: results/training_runs/pkd_skip_grid_search/pkd_skip_L6_B1000/checkpoint-3975
  L4 B100  epoch3: results/training_runs/pkd_skip_grid_search/pkd_skip_L4_B100/checkpoint-3975
  L3 B1000 epoch4: results/training_runs/pkd_skip_grid_search/pkd_skip_L3_B1000/checkpoint-5300
  L2 B500  epoch4: results/training_runs/pkd_skip_grid_search/pkd_skip_L2_B500/checkpoint-5300

PKD-Last best per size:
  L6 B500 epoch3: results/training_runs/pkd_last_grid_search/pkd_last_L6_B500/checkpoint-3975
  L4 B500 epoch3: results/training_runs/pkd_last_grid_search/pkd_last_L4_B500/checkpoint-3975
  L3 B500 epoch3: results/training_runs/pkd_last_grid_search/pkd_last_L3_B500/checkpoint-3975
  L2 B500 epoch4: results/training_runs/pkd_last_grid_search/pkd_last_L2_B500/checkpoint-5300
```

---

## Output Directory Structure

All new output goes under `results/interpretability_v2/` — v1 results in `integrated_gradients_pkd_skip_results/` are never touched.

```
results/interpretability_v2/
    {model_name}/
        predictions.csv
        quantitative_correct.csv
        quantitative_incorrect.csv
        summary_stats_correct.txt
        summary_stats_incorrect.txt
        qualitative/
            correct/sample_000/{metadata.json, attribution_heatmap_filtered.png, top15_tokens_combined.png}
            incorrect/sample_000/...
    summary_all_models.csv
    pkd_last_similarity_summary.txt
    plots/
        accuracy_by_model.png
        confidence_correct_vs_incorrect.png
        cosine_sim_by_layers.png
        inference_time_by_model.png
```

---

## Task 1: Notebook scaffold — Cell 1 (Setup & Config)

**File:** Create `Interpretability_Dev_v2.ipynb`

**Step 1: Create the notebook with Cell 1 content**

The very first cell sets every tunable constant. Nothing else in the notebook uses magic numbers.

```python
# ============================================================
# CONFIG — change MODE here before running
# ============================================================
MODE = "mini"   # "mini" = smoke test (~2 min), "full" = production run

SEED = 42
OUTPUT_BASE = "results/interpretability_v2"
N_QUAL  = {"mini": 3, "full": 15}[MODE]   # qualitative samples per error group
N_QUANT = {"mini": 6, "full": 30}[MODE]   # quantitative samples per error group
N_PRED  = {"mini": 20, "full": None}[MODE] # examples for prediction pass (None = all)

import os, gc, json, time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoModelForMultipleChoice, AutoTokenizer, Trainer, TrainingArguments
from captum.attr import IntegratedGradients
from datasets import load_dataset
from data_loader import get_dataloaders
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

os.makedirs(OUTPUT_BASE, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"MODE={MODE}  device={device}  N_qual={N_QUAL}  N_quant={N_QUANT}")
```

**Step 2: Verify cell runs without error**

Execute Cell 1. Expected output:
```
MODE=mini  device=cuda  N_qual=3  N_quant=6
```
(or `device=cpu` if no GPU — rest of notebook still works.)

---

## Task 2: Model Registry — Cell 2

**Step 1: Write Cell 2**

```python
# ============================================================
# MODEL REGISTRY
# ============================================================
MODELS = [
    {
        "name": "vanilla_L6_A0p7_T20",
        "path": "results/training_runs/vanilla_kd_grid_search/vanilla_L6_A0p7_T20/checkpoint-3975",
        "strategy": "vanilla_kd", "layers": 6
    },
    {
        "name": "pkd_skip_L6_B1000",
        "path": "results/training_runs/pkd_skip_grid_search/pkd_skip_L6_B1000/checkpoint-3975",
        "strategy": "pkd_skip", "layers": 6
    },
    {
        "name": "pkd_skip_L4_B100",
        "path": "results/training_runs/pkd_skip_grid_search/pkd_skip_L4_B100/checkpoint-3975",
        "strategy": "pkd_skip", "layers": 4
    },
    {
        "name": "pkd_skip_L3_B1000",
        "path": "results/training_runs/pkd_skip_grid_search/pkd_skip_L3_B1000/checkpoint-5300",
        "strategy": "pkd_skip", "layers": 3
    },
    {
        "name": "pkd_skip_L2_B500",
        "path": "results/training_runs/pkd_skip_grid_search/pkd_skip_L2_B500/checkpoint-5300",
        "strategy": "pkd_skip", "layers": 2
    },
    {
        "name": "pkd_last_L6_B500",
        "path": "results/training_runs/pkd_last_grid_search/pkd_last_L6_B500/checkpoint-3975",
        "strategy": "pkd_last", "layers": 6
    },
    {
        "name": "pkd_last_L4_B500",
        "path": "results/training_runs/pkd_last_grid_search/pkd_last_L4_B500/checkpoint-3975",
        "strategy": "pkd_last", "layers": 4
    },
    {
        "name": "pkd_last_L3_B500",
        "path": "results/training_runs/pkd_last_grid_search/pkd_last_L3_B500/checkpoint-3975",
        "strategy": "pkd_last", "layers": 3
    },
    {
        "name": "pkd_last_L2_B500",
        "path": "results/training_runs/pkd_last_grid_search/pkd_last_L2_B500/checkpoint-5300",
        "strategy": "pkd_last", "layers": 2
    },
]

TEACHER_PATH = "results/training_runs/fine_tuned_base_bert_legal_teacher/run_lr_1e-05/checkpoint-1325"

# Verify all paths exist
print("Verifying model paths...")
all_ok = True
for m in [{"name": "teacher", "path": TEACHER_PATH}] + MODELS:
    exists = Path(m["path"]).exists()
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {m['name']}")
    if not exists:
        all_ok = False
print("All paths OK" if all_ok else "WARNING: Some paths are missing!")
```

**Step 2: Execute and verify**

All lines should print `[OK]`. If any show `[MISSING]`, the checkpoint path is wrong — check the exact directory name with `ls results/training_runs/...`.

---

## Task 3: Data Loading — Cell 3

**Step 1: Write Cell 3**

```python
# ============================================================
# DATA LOADING
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Tokenized test set (for model inference & attribution)
datasets_all = get_dataloaders(tokenizer, return_dict=True)
test_dataset_tokenized = datasets_all['test']

# Raw test set (for metadata: case text, holdings)
test_dataset_raw = load_dataset(
    'csv',
    data_files={'test': 'data/casehold/test.csv'},
    split='test'
)

# In mini mode, slice to first N_PRED examples
if N_PRED is not None:
    test_dataset_tokenized = test_dataset_tokenized.select(range(N_PRED))
    test_dataset_raw = test_dataset_raw.select(range(N_PRED))

print(f"Test set size: {len(test_dataset_tokenized)} examples")
print(f"Raw test set size: {len(test_dataset_raw)} examples")

# Sanity check alignment
assert len(test_dataset_tokenized) == len(test_dataset_raw), "Sizes must match!"
print("Alignment check passed.")
```

**Step 2: Execute and verify**

Mini mode: prints `Test set size: 20 examples`. Full mode: ~5,900.

---

## Task 4: Load Teacher — Cell 4

**Step 1: Write Cell 4**

```python
# ============================================================
# LOAD TEACHER (stays in memory the whole notebook)
# ============================================================
print("Loading teacher model...")
teacher_model = AutoModelForMultipleChoice.from_pretrained(TEACHER_PATH).to(device).eval()
print(f"Teacher loaded: {len(teacher_model.bert.encoder.layer)} layers")
```

**Step 2: Execute and verify**

Expected: `Teacher loaded: 12 layers`

---

## Task 5: Reuse v1 helper functions — Cell 5

These are copied verbatim from `Interpretability_Dev_EvalxNLP.ipynb`. Do not modify them.

**Step 1: Write Cell 5**

```python
# ============================================================
# HELPER FUNCTIONS (unchanged from v1)
# ============================================================

def get_attributions(model, input_ids, attention_mask, method='ig'):
    """Get token attributions for a single input. Returns (seq_len,) numpy array."""
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    def forward_wrapper(input_embeds):
        input_embeds = input_embeds.unsqueeze(1)
        attention_mask_expanded = attention_mask.unsqueeze(1)
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask_expanded)
        return outputs.logits.squeeze(1)

    with torch.no_grad():
        embeddings = model.bert.embeddings(input_ids.squeeze(1))
    embeddings = embeddings.detach().clone().unsqueeze(0)
    embeddings.requires_grad = True

    attr_method = IntegratedGradients(forward_wrapper)
    attributions = attr_method.attribute(embeddings, n_steps=50)
    return attributions.sum(dim=-1).squeeze().detach().cpu().numpy()


def filter_tokens_and_attributions(tokens, attributions):
    """Filter out [CLS]/[SEP]/[PAD]/subwords/punctuation."""
    filtered_tokens, filtered_attr, filtered_indices = [], [], []
    for idx, (token, attr) in enumerate(zip(tokens, attributions)):
        if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
            continue
        if token.startswith('##'):
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        if all(c in '.,;:!?()[]{}"\'-/' for c in token):
            continue
        filtered_tokens.append(token)
        filtered_attr.append(attr)
        filtered_indices.append(idx)
    return filtered_tokens, np.array(filtered_attr), filtered_indices


def save_attribution_heatmap(tokens, teacher_attr, student_attr, output_path,
                              max_tokens=30, title="Attribution Comparison"):
    """Save side-by-side teacher/student attribution heatmap as PNG."""
    tokens = tokens[:max_tokens]
    teacher_attr = teacher_attr[:max_tokens]
    student_attr = student_attr[:max_tokens]

    teacher_norm = teacher_attr / (np.abs(teacher_attr).max() + 1e-10)
    student_norm = student_attr / (np.abs(student_attr).max() + 1e-10)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), dpi=150)
    sns.heatmap(teacher_norm.reshape(1, -1), cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=tokens, yticklabels=['Teacher'],
                cbar_kws={'label': 'Attribution'}, ax=ax1)
    ax1.set_title('Teacher Attributions', fontsize=14)
    ax1.set_xticklabels(tokens, rotation=90, fontsize=8)
    sns.heatmap(student_norm.reshape(1, -1), cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=tokens, yticklabels=['Student'],
                cbar_kws={'label': 'Attribution'}, ax=ax2)
    ax2.set_title('Student Attributions', fontsize=14)
    ax2.set_xticklabels(tokens, rotation=90, fontsize=8)
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def save_top15_bar_chart(tokens, teacher_attr, student_attr, output_path, top_k=15):
    """Save top-k combined tokens bar chart as PNG."""
    teacher_top_idx = np.argsort(np.abs(teacher_attr))[-top_k:][::-1]
    student_top_idx = np.argsort(np.abs(student_attr))[-top_k:][::-1]
    all_top_idx = sorted(set(teacher_top_idx) | set(student_top_idx))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    x = np.arange(len(all_top_idx))
    width = 0.35
    ax.bar(x - width/2, [teacher_attr[i] for i in all_top_idx], width, label='Teacher', alpha=0.8)
    ax.bar(x + width/2, [student_attr[i] for i in all_top_idx], width, label='Student', alpha=0.8)
    ax.set_ylabel('Attribution Score', fontsize=10)
    ax.set_title(f'Top-{top_k} Most Important Tokens (Combined)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([tokens[i] for i in all_top_idx], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def extract_example_metadata(raw_example, label_idx):
    """Extract context + holdings from raw CSV row."""
    return {
        'context': raw_example['1'],
        'holdings': [raw_example[str(i+2)] for i in range(5)],
        'label': int(float(raw_example['12']))
    }


def compute_cosine_similarity(vec1, vec2):
    return sk_cosine(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]


print("v1 helper functions loaded.")
```

**Step 2: Execute and verify**

Expected: `v1 helper functions loaded.` with no errors.

---

## Task 6: New helper — Prediction Pass — Cell 6

**Step 1: Write Cell 6**

```python
# ============================================================
# NEW: PREDICTION PASS WITH CONFIDENCE + TIMING
# ============================================================

def run_prediction_pass(model, model_name, dataset_tokenized, dataset_raw):
    """
    Run full forward pass over dataset. Writes predictions.csv.
    Skips if file already exists (crash recovery).

    Returns: pd.DataFrame with columns:
      example_idx, true_label, predicted_label, is_correct,
      confidence, inference_time_sec
    """
    out_dir = Path(OUTPUT_BASE) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"

    if csv_path.exists():
        print(f"  [SKIP] predictions.csv already exists for {model_name}")
        return pd.read_csv(csv_path)

    print(f"  Running prediction pass for {model_name} ({len(dataset_tokenized)} examples)...")

    eval_args = TrainingArguments(
        output_dir="./temp_eval_v2",
        per_device_eval_batch_size=4,
        dataloader_num_workers=0,
        fp16=(device == "cuda"),
        no_cuda=(device != "cuda"),
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(model=model, args=eval_args)

    start = time.time()
    pred_output = trainer.predict(dataset_tokenized)
    elapsed = time.time() - start

    logits = pred_output.predictions          # (N, 5)
    true_labels = pred_output.label_ids       # (N,)
    predicted_labels = np.argmax(logits, axis=1)

    # Softmax for confidence
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)            # max softmax prob

    per_example_time = elapsed / len(dataset_tokenized)

    rows = []
    for i in range(len(dataset_tokenized)):
        rows.append({
            'example_idx': i,
            'true_label': int(true_labels[i]),
            'predicted_label': int(predicted_labels[i]),
            'is_correct': bool(true_labels[i] == predicted_labels[i]),
            'confidence': float(confidence[i]),
            'inference_time_sec': per_example_time,
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    acc = df['is_correct'].mean()
    print(f"  Accuracy: {acc:.4f} | Avg confidence: {df['confidence'].mean():.4f} | Time/example: {per_example_time*1000:.1f}ms")
    return df


print("Prediction pass helper defined.")
```

**Step 2: Smoke test — Cell 6b**

```python
# Smoke test: run prediction pass on teacher with mini data
print("=== SMOKE TEST: teacher prediction pass ===")
teacher_preds = run_prediction_pass(
    teacher_model, "_SMOKE_TEST_teacher",
    test_dataset_tokenized, test_dataset_raw
)
print(teacher_preds.head(3))
assert 'is_correct' in teacher_preds.columns
assert 'confidence' in teacher_preds.columns
assert len(teacher_preds) == len(test_dataset_tokenized)
print("Smoke test PASSED.")
```

**Step 3: Execute and verify**

Mini mode: runs over 20 examples in ~5 seconds. Expected output shows accuracy + confidence stats. All assertions pass.

---

## Task 7: New helper — Error-Group Sampling — Cell 7

**Step 1: Write Cell 7**

```python
# ============================================================
# NEW: ERROR-GROUP SAMPLING
# ============================================================

def sample_by_error_group(predictions_df, n, seed=SEED):
    """
    Sample n indices from each error group.

    Returns:
        correct_indices: list of example_idx values
        incorrect_indices: list of example_idx values

    If fewer than n examples exist in a group, returns all available.
    """
    np.random.seed(seed)

    correct_pool = predictions_df[predictions_df['is_correct'] == True]['example_idx'].values
    incorrect_pool = predictions_df[predictions_df['is_correct'] == False]['example_idx'].values

    n_correct = min(n, len(correct_pool))
    n_incorrect = min(n, len(incorrect_pool))

    correct_sample = np.random.choice(correct_pool, n_correct, replace=False).tolist()
    incorrect_sample = np.random.choice(incorrect_pool, n_incorrect, replace=False).tolist()

    print(f"  Sampled {n_correct} correct, {n_incorrect} incorrect examples")
    return correct_sample, incorrect_sample


print("Error-group sampling helper defined.")

# Quick unit test
_fake_preds = pd.DataFrame({
    'example_idx': range(10),
    'is_correct': [True]*7 + [False]*3
})
_c, _i = sample_by_error_group(_fake_preds, n=5, seed=42)
assert len(_c) == 5
assert len(_i) == 3   # only 3 incorrect available
print("Unit test PASSED.")
```

**Step 2: Execute and verify**

Expected: `Unit test PASSED.`

---

## Task 8: New helpers — Per-Group Attribution Analysis — Cell 8

**Step 1: Write Cell 8**

```python
# ============================================================
# NEW: ATTRIBUTION ANALYSIS WITH ERROR-GROUP LABELING
# ============================================================

def run_qualitative_for_group(teacher_model, student_model, student_name,
                               example_indices, error_group,
                               dataset_tokenized, dataset_raw,
                               predictions_df):
    """
    Run qualitative attribution analysis for one error group (correct/incorrect).
    Saves heatmaps + metadata to:
      results/interpretability_v2/{student_name}/qualitative/{error_group}/sample_NNN/
    """
    group_dir = Path(OUTPUT_BASE) / student_name / "qualitative" / error_group
    group_dir.mkdir(parents=True, exist_ok=True)

    for sample_num, idx in enumerate(example_indices):
        sample_dir = group_dir / f"sample_{sample_num:03d}"
        sample_dir.mkdir(exist_ok=True)

        example_tok = dataset_tokenized[idx]
        example_raw = dataset_raw[idx]
        pred_row = predictions_df[predictions_df['example_idx'] == idx].iloc[0]

        label = int(example_tok['labels'].item())
        tokens = tokenizer.convert_ids_to_tokens(example_tok['input_ids'][label])

        teacher_attr = get_attributions(teacher_model, example_tok['input_ids'][label],
                                        example_tok['attention_mask'][label], 'ig')
        student_attr = get_attributions(student_model, example_tok['input_ids'][label],
                                        example_tok['attention_mask'][label], 'ig')

        filtered_tokens, filtered_teacher, _ = filter_tokens_and_attributions(tokens, teacher_attr)
        _, filtered_student, _ = filter_tokens_and_attributions(tokens, student_attr)

        metadata = extract_example_metadata(example_raw, label)
        metadata.update({
            'example_idx': int(idx),
            'sample_num': sample_num,
            'error_group': error_group,
            'is_correct': bool(pred_row['is_correct']),
            'confidence': float(pred_row['confidence']),
            'inference_time_sec': float(pred_row['inference_time_sec']),
        })

        with open(sample_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        save_attribution_heatmap(
            filtered_tokens, filtered_teacher, filtered_student,
            sample_dir / 'attribution_heatmap_filtered.png',
            max_tokens=30,
            title=f"{student_name} [{error_group}] Sample {sample_num}"
        )
        save_top15_bar_chart(
            filtered_tokens, filtered_teacher, filtered_student,
            sample_dir / 'top15_tokens_combined.png',
            top_k=15
        )

    print(f"    Qualitative [{error_group}]: {len(example_indices)} samples saved to {group_dir}")


def run_quantitative_for_group(teacher_model, student_model, student_name,
                                example_indices, error_group, dataset_tokenized):
    """
    Run quantitative attribution metrics for one error group.
    Saves results to:
      results/interpretability_v2/{student_name}/quantitative_{error_group}.csv
      results/interpretability_v2/{student_name}/summary_stats_{error_group}.txt
    """
    results = []
    for idx in example_indices:
        example = dataset_tokenized[idx]
        label = int(example['labels'].item())

        teacher_attr = get_attributions(teacher_model, example['input_ids'][label],
                                        example['attention_mask'][label], 'ig')
        student_attr = get_attributions(student_model, example['input_ids'][label],
                                        example['attention_mask'][label], 'ig')

        cosine_sim = compute_cosine_similarity(teacher_attr, student_attr)
        corr = np.corrcoef(teacher_attr, student_attr)[0, 1]
        k = 10
        teacher_top = set(np.argsort(np.abs(teacher_attr))[-k:])
        student_top = set(np.argsort(np.abs(student_attr))[-k:])
        overlap = len(teacher_top & student_top) / k

        results.append({
            'example_idx': int(idx),
            'error_group': error_group,
            'cosine_similarity': cosine_sim,
            'correlation': corr,
            'top10_overlap': overlap
        })

    df = pd.DataFrame(results)
    out_dir = Path(OUTPUT_BASE) / student_name
    csv_path = out_dir / f"quantitative_{error_group}.csv"
    df.to_csv(csv_path, index=False)

    summary_path = out_dir / f"summary_stats_{error_group}.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Quantitative Summary: {student_name} [{error_group}]\n{'='*50}\n\n")
        for col in ['cosine_similarity', 'correlation', 'top10_overlap']:
            f.write(f"{col}:\n  Mean: {df[col].mean():.4f}\n  Std:  {df[col].std():.4f}\n\n")

    print(f"    Quantitative [{error_group}]: {len(results)} samples | "
          f"cosine={df['cosine_similarity'].mean():.3f} | "
          f"corr={df['correlation'].mean():.3f} | "
          f"overlap={df['top10_overlap'].mean():.3f}")
    return df


print("Attribution analysis helpers defined.")
```

**Step 2: Execute and verify**

Expected: `Attribution analysis helpers defined.`

---

## Task 9: Mini end-to-end test — Cell 9

This is the key verification cell. Run everything on one student model with `N_QUAL=3` before the main loop.

**Step 1: Write Cell 9**

```python
# ============================================================
# MINI END-TO-END TEST (verifies all helpers work together)
# ============================================================
print("=== MINI END-TO-END TEST ===")
_test_model_cfg = MODELS[0]   # vanilla_L6
_test_student = AutoModelForMultipleChoice.from_pretrained(_test_model_cfg["path"]).to(device).eval()

_preds = run_prediction_pass(_test_student, f"_TEST_{_test_model_cfg['name']}",
                              test_dataset_tokenized, test_dataset_raw)

_correct_idx, _incorrect_idx = sample_by_error_group(_preds, n=N_QUAL)

print("[1/2] Qualitative correct...")
run_qualitative_for_group(teacher_model, _test_student, f"_TEST_{_test_model_cfg['name']}",
                          _correct_idx, "correct",
                          test_dataset_tokenized, test_dataset_raw, _preds)

print("[2/2] Qualitative incorrect...")
if _incorrect_idx:
    run_qualitative_for_group(teacher_model, _test_student, f"_TEST_{_test_model_cfg['name']}",
                              _incorrect_idx, "incorrect",
                              test_dataset_tokenized, test_dataset_raw, _preds)

print("[3/3] Quantitative...")
run_quantitative_for_group(teacher_model, _test_student, f"_TEST_{_test_model_cfg['name']}",
                           _correct_idx + _incorrect_idx, "all", test_dataset_tokenized)

# Verify outputs exist
_test_dir = Path(OUTPUT_BASE) / f"_TEST_{_test_model_cfg['name']}"
assert (_test_dir / "predictions.csv").exists(), "predictions.csv missing"
assert (_test_dir / "qualitative" / "correct" / "sample_000" / "metadata.json").exists()
assert (_test_dir / "qualitative" / "correct" / "sample_000" / "attribution_heatmap_filtered.png").exists()
assert (_test_dir / "qualitative" / "correct" / "sample_000" / "top15_tokens_combined.png").exists()

del _test_student
torch.cuda.empty_cache()

print("=== MINI END-TO-END TEST PASSED ===")
```

**Step 2: Execute and verify**

All assertions pass. Mini run takes ~30–60 seconds. If an assertion fails, debug the specific helper before proceeding.

---

## Task 10: Main Model Loop — Cell 10

**Step 1: Write Cell 10**

```python
# ============================================================
# MAIN LOOP: All student models
# ============================================================
print(f"\nStarting main loop over {len(MODELS)} models...")

for model_cfg in MODELS:
    name = model_cfg["name"]
    print(f"\n{'='*60}")
    print(f"Processing: {name} ({model_cfg['layers']} layers, {model_cfg['strategy']})")
    print(f"{'='*60}")

    # Load student
    student = AutoModelForMultipleChoice.from_pretrained(model_cfg["path"]).to(device).eval()
    print(f"  Loaded: {len(student.bert.encoder.layer)} layers")

    # 1. Prediction pass (crash-recoverable)
    print("  [1/3] Prediction pass...")
    preds_df = run_prediction_pass(student, name, test_dataset_tokenized, test_dataset_raw)

    # 2. Sample error groups
    correct_idx, incorrect_idx = sample_by_error_group(preds_df, n=N_QUAL)

    # 3. Qualitative analysis
    print("  [2/3] Qualitative attribution...")
    run_qualitative_for_group(teacher_model, student, name, correct_idx, "correct",
                              test_dataset_tokenized, test_dataset_raw, preds_df)
    if incorrect_idx:
        run_qualitative_for_group(teacher_model, student, name, incorrect_idx, "incorrect",
                                  test_dataset_tokenized, test_dataset_raw, preds_df)
    else:
        print("    WARNING: No incorrect examples — model may be perfect on mini slice")

    # 4. Quantitative analysis (separate per group + combined)
    print("  [3/3] Quantitative attribution...")
    quant_sample_size = N_QUANT // 2
    q_correct, q_incorrect = sample_by_error_group(preds_df, n=quant_sample_size, seed=SEED + 1)
    if q_correct:
        run_quantitative_for_group(teacher_model, student, name, q_correct,
                                   "correct", test_dataset_tokenized)
    if q_incorrect:
        run_quantitative_for_group(teacher_model, student, name, q_incorrect,
                                   "incorrect", test_dataset_tokenized)

    # Free memory
    del student
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Done: {name}")

print("\n" + "="*60)
print("MAIN LOOP COMPLETE")
print("="*60)
```

**Step 2: Execute**

In `"mini"` mode this runs in ~5–10 minutes total. In `"full"` mode expect several hours. Watch for any model that prints `[SKIP]` — that means crash recovery kicked in correctly.

---

## Task 11: Cross-Model Aggregation — Cell 11

**Step 1: Write Cell 11**

```python
# ============================================================
# CROSS-MODEL AGGREGATION
# ============================================================
print("Aggregating results across all models...")

# Load teacher prediction pass (run it now if not done)
print("Running teacher prediction pass...")
teacher_preds = run_prediction_pass(teacher_model, "teacher",
                                    test_dataset_tokenized, test_dataset_raw)

summary_rows = []

# Teacher row (no student comparison — just accuracy + confidence)
teacher_acc = teacher_preds['is_correct'].mean()
teacher_conf = teacher_preds['confidence'].mean()
teacher_time = teacher_preds['inference_time_sec'].mean()
for group in ['correct', 'incorrect']:
    group_df = teacher_preds[teacher_preds['is_correct'] == (group == 'correct')]
    summary_rows.append({
        'model_name': 'teacher',
        'strategy': 'teacher',
        'layers': 12,
        'error_group': group,
        'n_samples': len(group_df),
        'mean_cosine_sim': None,
        'mean_correlation': None,
        'mean_top10_overlap': None,
        'mean_confidence': group_df['confidence'].mean(),
        'mean_inference_time_sec': teacher_time,
        'overall_accuracy': teacher_acc,
    })

# Student models
for model_cfg in MODELS:
    name = model_cfg["name"]
    out_dir = Path(OUTPUT_BASE) / name
    preds_path = out_dir / "predictions.csv"

    if not preds_path.exists():
        print(f"  WARNING: predictions.csv missing for {name} — skipping")
        continue

    preds_df = pd.read_csv(preds_path)
    overall_acc = preds_df['is_correct'].mean()
    per_example_time = preds_df['inference_time_sec'].mean()

    for group in ['correct', 'incorrect']:
        quant_path = out_dir / f"quantitative_{group}.csv"
        group_preds = preds_df[preds_df['is_correct'] == (group == 'correct')]

        quant_df = pd.read_csv(quant_path) if quant_path.exists() else pd.DataFrame()

        summary_rows.append({
            'model_name': name,
            'strategy': model_cfg['strategy'],
            'layers': model_cfg['layers'],
            'error_group': group,
            'n_samples': len(group_preds),
            'mean_cosine_sim': quant_df['cosine_similarity'].mean() if len(quant_df) else None,
            'mean_correlation': quant_df['correlation'].mean() if len(quant_df) else None,
            'mean_top10_overlap': quant_df['top10_overlap'].mean() if len(quant_df) else None,
            'mean_confidence': group_preds['confidence'].mean() if len(group_preds) else None,
            'mean_inference_time_sec': per_example_time,
            'overall_accuracy': overall_acc,
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = Path(OUTPUT_BASE) / "summary_all_models.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved: {summary_path}")
print(summary_df[['model_name', 'layers', 'error_group', 'overall_accuracy', 'mean_confidence']].to_string())
```

**Step 2: Execute and verify**

Expected: summary table printed with all 9 student models + teacher, each with correct/incorrect rows. No NaN in `overall_accuracy`.

---

## Task 12: Summary Plots — Cell 12

**Step 1: Write Cell 12**

```python
# ============================================================
# SUMMARY PLOTS
# ============================================================
plots_dir = Path(OUTPUT_BASE) / "plots"
plots_dir.mkdir(exist_ok=True)

# Use only one row per model for accuracy/timing plots
per_model = summary_df.drop_duplicates(subset='model_name').copy()
per_model = per_model.sort_values(['layers', 'strategy'])

strategy_colors = {'teacher': '#2c7bb6', 'vanilla_kd': '#abd9e9',
                   'pkd_skip': '#fdae61', 'pkd_last': '#d7191c'}

# Plot 1: Accuracy by model
fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
colors = [strategy_colors.get(s, 'gray') for s in per_model['strategy']]
bars = ax.bar(per_model['model_name'], per_model['overall_accuracy'], color=colors)
ax.set_xticklabels(per_model['model_name'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy by Model')
ax.set_ylim(0.6, 0.8)
ax.axhline(y=per_model[per_model['strategy']=='teacher']['overall_accuracy'].values[0],
           color='gray', linestyle='--', linewidth=1, label='Teacher baseline')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=s) for s, c in strategy_colors.items()]
ax.legend(handles=legend_elements, fontsize=8)
plt.tight_layout()
plt.savefig(plots_dir / 'accuracy_by_model.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved: accuracy_by_model.png")

# Plot 2: Confidence correct vs incorrect
conf_data = summary_df[summary_df['strategy'] != 'teacher'].copy()
conf_pivot = conf_data.pivot(index='model_name', columns='error_group', values='mean_confidence')
fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
x = np.arange(len(conf_pivot))
width = 0.35
ax.bar(x - width/2, conf_pivot.get('correct', [0]*len(conf_pivot)), width,
       label='Correct predictions', color='#2ca02c', alpha=0.8)
ax.bar(x + width/2, conf_pivot.get('incorrect', [0]*len(conf_pivot)), width,
       label='Incorrect predictions', color='#d62728', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(conf_pivot.index, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean Max Softmax Probability')
ax.set_title('Model Confidence: Correct vs Incorrect Predictions')
ax.legend()
plt.tight_layout()
plt.savefig(plots_dir / 'confidence_correct_vs_incorrect.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved: confidence_correct_vs_incorrect.png")

# Plot 3: Cosine similarity by layers (skip vs last)
cosine_data = summary_df[
    (summary_df['strategy'].isin(['pkd_skip', 'pkd_last'])) &
    (summary_df['error_group'] == 'correct') &
    (summary_df['mean_cosine_sim'].notna())
].copy()
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
for strategy, grp in cosine_data.groupby('strategy'):
    grp_sorted = grp.sort_values('layers')
    ax.plot(grp_sorted['layers'], grp_sorted['mean_cosine_sim'],
            marker='o', label=strategy, linewidth=2)
ax.set_xlabel('Student Layers')
ax.set_ylabel('Mean Cosine Similarity (Teacher vs Student)')
ax.set_title('Attribution Similarity by Layer Count\n(correct predictions)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'cosine_sim_by_layers.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved: cosine_sim_by_layers.png")

# Plot 4: Inference time by model
fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
per_model_sorted = per_model.sort_values('layers')
colors4 = [strategy_colors.get(s, 'gray') for s in per_model_sorted['strategy']]
ax.bar(per_model_sorted['model_name'],
       per_model_sorted['mean_inference_time_sec'] * 1000,   # convert to ms
       color=colors4)
ax.set_xticklabels(per_model_sorted['model_name'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Per-Example Inference Time (ms)')
ax.set_title('Inference Time by Model')
legend_elements2 = [Patch(facecolor=c, label=s) for s, c in strategy_colors.items()]
ax.legend(handles=legend_elements2, fontsize=8)
plt.tight_layout()
plt.savefig(plots_dir / 'inference_time_by_model.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved: inference_time_by_model.png")

print("\nAll 4 plots saved to:", plots_dir)
```

**Step 2: Execute and verify**

All 4 `.png` files appear in `results/interpretability_v2/plots/`. Open them to confirm they render correctly (axes labeled, no empty plots).

---

## Task 13: PKD-Last Similarity Report — Cell 13

**Step 1: Write Cell 13**

```python
# ============================================================
# PKD-LAST SIMILARITY REPORT (matches format of v1 summary_readable.txt)
# ============================================================
report_path = Path(OUTPUT_BASE) / "pkd_last_similarity_summary.txt"

pkd_last_models = [m for m in MODELS if m['strategy'] == 'pkd_last']

with open(report_path, 'w') as f:
    f.write("Cross-Model Attribution Comparison Summary — PKD-Last\n")
    f.write("="*60 + "\n\n")

    for model_cfg in pkd_last_models:
        name = model_cfg["name"]
        layers = model_cfg["layers"]

        for group in ['correct', 'incorrect']:
            quant_path = Path(OUTPUT_BASE) / name / f"quantitative_{group}.csv"
            if not quant_path.exists():
                continue
            df = pd.read_csv(quant_path)

            f.write(f"{name} ({layers} layers) — [{group}]\n")
            f.write("-" * 40 + "\n")
            f.write(f"  N samples:         {len(df)}\n")
            f.write(f"  Cosine Similarity: {df['cosine_similarity'].mean():.4f} "
                    f"± {df['cosine_similarity'].std():.4f}\n")
            f.write(f"  Correlation:       {df['correlation'].mean():.4f} "
                    f"± {df['correlation'].std():.4f}\n")
            f.write(f"  Top-10 Overlap:    {df['top10_overlap'].mean():.4f} "
                    f"± {df['top10_overlap'].std():.4f}\n\n")

    # Append PKD-Skip for direct comparison
    f.write("\n" + "="*60 + "\n")
    f.write("PKD-Skip (best per size) — for comparison\n")
    f.write("="*60 + "\n\n")

    pkd_skip_models = [m for m in MODELS if m['strategy'] == 'pkd_skip']
    for model_cfg in pkd_skip_models:
        name = model_cfg["name"]
        layers = model_cfg["layers"]
        for group in ['correct', 'incorrect']:
            quant_path = Path(OUTPUT_BASE) / name / f"quantitative_{group}.csv"
            if not quant_path.exists():
                continue
            df = pd.read_csv(quant_path)
            f.write(f"{name} ({layers} layers) — [{group}]\n")
            f.write("-" * 40 + "\n")
            f.write(f"  N samples:         {len(df)}\n")
            f.write(f"  Cosine Similarity: {df['cosine_similarity'].mean():.4f} "
                    f"± {df['cosine_similarity'].std():.4f}\n")
            f.write(f"  Correlation:       {df['correlation'].mean():.4f} "
                    f"± {df['correlation'].std():.4f}\n")
            f.write(f"  Top-10 Overlap:    {df['top10_overlap'].mean():.4f} "
                    f"± {df['top10_overlap'].std():.4f}\n\n")

print(f"PKD-Last similarity report saved to: {report_path}")
```

**Step 2: Execute and verify**

File created at `results/interpretability_v2/pkd_last_similarity_summary.txt`. Open it and confirm it reads like the existing `summary_readable.txt` in `integrated_gradients_pkd_skip_results/`.

---

## Task 14: Switch to Full Mode & Run

**Step 1: Change the MODE variable**

In Cell 1, change:
```python
MODE = "mini"
```
to:
```python
MODE = "full"
```

**Step 2: Kernel → Restart & Run All**

This is the production run. Expected total time: several hours on GPU. The loop prints progress per model. If it crashes, simply re-run — all completed models will show `[SKIP]` on the prediction pass.

**Step 3: Verify final outputs**

After the run completes:
```
results/interpretability_v2/
    summary_all_models.csv          ← should have 20 rows (10 models × 2 groups)
    pkd_last_similarity_summary.txt
    plots/
        accuracy_by_model.png
        confidence_correct_vs_incorrect.png
        cosine_sim_by_layers.png
        inference_time_by_model.png
    {each model_name}/
        predictions.csv
        quantitative_correct.csv
        quantitative_incorrect.csv
        qualitative/correct/sample_000..sample_014/
        qualitative/incorrect/sample_000..sample_014/
```

---

## Task 15: Commit results

**Step 1: Stage and commit the notebook + design docs**

```bash
git add Interpretability_Dev_v2.ipynb
git add docs/plans/2026-02-19-interpretability-v2-plan.md
git commit -m "add interpretability v2 notebook with error analysis, confidence, and PKD-last coverage"
```

Note: `results/interpretability_v2/` is gitignored (large files). Commit only the notebook itself.

**Step 2: Verify git status is clean**

```bash
git status
```

---

## Notes on Debugging

- **`incorrect_idx` is empty in mini mode:** Normal if the model gets all 20 test examples correct. Switch to `N_PRED = 50` temporarily in mini mode to see more diversity.
- **CUDA OOM during attribution:** `n_steps=50` in IG is the cost driver. Reduce to `n_steps=25` if memory is tight.
- **Prediction pass hangs:** The `TrainingArguments` uses `fp16=True` — ensure the GPU supports it. On CPU fallback, remove `fp16`.
- **Mismatched dataset sizes:** The raw and tokenized datasets must have the same length. This is guaranteed as long as both use `test.csv` — verify with the assertion in Cell 3.
