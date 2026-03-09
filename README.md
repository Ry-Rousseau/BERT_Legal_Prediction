# Legal-BERT Knowledge Distillation on CaseHOLD

This repository studies **Knowledge Distillation** to compress `nlpaueb/legal-bert-base-uncased` while preserving performance on the [CaseHOLD](https://huggingface.co/datasets/casehold/casehold) legal reasoning benchmark (multiple-choice, 5 options per prompt).

---

## Quick Start: What to Look At

This repo is a research workspace — most scripts are one-off training/eval runs and are effectively deprecated. The primary artifacts are:

| File | Purpose |
|---|---|
| **[`Interpretability_Dev_v2.ipynb`](Interpretability_Dev_v2.ipynb)** | **Main analysis notebook.** Integrated Gradients attribution analysis comparing teacher vs all student models (skip + last strategies), with quantitative metrics, error-group analysis, and summary plots. Start here. |
| [`Interpretability_Dev_EvalxNLP.ipynb`](Interpretability_Dev_EvalxNLP.ipynb) | Earlier interpretability notebook using the EvalxNLP framework. Supplementary. |
| [`distill_cloud.py`](distill_cloud.py) | Stage 2: PKD-skip grid search (16 runs × 4 epochs). The main distillation script. |
| [`vanilla_distill_cloud.py`](vanilla_distill_cloud.py) | Stage 1: Vanilla KD grid search (9 runs × 4 epochs). |
| [`train_teacher_cloud.py`](train_teacher_cloud.py) | Fine-tunes the 12-layer Legal-BERT teacher. |
| [`data_loader.py`](data_loader.py) | HuggingFace Datasets pipeline — required by the interpretability notebook. |
| [`model_utils.py`](model_utils.py) | Student initialization by layer truncation from teacher. |
| [`pkd_loss.py`](pkd_loss.py) | Patient KD loss (normalized MSE on intermediate hidden states). |
| [`evaluate_on_test.py`](evaluate_on_test.py) | Final test-set evaluation (run once only). |
| [`create_splits.py`](create_splits.py) | Generates deterministic 80/10/10 splits from raw data. |

### Deprecated / Legacy Files

The following exist but are **not needed** for reproducing the main results:

- `distill.py`, `train_teacher.py` — early local versions, superseded by the `_cloud` variants
- `distill_cloud_last.py` — PKD-last strategy, separate from the primary PKD-skip analysis
- `evaluate_pkd_last_grid_search.py`, `analyze_pkd_last_grid_search.py` — for the last strategy
- `evaluate_grid_search.py`, `evaluate_pkd_grid_search.py` — post-hoc eval runners (already done, results in `results/eval_results/`)
- `analyze_grid_search.py`, `analyze_pkd_grid_search.py` — visualization scripts (already run)
- `old_code/` — prior iterations, ignore entirely
- `EvalxNLP/` — external framework vendored for the first interpretability notebook

---

## Gitignored Assets: What's Missing and How to Restore It

Several large directories are excluded from version control. Here's what each contains and how to reconstruct it.

### 1. Dataset splits — `data/casehold/`

The processed CSV splits are gitignored. To recreate them:

1. Download `casehold.csv` from [HuggingFace](https://huggingface.co/datasets/casehold/casehold) or the original source and place it in the project root.
2. Run:
   ```bash
   python create_splits.py
   ```
   This produces `data/casehold/train.csv`, `dev.csv`, and `test.csv` using a fixed random seed for reproducibility.

**Expected sizes:** ~42,382 train / ~5,298 dev / ~5,298 test examples.

### 2. Model checkpoints — `results/training_runs/`

All trained model weights (`.safetensors`) are gitignored. The directory structure expected by the interpretability notebook is:

```
results/training_runs/
├── fine_tuned_base_bert_legal_teacher/
│   └── run_lr_1e-05/
│       └── checkpoint-1325/          ← teacher model (12-layer, 75.48% dev acc)
├── vanilla_kd_grid_search/
│   └── vanilla_L6_A0p7_T20/
│       └── checkpoint-3975/          ← best vanilla KD (6-layer, 75.63% dev acc)
└── pkd_skip_grid_search/
    ├── pkd_skip_L6_B1000/checkpoint-3975/
    ├── pkd_skip_L4_B100/checkpoint-3975/
    ├── pkd_skip_L3_B1000/checkpoint-5300/
    └── pkd_skip_L2_B500/checkpoint-5300/
```

If running from scratch, execute in order:
```bash
# Stage 0: train teacher (~4 epochs on GPU)
python train_teacher_cloud.py

# Stage 1: vanilla KD grid search (9 configs × 4 epochs)
python vanilla_distill_cloud.py
python evaluate_grid_search.py   # populates results/eval_results/vanilla_kd_grid_search_eval.csv

# Stage 2: PKD-skip grid search (16 configs × 4 epochs)
python distill_cloud.py
python evaluate_pkd_grid_search.py   # populates results/eval_results/pkd_skip_grid_search_eval.csv
```

Training was originally run on a cloud GPU (A100). Local CPU training is not recommended.

### 3. Interpretability outputs — `results/interpretability_v2/`

The outputs of `Interpretability_Dev_v2.ipynb` (attribution heatmaps, quantitative CSVs, summary plots) are gitignored. Re-run the notebook with `MODE = "full"` to regenerate them. With a GPU this takes roughly 2–3 hours for all 9 student models.

Set `MODE = "mini"` for a smoke test (~2 minutes).

### 4. Other gitignored paths

| Path | Contents |
|---|---|
| `results/integrated_gradients_pkd_skip_results/` | Outputs from the earlier `Interpretability_Dev_EvalxNLP.ipynb` |
| `results/outputs/test_run/` | Temporary test-run checkpoints |
| `archive/` | Old model snapshots, safe to ignore |
| `*.safetensors`, `*.pt` | All model weights |

---

## Setup

```bash
pip install torch transformers datasets accelerate scikit-learn numpy captum seaborn matplotlib
```

The interpretability notebook additionally requires `captum`:
```bash
pip install captum
```

Python 3.9+ recommended. A CUDA GPU is required for the full interpretability run; inference on CPU will work but is slow.

---

## Key Results (Test Set)

| Model | Layers | Test Accuracy | Inference Time |
|---|---|---|---|
| Teacher (Legal-BERT) | 12 | ~74.3% | ~10.2 ms/example |
| Vanilla KD (best) | 6 | ~74.9% | ~5.2 ms/example |
| PKD-Skip L6 B1000 | 6 | ~74.8% | ~5.2 ms/example |
| PKD-Skip L4 B100 | 4 | ~74.5% | ~3.6 ms/example |
| PKD-Skip L3 B1000 | 3 | ~71.8% | ~2.8 ms/example |
| PKD-Skip L2 B500 | 2 | ~67.9% | ~2.0 ms/example |

Full results: [`results/eval_results/test_set_evaluation.csv`](results/eval_results/test_set_evaluation.csv)
Comparison summary: [`results/eval_results/pkd_vs_vanilla_comparison.txt`](results/eval_results/pkd_vs_vanilla_comparison.txt)

---

## Interpretability Findings (from `Interpretability_Dev_v2.ipynb`)

Attribution analysis uses Integrated Gradients (Captum) comparing teacher and student token attributions on the test set, split by correct vs. incorrect predictions.

Key metrics per model pair:
- **Cosine similarity** of attribution vectors
- **Pearson correlation** of attribution vectors
- **Top-10 token overlap** (fraction of top-10 most important tokens shared)

Attribution similarity degrades sharply below 4 layers — the 2-layer models show near-zero cosine similarity with the teacher, indicating qualitatively different reasoning. The 6-layer students (both vanilla KD and PKD-skip) maintain moderate attribution similarity (~0.48–0.55) while matching teacher accuracy within 0.2%.

Summary CSV: `results/interpretability_v2/summary_all_models.csv` (gitignored, regenerated by notebook)
