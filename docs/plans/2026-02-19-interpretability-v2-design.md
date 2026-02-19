# Interpretability Dev v2 — Design Document

**Date:** 2026-02-19
**Scope:** New notebook extending the v1 interpretability pipeline with error analysis, confidence metrics, inference timing, and PKD-Last attribution coverage.

---

## 1. Goals

Produce a richer results battery from the existing trained models without any retraining. Specifically:

1. **Error analysis** — separate attribution analysis for correctly vs. incorrectly predicted test examples per model.
2. **Confidence metric** — extract max softmax probability before argmax as a per-example confidence score.
3. **Inference time** — measure per-example wall-clock latency on GPU.
4. **PKD-Last coverage** — run the same similarity/attribution pipeline on PKD-Last models (currently missing from v1).
5. **Test set** — all analysis runs on `data/casehold/test.csv` (the vault), not the dev set used in v1.

---

## 2. Approach: Single Notebook with MODE Variable

**File:** `Interpretability_Dev_v2.ipynb`

A single `MODE = "mini"` / `MODE = "full"` variable at the top of the notebook controls all loop sizes. In `"mini"` mode the entire notebook runs in under a few minutes on a small slice of data, verifying correctness before the full GPU run. All results write incrementally to CSV so the notebook is safely restartable after a crash.

---

## 3. Model Registry

9 models total (teacher is the shared reference):

| Name | Strategy | Layers | Notes |
|------|----------|--------|-------|
| `teacher` | teacher | 12 | Baseline reference for all comparisons |
| `vanilla_L6_A0p7_T20` | vanilla_kd | 6 | Best vanilla KD (α=0.7, T=20, epoch 3) |
| `pkd_skip_L6_B1000` | pkd_skip | 6 | Best PKD-Skip 6-layer |
| `pkd_skip_L4_B100` | pkd_skip | 4 | Best PKD-Skip 4-layer |
| `pkd_skip_L3_B1000` | pkd_skip | 3 | Best PKD-Skip 3-layer |
| `pkd_skip_L2_B500` | pkd_skip | 2 | Best PKD-Skip 2-layer |
| `pkd_last_L6_B500` | pkd_last | 6 | Only β available |
| `pkd_last_L4_B500` | pkd_last | 4 | Only β available |
| `pkd_last_L3_B500` | pkd_last | 3 | Only β available |
| `pkd_last_L2_B500` | pkd_last | 2 | Only β available |

Teacher is loaded once and kept in memory throughout. Student models are loaded, processed, then deleted (`del model; torch.cuda.empty_cache()`) before loading the next.

---

## 4. Data Pipeline

- **Test set:** `data/casehold/test.csv` — deterministic static file (created with `random_state=42`), no runtime seed needed.
- **Tokenized dataset:** loaded via existing `get_dataloaders(tokenizer, return_dict=True)['test']`.
- **Raw dataset:** loaded via `load_dataset('csv', data_files={'test': 'data/casehold/test.csv'}, split='test')` for metadata extraction.
- **Mini mode:** prediction pass runs on first 20 test examples only.
- **Full mode:** prediction pass runs on all ~5,900 test examples.

---

## 5. Prediction Pass (per model)

Before any attribution work, a full prediction pass runs for each model using `Trainer.predict()` (batch size 4, fp16). Produces `results/interpretability_v2/{model_name}/predictions.csv` with columns:

| Column | Description |
|--------|-------------|
| `example_idx` | Index into test dataset |
| `true_label` | Ground truth (0–4) |
| `predicted_label` | Model's argmax prediction |
| `is_correct` | `true_label == predicted_label` |
| `confidence` | Max softmax probability across 5 choices |
| `inference_time_sec` | Total prediction time / N examples |

**Crash recovery:** if `predictions.csv` already exists for a model, the prediction pass is skipped entirely for that model.

---

## 6. Attribution Analysis (per model)

Reuses `get_attributions()`, `filter_tokens_and_attributions()`, `save_attribution_heatmap()`, and `save_top15_bar_chart()` from v1 unchanged.

### Sampling

For each student model, attribution runs in two groups drawn from `predictions.csv`:

| Mode | N correct examples | N incorrect examples |
|------|-------------------|---------------------|
| mini | 3 | 3 |
| full | 15 | 15 |

Sampling uses `random_state=SEED` (42) for reproducibility.

### Output Structure

```
results/interpretability_v2/{model_name}/
    predictions.csv
    qualitative/
        correct/
            sample_000/
                metadata.json
                attribution_heatmap_filtered.png
                top15_tokens_combined.png
            sample_001/ ...
        incorrect/
            sample_000/ ...
    quantitative_correct.csv
    quantitative_incorrect.csv
    summary_stats_correct.txt
    summary_stats_incorrect.txt
```

`metadata.json` extends v1 with three new fields: `is_correct`, `confidence`, `inference_time_sec`.

Quantitative metrics per example (same as v1): cosine similarity, Pearson correlation, top-10 token overlap between teacher and student attributions.

---

## 7. Cross-Model Comparison & Summary Outputs

Runs entirely from CSVs — no model loading. Produces:

### Summary CSV
`results/interpretability_v2/summary_all_models.csv`

Columns: `model_name`, `strategy`, `layers`, `error_group`, `n_samples`, `mean_cosine_sim`, `mean_correlation`, `mean_top10_overlap`, `mean_confidence`, `mean_inference_time_sec`, `overall_accuracy`.

### Plots (`results/interpretability_v2/plots/`)

| File | Content |
|------|---------|
| `accuracy_by_model.png` | Bar chart of test accuracy, colored by strategy |
| `confidence_correct_vs_incorrect.png` | Paired bars: mean confidence on correct vs. incorrect examples per model |
| `cosine_sim_by_layers.png` | Line plot: mean cosine similarity vs. layer count, split by PKD-Skip vs. PKD-Last |
| `inference_time_by_model.png` | Bar chart of per-example inference time ordered by layer count |

### PKD-Last Similarity Report
`results/interpretability_v2/pkd_last_similarity_summary.txt` — same format as the existing `summary_readable.txt` for PKD-Skip, enabling direct write-up comparison.

---

## 8. Sample Counts Summary

| Mode | Prediction pass | Correct samples | Incorrect samples | Quantitative samples |
|------|----------------|-----------------|-------------------|---------------------|
| mini | 20 examples | 3 | 3 | 6 total |
| full | all ~5,900 | 15 | 15 | 30 total (teacher: no student comparison needed) |

---

## 9. Reused vs. New Code

| Component | Source |
|-----------|--------|
| `get_attributions()` | Reused from v1 unchanged |
| `filter_tokens_and_attributions()` | Reused from v1 unchanged |
| `save_attribution_heatmap()` | Reused from v1 unchanged |
| `save_top15_bar_chart()` | Reused from v1 unchanged |
| `compute_cosine_similarity()` | Reused from v1 unchanged |
| `extract_example_metadata()` | Reused from v1 unchanged |
| Prediction pass with confidence + timing | **New** |
| Error-group sampling (`is_correct` split) | **New** |
| Cross-model summary plots | **New** (4 plots) |
| PKD-Last similarity report | **New** |
| `MODE` / crash-recovery scaffolding | **New** |

---

## 10. Key Constraints

- Test set (`test.csv`) must not be used for any model selection — this notebook is read-only analysis only.
- The teacher model stays loaded in memory throughout the run; student models are loaded/deleted one at a time.
- All output paths are under `results/interpretability_v2/` to keep v1 results (`integrated_gradients_pkd_skip_results/`) untouched.
