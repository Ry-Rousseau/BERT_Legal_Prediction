

# CaseHOLD Distillation Research

This repository implements a comprehensive **Knowledge Distillation** study to compress the `Legal-BERT` model while retaining performance on the **CaseHOLD** legal reasoning benchmark.

The project follows a rigorous two-stage experimental design:
- **Stage 1:** Vanilla Knowledge Distillation (grid search over α and temperature)
- **Stage 2:** Patient Knowledge Distillation with skip strategy (grid search over student sizes and β)

All experiments utilize a strict **80/10/10 data split** to prevent test set leakage ("The Vault").

## 📂 Project Structure

```text
.
├── data/                          # Data storage (Gitignored)
│   └── casehold/                  # Created by create_splits.py
│       ├── train.csv              # 80% Training set
│       ├── dev.csv                # 10% Validation set (for hyperparameter tuning)
│       └── test.csv               # 10% Test set (THE VAULT - Only for final evaluation)
│
├── results/
│   ├── training_runs/             # All trained models
│   │   ├── fine_tuned_base_bert_legal_teacher/  # Teacher model (12-layer)
│   │   ├── vanilla_kd_grid_search/              # Stage 1: 9 vanilla KD runs
│   │   └── pkd_skip_grid_search/                # Stage 2: 16 PKD runs
│   ├── eval_results/              # Evaluation CSVs and summaries
│   │   ├── vanilla_kd_grid_search_eval.csv
│   │   ├── pkd_skip_grid_search_eval.csv
│   │   ├── test_set_evaluation.csv
│   │   └── *.txt (summary reports)
│   └── plots/                     # Visualizations (heatmaps, learning curves)
│
├── create_splits.py               # Deterministic data splitting script
├── data_loader.py                 # HF Datasets pipeline for Multiple Choice
├── model_utils.py                 # Student model initialization (layer truncation)
├── pkd_loss.py                    # Patient Distillation Loss (Cheng et al. 2019)
│
├── train_teacher_cloud.py         # Teacher training script
├── vanilla_distill_cloud.py       # Stage 1: Vanilla KD grid search
├── distill_cloud.py               # Stage 2: PKD-skip grid search
│
├── evaluate_grid_search.py        # Vanilla KD post-hoc evaluation
├── evaluate_pkd_grid_search.py    # PKD post-hoc evaluation
├── analyze_grid_search.py         # Vanilla KD analysis & visualization
├── analyze_pkd_grid_search.py     # PKD analysis & visualization
├── evaluate_on_test.py            # Final test set evaluation (THE VAULT)
└── Interpretability_Dev_EvalxNLP.ipynb  # Attribution analysis (teacher vs student)
```

## 🚀 Setup

1.  **Install Dependencies**

    ```bash
    pip install torch transformers datasets accelerate scikit-learn numpy
    ```

2.  **Prepare the Data**
    Download `casehold.csv` and place it in the root (or update the script path). Then run the splitter to generate the canonical datasets.

    ```bash
    python create_splits.py
    ```

    *Output:* Creates `data/casehold/train.csv`, `dev.csv`, and `test.csv`.

## 🧪 Research Workflow

### Stage 0: Train the Teacher Model

Fine-tune the full 12-layer `nlpaueb/legal-bert-base-uncased` model on CaseHOLD.

**Script:** `train_teacher_cloud.py`

```bash
python train_teacher_cloud.py
```

**Configuration:**
- Learning rate: 1e-5 (optimal from LR search)
- Batch size: 32
- Epochs: 4
- Evaluation: On dev set only
- **Result:** 75.48% dev accuracy

**Output:** `results/training_runs/fine_tuned_base_bert_legal_teacher/run_lr_1e-05/checkpoint-1325`

---

### Stage 1: Vanilla Knowledge Distillation Grid Search

Find optimal α (distillation weight) and temperature for 6-layer student.

**Script:** `vanilla_distill_cloud.py`

```bash
python vanilla_distill_cloud.py
```

**Grid Search:**
- Alphas: [0.2, 0.5, 0.7]
- Temperatures: [5, 10, 20]
- **Total:** 9 runs × 4 epochs = 36 checkpoints

**Loss Function:**
```
L_total = (1-α)·L_CE + α·L_KD
where L_KD = KL_div(student_logits/T || teacher_logits/T) × T²
```

**Post-Training Evaluation:**
```bash
# Evaluate all checkpoints on dev set
python evaluate_grid_search.py

# Analyze results and create visualizations
python analyze_grid_search.py
```

**Best Result:** α=0.7, T=20, Epoch 3 → **75.63% dev accuracy**

---

### Stage 2: Patient Knowledge Distillation Grid Search

Find optimal β (patient loss weight) for each student size, using best α and T from Stage 1.

**Script:** `distill_cloud.py`

```bash
python distill_cloud.py
```

**Grid Search:**
- Student layers: [6, 4, 3, 2]
- Betas: [10, 100, 500, 1000]
- **Total:** 16 runs × 4 epochs = 64 checkpoints
- **Fixed:** α=0.7, T=20 (from Stage 1)

**Loss Function:**
```
L_total = (1-α)·L_CE + α·L_KD + β·L_PT
where L_PT = Σ ||normalize(h_student) - normalize(h_teacher)||²
```

**PKD Strategy:** Skip - student layers match evenly-spaced teacher layers
- 6-layer: matches teacher layers 0, 2, 4, 6, 8, 10
- 4-layer: matches teacher layers 0, 3, 6, 9
- 3-layer: matches teacher layers 0, 4, 8
- 2-layer: matches teacher layers 0, 6

**Post-Training Evaluation:**
```bash
# Evaluate all PKD checkpoints on dev set
python evaluate_pkd_grid_search.py

# Analyze results and create visualizations
python analyze_pkd_grid_search.py
```

---

### Stage 3: Final Test Set Evaluation

Evaluate best models on the held-out test set (THE VAULT).

**Script:** `evaluate_on_test.py`

```bash
# Recommended: Only evaluate best models
python evaluate_on_test.py --best-only

# Or evaluate a specific model
python evaluate_on_test.py --model-path "path/to/checkpoint"
```

**Models Evaluated:**
- Teacher (12-layer baseline)
- Best vanilla KD (6-layer)
- Best PKD per student size (6, 4, 3, 2 layers)

**IMPORTANT:** Run this **only once** for final results. Test set should never be used for model selection or hyperparameter tuning.

## 🛠️ Technical Implementation Details

### Data Loading (`data_loader.py`)

  * **Task:** Multiple Choice (5 options per prompt).
  * **Formatting:** Expands each example into 5 separate input sequences `[CLS] Context [SEP] Option [SEP]`.
  * **Labels:** Converts the dataset's string labels ("0"-"4") into integer indices for CrossEntropyLoss.

### Student Initialization (`model_utils.py`)

We do not use a generic "small BERT." We explicitly construct the student by slicing the Teacher:

```python
# Copies layers 0, 1, 2, 3, 4, 5 from Teacher -> Student
student.bert.encoder.layer[i].load_state_dict(teacher.bert.encoder.layer[i].state_dict())
```

### Patient Loss (`pkd_loss.py`)

Implements the normalized Mean Squared Error from *Cheng et al. (2019)*:
$$L_{PT} = \sum || \frac{h_s}{||h_s||_2} - \frac{h_t}{||h_t||_2} ||_2^2$$

**Strategies:**
- `skip`: Student layers match evenly-spaced teacher layers (used in this project)
- `last`: Student layers match final k teacher layers

### Custom Trainers

**`PKDTrainer` (in `distill_cloud.py`):**
Combines three loss components:
```python
total_loss = (1-α)·task_loss + α·distill_loss + β·pkd_loss
```
- Task loss: Cross-entropy with hard labels
- Distillation loss: KL divergence with soft teacher predictions
- Patient loss: Intermediate layer matching

**Memory Optimization:**
- `save_only_model=True`: Only saves model weights, not optimizer/scheduler states
- Reduces checkpoint size by ~70% (critical for grid searches)
- Enables 16 PKD runs × 4 checkpoints to fit in 50GB storage

## 📊 Evaluation Strategy

### Dev Set Evaluation (Model Selection)

All hyperparameter tuning and model selection uses **only the dev set**.

**Post-hoc evaluation approach:**
- Training runs use `eval_strategy="no"` to avoid memory issues
- After training completes, evaluate all checkpoints separately
- Enables memory-efficient evaluation with aggressive cleanup

**Scripts:**
- `evaluate_grid_search.py` - Vanilla KD checkpoints
- `evaluate_pkd_grid_search.py` - PKD checkpoints

**Configuration:**
- Batch size: 4 (conservative for memory constraints)
- FP16 precision
- Aggressive memory cleanup between evaluations
- CSV crash recovery (can resume if interrupted)

### Test Set Evaluation (Final Results)

The test set ("THE VAULT") is **only** used for final evaluation.

**Rules:**
1. Never loaded during training
2. Never used for hyperparameter tuning
3. Only evaluated **once** after all model selection is complete
4. Results reported in final paper/analysis

**Script:** `evaluate_on_test.py --best-only`

## 📈 Key Results

### Teacher (Baseline)
- 12-layer Legal-BERT
- **75.48% dev accuracy**

### Stage 1: Vanilla KD (6-layer)
- Best config: α=0.7, T=20, Epoch 3
- **75.63% dev accuracy**
- Matches teacher performance with 50% model compression

### Stage 2: Patient KD
- Results vary by student size and β
- See `results/eval_results/pkd_best_models_summary.txt` for details
- Comparison with vanilla KD in `pkd_vs_vanilla_comparison.txt`

## 🎯 Best Practices Learned

### Training at Scale
1. **Disable evaluation during training** when memory-constrained
2. **Save checkpoints frequently** (every epoch) for post-hoc analysis
3. **Use `save_only_model=True`** to reduce storage by 70%
4. **Grid search systematically** - Stage 1 (α, T) → Stage 2 (β per size)

### Evaluation
1. **Two-tiered strategy** reduces evaluations by 75%:
   - Tier 1: Final checkpoints only
   - Tier 2: All epochs for top performers
2. **Aggressive memory cleanup** between evaluations
3. **CSV checkpointing** for crash recovery
4. **Separate dev/test strictly** - never touch test set until final

### Student Model Architecture
- **Factor of 12 preferred** for clean layer matching (6, 4, 3, 2)
- **Layer truncation initialization** from fine-tuned teacher
- **Skip strategy** works well for varying student sizes