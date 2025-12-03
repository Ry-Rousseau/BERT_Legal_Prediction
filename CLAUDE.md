

# CaseHOLD Distillation Research

This repository implements **Patient Knowledge Distillation (PKD)** to compress the `Legal-BERT` model while retaining performance on the **CaseHOLD** legal reasoning benchmark.

The project is designed for a rigorous scientific study on the effects of distillation in the legal domain, utilizing a strict **80/10/10 data split** to prevent test set leakage ("The Vault").

## 📂 Project Structure

```text
.
├── data/                   # Data storage (Gitignored)
│   └── casehold/           # Created by create_splits.py
│       ├── train.csv       # 80% Training set
│       ├── dev.csv         # 10% Validation set (for Early Stopping)
│       └── test.csv        # 10% Test set (THE VAULT - Never touched during training)
├── checkpoints/            # Model artifacts
│   ├── teacher/            # Fine-tuned 12-layer Legal-BERT
│   └── student/            # Distilled 6-layer models
├── create_splits.py        # Deterministic data splitting script
├── data_loader.py          # Modern HF Datasets pipeline for Multiple Choice
├── model_utils.py          # Student model initialization (Layer Truncation)
├── pkd_loss.py             # Implementation of Patient Distillation Loss (Cheng et al. 2019)
├── train_teacher.py        # Standard fine-tuning script for the Teacher
└── distill.py              # Custom Trainer for PKD Distillation
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

### Step 1: Train the "Gold Standard" Teacher

We fine-tune the full 12-layer `nlpaueb/legal-bert-base-uncased` model on the CaseHOLD task.

  * **Script:** `train_teacher.py`
  * **Config:** 3 Epochs, Batch Size 16, LR 5e-6.
  * **Control:** Evaluates on `dev.csv` only. Saves best model to `checkpoints/models/fine_tuned_base_bert_legal_teacher`.

<!-- end list -->

```bash
python train_teacher.py
```

### Step 2: Distill the Student (The Experiment)

We train a 6-layer student model to mimic the Teacher using **Patient Knowledge Distillation (PKD)**.

  * **Initialization:** The student is initialized by copying the first 6 layers of the *fine-tuned* Teacher (not random weights).
  * **Objective:** $L_{Total} = (1-\alpha)L_{CE} + \alpha L_{Distill} + \beta L_{Patient}$
  * **Script:** `distill.py`

<!-- end list -->

```bash
python distill.py
```

*Note: Ensure `teacher_path` in `distill.py` points to your fine-tuned teacher folder.*

### Step 3: Train the Baseline (The Control)

To prove distillation works, we train an identical 6-layer student *without* looking at the Teacher.

  * **Config:** Set `alpha=0` and `beta=0` in `distill.py`.
  * **Why:** This isolates the effect of the distillation loss from the benefit of the layer initialization.

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

  * **Strategy:** `skip` (matches every 2nd layer) or `last` (matches last $k$ layers).

## 📊 Evaluation

Final results should be reported on `data/casehold/test.csv`.

  * **Metric:** Accuracy (or Macro-F1).
  * **Rule:** The Test set is **never** loaded during the training of the Teacher or the Student. It is reserved strictly for the final inference pass.