"""
Patient Knowledge Distillation (PKD-Skip) Grid Search - Stage 2

STAGE 2: Patient KD Search (16 Runs)
Goal: Find the best β (Beta) for each specific student depth.
Grid: 4 Student Sizes × 4 Betas = 16 Runs.

FIXED HYPERPARAMETERS FROM STAGE 1:
- Alpha (α) = 0.7 (distillation weight) - FIXED from vanilla KD grid search
- Temperature (T) = 20 - FIXED from vanilla KD grid search

GRID SEARCH PARAMETERS:
- Student Layers: [6, 4, 3, 2] - 4 different model sizes
- Beta (β): [10, 100, 500, 1000] - 4 different patient loss weights

This script implements PKD with "skip" strategy, where student layers are matched
to teacher layers by skipping intermediate teacher layers.

Architecture:
- Teacher: 12 layers (nlpaueb/legal-bert-base-uncased)
- Students: 6-layer, 4-layer, 3-layer, 2-layer variants

Training Details:
- 4 epochs (same as Stage 1)
- Learning rate: 1e-5 (same as best teacher run)
- Batch size: 32
- Saves checkpoints every epoch for post-hoc evaluation
"""

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForMultipleChoice
from data_loader import get_dataloaders
from model_utils import create_student_model
from pkd_loss import compute_pkd_loss
from train_teacher_cloud import compute_metrics
import csv
import os
from datetime import datetime

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable synchronous CUDA for better error messages


class PKDTrainer(Trainer):
    """
    Custom Trainer for Patient Knowledge Distillation with Skip Strategy

    Implements the full PKD loss function:
    L_total = (1-α)·L_CE + α·L_KD + β·L_PT

    Where:
    - L_CE: Cross-entropy loss (hard labels)
    - L_KD: KL divergence loss (soft labels from teacher)
    - L_PT: Patient loss (intermediate layer matching)
    """

    def __init__(self, teacher_model=None, pkd_strategy="skip", alpha=0.7, beta=100.0, temperature=20.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.pkd_strategy = pkd_strategy
        self.alpha = alpha       # Weight for Soft Target Loss (KL Divergence)
        self.beta = beta         # Weight for Patient Loss (Intermediate Layers)
        self.temperature = temperature  # Temperature for knowledge distillation softmax

        # Freeze teacher and move to same device
        self.teacher.eval()
        self.teacher.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute PKD loss combining task loss, distillation loss, and patient loss
        """
        # Ensure labels are Long type (required for CrossEntropyLoss)
        if "labels" in inputs:
            inputs["labels"] = inputs["labels"].long()

            # Debug: Check label values to catch out-of-bounds issues
            if torch.any(inputs["labels"] < 0) or torch.any(inputs["labels"] > 4):
                print(f"WARNING: Invalid label detected! Labels: {inputs['labels']}")
                print(f"  Min: {inputs['labels'].min()}, Max: {inputs['labels'].max()}")
                raise ValueError(f"Labels must be in range [0, 4], got {inputs['labels']}")

        # 1. Forward pass Student (with hidden states for PKD)
        student_outputs = model(**inputs, output_hidden_states=True)

        # 2. Forward pass Teacher (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_hidden_states=True)

        # 3. Calculate Losses

        # A. Task Loss (Cross Entropy) - "Hard Labels"
        task_loss = student_outputs.loss

        # B. Distillation Loss (Soft Targets) - "Knowledge Distillation"
        T = self.temperature
        distill_loss = F.kl_div(
            F.log_softmax(student_outputs.logits / T, dim=-1),
            F.softmax(teacher_outputs.logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        # C. Patient Loss (Intermediate Layers) - "PKD"
        # Skip first hidden state (embeddings), so we take [1:]
        pkd_loss = compute_pkd_loss(
            student_outputs.hidden_states[1:],
            teacher_outputs.hidden_states[1:],
            strategy=self.pkd_strategy
        )

        # 4. Combine losses
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss + self.beta * pkd_loss

        return (total_loss, student_outputs) if return_outputs else total_loss


def log_to_csv(training_args, hyperparams, metrics, csv_path="results/distillation_experiments.csv"):
    """
    Logs experiment results to a CSV file.

    Args:
        training_args: TrainingArguments object
        hyperparams: Dictionary containing hyperparameters (alpha, beta, temperature, pkd_strategy, student_layers)
        metrics: Dictionary containing performance metrics (best_dev_accuracy, best_dev_loss, train_runtime_sec, train_samples_per_second)
        csv_path: Path to the CSV file
    """
    # Define column order
    columns = [
        "timestamp",
        "strategy",
        "student_layers",
        "alpha",
        "beta",
        "temperature",
        "learning_rate",
        "batch_size",
        "num_epochs",
        "best_dev_accuracy",
        "best_dev_loss",
        "train_runtime_sec",
        "train_samples_per_second"
    ]

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_path)

    # Prepare the row data
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": hyperparams.get("pkd_strategy", "unknown"),
        "student_layers": hyperparams.get("student_layers", 0),
        "alpha": hyperparams.get("alpha", 0.0),
        "beta": hyperparams.get("beta", 0.0),
        "temperature": hyperparams.get("temperature", 1.0),
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "num_epochs": int(training_args.num_train_epochs),
        "best_dev_accuracy": round(metrics.get("best_dev_accuracy", 0.0), 4),
        "best_dev_loss": round(metrics.get("best_dev_loss", 0.0), 4),
        "train_runtime_sec": round(metrics.get("train_runtime_sec", 0.0), 1),
        "train_samples_per_second": metrics.get("train_samples_per_second", 0.0)
    }

    # Write to CSV
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow(row)

    print(f"[SUCCESS] Logged experiment to {csv_path}")


def main():
    """
    Main execution function for PKD-Skip grid search
    """

    print("="*70)
    print("STAGE 2: PATIENT KNOWLEDGE DISTILLATION (PKD-SKIP) GRID SEARCH")
    print("="*70)

    ######################################
    # FIXED PARAMETERS FROM STAGE 1
    ######################################
    pkd_strategy = "last"  # Skip strategy for layer matching
    alpha = 0.7            # FIXED - optimal from Stage 1 vanilla KD
    temperature = 20.0     # FIXED - optimal from Stage 1 vanilla KD

    ######################################
    # GRID SEARCH PARAMETERS
    ######################################
    student_layer_sizes = [6, 4, 3, 2]  # 4 different student architectures
    betas = [500]        # for efficiency, set beta as optimal from the results of pkd-skip

    ######################################
    # TRAINING CONFIGURATION
    ######################################
    learning_rate = 1e-5   # Same as best teacher run
    batch_size = 32
    num_epochs = 4

    ######################################
    # PATHS AND SETUP
    ######################################
    base_output_dir = "results/training_runs/pkd_last_grid_search"

    # Load Tokenizer & Teacher
    teacher_path = "results/training_runs/fine_tuned_base_bert_legal_teacher/run_lr_1e-05/checkpoint-1325"
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    print("\nLoading Teacher model...")
    teacher_model = AutoModelForMultipleChoice.from_pretrained(teacher_path)

    # Load Data (using pre-split files)
    print("Processing Data...")
    datasets = get_dataloaders(tokenizer, return_dict=True)
    train_dataset = datasets['train']
    eval_dataset = datasets['dev']

    print(f"Train examples: {len(train_dataset)}")
    print(f"Dev examples: {len(eval_dataset)}")

    ######################################
    # GRID SEARCH EXECUTION
    ######################################
    total_runs = len(student_layer_sizes) * len(betas)

    print(f"\n{'='*70}")
    print(f"STARTING GRID SEARCH: {len(student_layer_sizes)} sizes × {len(betas)} betas = {total_runs} runs")
    print(f"Student Layers: {student_layer_sizes}")
    print(f"Betas: {betas}")
    print(f"Fixed Alpha: {alpha}")
    print(f"Fixed Temperature: {temperature}")
    print(f"{'='*70}\n")

    run_counter = 0

    for num_student_layers in student_layer_sizes:
        for beta in betas:
            run_counter += 1

            print(f"\n" + "="*70)
            print(f"RUN {run_counter}/{total_runs}: L{num_student_layers}_B{beta}")
            print(f"Student Layers: {num_student_layers}, Beta: {beta}")
            print(f"(Alpha={alpha}, Temp={temperature})")
            print("="*70)

            # Create Student model
            print(f"Creating {num_student_layers}-layer student model...")
            student_model = create_student_model(teacher_model, num_student_layers=num_student_layers)

            # Generate run name
            run_name = f"pkd_last_L{num_student_layers}_B{beta}"
            run_output_dir = os.path.join(base_output_dir, run_name)

            # Training Arguments
            training_args = TrainingArguments(
                output_dir=run_output_dir,

                num_train_epochs=num_epochs,
                max_steps=-1,

                # Batch size and optimization
                per_device_train_batch_size=batch_size,
                gradient_checkpointing=False,  # Faster on A40
                dataloader_num_workers=8,      # Use 8 CPU cores
                learning_rate=learning_rate,

                # Evaluation (disabled during training, will evaluate post-hoc)
                eval_strategy="no",
                save_strategy="epoch",
                load_best_model_at_end=False,
                per_device_eval_batch_size=4,
                save_only_model=True,  # CRITICAL: Only save model weights, not optimizer/scheduler

                # Logging
                logging_steps=100,
                remove_unused_columns=False,
                report_to="none",

                # Hardware optimization
                use_cpu=False,
                no_cuda=False,
                fp16=True,
                tf32=True,
                optim="adamw_torch_fused"
            )

            # Create custom data collator that doesn't pad labels
            from dataclasses import dataclass
            from typing import Any, Dict, List

            @dataclass
            class DataCollatorForMultipleChoice:
                """
                Data collator for multiple choice tasks.
                Pads inputs but keeps labels as-is (no padding).
                """
                tokenizer: Any

                def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
                    batch_size = len(features)
                    num_choices = len(features[0]["input_ids"])

                    # Flatten for padding
                    flattened_features = [
                        {k: v[i] for k, v in feature.items() if k != "labels"}
                        for feature in features
                        for i in range(num_choices)
                    ]

                    # Pad the flattened features
                    batch = self.tokenizer.pad(
                        flattened_features,
                        padding=True,
                        return_tensors="pt"
                    )

                    # Unflatten
                    batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}

                    # Add labels (no padding needed - just stack)
                    batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)

                    return batch

            data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

            # Initialize PKD Trainer
            trainer = PKDTrainer(
                teacher_model=teacher_model,
                eval_dataset=eval_dataset,
                pkd_strategy=pkd_strategy,
                model=student_model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,  # Add custom data collator
                compute_metrics=compute_metrics,

                # HYPERPARAMETERS
                alpha=alpha,           # FIXED from Stage 1
                beta=beta,             # GRID SEARCH PARAMETER
                temperature=temperature  # FIXED from Stage 1
            )

            # Train
            print("Starting PKD Distillation...")
            train_result = trainer.train()

            # Extract metrics (no evaluation during training)
            metrics = {
                "best_dev_accuracy": 0.0,  # Will evaluate separately after training
                "best_dev_loss": 0.0,
                "train_runtime_sec": train_result.metrics.get("train_runtime", 0.0),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0.0)
            }

            # Prepare hyperparameters dictionary
            hyperparams = {
                "pkd_strategy": pkd_strategy,
                "student_layers": num_student_layers,
                "alpha": alpha,
                "beta": beta,
                "temperature": temperature
            }

            # Log to CSV
            log_to_csv(training_args, hyperparams, metrics)

            # Save final model (optional - checkpoints already saved)
            final_model_path = os.path.join(run_output_dir, "final_model")
            student_model.save_pretrained(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            print(f"Final model saved to {final_model_path}")

            print(f"\n✓ Run {run_counter}/{total_runs} complete\n")

    # Final Summary
    print("\n" + "="*70)
    print("STAGE 2 GRID SEARCH COMPLETE")
    print("="*70)
    print(f"Total runs completed: {total_runs}")
    print(f"Results directory: {base_output_dir}")
    print(f"Experiment log: results/distillation_experiments.csv")
    print("\nNext steps:")
    print("  1. Run post-hoc evaluation on all 16 runs using evaluate_grid_search.py")
    print("  2. Analyze results to find optimal (student_size, beta) combinations")
    print("  3. Compare PKD performance vs vanilla KD from Stage 1")
    print("="*70)


if __name__ == "__main__":
    main()
