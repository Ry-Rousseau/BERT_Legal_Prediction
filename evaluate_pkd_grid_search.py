"""
Post-Training Evaluation Script for PKD-Skip Grid Search

This script evaluates the Patient Knowledge Distillation (PKD-Skip) grid search results.
Similar to vanilla KD evaluation, but adapted for PKD checkpoint structure.

Grid: 4 Student Sizes × 4 Betas = 16 Runs
- Student Layers: [6, 4, 3, 2]
- Betas: [10, 100, 500, 1000]
- Fixed: Alpha=0.7, Temperature=20

Each run has 4 checkpoints (epochs 1-4).
"""

import os
import gc
import csv
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForMultipleChoice, AutoTokenizer, Trainer, TrainingArguments
from data_loader import get_dataloaders

# Import compute_metrics from train_teacher_cloud
def compute_metrics(eval_pred):
    """Compute accuracy metric for evaluation"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == labels).mean()}


class CheckpointDiscovery:
    """Discovers and catalogs all PKD checkpoints in the grid search directory"""

    def __init__(self, base_dir="results/training_runs/pkd_skip_grid_search"):
        self.base_dir = Path(base_dir)

    def discover_all_checkpoints(self):
        """
        Scans the PKD grid search directory and returns structured list of all checkpoints

        Returns:
            List[Dict]: List of checkpoint metadata dicts with keys:
                - path: absolute path to checkpoint
                - run_name: name of the run (e.g., pkd_skip_L6_B10)
                - student_layers: number of student layers (int)
                - beta: beta value (float)
                - checkpoint_step: step number (int)
                - epoch: epoch number (int, 1-4)
        """
        checkpoints = []

        # Iterate through run directories
        for run_dir in sorted(self.base_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            run_name = run_dir.name

            # Parse run name to extract hyperparameters
            # Format: pkd_skip_L6_B10
            try:
                parts = run_name.split('_')
                layers_str = parts[2][1:]  # Remove 'L' prefix, get '6'
                beta_str = parts[3][1:]     # Remove 'B' prefix, get '10'

                student_layers = int(layers_str)
                beta = float(beta_str)
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse run name {run_name}: {e}")
                continue

            # Find all checkpoint subdirectories
            for checkpoint_dir in sorted(run_dir.iterdir()):
                if not checkpoint_dir.is_dir() or not checkpoint_dir.name.startswith('checkpoint-'):
                    continue

                # Extract step number
                try:
                    step = int(checkpoint_dir.name.split('-')[1])
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse checkpoint {checkpoint_dir.name}: {e}")
                    continue

                # Map step to epoch (steps 1325, 2650, 3975, 5300 -> epochs 1, 2, 3, 4)
                epoch = step // 1325

                checkpoints.append({
                    'path': str(checkpoint_dir.absolute()),
                    'run_name': run_name,
                    'student_layers': student_layers,
                    'beta': beta,
                    'checkpoint_step': step,
                    'epoch': epoch
                })

        return checkpoints

    def filter_final_checkpoints(self, checkpoints):
        """
        Filter checkpoints to only include final epoch (epoch 4, step 5300)

        Args:
            checkpoints: List of checkpoint dicts from discover_all_checkpoints()

        Returns:
            List[Dict]: Filtered list containing only final checkpoints
        """
        return [cp for cp in checkpoints if cp['checkpoint_step'] == 5300]


class ResultsTracker:
    """Manages CSV writing with crash recovery support"""

    def __init__(self, output_path="results/eval_results/pkd_skip_grid_search_eval.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV columns
        self.columns = [
            'run_name',
            'student_layers',
            'beta',
            'checkpoint_step',
            'epoch',
            'dev_accuracy',
            'dev_loss',
            'num_samples',
            'inference_time_sec',
            'timestamp'
        ]

        # Initialize file with headers if it doesn't exist
        if not self.output_path.exists():
            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()

    def add_result(self, result_dict):
        """
        Add a result to the CSV file (appends immediately for crash recovery)

        Args:
            result_dict: Dict with keys matching self.columns
        """
        with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(result_dict)

    def get_evaluated_checkpoints(self):
        """
        Load already-evaluated checkpoints from CSV

        Returns:
            Set[Tuple]: Set of (run_name, checkpoint_step) tuples already evaluated
        """
        if not self.output_path.exists():
            return set()

        evaluated = set()
        with open(self.output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                evaluated.add((row['run_name'], int(row['checkpoint_step'])))

        return evaluated


def evaluate_checkpoint(checkpoint_path, eval_dataset, tokenizer, batch_size=4):
    """
    Load model, evaluate on dev set, return metrics with aggressive memory cleanup

    Args:
        checkpoint_path: Path to checkpoint directory
        eval_dataset: HuggingFace Dataset for evaluation
        tokenizer: Tokenizer for the model
        batch_size: Evaluation batch size (default: 4 for memory efficiency)

    Returns:
        Dict with keys: accuracy, loss, num_samples, inference_time_sec
    """
    import time

    # Record start time
    start_time = time.time()

    # 1. Load model from checkpoint
    print(f"Loading model from {checkpoint_path}...")
    model = AutoModelForMultipleChoice.from_pretrained(checkpoint_path)

    # 2. Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    # 3. Create Trainer with memory-efficient settings
    eval_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=0,  # Minimize overhead
        fp16=True,  # Mixed precision
        use_cpu=False if device == "cuda" else True,
        no_cuda=False if device == "cuda" else True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
    )

    # 4. Evaluate (no gradients needed)
    print("Evaluating...")
    with torch.no_grad():
        metrics = trainer.evaluate(eval_dataset)

    # 5. Extract results
    accuracy = metrics.get('eval_accuracy', 0.0)
    loss = metrics.get('eval_loss', 0.0)
    num_samples = metrics.get('eval_samples', len(eval_dataset))

    # Record end time
    inference_time = time.time() - start_time

    print(f"Evaluation complete: Accuracy={accuracy:.4f}, Loss={loss:.4f}, Time={inference_time:.1f}s")

    # 6. CRITICAL: Aggressive memory cleanup
    del model
    del trainer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return {
        'accuracy': accuracy,
        'loss': loss,
        'num_samples': num_samples,
        'inference_time_sec': inference_time
    }


def main():
    """Main execution function"""

    print("="*70)
    print("PKD-SKIP GRID SEARCH - POST-TRAINING EVALUATION")
    print("="*70)

    # Configuration
    BATCH_SIZE = 4  # Conservative starting point
    EVAL_ALL = True  # Set to True to evaluate all checkpoints, False for final only

    print(f"\nConfiguration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Mode: {'ALL CHECKPOINTS' if EVAL_ALL else 'FINAL CHECKPOINTS ONLY'}")

    # 1. Load tokenizer and data
    print(f"\nLoading tokenizer and dev dataset...")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    datasets = get_dataloaders(tokenizer, return_dict=True)
    eval_dataset = datasets['dev']

    print(f"Dev set loaded: {len(eval_dataset)} examples")

    # 2. Discover checkpoints
    print(f"\nDiscovering checkpoints...")
    discovery = CheckpointDiscovery()
    all_checkpoints = discovery.discover_all_checkpoints()

    print(f"Found {len(all_checkpoints)} total checkpoints")

    # 3. Filter checkpoints based on mode
    if EVAL_ALL:
        checkpoints_to_eval = all_checkpoints
        print(f"Evaluating ALL {len(checkpoints_to_eval)} checkpoints (all epochs)")
    else:
        checkpoints_to_eval = discovery.filter_final_checkpoints(all_checkpoints)
        print(f"Evaluating FINAL {len(checkpoints_to_eval)} checkpoints (epoch 4 only)")

    # 4. Run evaluation
    tracker = ResultsTracker()
    evaluated = tracker.get_evaluated_checkpoints()

    print(f"\n{'='*70}")
    print(f"STARTING EVALUATION")
    print(f"{'='*70}\n")

    for checkpoint in tqdm(checkpoints_to_eval, desc="Evaluating Checkpoints"):
        # Skip if already evaluated
        if (checkpoint['run_name'], checkpoint['checkpoint_step']) in evaluated:
            print(f"\nSkipping {checkpoint['run_name']} Step {checkpoint['checkpoint_step']} (already evaluated)")
            continue

        print(f"\nEvaluating: {checkpoint['run_name']} (Epoch {checkpoint['epoch']}, Step {checkpoint['checkpoint_step']})")
        print(f"Student Layers: {checkpoint['student_layers']}, Beta: {checkpoint['beta']}")

        try:
            # Evaluate checkpoint
            metrics = evaluate_checkpoint(
                checkpoint['path'],
                eval_dataset,
                tokenizer,
                batch_size=BATCH_SIZE
            )

            # Prepare result dict
            result = {
                'run_name': checkpoint['run_name'],
                'student_layers': checkpoint['student_layers'],
                'beta': checkpoint['beta'],
                'checkpoint_step': checkpoint['checkpoint_step'],
                'epoch': checkpoint['epoch'],
                'dev_accuracy': round(metrics['accuracy'], 4),
                'dev_loss': round(metrics['loss'], 4),
                'num_samples': metrics['num_samples'],
                'inference_time_sec': round(metrics['inference_time_sec'], 1),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Save immediately (crash recovery)
            tracker.add_result(result)

            print(f"✓ Saved to CSV")

        except Exception as e:
            print(f"ERROR evaluating {checkpoint['run_name']}: {e}")
            print("Continuing to next checkpoint...")
            continue

    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {tracker.output_path}")
    print(f"\nNext steps:")
    print(f"  1. Run analyze_pkd_grid_search.py to analyze results")
    print(f"  2. Compare PKD performance vs vanilla KD")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
