"""
Post-Training Evaluation Script for Vanilla Knowledge Distillation Grid Search

STRATEGIC DECISION (User-Approved):
This script implements a two-tiered evaluation strategy to reduce computational load by 75%
while maintaining scientific rigor. We evaluate final checkpoints first (Tier 1), then
perform detailed learning curve analysis for top performers (Tier 2).

DEPLOYMENT CONTEXT:
Running locally with memory constraints. Previous OOM issues occurred during distillation training.
Evaluation-only with pre-saved models should work better. Implements aggressive memory management.

Tier 1: Evaluate 9 final checkpoints (epoch 4, step 5300) from each alpha/temp combination
Tier 2: Evaluate all 4 checkpoints for top 3-4 performers to understand convergence dynamics
"""

import os
import gc
import json
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
    """Discovers and catalogs all checkpoints in the grid search directory"""

    def __init__(self, base_dir="results/training_runs/vanilla_kd_grid_search"):
        self.base_dir = Path(base_dir)

    def discover_all_checkpoints(self):
        """
        Scans the grid search directory and returns structured list of all checkpoints

        Returns:
            List[Dict]: List of checkpoint metadata dicts with keys:
                - path: absolute path to checkpoint
                - run_name: name of the run (e.g., vanilla_L6_A0p5_T10)
                - alpha: alpha value (float)
                - temperature: temperature value (float)
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
            # Format: vanilla_L6_A0p5_T10
            try:
                parts = run_name.split('_')
                alpha_str = parts[2][1:]  # Remove 'A' prefix, get '0p5'
                temp_str = parts[3][1:]   # Remove 'T' prefix, get '10'

                alpha = float(alpha_str.replace('p', '.'))
                temperature = float(temp_str)
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
                    'alpha': alpha,
                    'temperature': temperature,
                    'checkpoint_step': step,
                    'epoch': epoch
                })

        return checkpoints

    def filter_tier1(self, checkpoints):
        """
        Filter checkpoints to only include final epoch (epoch 4, step 5300)

        Args:
            checkpoints: List of checkpoint dicts from discover_all_checkpoints()

        Returns:
            List[Dict]: Filtered list containing only final checkpoints (9 total)
        """
        return [cp for cp in checkpoints if cp['checkpoint_step'] == 5300]

    def filter_tier2(self, checkpoints, top_run_names):
        """
        Filter checkpoints for Tier 2 evaluation (all epochs for specific runs)

        Args:
            checkpoints: List of checkpoint dicts from discover_all_checkpoints()
            top_run_names: List of run names to include (e.g., ['vanilla_L6_A0p5_T10'])

        Returns:
            List[Dict]: Filtered list containing all checkpoints for specified runs
        """
        return [cp for cp in checkpoints if cp['run_name'] in top_run_names]


class ResultsTracker:
    """Manages CSV writing with crash recovery support"""

    def __init__(self, output_path="results/eval_results/vanilla_kd_grid_search_eval.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV columns
        self.columns = [
            'run_name',
            'alpha',
            'temperature',
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


def run_tier1_evaluation(checkpoints, eval_dataset, tokenizer, batch_size=4):
    """
    Run Tier 1 evaluation: Evaluate 9 final checkpoints (epoch 4)

    Args:
        checkpoints: List of checkpoint dicts (should be filtered to tier 1)
        eval_dataset: HuggingFace Dataset for evaluation
        tokenizer: Tokenizer
        batch_size: Evaluation batch size

    Returns:
        List[str]: Run names of top 3-4 performers for Tier 2
    """
    tracker = ResultsTracker()
    evaluated = tracker.get_evaluated_checkpoints()

    print(f"\n{'='*70}")
    print(f"TIER 1 EVALUATION: {len(checkpoints)} final checkpoints (Epoch 4)")
    print(f"{'='*70}\n")

    results = []

    for checkpoint in tqdm(checkpoints, desc="Tier 1 Evaluation"):
        # Skip if already evaluated
        if (checkpoint['run_name'], checkpoint['checkpoint_step']) in evaluated:
            print(f"\nSkipping {checkpoint['run_name']} (already evaluated)")
            continue

        print(f"\nEvaluating: {checkpoint['run_name']} (Step {checkpoint['checkpoint_step']})")
        print(f"Alpha={checkpoint['alpha']}, Temp={checkpoint['temperature']}")

        try:
            # Evaluate checkpoint
            metrics = evaluate_checkpoint(
                checkpoint['path'],
                eval_dataset,
                tokenizer,
                batch_size=batch_size
            )

            # Prepare result dict
            result = {
                'run_name': checkpoint['run_name'],
                'alpha': checkpoint['alpha'],
                'temperature': checkpoint['temperature'],
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
            results.append(result)

            print(f" Saved to CSV")

        except Exception as e:
            print(f"ERROR evaluating {checkpoint['run_name']}: {e}")
            print("Continuing to next checkpoint...")
            continue

    # Identify top 3-4 performers
    if results:
        sorted_results = sorted(results, key=lambda x: x['dev_accuracy'], reverse=True)

        print(f"\n{'='*70}")
        print(f"TIER 1 RESULTS SUMMARY")
        print(f"{'='*70}\n")

        for i, result in enumerate(sorted_results, 1):
            print(f"{i}. {result['run_name']}: {result['dev_accuracy']:.4f} "
                  f"(Alpha={result['alpha']}, Temp={result['temperature']})")

        # Select top 3-4 (or top 3 if 9 runs, top 4 if more variation)
        top_n = min(4, max(3, len(sorted_results) // 3))
        top_performers = [r['run_name'] for r in sorted_results[:top_n]]

        print(f"\nTop {top_n} performers selected for Tier 2 evaluation:")
        for name in top_performers:
            print(f"  - {name}")

        return top_performers
    else:
        print("No new results generated (all already evaluated)")
        return []


def run_tier2_evaluation(checkpoints, top_run_names, eval_dataset, tokenizer, batch_size=4):
    """
    Run Tier 2 evaluation: Evaluate all epochs for top performers

    Args:
        checkpoints: List of all checkpoint dicts
        top_run_names: List of run names to evaluate (from Tier 1)
        eval_dataset: HuggingFace Dataset for evaluation
        tokenizer: Tokenizer
        batch_size: Evaluation batch size
    """
    tracker = ResultsTracker()
    evaluated = tracker.get_evaluated_checkpoints()

    # Filter to tier 2 checkpoints
    discovery = CheckpointDiscovery()
    tier2_checkpoints = discovery.filter_tier2(checkpoints, top_run_names)

    print(f"\n{'='*70}")
    print(f"TIER 2 EVALUATION: Learning curve analysis for top performers")
    print(f"Evaluating {len(tier2_checkpoints)} checkpoints across {len(top_run_names)} runs")
    print(f"{'='*70}\n")

    for checkpoint in tqdm(tier2_checkpoints, desc="Tier 2 Evaluation"):
        # Skip if already evaluated
        if (checkpoint['run_name'], checkpoint['checkpoint_step']) in evaluated:
            print(f"\nSkipping {checkpoint['run_name']} Step {checkpoint['checkpoint_step']} (already evaluated)")
            continue

        print(f"\nEvaluating: {checkpoint['run_name']} (Epoch {checkpoint['epoch']}, Step {checkpoint['checkpoint_step']})")

        try:
            # Evaluate checkpoint
            metrics = evaluate_checkpoint(
                checkpoint['path'],
                eval_dataset,
                tokenizer,
                batch_size=batch_size
            )

            # Prepare result dict
            result = {
                'run_name': checkpoint['run_name'],
                'alpha': checkpoint['alpha'],
                'temperature': checkpoint['temperature'],
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

            print(f" Saved to CSV")

        except Exception as e:
            print(f"ERROR evaluating {checkpoint['run_name']}: {e}")
            print("Continuing to next checkpoint...")
            continue

    print(f"\n{'='*70}")
    print(f"TIER 2 EVALUATION COMPLETE")
    print(f"{'='*70}")


def main():
    """Main execution function"""

    print("="*70)
    print("VANILLA KD GRID SEARCH - POST-TRAINING EVALUATION")
    print("="*70)

    # Configuration
    BATCH_SIZE = 4  # Conservative starting point; can adjust if stable
    TIER = 2  # Start with 1 for Tier 1, then set to 2 for Tier 2

    print(f"\nConfiguration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # 1. Load tokenizer and data
    print(f"\nLoading tokenizer and dev dataset...")
    # Use base Legal-BERT model for tokenizer (same as what teacher was trained with)
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    datasets = get_dataloaders(tokenizer, return_dict=True)
    eval_dataset = datasets['dev']

    print(f"Dev set loaded: {len(eval_dataset)} examples")

    # 2. Discover checkpoints
    print(f"\nDiscovering checkpoints...")
    discovery = CheckpointDiscovery()
    all_checkpoints = discovery.discover_all_checkpoints()

    print(f"Found {len(all_checkpoints)} total checkpoints")

    # 3. Run evaluation based on TIER setting
    if TIER == 1:
        # Tier 1: Evaluate only final checkpoints
        tier1_checkpoints = discovery.filter_tier1(all_checkpoints)
        print(f"Tier 1: {len(tier1_checkpoints)} final checkpoints (epoch 4)")

        top_performers = run_tier1_evaluation(
            tier1_checkpoints,
            eval_dataset,
            tokenizer,
            batch_size=BATCH_SIZE
        )

        print(f"\n{'='*70}")
        print(f"NEXT STEPS:")
        print(f"1. Review Tier 1 results in: results/eval_results/vanilla_kd_grid_search_eval.csv")
        print(f"2. Set TIER=2 in this script and re-run to evaluate learning curves")
        print(f"3. Top performers: {', '.join(top_performers)}")
        print(f"{'='*70}")

    elif TIER == 2:
        # Load Tier 1 results to identify top performers
        tracker = ResultsTracker()
        tier1_checkpoints = discovery.filter_tier1(all_checkpoints)

        # Get top performers from CSV
        import pandas as pd
        df = pd.read_csv(tracker.output_path)
        tier1_df = df[df['checkpoint_step'] == 5300]
        top_n = min(4, max(3, len(tier1_df) // 3))
        top_performers = tier1_df.nlargest(top_n, 'dev_accuracy')['run_name'].tolist()

        print(f"Top performers from Tier 1: {', '.join(top_performers)}")

        # Run Tier 2
        run_tier2_evaluation(
            all_checkpoints,
            top_performers,
            eval_dataset,
            tokenizer,
            batch_size=BATCH_SIZE
        )

        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETE")
        print(f"Results saved to: results/eval_results/vanilla_kd_grid_search_eval.csv")
        print(f"Next: Run analyze_grid_search.py for visualization and analysis")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
