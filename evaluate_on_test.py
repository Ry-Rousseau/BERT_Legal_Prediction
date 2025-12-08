"""
Test Set Evaluation Script - THE VAULT

IMPORTANT: This script evaluates models on the held-out test set.
Only use this for FINAL evaluation after all model selection is complete.

This script can evaluate:
- Teacher model
- Vanilla KD models (any checkpoint)
- PKD models (any checkpoint)

The test set ("THE VAULT") should only be touched once for final results.
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
import argparse


def compute_metrics(eval_pred):
    """Compute accuracy metric for evaluation"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == labels).mean()}


class ResultsTracker:
    """Manages CSV writing for test set results"""

    def __init__(self, output_path="results/eval_results/test_set_evaluation.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV columns
        self.columns = [
            'model_type',  # 'teacher', 'vanilla_kd', 'pkd'
            'model_path',
            'model_name',
            'hyperparameters',  # JSON string with alpha/temp/beta/layers
            'test_accuracy',
            'test_loss',
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
        """Add a result to the CSV file"""
        with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(result_dict)


def evaluate_checkpoint(checkpoint_path, test_dataset, tokenizer, batch_size=4):
    """
    Load model, evaluate on test set, return metrics with aggressive memory cleanup

    Args:
        checkpoint_path: Path to checkpoint directory
        test_dataset: HuggingFace Dataset for evaluation
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
    print("Evaluating on TEST set...")
    with torch.no_grad():
        metrics = trainer.evaluate(test_dataset)

    # 5. Extract results
    accuracy = metrics.get('eval_accuracy', 0.0)
    loss = metrics.get('eval_loss', 0.0)
    num_samples = metrics.get('eval_samples', len(test_dataset))

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


def discover_all_models():
    """
    Discover all available models for test evaluation

    Returns:
        List[Dict]: List of model info dicts with keys:
            - path: absolute path to model
            - model_type: 'teacher', 'vanilla_kd', 'pkd'
            - model_name: descriptive name
            - hyperparameters: dict with hyperparameters
    """
    models = []

    # 1. Teacher model
    teacher_path = Path("results/training_runs/fine_tuned_base_bert_legal_teacher/run_lr_1e-05/checkpoint-1325")
    if teacher_path.exists():
        models.append({
            'path': str(teacher_path.absolute()),
            'model_type': 'teacher',
            'model_name': 'Teacher (12-layer Legal-BERT)',
            'hyperparameters': {'layers': 12, 'lr': '1e-05'}
        })

    # 2. Vanilla KD models
    vanilla_base = Path("results/training_runs/vanilla_kd_grid_search")
    if vanilla_base.exists():
        for run_dir in sorted(vanilla_base.iterdir()):
            if not run_dir.is_dir():
                continue

            # Parse run name (e.g., vanilla_L6_A0p7_T20)
            run_name = run_dir.name
            try:
                parts = run_name.split('_')
                alpha_str = parts[2][1:].replace('p', '.')
                temp_str = parts[3][1:]
                alpha = float(alpha_str)
                temperature = float(temp_str)
            except (IndexError, ValueError):
                continue

            # Find all checkpoints
            for checkpoint_dir in sorted(run_dir.iterdir()):
                if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith('checkpoint-'):
                    step = int(checkpoint_dir.name.split('-')[1])
                    epoch = step // 1325

                    models.append({
                        'path': str(checkpoint_dir.absolute()),
                        'model_type': 'vanilla_kd',
                        'model_name': f'{run_name}_epoch{epoch}',
                        'hyperparameters': {
                            'layers': 6,
                            'alpha': alpha,
                            'temperature': temperature,
                            'epoch': epoch
                        }
                    })

    # 3. PKD models
    pkd_base = Path("results/training_runs/pkd_skip_grid_search")
    if pkd_base.exists():
        for run_dir in sorted(pkd_base.iterdir()):
            if not run_dir.is_dir():
                continue

            # Parse run name (e.g., pkd_skip_L6_B10)
            run_name = run_dir.name
            try:
                parts = run_name.split('_')
                layers_str = parts[2][1:]
                beta_str = parts[3][1:]
                student_layers = int(layers_str)
                beta = float(beta_str)
            except (IndexError, ValueError):
                continue

            # Find all checkpoints
            for checkpoint_dir in sorted(run_dir.iterdir()):
                if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith('checkpoint-'):
                    step = int(checkpoint_dir.name.split('-')[1])
                    epoch = step // 1325

                    models.append({
                        'path': str(checkpoint_dir.absolute()),
                        'model_type': 'pkd',
                        'model_name': f'{run_name}_epoch{epoch}',
                        'hyperparameters': {
                            'layers': student_layers,
                            'alpha': 0.7,  # Fixed for PKD
                            'temperature': 20,  # Fixed for PKD
                            'beta': beta,
                            'epoch': epoch
                        }
                    })

    return models


def filter_best_models_only(all_models, dev_results_vanilla="results/eval_results/vanilla_kd_grid_search_eval.csv",
                            dev_results_pkd="results/eval_results/pkd_skip_grid_search_eval.csv"):
    """
    Filter to only best models based on dev set results

    Returns best:
    - Teacher
    - Best vanilla KD (6-layer)
    - Best PKD per student size (6, 4, 3, 2 layers)
    """
    import pandas as pd

    best_models = []

    # Always include teacher
    teacher_models = [m for m in all_models if m['model_type'] == 'teacher']
    best_models.extend(teacher_models)

    # Best vanilla KD
    try:
        vanilla_df = pd.read_csv(dev_results_vanilla)
        best_vanilla_row = vanilla_df.loc[vanilla_df['dev_accuracy'].idxmax()]
        best_vanilla_name = best_vanilla_row['run_name'] + f"_epoch{best_vanilla_row['epoch']}"

        for model in all_models:
            if model['model_type'] == 'vanilla_kd' and model['model_name'] == best_vanilla_name:
                best_models.append(model)
                break
    except Exception as e:
        print(f"Warning: Could not load vanilla KD dev results: {e}")

    # Best PKD per student size
    try:
        pkd_df = pd.read_csv(dev_results_pkd)

        for layers in sorted(pkd_df['student_layers'].unique()):
            subset = pkd_df[pkd_df['student_layers'] == layers]
            best_row = subset.loc[subset['dev_accuracy'].idxmax()]
            best_pkd_name = best_row['run_name'] + f"_epoch{best_row['epoch']}"

            for model in all_models:
                if model['model_type'] == 'pkd' and model['model_name'] == best_pkd_name:
                    best_models.append(model)
                    break
    except Exception as e:
        print(f"Warning: Could not load PKD dev results: {e}")

    return best_models


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(description='Evaluate models on test set')
    parser.add_argument('--best-only', action='store_true',
                       help='Only evaluate best models from each category (recommended)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Evaluation batch size (default: 4)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Evaluate specific model path only')

    args = parser.parse_args()

    print("="*70)
    print("TEST SET EVALUATION - THE VAULT")
    print("="*70)
    print("\nWARNING: You are about to evaluate on the held-out test set.")
    print("This should only be done ONCE for final results.")
    print()

    # Confirmation prompt
    confirm = input("Type 'YES' to confirm test set evaluation: ")
    if confirm != 'YES':
        print("Evaluation cancelled.")
        return

    print(f"\nConfiguration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Mode: {'BEST MODELS ONLY' if args.best_only else 'ALL MODELS'}")

    # 1. Load tokenizer and test data
    print(f"\nLoading tokenizer and TEST dataset...")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    datasets = get_dataloaders(tokenizer, return_dict=True)
    test_dataset = datasets['test']  # THE VAULT

    print(f"Test set loaded: {len(test_dataset)} examples")

    # 2. Discover models
    if args.model_path:
        print(f"\nEvaluating single model: {args.model_path}")
        models_to_eval = [{
            'path': args.model_path,
            'model_type': 'custom',
            'model_name': Path(args.model_path).parent.name,
            'hyperparameters': {}
        }]
    else:
        print(f"\nDiscovering models...")
        all_models = discover_all_models()
        print(f"Found {len(all_models)} total models")

        if args.best_only:
            models_to_eval = filter_best_models_only(all_models)
            print(f"Filtered to {len(models_to_eval)} best models")
        else:
            models_to_eval = all_models

    # 3. Run evaluation
    tracker = ResultsTracker()

    print(f"\n{'='*70}")
    print(f"STARTING TEST SET EVALUATION")
    print(f"{'='*70}\n")

    for model_info in tqdm(models_to_eval, desc="Evaluating Models"):
        print(f"\nEvaluating: {model_info['model_name']}")
        print(f"Type: {model_info['model_type']}")
        print(f"Hyperparameters: {model_info['hyperparameters']}")

        try:
            # Evaluate checkpoint
            metrics = evaluate_checkpoint(
                model_info['path'],
                test_dataset,
                tokenizer,
                batch_size=args.batch_size
            )

            # Prepare result dict
            import json
            result = {
                'model_type': model_info['model_type'],
                'model_path': model_info['path'],
                'model_name': model_info['model_name'],
                'hyperparameters': json.dumps(model_info['hyperparameters']),
                'test_accuracy': round(metrics['accuracy'], 4),
                'test_loss': round(metrics['loss'], 4),
                'num_samples': metrics['num_samples'],
                'inference_time_sec': round(metrics['inference_time_sec'], 1),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Save immediately
            tracker.add_result(result)

            print(f"✓ Saved to CSV")

        except Exception as e:
            print(f"ERROR evaluating {model_info['model_name']}: {e}")
            print("Continuing to next model...")
            continue

    print(f"\n{'='*70}")
    print(f"TEST SET EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {tracker.output_path}")
    print(f"\nIMPORTANT: These are FINAL test set results.")
    print(f"Do not re-run this evaluation. Use these results for publication/reporting.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
