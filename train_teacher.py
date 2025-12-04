# Old script for training teacher locally
# see train_teacher_cloud for updated version

import os
from transformers import AutoModelForMultipleChoice, AutoTokenizer, TrainingArguments, Trainer
from data_loader import get_dataloaders # Reuse your modern data loader!

def compute_metrics(eval_pred):
    # Optional: Add accuracy metric if you want to see it during training
    predictions, labels = eval_pred
    import numpy as np
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == labels).mean()}

def main(dry_run=False):
    # 1. Configuration
    model_name = "nlpaueb/legal-bert-base-uncased"
    output_dir = "checkpoints/models/fine_tuned_base_bert_legal_teacher"

    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - Testing without full training")
        print("="*60 + "\n")
    
    # 2. Load Tokenizer & Model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    
    # 3. Load Data (Uses pre-split files from create_splits.py)
    # Ensure you have run 'create_splits.py' first so train/dev/test.csv exist
    print("Loading datasets...")
    datasets = get_dataloaders(tokenizer, return_dict=True)

    # Returns dict with keys: 'train', 'dev', 'test'
    # Splits: 42,382 train / 5,298 dev / 5,298 test
    
    # 4. Training Arguments (Follow same parameters as Zheng paper)
    if dry_run:
        # Test configuration with minimal steps
        training_args = TrainingArguments(
            output_dir=output_dir + "_dry_run",
            eval_strategy="steps",
            eval_steps=5,
            save_strategy="no",
            max_steps=10,  # Only 10 steps for testing
            learning_rate=5e-6,
            per_device_train_batch_size=2,  # Small batch for testing
            per_device_eval_batch_size=2,
            logging_steps=1,
            report_to="none",
            use_cpu=False,
            no_cuda=False
        )
    else:
        # Full training configuration
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",      # Check validation every epoch
            save_strategy="epoch",      # Save model every epoch
            load_best_model_at_end=True, # Auto-load best checkpoint
            metric_for_best_model="eval_loss",
            learning_rate=5e-6,
            per_device_train_batch_size=16, # Standard for Legal-BERT
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_steps=1000,
            report_to="none",            # Change to "wandb" if you use it
            use_cpu=False,
            no_cuda=False
        )

    print(f"Training device: {training_args.device}")
    print(f"Number of GPUs: {training_args.n_gpu}")
    
    # 5. Initialize Trainer
    # Use subset for dry run
    train_data = datasets['train'].select(range(50)) if dry_run else datasets['train']
    eval_data = datasets['dev'].select(range(20)) if dry_run else datasets['dev']

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # 6. Train & Save
    if dry_run:
        print("\nStarting DRY RUN (10 steps only)...")
    else:
        print("\nStarting Full Teacher Fine-Tuning...")

    trainer.train()

    if not dry_run:
        print(f"\nSaving Best Teacher to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        print("\nDry run completed successfully! All components working.")
        print("To run full training, call: main(dry_run=False)")

if __name__ == "__main__":
    import sys
    # Check if --dry-run flag is passed
    dry_run_mode = "--dry-run" in sys.argv
    # main(dry_run=dry_run_mode)
    main()

"""
ACCESSING TRAINING RESULTS & FINE-TUNED MODEL:

1. Training Logs & Metrics:
   - Location: checkpoints/models/fine_tuned_base_bert_legal_teacher/
   - Files:
     * trainer_state.json - Contains training/eval loss and accuracy per epoch
     * training_args.bin - Training configuration used
     * checkpoint-{N}/ - Intermediate checkpoints (one per epoch)

2. Final Model (Best checkpoint auto-loaded):
   - Location: checkpoints/models/fine_tuned_base_bert_legal_teacher/
   - Files: pytorch_model.bin, config.json, tokenizer files

3. Load the fine-tuned model:
   from transformers import AutoModelForMultipleChoice, AutoTokenizer

   model_path = "checkpoints/models/fine_tuned_base_bert_legal_teacher"
   model = AutoModelForMultipleChoice.from_pretrained(model_path)
   tokenizer = AutoTokenizer.from_pretrained(model_path)

   # Use for inference or distillation
   model.eval()

4. Evaluate on test set:
   from transformers import Trainer
   from data_loader import get_dataloaders

   datasets = get_dataloaders(tokenizer, return_dict=True)
   trainer = Trainer(model=model, compute_metrics=compute_metrics)
   results = trainer.evaluate(datasets['test'])
   print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
   print(f"Test Loss: {results['eval_loss']:.4f}")

5. Quick accuracy check during training:
   - Check trainer_state.json for eval_accuracy at each epoch
   - Example: {"epoch": 3.0, "eval_accuracy": 0.6823, "eval_loss": 0.9234}
"""