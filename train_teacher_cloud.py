# Old script for training teacher locally
# see train_teacher_cloud for updated version

import os
import shutil
import sys
import numpy as np
from transformers import AutoModelForMultipleChoice, AutoTokenizer, TrainingArguments, Trainer
from data_loader import get_dataloaders
import torch

def compute_metrics(eval_pred):
    # Optional: Add accuracy metric if you want to see it during training
    predictions, labels = eval_pred
    import numpy as np
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == labels).mean()}

def main(dry_run=False):
    # Add cloud-friendly absolute paths
    import os
    WORKSPACE = os.getenv("WORKSPACE", ".")  # RunPod uses /workspace
    
    base_output_dir = os.path.join(WORKSPACE, "results/training_runs/fine_tuned_base_bert_legal_teacher")
    final_best_dir = os.path.join(WORKSPACE, "checkpoints/models/fine_tuned_base_bert_legal_teacher")
    
    # Create directories if they don't exist
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(final_best_dir), exist_ok=True)
    
    # 1. Configuration
    model_name = "nlpaueb/legal-bert-base-uncased"
    #base_output_dir = "results/training_runs/fine_tuned_base_bert_legal_teacher" # - use locally
    #final_best_dir = "checkpoints/models/fine_tuned_base_bert_legal_teacher" # - use locally

    # Define grid search space (from cheng et al. 2019)
    if dry_run:
        learning_rates = [5e-5]
    else:
        learning_rates = [5e-5, 2e-5, 1e-5]
    
    print(f"\n starting teacher grid search over LRs {learning_rates}")
    
    # 2. Load Tokenizer and data
    print(f"\n loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading datasets...")
    datasets = get_dataloaders(tokenizer, return_dict=True)
    
    # Prepare subsets for dry run if needed
    if dry_run:
        train_data = datasets['train'].select(range(50))
        eval_data = datasets['dev'].select(range(20))
    else:
        train_data = datasets['train']
        eval_data = datasets['dev']
    
    # Trackers for the Best Model
    best_accuracy = 0.0
    best_lr = None
    best_run_dir = None
    
    # Error checking robustness
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    # Verify data loaded correctly
    print(f"Train examples: {len(train_data)}")
    print(f"Dev examples: {len(eval_data)}")
    
    # 4. The Grid Search Loop
    for lr in learning_rates:
        print(f"\n TRAINING WITH LEARNING RATE: {lr}")
        
        # Unique output directory for this specific run
        run_output_dir = os.path.join(base_output_dir, f"run_lr_{lr}")
        
        # Reload fresh model
        model = AutoModelForMultipleChoice.from_pretrained(model_name)
    
        # Define Arguments (Optimized for A40)
        training_args = TrainingArguments(
            output_dir=run_output_dir,
            
            # Grid Search Parameter
            learning_rate=lr,
            
            # Paper Constants [Cheng et al. 2019]
            num_train_epochs=4,        # Paper used 4 epochs
            per_device_train_batch_size=32, # Paper used 32
            weight_decay=0.01,
            
            # A40 Cloud Optimization
            per_device_eval_batch_size=64,  # Faster eval
            dataloader_num_workers=8,       # Feed GPU fast
            gradient_checkpointing=False,
            fp16=True,                      # Mixed Precision
            tf32=True,                      # Ampere Speedup
            optim="adamw_torch_fused",      # Fast optimizer
            dataloader_pin_memory=True,
            
            # Evaluation Strategy
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,    # Load best epoch for THIS run
            metric_for_best_model="accuracy",
            
            # Housekeeping
            logging_steps=50,
            report_to="none",
            overwrite_output_dir=True,
            save_total_limit=1, # Only keep the best checkpoint per run to save space
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Evaluate Final Performance of this LR
        print(f"Evaluating LR {lr}...")
        metrics = trainer.evaluate()
        accuracy = metrics['eval_accuracy']
        
        print(f"Result for LR {lr}: {accuracy:.4f} Accuracy")
        
        # Compare to current champion
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lr = lr
            best_run_dir = run_output_dir
            print(f"New Best - (Accuracy: {best_accuracy:.4f})")
        else:
            print(f"Run did not beat current best ({best_accuracy:.4f})")
    
    # 5. Finalizing the Winner
    print("\n" + "="*60)
    print(f"Grid search complete")
    print(f"Winner: Learning Rate {best_lr} with Accuracy {best_accuracy:.4f}")
    
    print(f"Saving the best model to: {final_best_dir}...")
    
    if os.path.exists(final_best_dir):
        shutil.rmtree(final_best_dir)
    shutil.copytree(best_run_dir, final_best_dir)
    
    # Ensure tokenizer is also there (just in case)
    tokenizer.save_pretrained(final_best_dir)    

if __name__ == "__main__":
    import sys
    dry_run_mode = "--dry-run" in sys.argv
    main()