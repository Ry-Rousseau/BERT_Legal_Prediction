# distillation script for running on cloud A40 GPU with 48 GB VRAM
# implements grid search over hyperparameter search space 

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


# Implement the balance between alpha, beta and temperature
# 1. Define the Custom Distillation Trainer
class PKDTrainer(Trainer):
    def __init__(self, teacher_model=None, pkd_strategy="skip", alpha=0.5, beta=10.0, temperature=10.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.pkd_strategy = pkd_strategy
        self.alpha = alpha # Weight for Soft Target Loss (KL Divergence)
        self.beta = beta   # Weight for Patient Loss (Intermediate Layers)
        self.temperature = temperature # Temperature for knowledge distillation softmax

        # Freeze teacher and move to same device
        self.teacher.eval()
        self.teacher.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # Updated signature
        # 1. Forward pass Student
        # output_hidden_states=True is required for PKD
        student_outputs = model(**inputs, output_hidden_states=True)
        
        # 2. Forward pass Teacher (No Grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_hidden_states=True)

        # 3. Calculate Losses
        
        # A. Task Loss (Cross Entropy) - "Hard Labels"
        # HF models calculate this automatically if 'labels' are provided
        task_loss = student_outputs.loss 
        
        # B. Distillation Loss (Soft Targets) - "Knowledge Distillation"
        # KL Divergence between Student logits and Teacher logits
        T = self.temperature  # Use temperature from hyperparameter
        distill_loss = F.kl_div(
            F.log_softmax(student_outputs.logits / T, dim=-1),
            F.softmax(teacher_outputs.logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        # C. Patient Loss (Intermediate Layers) - "PKD"
        # We skip the first hidden state (embeddings) usually, so we take [1:]
        # You might need to adjust slicing depending on exact model config
        pkd_loss = compute_pkd_loss(
            student_outputs.hidden_states[1:], 
            teacher_outputs.hidden_states[1:], 
            strategy=self.pkd_strategy
        )

        # 4. Combine
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss + self.beta * pkd_loss
        
        return (total_loss, student_outputs) if return_outputs else total_loss

# ---------------------------------------------------------
# Experiment Logging
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
######################################    
    pkd_strategy = "skip" #skip or last
    num_student_layers = 6
    
    # PRIMARY PARAMETER NAMING SPACE
    #temperature = 10.0  # Temperature for KD softmax (Cheng et al. 2019: 5, 10, 20)
    #alpha = 0.7  # Weight for distillation loss (Cheng et al. 2019: 0.2, 0.5, 0.7)
    beta = 0  # For vanilla KD (no patient loss). Set to 100 for PKD
    learning_rate = 1e-5  # Same LR as best parent fine-tuning run 
    
    # SECONDARY PARAMETERS (those not in grid search in cheng et al. 2019)
    batch_size = 32
    num_epochs = 4

###########################################
    #Define grid search, from cheng et al 2019 values
    alphas = [0.2, 0.5, 0.7]
    temperatures = [5.0, 10.0, 20.0]

    base_output_dir = "results/training_runs/vanilla_kd_grid_search"


#####################################
    # Load Tokenizer & Teacher
    # REPLACE THIS with the path to your FINE-TUNED CaseHOLD Teacher
    teacher_path = "results/training_runs/fine_tuned_base_bert_legal_teacher/run_lr_1e-05/checkpoint-1325" 
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    
    print("Loading Teacher...")
    teacher_model = AutoModelForMultipleChoice.from_pretrained(teacher_path)
    
    # Load Data (using pre-split files)
    print("Processing Data...")
    datasets = get_dataloaders(tokenizer, return_dict=True)
    train_dataset = datasets['train']
    eval_dataset = datasets['dev']

    #Trackers
    best_accuracy = 0.0
    best_params = {}
    
    print(f"\n{'='*60}")
    print(f"STARTING GRID SEARCH: {len(alphas)}x{len(temperatures)} = {len(alphas)*len(temperatures)} runs")
    print(f"Alphas: {alphas}")
    print(f"Temperatures: {temperatures}")
    print(f"{'='*60}\n")
    
    for alpha in alphas:
        for T in temperatures:
            # Skip already completed run
            if alpha == 0.2 and T == 5.0:
                print(f"\n" + "="*50)
                print(f"SKIPPING: Alpha={alpha}, Temp={T} (already completed)")
                print("="*50)
                continue

            print(f"\n" + "="*50)
            print(f"Testing: Alpha={alpha}, Temp={T}")
            print("="*50)     
                   
            # Create Student
            print("Creating Student...")
            student_model = create_student_model(teacher_model, num_student_layers=num_student_layers)
            
            run_name = f"vanilla_L{num_student_layers}_A{str(alpha).replace('.','p')}_T{int(T)}"
            
            run_output_dir = os.path.join(base_output_dir, run_name)

            # Training Args
            training_args = TrainingArguments(

                output_dir=run_output_dir,
                
                num_train_epochs=num_epochs, # follow same epochs as cheng et al 2019 
                max_steps = -1,
                
                # If GPU runs out of memory, set this to 16 and add 'gradient_accumulation_steps=2'
                per_device_train_batch_size=batch_size,
                #gradient_accumulation_steps=1, remove for A40 run
                gradient_checkpointing=False, # significant speed up on A40
                #per_device_eval_batch_size=16, # drop this to prevent memory overspilling from gpu to ram
                dataloader_num_workers=8, # use 8 CPU cores to feed the GPU faster
                #Learning rate
                learning_rate=learning_rate,
                
                # Evaluation
                eval_strategy="no", # change evaluation 
                save_strategy="epoch",
                load_best_model_at_end=False,
                #metric_for_best_model="accuracy",
                per_device_eval_batch_size=4,
                        
                # Logging
                logging_steps=100, # Log every 100 steps to track progress
                remove_unused_columns=False, # Important so we don't drop columns needed for logic
                report_to= "none", # Must be string, not boolean
                
                # Hardware
                use_cpu=False, # Explicitly use GPU
                no_cuda=False,  # Ensure CUDA is not disabled
                fp16=True,
                tf32=True, # Use tensorflow-32 for A40
                optim="adamw_torch_fused" #Use a faster fused optimized for A40
            )

            # Initialize Trainer
            trainer = PKDTrainer(
                teacher_model=teacher_model,
                eval_dataset=eval_dataset,
                pkd_strategy=pkd_strategy,
                model=student_model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,  # Add metrics computation

                # HYPERPARAMETERS HERE
                alpha=alpha,  # Balances Hard Labels vs Soft Labels (Distillation)
                beta=beta,  # Balances the "Patient" intermediate layer loss
                temperature=T,  # Temperature for KD softmax (higher = softer)
            )
            
            # Train
            print("Starting Distillation...")
            train_result = trainer.train()

            # Evaluate on dev set to get best metrics
            # print("Evaluating on dev set...")
            #eval_result = trainer.evaluate()

            # Extract metrics for logging (no evaluation due to OOM issues)
            metrics = {
                "best_dev_accuracy": 0.0,  # Will evaluate separately after training
                "best_dev_loss": 0.0,
                "train_runtime_sec": train_result.metrics.get("train_runtime", 0.0),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0.0)
            }

            acc = 0.0  # No accuracy since we skipped evaluation
            
            # Prepare hyperparameters dictionary
            hyperparams = {
                "pkd_strategy": pkd_strategy,
                "student_layers": num_student_layers,
                "alpha": alpha,
                "beta": beta,
                "temperature": T
            }
            
            # Log to CSV for this run
            log_to_csv(training_args, hyperparams, metrics)

            # Update Winner
            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {"alpha": alpha, "temperature": T}
                print(f" NEW BEST FOUND: {acc:.4f}")

            # Save this model
            model_save_path = f"checkpoints/models/student_L{hyperparams['student_layers']}_A{str(hyperparams['alpha']).replace('.', 'p')}_B{int(hyperparams['beta'])}_T{int(hyperparams['temperature'])}_LR{learning_rate}"
            student_model.save_pretrained(model_save_path)
            print(f"Model saved to {model_save_path}\n")

    # Final summary
    print("\n" + "="*60)
    print("GRID SEARCH COMPLETE")
    print("="*60)
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*60)

if __name__ == "__main__":
    main()