import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForMultipleChoice
from data_loader import get_dataloaders
from model_utils import create_student_model
from pkd_loss import compute_pkd_loss
from train_teacher import compute_metrics
import csv
import os
from datetime import datetime

# Implement the balance between alpha, beta and temperature
# 1. Define the Custom Distillation Trainer
class PKDTrainer(Trainer):
    def __init__(self, teacher_model=None, pkd_strategy="skip", alpha=0.5, beta=10.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.pkd_strategy = pkd_strategy
        self.alpha = alpha # Weight for Soft Target Loss (KL Divergence)
        self.beta = beta   # Weight for Patient Loss (Intermediate Layers)
        
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
        T = 10.0 # Temperature, use the baseline from 
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
    temperature = 10 # set above
    alpha = 0.7
    beta = 100
    learning_rate = 5e-5
    # note that temperature is set above 
    
    # SECONDARY PARAMETERS (those not in grid search in cheng et al. 2019)
    batch_size = 32
    num_epochs = 4

###########################################
    # Load Tokenizer & Teacher
    # REPLACE THIS with the path to your FINE-TUNED CaseHOLD Teacher
    teacher_path = "checkpoints/models/fine_tuned_base_bert_legal_teacher" 
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    
    print("Loading Teacher...")
    teacher_model = AutoModelForMultipleChoice.from_pretrained(teacher_path)
    
    # Create Student
    print("Creating Student...")
    student_model = create_student_model(teacher_model, num_student_layers=num_student_layers)

    # Load Data (using pre-split files)
    print("Processing Data...")
    datasets = get_dataloaders(tokenizer, return_dict=True)
    train_dataset = datasets['train']
    eval_dataset = datasets['dev']

    # Training Args
    training_args = TrainingArguments(
        
        output_dir="results/outputs/pkd_skip_beta100",
        
        num_train_epochs=num_epochs, # follow same epochs as cheng et al 2019 
        max_steps = -1,
        
        # If GPU runs out of memory, set this to 16 and add 'gradient_accumulation_steps=2'
        per_device_train_batch_size=batch_size, # increase for full training
        per_device_eval_batch_size=batch_size,
        
        #Learning rate
        learning_rate=learning_rate,

        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
                
        # Logging
        logging_steps=100, # Log every 100 steps to track progress
        remove_unused_columns=False, # Important so we don't drop columns needed for logic
        report_to= "none", # Must be string, not boolean
        
        # Hardware
        use_cpu=False, # Explicitly use GPU
        no_cuda=False,  # Ensure CUDA is not disabled
        fp16=True
    )

    print(f"Training device: {training_args.device}")
    print(f"Number of GPUs: {training_args.n_gpu}")

    # Initialize Trainer
    trainer = PKDTrainer(
        teacher_model=teacher_model,
        eval_dataset=eval_dataset,
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        pkd_strategy=pkd_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Add metrics computation

        # HYPERPARAMETERS HERE
        alpha=alpha,  # Balances Hard Labels vs Soft Labels (Distillation)
        beta=beta,  # Balances the "Patient" intermediate layer loss
    )

    # Train
    print("Starting Distillation...")
    train_result = trainer.train()

    # Evaluate on dev set to get best metrics
    print("Evaluating on dev set...")
    eval_result = trainer.evaluate()

    # Extract metrics for logging
    metrics = {
        "best_dev_accuracy": eval_result.get("eval_accuracy", 0.0),
        "best_dev_loss": eval_result.get("eval_loss", 0.0),
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

    # Save
    student_model.save_pretrained(f"checkpoints/models/student_L{hyperparams['student_layers']}_PKD-{hyperparams['pkd_strategy']}_A{str(hyperparams['alpha']).replace('.', 'p')}_B{int(hyperparams['beta'])}_T{int(hyperparams['temperature'])}")
    print("Training complete!")

if __name__ == "__main__":
    main()