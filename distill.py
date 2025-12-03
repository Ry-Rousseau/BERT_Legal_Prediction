import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForMultipleChoice
from data_loader import get_dataloaders
from model_utils import create_student_model
from pkd_loss import compute_pkd_loss

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
        T = 2.0 # Temperature
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
# Main Execution
# ---------------------------------------------------------
def main():
    # Load Tokenizer & Teacher
    # REPLACE THIS with the path to your FINE-TUNED CaseHOLD Teacher
    teacher_path = "nlpaueb/legal-bert-base-uncased" 
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    
    print("Loading Teacher...")
    teacher_model = AutoModelForMultipleChoice.from_pretrained(teacher_path)
    
    # Create Student
    print("Creating Student...")
    student_model = create_student_model(teacher_model, num_student_layers=6)

    # Load Data (using pre-split files)
    print("Processing Data...")
    datasets = get_dataloaders(tokenizer, return_dict=True)
    train_dataset = datasets['train']
    eval_dataset = datasets['dev']

    # Training Args
    training_args = TrainingArguments(
        output_dir="results/outputs/test_run",
        max_steps = 10, # remove for full training
        # num_train_epochs=4,
        per_device_train_batch_size=2, # increase for full training
        logging_steps=1,
        remove_unused_columns=False, # Important so we don't drop columns needed for logic
        learning_rate=5e-5,
        report_to= "none", # Must be string, not boolean
        use_cpu=False, # Explicitly use GPU
        no_cuda=False  # Ensure CUDA is not disabled
    )

    print(f"Training device: {training_args.device}")
    print(f"Number of GPUs: {training_args.n_gpu}")

    # Initialize Trainer
    trainer = PKDTrainer(
        teacher_model=teacher_model,
        pkd_strategy="skip", # or "last"
        model=student_model,
        args=training_args,
        train_dataset=train_dataset.select(range(50)), # limited small slice of full data
        tokenizer=tokenizer,
        
        # HYPERPARAMETERS HERE
        alpha=0.5,  # Balances Hard Labels vs Soft Labels (Distillation)
        beta=10.0,  # Balances the "Patient" intermediate layer loss
    )

    # Train
    print("Starting Distillation...")
    trainer.train()
    
    # Save
    student_model.save_pretrained("checkpoints/models/dry_test_run")

if __name__ == "__main__":
    main()