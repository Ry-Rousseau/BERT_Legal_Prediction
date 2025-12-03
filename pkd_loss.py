import torch
import torch.nn.functional as F

def compute_pkd_loss(student_states, teacher_states, strategy="skip"):
    """
    Computes Patient Knowledge Distillation loss.
    """
    loss = 0
    num_student_layers = len(student_states)
    num_teacher_layers = len(teacher_states)

    # 1. Define Mapping Strategy
    if strategy == "skip":
        # Maps student 1->2, 2->4... (Every 2nd layer)
        # Note: We slice [1:] to skip the embedding layer output if included
        teacher_indices = [i * 2 + 1 for i in range(num_student_layers)]
    elif strategy == "last":
        # Maps student 1->7, 2->8... (Last k layers)
        teacher_indices = [num_teacher_layers - num_student_layers + i for i in range(num_student_layers)]

    # 2. Compute Loss - only iterate over valid indices
    for s_idx, t_idx in enumerate(teacher_indices):
        # Check if teacher index is valid
        if t_idx >= num_teacher_layers:
            break

        # Normalize vectors (L2 norm) - Critical step from the paper
        s_norm = F.normalize(student_states[s_idx], p=2, dim=-1)
        t_norm = F.normalize(teacher_states[t_idx], p=2, dim=-1)

        # Mean Squared Error
        loss += F.mse_loss(s_norm, t_norm)

    return loss

def test_loss_math():
    # Simulate data: Batch size 2, Sequence Length 10, Hidden Dim 768
    # Teacher has 12 layers, Student has 6
    print(" Generating dummy hidden states...")
    
    # Create list of 13 tensors (Embeddings + 12 layers) for Teacher
    teacher_states = [torch.randn(2, 10, 768) for _ in range(13)]
    
    # Create list of 7 tensors (Embeddings + 6 layers) for Student
    student_states = [torch.randn(2, 10, 768) for _ in range(7)]
    
    # Test "Skip" Strategy
    loss = compute_pkd_loss(student_states[1:], teacher_states[1:], strategy="skip") # be sure to slice off the embeddings, keep only the layers   
    print(f"PKD-Skip Loss: {loss.item()}")
    
    # Verify it's not zero (unless random tensors happened to match perfectly, which is impossible)
    assert loss > 0, "Loss should be non-zero"

    # Test "last" Strategy
    loss_last = compute_pkd_loss(student_states[1:], teacher_states[1:], strategy="last")
    print(f"PKD-Last Loss: {loss_last.item()}")
    assert loss_last > 0, "Loss should be non-zero"

    # Test "Identity" (Pass same tensors for both using "last" strategy)
    # The loss between X and X should be 0
    zero_loss = compute_pkd_loss(student_states[1:], student_states[1:], strategy="last")
    print(f"Identity Loss (Should be close to 0): {zero_loss.item()}")

    if zero_loss < 1e-6:
        print("Loss logic is mathematically sound (Identity = 0).")
    else:
        print("Something is wrong with the math.")

if __name__ == "__main__":
    test_loss_math()