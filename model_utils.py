import torch
from transformers import AutoModelForMultipleChoice, AutoConfig

def create_student_model(teacher_model, num_student_layers=6):
    """
    Creates a student model by copying the first 'num_student_layers' from the teacher.
    """
    # 1. Create a configuration for the student (same as teacher but fewer layers)
    # IMPORTANT: Copy the config to avoid modifying the teacher's config
    import copy
    student_config = copy.deepcopy(teacher_model.config)
    student_config.num_hidden_layers = num_student_layers

    # 2. Initialize a "blank" student with this config
    student_model = AutoModelForMultipleChoice.from_config(student_config)
    
    # 3. Copy the Embeddings (CRITICAL: Student must speak the same "language")
    student_model.bert.embeddings.load_state_dict(teacher_model.bert.embeddings.state_dict())
    
    # 4. Copy the Encoder Layers (The "Truncation" Strategy)
    # We copy the first N layers from the teacher to the student
    for i in range(num_student_layers):
        student_model.bert.encoder.layer[i].load_state_dict(teacher_model.bert.encoder.layer[i].state_dict())
        
    # 5. Copy the Classification Head
    student_model.classifier.load_state_dict(teacher_model.classifier.state_dict())
    
    return student_model

def test_student_init():
    print("Loading Teacher...")
    # Use a small model for the test to be fast
    teacher = AutoModelForMultipleChoice.from_pretrained("nlpaueb/legal-bert-base-uncased")
    
    print("Creating Student (6 layers)...")
    student = create_student_model(teacher, num_student_layers=6)
    
    # Test 1: Layer Count
    student_layers = len(student.bert.encoder.layer)
    print(f"Student Layers: {student_layers}")
    assert student_layers == 6, "Student should have 6 layers!"
    print("Student has correct depth.")

    # Test 2: Weight Equality (The "Copy" Check)
    # The student's Layer 0 should be identical to Teacher's Layer 0
    teacher_w = teacher.bert.encoder.layer[0].output.dense.weight
    student_w = student.bert.encoder.layer[0].output.dense.weight
    
    if torch.equal(teacher_w, student_w):
        print("Layer 0 weights match exactly (Copy successful).")
    else:
        print("Layer 0 weights are different! Initialization failed.")

    # Test 3: Structural Difference (The "Truncation" Check)
    # The student should NOT have a Layer 11
    try:
        _ = student.bert.encoder.layer[11]
        print("Student still has Layer 11! Not good")
    except IndexError:
        print("Student correctly ends at Layer 6 (IndexError on layer 11).")

if __name__ == "__main__":
    test_student_init()