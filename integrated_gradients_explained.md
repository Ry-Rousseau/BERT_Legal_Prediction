# Understanding Integrated Gradients in Astrid

## What Problem Is Astrid Trying to Solve?

After training multiple models of different sizes, Astrid asks:

**"Do the smaller models focus on the same important words/phrases as the teacher model?"**

For example:
- Does the teacher BERT focus on "defendant" and "guilty" in a legal text?
- Does the tiny 2-layer model also focus on those same words?
- Or does it look at different, potentially less relevant words?

This is where **Integrated Gradients** comes in.

---

## What Are Integrated Gradients?

**Integrated Gradients (IG)** is an **attribution method** that answers:

> "Which input tokens contributed most to the model's prediction?"

### The Goal:
Given a text and a model's prediction, assign an **importance score** to each token showing how much it influenced the decision.

### Example:

**Input Text:** "The defendant was found guilty of theft."

**Model Prediction:** First party wins (class 1)

**Token Attributions (scores):**
```
The        [0.02]  ← Low importance
defendant  [0.45]  ← High importance! 
was        [0.01]  ← Low importance
found      [0.08]  ← Moderate importance
guilty     [0.52]  ← High importance!
of         [0.01]  ← Low importance
theft      [0.31]  ← Moderate-high importance
.          [0.00]  ← No importance
```

The model "paid attention" most to "defendant" and "guilty" - these drove the prediction.

---

## How Integrated Gradients Work

### The Core Idea

IG measures how much the prediction changes as each input feature goes from a **baseline** (usually zeros) to the **actual value**.

### Step-by-Step Process

```
1. Start with a baseline input (all zeros - meaningless embedding)
2. Gradually interpolate from baseline → actual input
3. Measure how prediction changes along this path
4. Integrate (sum up) all the gradients along the path
5. Result: Attribution score for each token
```

### Mathematical Formula

For each token *i*:

```
Attribution_i = (x_i - baseline_i) × ∫[α=0 to 1] ∂F/∂x_i (baseline + α × (x - baseline)) dα
```

Where:
- `x_i` = actual token embedding
- `baseline_i` = zero embedding
- `F` = model's prediction function
- `∂F/∂x_i` = gradient of prediction with respect to token i
- `α` = interpolation coefficient (0 to 1)

**Translation:** "How much does the prediction change as we gradually add each token?"

---

## Visual Explanation

### The Interpolation Path

```
Baseline (meaningless)              Actual Input (meaningful)
[0, 0, 0, ..., 0]  ────────────→   [actual embeddings]
      ↑                                      ↑
      |                                      |
   Step 0                                 Step 50
   (α=0.00)                              (α=1.00)

Intermediate steps:
Step 1:  [0.02 × actual embeddings]  (α=0.02)
Step 2:  [0.04 × actual embeddings]  (α=0.04)
...
Step 25: [0.50 × actual embeddings]  (α=0.50)
...
Step 49: [0.98 × actual embeddings]  (α=0.98)
Step 50: [1.00 × actual embeddings]  (α=1.00)
```

At each step, compute:
- Forward pass → prediction
- Backward pass → gradients for each token
- Record how much each token matters at this interpolation level

### What Each Step Represents

```
α = 0.0:  Complete nonsense (all zeros)
          Model output: ~50% (random guess)
          
α = 0.2:  Faint signal ("The ... ... ... guilty")
          Model output: ~55%
          Gradient: Some tokens starting to matter
          
α = 0.5:  Half-strength input ("The defendant ... guilty")
          Model output: ~70%
          Gradient: "guilty" shows high gradient
          
α = 0.8:  Almost full input
          Model output: ~88%
          Gradient: "defendant" and "guilty" peak
          
α = 1.0:  Full input ("The defendant was found guilty...")
          Model output: 92% (confident prediction)
```

---

## Astrid's Implementation

### The Code

```python
def explain_prediction(model, tokenizer, text, label=None):
    model.eval()
    
    # 1. Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    
    # 2. Get model's prediction (if not provided)
    if label is None:
        with torch.no_grad():
            outputs = model(**inputs)
            label = outputs.logits.argmax(-1).item()
    
    # 3. Define forward function for IG
    def forward_func(embeddings, attention_mask=None):
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        return torch.softmax(outputs.logits, dim=1)[:, label]
    
    # 4. Get input embeddings
    embeddings = model.get_input_embeddings()(inputs["input_ids"])
    
    # 5. Compute Integrated Gradients
    ig = IntegratedGradients(forward_func)
    attributions, _ = ig.attribute(
        embeddings,
        additional_forward_args=(inputs["attention_mask"],),
        return_convergence_delta=True
    )
    
    # 6. Sum attributions across embedding dimensions
    scores = attributions.sum(dim=-1).squeeze(0).detach().numpy()
    
    return scores  # Shape: [sequence_length]
```

### What This Returns

For input: "The defendant was found guilty"

Returns array of shape `[128]` (max_length):
```python
[
  0.02,   # [CLS]
  0.02,   # The
  0.45,   # defendant
  0.01,   # was
  0.08,   # found
  0.52,   # guilty
  0.00,   # [SEP]
  0.00,   # [PAD]
  0.00,   # [PAD]
  ...     # (more padding)
]
```

Each number = importance of that token position.

---

## How Astrid Uses This for Comparison

### The Teacher-Student Comparison

```python
# After training all models independently:

# 1. Get teacher's explanations
teacher_vectors = []
for text in validation_texts[:50]:
    scores = explain_prediction(teacher_model, tokenizer, text)
    teacher_vectors.append(scores)  # [50 texts × 128 scores]

# 2. For each student model
for student_model in [Step1, Step2, ..., Step5]:
    student_vectors = []
    
    for text in validation_texts[:50]:
        scores = explain_prediction(student_model, tokenizer, text)
        student_vectors.append(scores)
    
    # 3. Compare attribution patterns
    similarities = []
    for teacher_scores, student_scores in zip(teacher_vectors, student_vectors):
        sim = cosine_similarity([teacher_scores], [student_scores])[0][0]
        similarities.append(sim)
    
    avg_similarity = mean(similarities)
    # e.g., 0.87 means 87% similar attention patterns
```

### What Cosine Similarity Measures

Given two attribution vectors:
```
Teacher:  [0.02, 0.45, 0.01, 0.08, 0.52, 0.00, ...]
Student:  [0.03, 0.41, 0.02, 0.10, 0.48, 0.01, ...]
```

Cosine similarity = angle between vectors:
- **1.0** = Identical attention patterns
- **0.9** = Very similar (slightly different weights)
- **0.5** = Somewhat similar
- **0.0** = Completely different focus
- **-1.0** = Opposite patterns

### Example Results

```
Model                Accuracy    Similarity to Teacher
Teacher_BERT        75.2%       1.000 (itself)
Step1_DistilBERT    73.8%       0.912 ← Very similar reasoning
Step2_Custom6x512   71.5%       0.854 ← Pretty similar
Step3_TinyBERT      69.2%       0.781 ← Somewhat similar
Step4_MiniLM        72.1%       0.823 ← Pretty similar  
Step5_Custom2x384   65.8%       0.642 ← Quite different!
```

**Interpretation:**
- DistilBERT (6 layers) maintains similar reasoning to teacher despite 40% compression
- The tiny 2-layer model performs worse AND reasons differently
- Conclusion: DistilBERT or Custom6x512 are best for deployment (good accuracy + similar reasoning)

---

## Why This Matters

### The Research Question

**"Is a small model just memorizing patterns, or does it genuinely understand the task like the teacher?"**

### Two Scenarios

**Scenario A: High Similarity (Good!)**
```
Teacher focuses on:   ["defendant", "guilty", "evidence"]
Student focuses on:   ["defendant", "guilty", "evidence"]
→ Student learned the RIGHT features
→ Likely to generalize well
→ Safe for deployment
```

**Scenario B: Low Similarity (Concerning!)**
```
Teacher focuses on:   ["defendant", "guilty", "evidence"]
Student focuses on:   ["the", "was", "court"]
→ Student learned DIFFERENT features
→ Might be using shortcuts
→ Could fail on new cases
```

### Practical Implications

If you're deploying a compressed model in production:

**High attribution similarity means:**
- ✅ Model reasoning is preserved
- ✅ Likely to handle edge cases similarly
- ✅ Fewer unexpected behaviors
- ✅ More trustworthy for legal/medical applications

**Low attribution similarity means:**
- ⚠️ Model might be using shortcuts
- ⚠️ Could fail in ways the teacher wouldn't
- ⚠️ Needs more careful validation
- ⚠️ Higher risk for critical applications

---

## Integrated Gradients vs Other Attribution Methods

### Comparison Table

| Method | How It Works | Pros | Cons |
|--------|-------------|------|------|
| **Gradient × Input** | Multiply gradient by input | Fast, simple | Violates some axioms |
| **Saliency Maps** | Raw gradient magnitude | Very fast | Noisy, not theoretically sound |
| **Integrated Gradients** | Integrate gradients along path | Satisfies axioms, clean | Slower (50+ forward passes) |
| **Attention Weights** | Use model's attention | Model-native | Not always interpretable |
| **LIME** | Local linear approximation | Model-agnostic | Can be unstable |
| **SHAP** | Game-theoretic approach | Theoretically rigorous | Very slow |

### Why Astrid Chose Integrated Gradients

1. **Axiomatically sound**: Satisfies sensitivity and implementation invariance
2. **Baseline-invariant**: Results don't depend on arbitrary choices
3. **Widely adopted**: Standard method in NLP interpretability
4. **Good library support**: Captum provides easy implementation
5. **Path-based**: Captures how prediction builds up, not just final gradient

---

## The Mathematics Behind Attribution

### Why Not Just Use Gradients?

**Simple gradient approach:**
```python
gradient = ∂prediction/∂token_embedding
attribution = gradient × token_embedding
```

**Problem:** This violates **implementation invariance**.

Two functionally identical models can give different attributions!

**Example:**
```
Model A: f(x) = x²
Model B: g(x) = x² + 1 - 1  (functionally identical!)

At x=2:
Gradient of A: 2x = 4
Gradient of B: 2x = 4

Attribution with gradient×input:
A: 4 × 2 = 8
B: 4 × 2 = 8  ✓ Same, but only by coincidence

For more complex models, this breaks!
```

### How IG Fixes This

By integrating along a path, IG satisfies:

**1. Sensitivity:** If changing a feature changes output, it gets non-zero attribution

**2. Implementation Invariance:** Functionally equivalent networks give identical attributions

**3. Completeness:** Sum of attributions = (output - baseline_output)

---

## Practical Example: Legal Case

### Input Text
```
"The defendant was charged with theft. The prosecution presented 
strong evidence including video footage. The jury found the 
defendant guilty beyond reasonable doubt."
```

### Teacher BERT Attributions
```
Token              Score   Interpretation
-------------------------------------------
[CLS]             0.01    Special token
The               0.02    Article (low importance)
defendant         0.48    ← Key legal entity
was               0.01    Verb (low importance)
charged           0.15    Legal action word
with              0.01    Preposition
theft             0.32    ← Crime type important
.                 0.00    Punctuation
The               0.02    Article
prosecution       0.12    Legal entity
presented         0.05    Verb
strong            0.08    Descriptor
evidence          0.41    ← Key concept
including         0.03    Conjunction
video             0.22    Evidence type
footage           0.19    Evidence type
.                 0.00    Punctuation
The               0.02    Article
jury              0.25    Decision maker
found             0.11    Verdict verb
the               0.01    Article
defendant         0.38    ← Key entity again
guilty            0.67    ← Strongest signal!
beyond            0.14    Legal standard
reasonable        0.18    Legal standard
doubt             0.21    Legal standard
.                 0.00    Punctuation
```

### DistilBERT (Student) Attributions
```
Token              Score   Similar to Teacher?
------------------------------------------------
defendant         0.45    ✓ (teacher: 0.48)
theft             0.35    ✓ (teacher: 0.32)
evidence          0.39    ✓ (teacher: 0.41)
guilty            0.62    ✓ (teacher: 0.67)
jury              0.23    ✓ (teacher: 0.25)
```

**Cosine Similarity: 0.91** → Very similar reasoning!

### Tiny 2-Layer Model Attributions
```
Token              Score   Similar to Teacher?
------------------------------------------------
The               0.18    ✗ (teacher: 0.02) - focusing on articles!
was               0.22    ✗ (teacher: 0.01) - focusing on verbs!
found             0.31    ✗ (teacher: 0.11) - wrong focus
guilty            0.28    ✗ (teacher: 0.67) - underweighting key term!
defendant         0.12    ✗ (teacher: 0.48) - underweighting key entity!
```

**Cosine Similarity: 0.64** → Different reasoning! ⚠️

---

## How Astrid Uses This Information

### Decision Framework

```python
# After getting all results:
models_ranked = [
    ("Teacher_BERT", 75.2%, 1.000, 110M, "baseline"),
    ("Step1_DistilBERT", 73.8%, 0.912, 66M, "RECOMMENDED"),
    ("Step4_MiniLM", 72.1%, 0.823, 33M, "good option"),
    ("Step2_Custom6x512", 71.5%, 0.854, 29M, "good option"),
    ("Step3_TinyBERT", 69.2%, 0.781, 14M, "acceptable"),
    ("Step5_Custom2x384", 65.8%, 0.642, 11M, "not recommended"),
]

# Selection criteria:
# 1. Accuracy > 70%
# 2. Similarity > 0.80
# 3. Minimize parameters

# Winner: DistilBERT
# - Only 1.4% accuracy drop
# - 91% attribution similarity (very close reasoning)
# - 40% smaller (66M vs 110M params)
# - 2x faster inference
```

---

## Limitations of This Approach

### What IG Doesn't Tell You

1. **Correlation ≠ Causation**
   - High similarity doesn't guarantee correct reasoning
   - Both models could be wrong in similar ways

2. **Token-level focus**
   - Doesn't capture phrase-level or syntactic understanding
   - Might miss higher-order reasoning patterns

3. **Computational cost**
   - Requires 50+ forward passes per example
   - Only practical for small validation set (50 examples in Astrid)

4. **Baseline choice matters**
   - Zero baseline is standard but arbitrary
   - Different baselines can give different attributions

5. **Post-hoc analysis only**
   - Doesn't improve training
   - Just diagnostic tool

---

## Summary

### What Integrated Gradients Does:
- Assigns importance scores to each input token
- Shows which words the model "paid attention to"
- Provides interpretability for black-box models

### How Astrid Uses It:
1. Train all models independently
2. After training, compute attributions for validation set
3. Compare attribution patterns between teacher and students
4. Use cosine similarity to measure alignment
5. Select model with good accuracy AND high similarity

### Why It Matters:
- Validates that compressed models reason similarly to teacher
- Provides confidence in deployment decisions
- Identifies when small models use incorrect shortcuts
- Helps choose the optimal compression level

### The Key Insight:
**A model might have good accuracy but achieve it through wrong reasoning. IG helps detect this.**

If DistilBERT gets 73% accuracy with 91% attribution similarity, it's safer than a model with 74% accuracy but 65% similarity—because it reasons more like the trusted teacher model.
