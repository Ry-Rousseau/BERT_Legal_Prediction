# Research Framework: Exploring Interpretability, Compression, and Accuracy

## Can You Change the Classification Layer in BertForSequenceClassification?

**Short answer: YES!**

`BertForSequenceClassification` is just a convenience wrapper. You can:

1. **Subclass and override the classifier**
2. **Use base `BertModel` and add custom heads**
3. **Replace the linear layer with anything you want**

### Quick Examples

```python
# Option 1: Subclass and customize
class BertWithCustomClassifier(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # Replace the simple linear classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, 2)
        )

# Option 2: Build from scratch
class BertWithAttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.attention_head = AttentionClassificationHead(768, 2)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state  # All tokens
        logits = self.attention_head(sequence_output, attention_mask)
        return logits
```

**The key insight:** BERT gives you representations. What you do with them is up to you!

---

## The Research Pipeline: Where Can We Vary?

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT TEXT                               │
│              "The defendant was found guilty..."                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    VARIATION POINT #1                            │
│                   ENCODER ARCHITECTURE                           │
│                                                                  │
│   • BERT-base (12 layers, 768-dim)                              │
│   • DistilBERT (6 layers, 768-dim)                              │
│   • TinyBERT (4 layers, 312-dim)                                │
│   • Custom (2 layers, 384-dim)                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    VARIATION POINT #2                            │
│                  CLASSIFICATION HEAD                             │
│                                                                  │
│   • Linear: Direct [768] → [2]                                  │
│   • Deep MLP: [768] → [384] → [192] → [2]                       │
│   • Attention-based: Weighted combination of all tokens         │
│   • Dual-head: Separate paths for different features           │
│   • Ensemble: Multiple classifiers voting                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         PREDICTIONS                              │
│                    [logit_0, logit_1]                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    VARIATION POINT #3                            │
│                  INTERPRETABILITY METHOD                         │
│                                                                  │
│   • Integrated Gradients: Path-based attribution                │
│   • Attention Weights: Model's internal attention              │
│   • LIME: Local approximation                                   │
│   • SHAP: Game-theoretic attribution                            │
│   • Layer-wise Relevance: Backprop variants                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS & COMPARISON                         │
│     • Accuracy vs Model Size                                    │
│     • Interpretability Similarity                               │
│     • Reasoning Pattern Analysis                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Variation Point #1: Encoder Architecture (Already Explored in Astrid)

This is what Astrid does - varying the encoder size.

**Keep this dimension, but now we add two more:**

---

## Variation Point #2: Classification Head Architectures

### Why Vary the Classification Head?

**Research Questions:**
- Does a complex classifier help smaller encoders?
- Can we trade encoder complexity for classifier complexity?
- Do different heads produce different interpretability patterns?
- Which architectures are more robust?

### Classification Head Options

#### 1. **Linear (Baseline - Current Approach)**

```python
class LinearClassifier(nn.Module):
    """Simple linear projection"""
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, pooled_output):
        x = self.dropout(pooled_output)
        return self.classifier(x)

# Characteristics:
# - Fastest inference
# - Fewest parameters (~1.5K)
# - Most interpretable (direct weights)
# - May underfit with small encoders
```

#### 2. **Multi-Layer Perceptron (MLP)**

```python
class MLPClassifier(nn.Module):
    """Deep classification with multiple hidden layers"""
    def __init__(self, hidden_size, num_labels, mlp_dims=[384, 192]):
        super().__init__()
        layers = []
        input_dim = hidden_size
        
        for mlp_dim in mlp_dims:
            layers.extend([
                nn.Linear(input_dim, mlp_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = mlp_dim
        
        layers.append(nn.Linear(input_dim, num_labels))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, pooled_output):
        return self.classifier(pooled_output)

# Characteristics:
# - More expressive (can learn complex boundaries)
# - More parameters (~150K)
# - May help smaller encoders compensate
# - Less directly interpretable
```

#### 3. **Attention-Based Classifier**

```python
class AttentionClassifier(nn.Module):
    """Uses attention to weight token importance"""
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, sequence_output, attention_mask):
        # sequence_output: [batch, seq_len, hidden_size]
        
        # Compute attention scores
        attention_scores = self.attention(sequence_output)  # [batch, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)      # [batch, seq_len]
        
        # Mask padding tokens
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, -1e9
        )
        
        # Softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len]
        
        # Weighted sum of token representations
        weighted_output = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch, 1, seq_len]
            sequence_output                   # [batch, seq_len, hidden_size]
        ).squeeze(1)  # [batch, hidden_size]
        
        return self.classifier(weighted_output), attention_weights

# Characteristics:
# - Learns which tokens matter
# - Attention weights are interpretable!
# - More flexible than [CLS] token
# - Moderate parameters (~3K)
```

#### 4. **Dual-Head Classifier**

```python
class DualHeadClassifier(nn.Module):
    """Separate paths for different feature types"""
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        # Path 1: Focus on syntax/structure
        self.structural_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Path 2: Focus on semantics/content
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Combine both paths
        self.fusion = nn.Linear(512, num_labels)
    
    def forward(self, pooled_output):
        structural = self.structural_head(pooled_output)
        semantic = self.semantic_head(pooled_output)
        combined = torch.cat([structural, semantic], dim=-1)
        return self.fusion(combined)

# Characteristics:
# - Can learn complementary features
# - More interpretable (can analyze each path)
# - Good for ablation studies
# - Moderate parameters (~200K)
```

#### 5. **Ensemble Classifier**

```python
class EnsembleClassifier(nn.Module):
    """Multiple classifiers voting"""
    def __init__(self, hidden_size, num_labels, num_heads=3):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) 
            for _ in range(num_heads)
        ])
    
    def forward(self, pooled_output):
        # Get predictions from each head
        logits = [head(pooled_output) for head in self.heads]
        # Average
        return torch.mean(torch.stack(logits), dim=0)

# Characteristics:
# - More robust predictions
# - Can measure disagreement (uncertainty)
# - Moderate parameters (3-5x linear)
# - Interpretable diversity
```

---

## Variation Point #3: Interpretability Methods

### Why Vary Interpretability Methods?

**Research Questions:**
- Do different methods reveal different reasoning patterns?
- Which methods best capture model compression effects?
- Are some methods more stable across model sizes?
- Can we find interpretability-accuracy tradeoffs?

### Interpretability Method Options

#### 1. **Integrated Gradients (Current Approach)**

```python
def integrated_gradients_attribution(model, inputs, baseline=None):
    """Path-based gradient attribution"""
    from captum.attr import IntegratedGradients
    
    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        inputs,
        baselines=baseline,
        n_steps=50
    )
    return attributions

# Pros:
# - Theoretically grounded (axioms)
# - Stable and reliable
# - Works for any differentiable model

# Cons:
# - Computationally expensive (50+ forward passes)
# - Baseline choice can affect results
```

#### 2. **Attention Weights (Model-Native)**

```python
def attention_attribution(model, inputs):
    """Use model's internal attention mechanisms"""
    outputs = model(
        inputs, 
        output_attentions=True
    )
    
    # Get attention weights from all layers
    attentions = outputs.attentions  # Tuple of [batch, heads, seq, seq]
    
    # Average across heads and layers
    avg_attention = torch.mean(torch.stack(attentions), dim=[0, 1])
    
    # Attention to [CLS] token shows token importance
    cls_attention = avg_attention[:, 0, :]  # [batch, seq_len]
    
    return cls_attention

# Pros:
# - Free (computed during forward pass)
# - Model's actual mechanism
# - No extra computation

# Cons:
# - Only available in transformer models
# - Not always faithful to predictions
# - Can be noisy
```

#### 3. **LIME (Local Interpretable Model-agnostic Explanations)**

```python
def lime_attribution(model, tokenizer, text):
    """Local linear approximation"""
    from lime.lime_text import LimeTextExplainer
    
    explainer = LimeTextExplainer(class_names=['class0', 'class1'])
    
    def predict_fn(texts):
        # Tokenize and predict
        encodings = tokenizer(texts, return_tensors='pt', padding=True)
        with torch.no_grad():
            logits = model(**encodings).logits
        return F.softmax(logits, dim=-1).numpy()
    
    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=10,
        num_samples=1000
    )
    
    return explanation.as_list()

# Pros:
# - Model-agnostic (works with any black box)
# - Human-interpretable word importance
# - Shows local behavior

# Cons:
# - Stochastic (varies between runs)
# - Computationally expensive (1000+ predictions)
# - Local only (doesn't capture global patterns)
```

#### 4. **SHAP (SHapley Additive exPlanations)**

```python
def shap_attribution(model, tokenizer, texts):
    """Game-theoretic attribution"""
    import shap
    
    def predict_fn(texts):
        encodings = tokenizer(texts, return_tensors='pt', padding=True)
        with torch.no_grad():
            logits = model(**encodings).logits
        return logits.numpy()
    
    explainer = shap.Explainer(predict_fn, tokenizer)
    shap_values = explainer(texts)
    
    return shap_values

# Pros:
# - Theoretically optimal (Shapley values)
# - Fair attribution
# - Consistent across methods

# Cons:
# - Very computationally expensive
# - Exponential complexity
# - Approximations needed for large inputs
```

#### 5. **Gradient × Input (Simple Baseline)**

```python
def gradient_input_attribution(model, inputs, labels):
    """Multiply gradient by input (fast but flawed)"""
    inputs.requires_grad = True
    
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    
    # Gradient × input
    attributions = inputs.grad * inputs
    
    return attributions

# Pros:
# - Very fast (single backward pass)
# - Simple to implement
# - Intuitive

# Cons:
# - Violates theoretical axioms
# - Can be misleading
# - Not as reliable as IG
```

---

## Experimental Design: Systematic Exploration

### The Research Matrix

```
DIMENSIONS TO VARY:
1. Encoder Size: [BERT-base, DistilBERT, TinyBERT, Custom2L]
2. Classifier Type: [Linear, MLP, Attention, DualHead, Ensemble]
3. Interpretability: [IG, Attention, LIME, SHAP, Grad×Input]

TOTAL COMBINATIONS: 4 × 5 × 5 = 100 experiments
```

### Recommended Experimental Pipeline

```python
# Pseudocode for systematic exploration

ENCODERS = {
    'BERT': BertModel (12L, 768d, 110M params),
    'DistilBERT': DistilBertModel (6L, 768d, 66M params),
    'TinyBERT': TinyBertModel (4L, 312d, 14M params),
    'Mini': CustomBert (2L, 384d, 11M params)
}

CLASSIFIERS = {
    'Linear': LinearClassifier,
    'MLP': MLPClassifier,
    'Attention': AttentionClassifier,
    'DualHead': DualHeadClassifier,
    'Ensemble': EnsembleClassifier
}

INTERPRETABILITY_METHODS = {
    'IG': integrated_gradients_attribution,
    'Attention': attention_attribution,
    'LIME': lime_attribution,
    'SHAP': shap_attribution,
    'GradInput': gradient_input_attribution
}

# Experimental loop
results = []

for encoder_name, encoder_class in ENCODERS.items():
    for classifier_name, classifier_class in CLASSIFIERS.items():
        
        # Build model
        model = create_model(encoder_class, classifier_class)
        
        # Train
        trainer = Trainer(model, train_data)
        trainer.train()
        
        # Evaluate accuracy
        accuracy = evaluate(model, test_data)
        
        # Test each interpretability method
        for interp_name, interp_fn in INTERPRETABILITY_METHODS.items():
            
            # Get attributions
            attributions = interp_fn(model, validation_texts)
            
            # Compare to baseline (BERT + Linear + IG)
            similarity = compare_attributions(
                attributions, 
                baseline_attributions
            )
            
            # Store results
            results.append({
                'encoder': encoder_name,
                'encoder_params': encoder.num_parameters(),
                'classifier': classifier_name,
                'interpretability': interp_name,
                'accuracy': accuracy,
                'attribution_similarity': similarity,
                'inference_time': measure_speed(model),
                'interpretability_compute': measure_interp_cost(interp_fn)
            })

# Analyze
df = pd.DataFrame(results)
```

---

## Research Questions to Explore

### 1. **Does Classifier Complexity Compensate for Encoder Compression?**

**Hypothesis:** A smaller encoder + complex classifier might match a larger encoder + simple classifier.

**Test:**
```
Compare:
  • BERT (110M) + Linear (1.5K) = 110M total
  • TinyBERT (14M) + MLP (150K) = 14.2M total

If similar accuracy → classifier complexity can compensate!
```

**Measurement:**
- Plot accuracy vs total parameters
- Color by encoder-classifier combinations
- Find Pareto frontier

### 2. **Which Classification Heads Preserve Interpretability Best?**

**Hypothesis:** Attention-based classifiers maintain similar reasoning patterns across compression.

**Test:**
```
For each encoder size:
  • Train with all 5 classifier types
  • Compute attribution similarity to baseline
  • Rank classifiers by similarity

Expected: Attention classifier > MLP > Linear
```

**Measurement:**
- Heatmap: Encoder (rows) × Classifier (cols) → Similarity score
- Identify which classifiers preserve teacher's reasoning

### 3. **Do Different Interpretability Methods Agree?**

**Hypothesis:** Methods should correlate, but may reveal different aspects.

**Test:**
```
For each model:
  • Compute attributions with all 5 methods
  • Calculate pairwise correlations
  • Identify consensus vs disagreement cases

High correlation → robust reasoning
Low correlation → method-dependent or unstable reasoning
```

**Measurement:**
- Correlation matrix between methods
- Identify stable vs unstable models
- Case studies where methods disagree

### 4. **Is There an Interpretability-Accuracy Tradeoff?**

**Hypothesis:** More interpretable models (e.g., attention classifier) may sacrifice some accuracy.

**Test:**
```
Measure "interpretability stability":
  • Standard deviation of attributions across methods
  • Lower std = more interpretable (methods agree)

Plot: Accuracy vs Interpretability Stability
```

**Measurement:**
- Scatter plot with Pareto frontier
- Identify models that optimize both
- Quantify tradeoff slope

### 5. **Do Compression Effects Depend on Classifier Architecture?**

**Hypothesis:** Complex classifiers help maintain accuracy as encoders shrink.

**Test:**
```
For each classifier type:
  • Plot accuracy vs encoder size
  • Calculate "compression slope"
  • Compare slopes across classifiers

Shallower slope = classifier helps with compression
```

**Measurement:**
- Line plots: Encoder size (x) vs Accuracy (y), one line per classifier
- Calculate regression slopes
- Test for significant differences

---

## Visualization Framework

### Key Plots to Generate

#### 1. **3D Landscape Plot**
```
X-axis: Encoder Size (110M → 11M)
Y-axis: Classifier Complexity (Linear → Ensemble)
Z-axis: Accuracy
Color: Attribution Similarity to Teacher

Shows: Where the sweet spots are
```

#### 2. **Attribution Similarity Heatmap**
```
Rows: Encoder types
Cols: Classifier types
Cell value: Attribution similarity
Cell color: Red (low) → Green (high)

Shows: Which combinations preserve reasoning
```

#### 3. **Method Agreement Matrix**
```
Rows: Interpretability methods
Cols: Interpretability methods
Cell value: Correlation between methods
For each model size

Shows: Which methods are reliable
```

#### 4. **Pareto Frontier**
```
X-axis: Model size (parameters)
Y-axis: Accuracy
Points: Each encoder-classifier combination
Line: Pareto frontier (best accuracy for each size)

Shows: Optimal configurations
```

#### 5. **Interpretability Stability**
```
X-axis: Model
Y-axis: Standard deviation of attributions across methods
Lower = more stable interpretability

Shows: Which models have consistent explanations
```

---

## Analysis Framework

### Quantitative Metrics

```python
# For each model configuration:

metrics = {
    # Performance
    'accuracy': test_accuracy,
    'f1_score': f1_score,
    'inference_time': avg_prediction_time,
    
    # Compression
    'encoder_params': num_encoder_parameters,
    'classifier_params': num_classifier_parameters,
    'total_params': total_parameters,
    'compression_ratio': teacher_params / total_params,
    
    # Interpretability
    'attribution_similarity': cosine_sim_to_teacher,
    'method_agreement': avg_correlation_across_methods,
    'interpretation_stability': std_of_attributions,
    'interpretation_cost': time_to_compute_attributions,
    
    # Efficiency
    'accuracy_per_param': accuracy / total_params,
    'accuracy_per_second': accuracy / inference_time,
    
    # Robustness
    'prediction_confidence': avg_max_probability,
    'attribution_magnitude': mean_absolute_attribution
}
```

### Qualitative Analysis

**Case Study Examples:**

1. **Where methods agree**: Show examples where all 5 interpretability methods highlight the same tokens
2. **Where methods disagree**: Analyze why different methods give different explanations
3. **Failure modes**: Identify cases where compressed models focus on wrong features
4. **Success stories**: Highlight tiny models that reason like the teacher

---

## Example Research Workflow

### Phase 1: Baseline Establishment
```python
# Train Teacher with all classifier types
teacher_results = train_and_evaluate(
    encoder='BERT-base',
    classifiers=['Linear', 'MLP', 'Attention', 'DualHead', 'Ensemble'],
    interpretability_methods='all'
)

# Establish ground truth
baseline_accuracy = teacher_results['Linear']['accuracy']
baseline_attributions = teacher_results['Linear']['attributions']
```

### Phase 2: Compression Study
```python
# For each compression level
for encoder in ['DistilBERT', 'TinyBERT', 'Mini']:
    for classifier in ['Linear', 'MLP', 'Attention', 'DualHead', 'Ensemble']:
        
        # Train model
        model = train(encoder, classifier)
        
        # Measure performance
        accuracy = evaluate(model)
        
        # Measure interpretability (all methods)
        attributions = {
            method: get_attributions(model, method)
            for method in ['IG', 'Attention', 'LIME', 'SHAP', 'GradInput']
        }
        
        # Compare to baseline
        similarities = {
            method: compare(attributions[method], baseline_attributions)
            for method in attributions
        }
        
        # Store results
        log_results(encoder, classifier, accuracy, similarities)
```

### Phase 3: Analysis
```python
# Load all results
df = pd.DataFrame(all_results)

# Key analyses:
# 1. Best accuracy for each size budget
best_per_size = df.groupby('encoder_params')['accuracy'].max()

# 2. Most interpretable configurations
most_interpretable = df.nlargest(10, 'method_agreement')

# 3. Best tradeoffs
pareto_optimal = find_pareto_frontier(
    df, 
    objectives=['accuracy', 'attribution_similarity'],
    minimize=['total_params']
)

# 4. Classifier impact
classifier_effects = df.groupby('classifier').agg({
    'accuracy': 'mean',
    'attribution_similarity': 'mean',
    'inference_time': 'mean'
})

# 5. Method reliability
method_correlations = compute_method_correlation_matrix(df)
```

---

## Expected Insights

### Potential Findings:

1. **"Classifier compensation effect"**: Complex classifiers may help small encoders maintain accuracy

2. **"Interpretability preservation"**: Attention-based classifiers might better preserve reasoning patterns across compression

3. **"Method stability hierarchy"**: Some interpretability methods (e.g., IG, Attention) may be more stable than others (LIME, SHAP)

4. **"Sweet spot identification"**: Optimal encoder-classifier combinations for different deployment scenarios

5. **"Reasoning divergence points"**: Specific model sizes where reasoning patterns break down

---

## Practical Outcomes

### Decision Framework for Deployment

```
IF deployment_priority == 'accuracy':
    → Choose: Larger encoder + simple classifier
    
ELIF deployment_priority == 'speed':
    → Choose: Tiny encoder + optimized classifier
    
ELIF deployment_priority == 'interpretability':
    → Choose: Medium encoder + attention classifier
    
ELIF deployment_priority == 'robustness':
    → Choose: Model with highest method agreement
```

### Model Selection Tool

```python
def select_model(requirements):
    """
    requirements = {
        'max_params': 50M,
        'min_accuracy': 0.70,
        'min_similarity': 0.80,
        'max_inference_time': 100ms
    }
    """
    candidates = df[
        (df['total_params'] <= requirements['max_params']) &
        (df['accuracy'] >= requirements['min_accuracy']) &
        (df['attribution_similarity'] >= requirements['min_similarity']) &
        (df['inference_time'] <= requirements['max_inference_time'])
    ]
    
    # Rank by composite score
    candidates['score'] = (
        0.4 * candidates['accuracy'] +
        0.3 * candidates['attribution_similarity'] +
        0.2 * (1 - candidates['total_params'] / max_params) +
        0.1 * (1 - candidates['inference_time'] / max_time)
    )
    
    return candidates.nlargest(5, 'score')
```

---

## Summary: The Research Framework

### Three Dimensions of Variation:

1. **Encoder Architecture** (size/compression)
   - BERT-base → DistilBERT → TinyBERT → Custom2L
   - Tests: How much can we compress?

2. **Classification Head** (complexity/design)
   - Linear → MLP → Attention → DualHead → Ensemble
   - Tests: Can classifiers compensate for smaller encoders?

3. **Interpretability Method** (analysis approach)
   - IG → Attention → LIME → SHAP → Grad×Input
   - Tests: Which methods reveal reliable reasoning patterns?

### Key Research Questions:

- Can classifier complexity compensate for encoder compression?
- Which architectures preserve interpretable reasoning?
- Do different interpretability methods agree?
- What are the accuracy-interpretability-efficiency tradeoffs?
- Where are the optimal deployment configurations?

### Methodological Approach:

1. **Systematic**: Test all combinations
2. **Comparative**: Benchmark against teacher
3. **Multi-metric**: Accuracy + interpretability + efficiency
4. **Practical**: Produce deployment guidelines

This framework transforms "distillation analysis" into a comprehensive study of the relationships between model architecture, reasoning patterns, and practical performance.
