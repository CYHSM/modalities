# LLM Activation Analysis Framework

Statistical analysis of LLM activations using fMRI-inspired methods. Compare how different types of inputs activate different parts of the model.

## 📂 Project Structure

```
llm-activation-analysis/
├── core/                      # Reusable infrastructure
│   ├── capture.py            # Activation capture & batch processing
│   └── stats.py              # Statistical analysis tools
│
├── experiments/              # Specific experiments
│   ├── math_vs_nonmath.py   # Math vs non-math comparison
│   └── [your_experiment].py # Add new experiments here
│
├── run_experiments.py        # Main runner script
└── experiments/              # Output directory
    └── math_vs_nonmath/     # Experiment results
        ├── data.pkl         # Raw captured data
        ├── metadata.txt     # Experiment info
        └── figures/         # Visualizations
```

## 🚀 Quick Start

### Run the Math vs Non-Math Experiment

```bash
# Basic run with GPT-2
python run_experiments.py --experiment math --model gpt2

# Use a different model
python run_experiments.py --experiment math --model microsoft/phi-2

# Re-analyze existing data
python run_experiments.py --experiment math --load
```

### What It Does

1. **Captures activations** from two conditions (math prompts vs general prompts)
2. **Computes statistical contrasts** between conditions:
   - T-tests at each activation dimension
   - Cohen's d effect sizes
   - Multiple comparison correction
3. **Identifies significant differences** across layers
4. **Visualizes results** with heatmaps and bar charts
5. **Finds most discriminative neurons** that differentiate conditions

## 📊 Statistical Methods

### Activation Comparison
- **Between-group t-tests**: Compare mean activations for each neuron
- **Effect sizes (Cohen's d)**: Measure magnitude of differences
- **Multiple comparison correction**: FDR/Bonferroni for controlling false positives
- **Layer-wise analysis**: Aggregate statistics per layer

### Inspired by fMRI Analysis
- Treats each activation dimension like a "voxel" in brain imaging
- Performs mass univariate testing with correction
- Creates activation "contrast maps" between conditions
- Identifies regions (layers/neurons) that respond differently

## 🔬 Creating New Experiments

### 1. Define Your Conditions

```python
# experiments/emotion_analysis.py
POSITIVE_PROMPTS = [
    "I feel happy because",
    "The best day of my life was",
    # ...
]

NEGATIVE_PROMPTS = [
    "I feel sad because", 
    "The worst day was when",
    # ...
]
```

### 2. Create Experiment Class

```python
from capture import ActivationCapture, DataStore
from stats import ActivationStats

class EmotionExperiment:
    def __init__(self, model_path="gpt2"):
        self.capture = ActivationCapture(model_path)
    
    def run_experiment(self):
        # Capture both conditions
        positive_data = self.capture.capture_batch(POSITIVE_PROMPTS)
        negative_data = self.capture.capture_batch(NEGATIVE_PROMPTS)
        
        # Save data
        experiment_data = {
            'positive': positive_data,
            'negative': negative_data
        }
        DataStore.save_experiment(experiment_data, "emotion_analysis")
        return experiment_data
    
    def analyze_experiment(self, data):
        # Statistical comparison
        layer_stats = ActivationStats.compute_layer_wise_stats(
            data['positive'],
            data['negative']
        )
        # ... visualization code
```

### 3. Advanced Analysis Options

```python
# Multiple comparison correction
corrected = ActivationStats.multiple_comparison_correction(
    p_values, 
    method='fdr_bh',  # or 'bonferroni'
    alpha=0.05
)

# RSA-style similarity analysis
similarity = ActivationStats.compute_rsa_similarity(
    acts1, acts2,
    metric='correlation'  # or 'cosine', 'euclidean'
)

# Find top discriminative dimensions
summary = ActivationStats.summarize_contrast(
    contrast_results,
    top_n=20
)
```

## 📈 Interpreting Results

### Layer Analysis Plot
- **Top**: Effect sizes (Cohen's d) - larger = bigger difference
- **Bottom**: Proportion of significant neurons - higher = more widespread effect

### Activation Heatmaps
- **Mean Difference**: Blue = higher in condition 2, Red = higher in condition 1
- **Cohen's d**: Effect size magnitude (yellow = large effect)
- **-log10(p)**: Statistical significance (brighter = more significant)
- **Significant Mask**: Binary map of p < 0.05

### Top Discriminative Neurons
Lists individual neurons with strongest differential responses:
- **Positive d**: More active for condition 1
- **Negative d**: More active for condition 2
- **p-value**: Statistical confidence

## 🎯 Experiment Ideas

### Language Tasks
- **Syntax**: Grammatical vs ungrammatical sentences
- **Semantics**: Concrete vs abstract words
- **Style**: Formal vs informal text
- **Languages**: English vs code vs other languages

### Reasoning Tasks
- **Logic**: Valid vs invalid arguments
- **Causality**: Causal vs correlational statements
- **Factual**: True vs false claims
- **Counterfactual**: Real vs hypothetical scenarios

### Domain-Specific
- **Scientific**: Physics vs biology vs chemistry
- **Creative**: Poetry vs prose vs dialogue
- **Technical**: Code vs documentation vs comments
- **Temporal**: Past vs present vs future tense

## 🔧 Core API Reference

### ActivationCapture
```python
capture = ActivationCapture(model_path, device='cuda')

# Single sample
result = capture.capture_single(text, max_new_tokens=1)

# Batch processing
results = capture.capture_batch(texts, layers_to_capture=['mlp', 'attn'])
```

### ActivationStats
```python
# Compute contrast between groups
contrast = ActivationStats.compute_contrast(
    group1_activations, 
    group2_activations,
    test='ttest'  # or 'welch', 'mannwhitney'
)

# Layer-wise analysis
layer_stats = ActivationStats.compute_layer_wise_stats(
    condition1_data,
    condition2_data
)
```

### DataStore
```python
# Save experiment
DataStore.save_experiment(data, "experiment_name")

# Load experiment
data = DataStore.load_experiment("experiment_name")
```

## 💡 Tips

1. **Sample Size**: Use at least 15-20 prompts per condition for statistical power
2. **Prompt Design**: Keep prompts similar length and complexity within conditions
3. **Multiple Comparisons**: Always apply correction when testing many neurons
4. **Effect Sizes**: Focus on Cohen's d > 0.5 for meaningful differences
5. **Layer Selection**: Can limit analysis to specific layers for efficiency

## 📚 References

This framework adapts classical neuroimaging analysis methods:
- Mass univariate testing (Friston et al., 1994)
- FDR correction (Benjamini & Hochberg, 1995)
- Representational Similarity Analysis (Kriegeskorte et al., 2008)

## 🤝 Contributing

To add new statistical methods or visualization types:
1. Core functions go in `core/stats.py` or `core/capture.py`
2. Experiment-specific code goes in `experiments/`
3. Keep experiments self-contained and well-documented

## 📜 License

MIT