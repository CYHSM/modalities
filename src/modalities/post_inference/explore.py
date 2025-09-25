import numpy as np
from visualize import load_activations, group_by_layer, combine_layer_activations

def stats(activations, step_texts=None):
    first_step = activations[0]
    layer_groups = group_by_layer(first_step)
    
    int_keys = [k for k in layer_groups.keys() if isinstance(k, int)]
    str_keys = [k for k in layer_groups.keys() if isinstance(k, str)]
    all_keys = sorted(int_keys) + sorted(str_keys)
    
    print(f"Steps: {len(activations)}")
    print(f"Layers: {len(all_keys)}")
    
    if step_texts:
        print(f"Text progression:")
        for i, text in enumerate(step_texts[:5]):
            print(f"  Step {i}: {text[:100]}{'...' if len(text) > 100 else ''}")
        if len(step_texts) > 5:
            print(f"  ... and {len(step_texts) - 5} more steps")
    
    print(f"\nLayer breakdown:")
    for layer_id in all_keys:
        layer_dict = layer_groups[layer_id]
        total_elements = sum(act.numel() for act in layer_dict.values())
        
        layer_name = f"Layer_{layer_id}" if isinstance(layer_id, int) else str(layer_id)
        print(f"{layer_name:20} {len(layer_dict):3} modules, {total_elements:10,} elements")
        
        for name in sorted(layer_dict.keys())[:3]:
            shape = layer_dict[name].shape
            print(f"  └─ {name:45} {str(shape):15}")
        
        if len(layer_dict) > 3:
            print(f"  └─ ... and {len(layer_dict) - 3} more")

def compare_normalization_effects(activations):
    import matplotlib.pyplot as plt
    from visualize import normalize
    
    first_step = activations[0]
    layer_groups = group_by_layer(first_step)
    
    sample_layer_id = list(layer_groups.keys())[0]
    values = combine_layer_activations(layer_groups[sample_layer_id])[:1000]
    
    methods = ['minmax', 'percentile', 'zscore', 'log']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, method in enumerate(methods):
        normalized = normalize(values, method)
        axes[i].hist(normalized, bins=50, alpha=0.7)
        axes[i].set_title(f'Normalization: {method}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=150)
    plt.show()

def activation_dynamics(activations):
    import matplotlib.pyplot as plt
    
    first_groups = group_by_layer(activations[0])
    sample_layers = list(first_groups.keys())[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, layer_id in enumerate(sample_layers):
        layer_means = []
        layer_stds = []
        
        for step_acts in activations:
            groups = group_by_layer(step_acts)
            if layer_id in groups:
                values = combine_layer_activations(groups[layer_id])
                layer_means.append(values.mean())
                layer_stds.append(values.std())
        
        ax = axes[i]
        steps = range(len(layer_means))
        ax.errorbar(steps, layer_means, yerr=layer_stds, alpha=0.7, marker='o')
        
        layer_name = f"Layer_{layer_id}" if isinstance(layer_id, int) else str(layer_id)
        ax.set_title(layer_name)
        ax.set_xlabel('Generation Step')
        ax.set_ylabel('Mean ± Std')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('activation_dynamics.png', dpi=150)
    plt.show()

def text_analysis(step_texts):
    if not step_texts:
        print("No text data available")
        return
    
    print("Text generation analysis:")
    print(f"Total steps: {len(step_texts)}")
    
    token_counts = [len(text.split()) for text in step_texts]
    print(f"Token growth: {token_counts[0]} -> {token_counts[-1]} words")
    
    for i in range(1, min(len(step_texts), 6)):
        new_text = step_texts[i][len(step_texts[i-1]):] if step_texts[i].startswith(step_texts[i-1]) else step_texts[i]
        print(f"Step {i}: +'{new_text.strip()}'")

if __name__ == "__main__":
    activations, step_texts = load_activations("data/activations.pkl")
    
    stats(activations, step_texts)
    text_analysis(step_texts)
    compare_normalization_effects(activations)
    activation_dynamics(activations)