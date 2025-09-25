import numpy as np
from PIL import Image
import pickle
from pathlib import Path
import math

def load_activations(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def group_by_layer(step_activations):
    layer_groups = {}
    
    for name, activation in step_activations.items():
        if '.layers.' in name:
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i+1 < len(parts):
                    try:
                        layer_idx = int(parts[i+1])
                        if layer_idx not in layer_groups:
                            layer_groups[layer_idx] = {}
                        layer_groups[layer_idx][name] = activation
                        break
                    except ValueError:
                        continue
        else:
            key = name.split('.')[0] if '.' in name else name
            if key not in layer_groups:
                layer_groups[key] = {}
            layer_groups[key][name] = activation
    
    return layer_groups

def normalize(values, method='percentile'):
    if method == 'minmax':
        vmin, vmax = values.min(), values.max()
        return (values - vmin) / (vmax - vmin + 1e-8)
    elif method == 'percentile':
        p1, p99 = np.percentile(values, [1, 99])
        clipped = np.clip(values, p1, p99)
        return (clipped - p1) / (p99 - p1 + 1e-8)
    elif method == 'zscore':
        mean, std = values.mean(), values.std()
        normalized = (values - mean) / (std + 1e-8)
        return (np.tanh(normalized) + 1) / 2
    elif method == 'log':
        shifted = values - values.min() + 1
        log_values = np.log(shifted)
        return (log_values - log_values.min()) / (log_values.max() - log_values.min() + 1e-8)
    return values

def combine_layer_activations(layer_dict):
    all_values = []
    for name in sorted(layer_dict.keys()):
        values = layer_dict[name].flatten().numpy()
        all_values.extend(values)
    return np.array(all_values)

def to_256x256(values):
    target_size = 256 * 256
    if len(values) < target_size:
        padded = np.zeros(target_size)
        padded[:len(values)] = values
        values = padded
    else:
        values = values[:target_size]
    return values.reshape(256, 256)

def apply_colormap(image, colormap='viridis'):
    from matplotlib import cm
    import matplotlib.pyplot as plt
    
    cmap = plt.get_cmap(colormap)
    colored = cmap(image)
    return (colored[:, :, :3] * 255).astype(np.uint8)

def create_grid(layer_images, colormap=None):
    n_layers = len(layer_images)
    grid_cols = int(math.ceil(math.sqrt(n_layers)))
    grid_rows = int(math.ceil(n_layers / grid_cols))
    
    height = grid_rows * 256
    width = grid_cols * 256
    
    if colormap:
        grid = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        grid = np.zeros((height, width), dtype=np.uint8)
    
    for idx, (_, img) in enumerate(layer_images):
        row = idx // grid_cols
        col = idx % grid_cols
        
        y_start = row * 256
        y_end = y_start + 256
        x_start = col * 256
        x_end = x_start + 256
        
        if colormap:
            colored = apply_colormap(img, colormap)
            grid[y_start:y_end, x_start:x_end] = colored
        else:
            grid[y_start:y_end, x_start:x_end] = (img * 255).astype(np.uint8)
    
    return grid

def visualize_step(step_activations, normalization='percentile', colormap=None):
    layer_groups = group_by_layer(step_activations)
    
    int_keys = [k for k in layer_groups.keys() if isinstance(k, int)]
    str_keys = [k for k in layer_groups.keys() if isinstance(k, str)]
    all_keys = sorted(int_keys) + sorted(str_keys)
    
    layer_images = []
    for layer_id in all_keys:
        combined = combine_layer_activations(layer_groups[layer_id])
        normalized = normalize(combined, normalization)
        img_array = to_256x256(normalized)
        
        layer_name = f"Layer_{layer_id}" if isinstance(layer_id, int) else str(layer_id)
        layer_images.append((layer_name, img_array))
    
    grid = create_grid(layer_images, colormap)
    
    if colormap:
        return Image.fromarray(grid, mode='RGB')
    else:
        return Image.fromarray(grid, mode='L')

def process_all_steps(activations, output_dir="viz", normalization='percentile', colormap=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    images = []
    for step_idx, step_acts in enumerate(activations):
        img = visualize_step(step_acts, normalization, colormap)
        filepath = f"{output_dir}/step_{step_idx:03d}.png"
        img.save(filepath)
        images.append(img)
        print(f"Saved step {step_idx} -> {filepath}")
    
    return images

if __name__ == "__main__":
    activations = load_activations("data/activations.pkl")
    
    process_all_steps(activations, "viz_gray", normalization='percentile')
    process_all_steps(activations, "viz_viridis", normalization='percentile', colormap='viridis')
    process_all_steps(activations, "viz_inferno", normalization='zscore', colormap='inferno')