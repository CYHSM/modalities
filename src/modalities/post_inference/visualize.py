import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pickle
from pathlib import Path
import math
import textwrap
from animate import create_gif

def load_activations(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, list):
        return data, None
    return data['activations'], data.get('texts', None)

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

def get_default_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Arial.ttf", size)
    except:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except:
            return ImageFont.load_default()

def add_text_overlay(image, step_num, current_text, prev_text=None, colormap=None, layer_stats=None):
    is_color = colormap is not None
    
    if is_color:
        img_array = np.array(image)
        overlay_height = 300
        new_height = img_array.shape[0] + overlay_height
        new_img = np.ones((new_height, img_array.shape[1], 3), dtype=np.uint8) * 40
        new_img[:img_array.shape[0]] = img_array
        result = Image.fromarray(new_img)
    else:
        img_array = np.array(image)
        overlay_height = 300
        new_height = img_array.shape[0] + overlay_height
        new_img = np.ones((new_height, img_array.shape[1]), dtype=np.uint8) * 40
        new_img[:img_array.shape[0]] = img_array
        result = Image.fromarray(new_img, mode='L').convert('RGB')
    
    draw = ImageDraw.Draw(result)
    
    font_large = get_default_font(24)
    font_medium = get_default_font(16)
    font_small = get_default_font(14)
    
    step_text = f"STEP {step_num}"
    bbox = draw.textbbox((0, 0), step_text, font=font_large)
    step_width = bbox[2] - bbox[0]
    
    y_text_start = img_array.shape[0] + 20
    draw.text((20, y_text_start), step_text, fill=(255, 255, 255), font=font_large)
    
    if layer_stats:
        stats_x = 20
        stats_y = y_text_start + 40
        
        draw.text((stats_x, stats_y), f"Global: min={layer_stats['global_min']:.3f}, max={layer_stats['global_max']:.3f}, layers={layer_stats['num_layers']}", 
                 fill=(200, 200, 255), font=font_small)
        stats_y += 18
        
        if 'lm_head' in layer_stats:
            draw.text((stats_x, stats_y), f"LM Head: min={layer_stats['lm_head']['min']:.3f}, max={layer_stats['lm_head']['max']:.3f}, mean={layer_stats['lm_head']['mean']:.3f}", 
                     fill=(255, 200, 200), font=font_small)
            stats_y += 18
            
        if 'embed_tokens' in layer_stats:
            draw.text((stats_x, stats_y), f"Embed: min={layer_stats['embed_tokens']['min']:.3f}, max={layer_stats['embed_tokens']['max']:.3f}, mean={layer_stats['embed_tokens']['mean']:.3f}", 
                     fill=(200, 255, 200), font=font_small)
            stats_y += 18
            
        if 'most_active_layer' in layer_stats:
            draw.text((stats_x, stats_y), f"Most Active: {layer_stats['most_active_layer']} (max={layer_stats['most_active_max']:.3f})", 
                     fill=(255, 255, 150), font=font_small)
            stats_y += 25
        
        text_y = stats_y
    else:
        text_y = y_text_start + 40
    
    if current_text:
        max_width = result.width - 40
        
        if prev_text and current_text.startswith(prev_text):
            old_part = prev_text
            new_part = current_text[len(prev_text):]
            
            lines_old = textwrap.fill(old_part, width=140).split('\n')
            lines_new = textwrap.fill(new_part, width=140).split('\n')
            
            for line in lines_old:
                if text_y + 20 < result.height - 10:
                    draw.text((20, text_y), line, fill=(180, 180, 180), font=font_medium)
                    text_y += 20
            
            for line in lines_new:
                if text_y + 20 < result.height - 10:
                    draw.text((20, text_y), line, fill=(100, 255, 100), font=font_medium)
                    text_y += 20
        else:
            wrapped_lines = textwrap.fill(current_text, width=140).split('\n')
            for line in wrapped_lines:
                if text_y + 20 < result.height - 10:
                    draw.text((20, text_y), line, fill=(255, 255, 255), font=font_medium)
                    text_y += 20
    
    border_color = (80, 80, 80)
    draw.rectangle([(0, img_array.shape[0]), (result.width-1, result.height-1)], 
                  outline=border_color, width=2)
    
    return result

def visualize_step(step_activations, step_num, current_text=None, prev_text=None, 
                  normalization='percentile', colormap=None):
    layer_groups = group_by_layer(step_activations)
    
    int_keys = [k for k in layer_groups.keys() if isinstance(k, int)]
    str_keys = [k for k in layer_groups.keys() if isinstance(k, str)]
    all_keys = sorted(int_keys) + sorted(str_keys)
    
    layer_stats = {
        'num_layers': len(all_keys),
        'global_min': float('inf'),
        'global_max': float('-inf'),
        'most_active_max': float('-inf'),
        'most_active_layer': None
    }
    
    layer_images = []
    for layer_id in all_keys:
        layer_dict = layer_groups[layer_id]
        combined = combine_layer_activations(layer_dict)
        normalized = normalize(combined, normalization)
        
        layer_min, layer_max = combined.min(), combined.max()
        layer_mean = combined.mean()
        
        layer_stats['global_min'] = min(layer_stats['global_min'], float(layer_min))
        layer_stats['global_max'] = max(layer_stats['global_max'], float(layer_max))
        
        if layer_max > layer_stats['most_active_max']:
            layer_stats['most_active_max'] = float(layer_max)
            layer_stats['most_active_layer'] = f"Layer_{layer_id}" if isinstance(layer_id, int) else str(layer_id)
        
        layer_name = f"Layer_{layer_id}" if isinstance(layer_id, int) else str(layer_id)
        if 'lm_head' in str(layer_id).lower():
            layer_stats['lm_head'] = {'min': float(layer_min), 'max': float(layer_max), 'mean': float(layer_mean)}
        elif 'embed' in str(layer_id).lower():
            layer_stats['embed_tokens'] = {'min': float(layer_min), 'max': float(layer_max), 'mean': float(layer_mean)}
        
        img_array = to_256x256(normalized)
        layer_images.append((layer_name, img_array))
    
    grid = create_grid(layer_images, colormap)
    
    if colormap:
        base_image = Image.fromarray(grid, mode='RGB')
    else:
        base_image = Image.fromarray(grid, mode='L')
    
    final_image = add_text_overlay(base_image, step_num, current_text, prev_text, colormap, layer_stats)
    
    return final_image

def process_all_steps(activations, step_texts=None, output_dir="viz", 
                     normalization='percentile', colormap=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    images = []
    for step_idx, step_acts in enumerate(activations):
        current_text = step_texts[step_idx + 1] if step_texts and step_idx + 1 < len(step_texts) else None
        prev_text = step_texts[step_idx] if step_texts and step_idx < len(step_texts) else None
        
        img = visualize_step(step_acts, step_idx + 1, current_text, prev_text, 
                           normalization, colormap)
        filepath = f"{output_dir}/step_{step_idx:03d}.png"
        img.save(filepath)
        images.append(img)
        print(f"Saved step {step_idx + 1} -> {filepath}")
    
    return images

if __name__ == "__main__":
    activations, step_texts = load_activations("/raid/s3/opengptx/mfrey/cp_analysis/inference_qwen/activations.pkl")

    normalization = "nonorm"
    colormap = "inferno"
    output_dir = f"/raid/s3/opengptx/mfrey/cp_analysis/inference_qwen/viz_{colormap}_{normalization}"
    process_all_steps(activations, step_texts, output_dir, normalization=normalization, colormap=colormap)

    gif_path = f"/raid/s3/opengptx/mfrey/cp_analysis/inference_qwen/activation_{colormap}_{normalization}.gif"
    create_gif(output_dir, gif_path, duration=800)