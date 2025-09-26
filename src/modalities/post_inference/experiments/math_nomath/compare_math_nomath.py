import numpy as np
import h5py
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict
import pickle

MATH_PROMPTS = [
    "Calculate the integral of x^2 from 0 to 1:",
    "Solve for x: 2x + 5 = 13. The answer is",
    "The derivative of sin(x) is",
    "If a triangle has sides 3, 4, and 5, its area is",
    "The quadratic formula is x equals",
    "The limit of (x^2 - 1)/(x - 1) as x approaches 1 is",
    "The eigenvalues of a 2x2 identity matrix are",
    "The probability of rolling a 6 on a fair die is",
    "The sum of angles in a triangle equals",
    "The factorial of 5 equals",
    "The logarithm base 10 of 1000 is",
    "The square root of 144 is",
    "The greatest common divisor of 24 and 36 is",
    "The circumference of a circle with radius r is",
    "The Pythagorean theorem states that",
]

NONMATH_PROMPTS = [
    "The capital of France is",
    "Shakespeare wrote his plays during the",
    "The color of the sky on a clear day is",
    "A typical greeting in English is",
    "The largest ocean on Earth is the",
    "Dogs are known for being loyal and",
    "The season after summer is",
    "Water freezes at a temperature of",
    "The opposite of hot is",
    "A common breakfast food is",
    "The sun rises in the",
    "Trees produce oxygen through",
    "A bicycle has how many wheels?",
    "The primary colors are red, blue, and",
    "Honey is made by",
]

class SimpleComparison:
    def __init__(self, model_path="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.model_path = model_path
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def capture_activations(self, prompts):
        all_activations = []
        
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            activations = OrderedDict()
            hooks = []
            
            def make_hook(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        activations[name] = output.detach().cpu().numpy()
                    elif isinstance(output, (tuple, list)) and len(output) > 0:
                        if isinstance(output[0], torch.Tensor):
                            activations[name] = output[0].detach().cpu().numpy()
                return hook
            
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:
                    hooks.append(module.register_forward_hook(make_hook(name)))
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            for hook in hooks:
                hook.remove()
            
            all_activations.append(activations)
        
        return all_activations
    
    def compute_average_activations(self, activation_list):
        avg_activations = OrderedDict()
        
        all_keys = set()
        for acts in activation_list:
            all_keys.update(acts.keys())
        
        for key in sorted(all_keys):
            values = []
            for acts in activation_list:
                if key in acts:
                    act = acts[key]
                    if len(act.shape) == 3:
                        act = act[:, -1, :]
                    elif len(act.shape) == 2:
                        act = act[-1, :]
                    values.append(act.flatten())
            
            if values:
                min_len = min(len(v) for v in values)
                values = [v[:min_len] for v in values]
                avg_activations[key] = np.mean(values, axis=0)
        
        return avg_activations
    
    def group_by_layer(self, activations):
        layer_groups = OrderedDict()
        
        for name, activation in activations.items():
            if '.layers.' in name or '.h.' in name:
                parts = name.split('.')
                layer_idx = None
                
                for i, part in enumerate(parts):
                    if part in ['layers', 'h'] and i+1 < len(parts):
                        try:
                            layer_idx = int(parts[i+1])
                            break
                        except ValueError:
                            continue
                
                if layer_idx is not None:
                    layer_key = f"layer_{layer_idx:02d}"
                    if layer_key not in layer_groups:
                        layer_groups[layer_key] = OrderedDict()
                    layer_groups[layer_key][name] = activation
                else:
                    if 'other' not in layer_groups:
                        layer_groups['other'] = OrderedDict()
                    layer_groups['other'][name] = activation
            else:
                key = 'embeddings' if 'embed' in name.lower() else 'other'
                if key not in layer_groups:
                    layer_groups[key] = OrderedDict()
                layer_groups[key][name] = activation
        
        return layer_groups
    
    def normalize_and_reshape(self, values, target_size=256*256):
        p1, p99 = np.percentile(values, [1, 99])
        clipped = np.clip(values, p1, p99)
        normalized = (clipped - p1) / (p99 - p1 + 1e-8)
        
        if len(values) < target_size:
            padded = np.zeros(target_size)
            padded[:len(values)] = normalized
            normalized = padded
        else:
            normalized = normalized[:target_size]
        
        return normalized.reshape(256, 256)
    
    def apply_colormap(self, image):
        cmap = plt.get_cmap('viridis')
        colored = cmap(image)
        return (colored[:, :, :3] * 255).astype(np.uint8)
    
    def create_comparison_grid(self, math_groups, nonmath_groups):
        all_keys = sorted(set(list(math_groups.keys()) + list(nonmath_groups.keys())))
        n_layers = len(all_keys)
        
        grid_cols = 2
        grid_rows = n_layers
        
        height = grid_rows * 256
        width = grid_cols * 256 + 100
        
        grid = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        for idx, layer_key in enumerate(all_keys):
            math_values = np.zeros(256*256)
            if layer_key in math_groups:
                for name, act in math_groups[layer_key].items():
                    flat = act.flatten()
                    math_values[:min(len(flat), len(math_values))] += flat[:min(len(flat), len(math_values))]
                if math_groups[layer_key]:
                    math_values /= len(math_groups[layer_key])
            
            nonmath_values = np.zeros(256*256)
            if layer_key in nonmath_groups:
                for name, act in nonmath_groups[layer_key].items():
                    flat = act.flatten()
                    nonmath_values[:min(len(flat), len(nonmath_values))] += flat[:min(len(flat), len(nonmath_values))]
                if nonmath_groups[layer_key]:
                    nonmath_values /= len(nonmath_groups[layer_key])
            
            math_img = self.normalize_and_reshape(math_values)
            nonmath_img = self.normalize_and_reshape(nonmath_values)
            
            math_colored = self.apply_colormap(math_img)
            nonmath_colored = self.apply_colormap(nonmath_img)
            
            y_start = idx * 256
            y_end = y_start + 256
            
            grid[y_start:y_end, 100:356] = math_colored
            grid[y_start:y_end, 356:612] = nonmath_colored
            
            label_img = Image.fromarray(grid)
            draw = ImageDraw.Draw(label_img)
            draw.text((5, y_start + 120), layer_key, fill=(0, 0, 0), font=font)
            grid = np.array(label_img)
        
        header_height = 50
        final_img = np.ones((height + header_height, width, 3), dtype=np.uint8) * 255
        final_img[header_height:] = grid
        
        final_pil = Image.fromarray(final_img)
        draw = ImageDraw.Draw(final_pil)
        
        try:
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            header_font = font
        
        draw.text((180, 15), "MATH", fill=(0, 0, 0), font=header_font)
        draw.text((430, 15), "NON-MATH", fill=(0, 0, 0), font=header_font)
        
        return np.array(final_pil)
    
    def save_to_h5(self, math_activations, nonmath_activations, output_path):
        h5_path = output_path / "activations.h5"
        
        with h5py.File(h5_path, 'w') as f:
            f.attrs['model'] = self.model_path
            f.attrs['n_math_samples'] = len(math_activations)
            f.attrs['n_nonmath_samples'] = len(nonmath_activations)
            
            math_avg = self.compute_average_activations(math_activations)
            nonmath_avg = self.compute_average_activations(nonmath_activations)
            
            math_groups = self.group_by_layer(math_avg)
            nonmath_groups = self.group_by_layer(nonmath_avg)
            
            math_grp = f.create_group('math')
            math_grp.attrs['n_samples'] = len(math_activations)
            
            for layer_key in sorted(math_groups.keys()):
                layer_grp = math_grp.create_group(layer_key)
                for component_name, values in math_groups[layer_key].items():
                    clean_name = component_name.replace('.', '_')
                    layer_grp.create_dataset(clean_name, data=values, compression='gzip')
            
            nonmath_grp = f.create_group('nonmath')
            nonmath_grp.attrs['n_samples'] = len(nonmath_activations)
            
            for layer_key in sorted(nonmath_groups.keys()):
                layer_grp = nonmath_grp.create_group(layer_key)
                for component_name, values in nonmath_groups[layer_key].items():
                    clean_name = component_name.replace('.', '_')
                    layer_grp.create_dataset(clean_name, data=values, compression='gzip')
            
            raw_math = f.create_group('raw_math')
            for i, sample_acts in enumerate(math_activations):
                sample_grp = raw_math.create_group(f'sample_{i:03d}')
                sample_grp.attrs['prompt'] = MATH_PROMPTS[i] if i < len(MATH_PROMPTS) else ""
                
                sample_groups = self.group_by_layer(sample_acts)
                for layer_key in sorted(sample_groups.keys()):
                    layer_grp = sample_grp.create_group(layer_key)
                    for component_name, values in sample_groups[layer_key].items():
                        clean_name = component_name.replace('.', '_')
                        if len(values.shape) == 3:
                            values = values[:, -1, :]
                        elif len(values.shape) == 2:
                            values = values[-1, :]
                        layer_grp.create_dataset(clean_name, data=values.flatten(), compression='gzip')
            
            raw_nonmath = f.create_group('raw_nonmath')
            for i, sample_acts in enumerate(nonmath_activations):
                sample_grp = raw_nonmath.create_group(f'sample_{i:03d}')
                sample_grp.attrs['prompt'] = NONMATH_PROMPTS[i] if i < len(NONMATH_PROMPTS) else ""
                
                sample_groups = self.group_by_layer(sample_acts)
                for layer_key in sorted(sample_groups.keys()):
                    layer_grp = sample_grp.create_group(layer_key)
                    for component_name, values in sample_groups[layer_key].items():
                        clean_name = component_name.replace('.', '_')
                        if len(values.shape) == 3:
                            values = values[:, -1, :]
                        elif len(values.shape) == 2:
                            values = values[-1, :]
                        layer_grp.create_dataset(clean_name, data=values.flatten(), compression='gzip')
        
        print(f"Saved H5 file: {h5_path}")
        
        with h5py.File(h5_path, 'r') as f:
            print("\nH5 Structure:")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"  {'  ' * name.count('/')}📁 {name.split('/')[-1] if '/' in name else name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"  {'  ' * name.count('/')}📄 {name.split('/')[-1]}: shape={obj.shape}")
            f.visititems(print_structure)
        
        return h5_path
    
    def run(self):
        output_path = Path("/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests/experiments") / "simple_comparison"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Capturing MATH activations...")
        math_activations = self.capture_activations(MATH_PROMPTS)
        
        print("\nCapturing NON-MATH activations...")
        nonmath_activations = self.capture_activations(NONMATH_PROMPTS)
        
        print("\nComputing averages...")
        math_avg = self.compute_average_activations(math_activations)
        nonmath_avg = self.compute_average_activations(nonmath_activations)
        
        print("\nGrouping by layer...")
        math_groups = self.group_by_layer(math_avg)
        nonmath_groups = self.group_by_layer(nonmath_avg)
        
        print(f"\nFound {len(math_groups)} layer groups")
        for key in sorted(math_groups.keys()):
            n_math = len(math_groups.get(key, {}))
            n_nonmath = len(nonmath_groups.get(key, {}))
            print(f"  {key}: {n_math} math components, {n_nonmath} non-math components")
        
        print("\nCreating visualization...")
        grid = self.create_comparison_grid(math_groups, nonmath_groups)
        
        img_path = output_path / "average_comparison.png"
        Image.fromarray(grid).save(img_path)
        print(f"Saved visualization: {img_path}")
        
        print("\nSaving to H5...")
        h5_path = self.save_to_h5(math_activations, nonmath_activations, output_path)
        
        with open(output_path / "prompts.txt", 'w') as f:
            f.write("MATH PROMPTS:\n")
            for i, p in enumerate(MATH_PROMPTS):
                f.write(f"{i+1}. {p}\n")
            f.write("\nNON-MATH PROMPTS:\n")
            for i, p in enumerate(NONMATH_PROMPTS):
                f.write(f"{i+1}. {p}\n")
        
        print(f"\n✓ Complete! All outputs in: {output_path}")
        return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    args = parser.parse_args()
    
    exp = SimpleComparison(model_path=args.model)
    exp.run()