import numpy as np
import h5py
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict

from modalities.post_inference.plot.visualize import visualize_step

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
]

class MathComparison:
    def __init__(self, *, model_path="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.model_path = model_path
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def capture_activations(self, *, prompts):
        all_activations = []
        
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            activations = OrderedDict()
            hooks = []
            
            def make_hook(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        activations[name] = output.detach().cpu()
                    elif isinstance(output, (tuple, list)) and len(output) > 0:
                        if isinstance(output[0], torch.Tensor):
                            activations[name] = output[0].detach().cpu()
                return hook
            
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:
                    hooks.append(module.register_forward_hook(make_hook(name)))
            
            with torch.no_grad():
                self.model(input_ids)
            
            for hook in hooks:
                hook.remove()
            
            all_activations.append(activations)
        
        return all_activations
    
    def compute_average_activations(self, *, activation_list):
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
                avg_activations[key] = torch.stack([torch.tensor(v) for v in values]).mean(dim=0)
        
        return avg_activations
    
    def save_to_h5(self, *, math_activations, nonmath_activations, output_path):
        h5_path = output_path / "activations.h5"
        
        with h5py.File(h5_path, 'w') as f:
            f.attrs['model'] = self.model_path
            f.attrs['n_math_samples'] = len(math_activations)
            f.attrs['n_nonmath_samples'] = len(nonmath_activations)
            
            raw_math = f.create_group('raw_math')
            for i, sample_acts in enumerate(math_activations):
                sample_grp = raw_math.create_group(f'sample_{i:03d}')
                sample_grp.attrs['prompt'] = MATH_PROMPTS[i] if i < len(MATH_PROMPTS) else ""
                
                for name, values in sample_acts.items():
                    clean_name = name.replace('.', '_')
                    if len(values.shape) == 3:
                        values = values[:, -1, :]
                    elif len(values.shape) == 2:
                        values = values[-1, :]
                    sample_grp.create_dataset(clean_name, data=values.flatten().numpy(), compression='gzip')
            
            raw_nonmath = f.create_group('raw_nonmath')
            for i, sample_acts in enumerate(nonmath_activations):
                sample_grp = raw_nonmath.create_group(f'sample_{i:03d}')
                sample_grp.attrs['prompt'] = NONMATH_PROMPTS[i] if i < len(NONMATH_PROMPTS) else ""
                
                for name, values in sample_acts.items():
                    clean_name = name.replace('.', '_')
                    if len(values.shape) == 3:
                        values = values[:, -1, :]
                    elif len(values.shape) == 2:
                        values = values[-1, :]
                    sample_grp.create_dataset(clean_name, data=values.flatten().numpy(), compression='gzip')
        
        return h5_path
    
    def run(self):
        output_path = Path("/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests/experiments") / "math_comparison"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Capturing MATH activations...")
        math_activations = self.capture_activations(prompts=MATH_PROMPTS)
        
        print("\nCapturing NON-MATH activations...")
        nonmath_activations = self.capture_activations(prompts=NONMATH_PROMPTS)
        
        print("\nComputing averages...")
        math_avg = self.compute_average_activations(activation_list=math_activations)
        nonmath_avg = self.compute_average_activations(activation_list=nonmath_activations)
        
        print("\nCreating visualizations...")
        math_img = visualize_step(math_avg, 1, "MATH PROMPTS (averaged)")
        nonmath_img = visualize_step(nonmath_avg, 2, "NON-MATH PROMPTS (averaged)")
        
        math_img.save(output_path / "math_activations.png")
        nonmath_img.save(output_path / "nonmath_activations.png")
        
        print("\nSaving to H5...")
        self.save_to_h5(
            math_activations=math_activations,
            nonmath_activations=nonmath_activations,
            output_path=output_path
        )
        
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
    
    exp = MathComparison(model_path=args.model)
    exp.run()