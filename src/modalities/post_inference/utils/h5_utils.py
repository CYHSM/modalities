from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np


class H5Store:
    @staticmethod
    def save_activations(*, activations_dict, texts_dict, prompts_dict, output_path, model_name="model"):
        h5_path = Path(output_path) / "activations.h5"
        
        with h5py.File(h5_path, "w") as f:
            f.attrs["model"] = model_name
            f.attrs["n_conditions"] = len(activations_dict)
            
            for condition, activations in activations_dict.items():
                texts = texts_dict[condition]
                prompts = prompts_dict.get(condition, [])
                
                group = f.create_group(condition)
                group.attrs["n_samples"] = len(activations)
                
                for i, (sample_acts, sample_texts) in enumerate(zip(activations, texts)):
                    sample_grp = group.create_group(f"sample_{i:03d}")
                    sample_grp.attrs["prompt"] = prompts[i] if i < len(prompts) else ""
                    sample_grp.attrs["n_steps"] = len(sample_acts)
                    
                    text_grp = sample_grp.create_group("texts")
                    for step_idx, text in enumerate(sample_texts):
                        text_grp.attrs[f"step_{step_idx}"] = text
                    
                    for step_idx, step_activations in enumerate(sample_acts):
                        step_grp = sample_grp.create_group(f"step_{step_idx}")
                        step_grp.attrs["step_number"] = step_idx
                        
                        for name, values in step_activations.items():
                            clean_name = name.replace(".", "_")
                            data = H5Store._prepare_tensor(values)
                            step_grp.create_dataset(clean_name, data=data, compression="gzip")
        
        return h5_path
    
    @staticmethod
    def load_activations(h5_path):
        activations_dict = {}
        texts_dict = {}
        prompts_dict = {}
        
        with h5py.File(h5_path, "r") as f:
            for condition in f.keys():
                group = f[condition]
                activations_dict[condition] = []
                texts_dict[condition] = []
                prompts_dict[condition] = []
                
                for sample_key in sorted(group.keys()):
                    sample_grp = group[sample_key]
                    prompts_dict[condition].append(sample_grp.attrs.get("prompt", ""))
                    
                    sample_acts = []
                    sample_texts = []
                    
                    text_grp = sample_grp.get("texts", {})
                    for step_idx in range(sample_grp.attrs["n_steps"]):
                        text_key = f"step_{step_idx}"
                        if text_key in text_grp.attrs:
                            sample_texts.append(text_grp.attrs[text_key])
                    
                    for step_idx in range(sample_grp.attrs["n_steps"]):
                        step_grp = sample_grp[f"step_{step_idx}"]
                        step_acts = OrderedDict()
                        
                        for layer_name in step_grp.keys():
                            original_name = layer_name.replace("_", ".")
                            step_acts[original_name] = step_grp[layer_name][:]
                        
                        sample_acts.append(step_acts)
                    
                    activations_dict[condition].append(sample_acts)
                    texts_dict[condition].append(sample_texts)
        
        return activations_dict, texts_dict, prompts_dict
    

    @staticmethod
    def _prepare_tensor(values):
        if hasattr(values, "shape"):
            if len(values.shape) == 3:
                values = values[:, -1, :]
            elif len(values.shape) == 2:
                values = values[-1, :]
            return values.flatten()
        return np.array(values).flatten()