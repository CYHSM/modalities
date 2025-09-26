import torch
import pickle
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict
import numpy as np

class ActivationCapture:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def capture_single(self, input_text, max_new_tokens=1, layers_to_capture=None):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        
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
            if layers_to_capture is None or any(pattern in name for pattern in layers_to_capture):
                if len(list(module.children())) == 0:
                    hooks.append(module.register_forward_hook(make_hook(name)))
        
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        for hook in hooks:
            hook.remove()
        
        logits = outputs.logits.detach().cpu().numpy()
        
        return {
            'activations': activations,
            'input_ids': input_ids.cpu().numpy(),
            'logits': logits,
            'text': input_text
        }
    
    def capture_batch(self, texts, max_new_tokens=1, layers_to_capture=None):
        results = []
        for i, text in enumerate(texts):
            print(f"  Processing {i+1}/{len(texts)}: {text[:50]}...")
            result = self.capture_single(text, max_new_tokens, layers_to_capture)
            results.append(result)
        return results

class DataStore:
    @staticmethod
    def save_experiment(data, experiment_name, output_dir="/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests"):
        output_path = Path(output_dir) / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / "data.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        metadata_file = output_path / "metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Conditions: {list(data.keys())}\n")
            for condition, samples in data.items():
                f.write(f"  {condition}: {len(samples)} samples\n")
        
        return output_path
    
    @staticmethod
    def load_experiment(experiment_name, output_dir="/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests"):
        filepath = Path(output_dir) / experiment_name / "data.pkl"
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def extract_layer_activations(activation_dict, layer_pattern):
    layer_acts = {}
    for name, act in activation_dict.items():
        if layer_pattern in name:
            layer_acts[name] = act
    return layer_acts

def get_activation_at_position(activation, position=-1):
    if len(activation.shape) == 3:
        return activation[:, position, :]
    elif len(activation.shape) == 2:
        return activation[position, :]
    else:
        return activation.flatten()