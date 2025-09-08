"""
Activation comparison between base and fine-tuned models
"""

import torch
import json
import gc
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, Any
from utils import calculate_distribution_stats, calculate_cosine_similarity


class ActivationComparator:
    """Compare activations between base and fine-tuned models"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        
        print(f"Loading base model from {config.base_model_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=config.device
        )
        
        print(f"Loading finetuned model from {config.finetuned_model_path}")
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            config.finetuned_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=config.device
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model.eval()
        self.finetuned_model.eval()
    
    def _get_activation_hook(self, activations_dict: Dict, name: str):
        """Create a hook to capture activations"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations_dict[name] = output.detach().cpu()
        return hook
    
    def _get_all_layers_to_monitor(self):
        """Get ALL layers to monitor, not just specific ones"""
        layers_to_monitor = []
        
        # Get all named modules
        for name, module in self.base_model.named_modules():
            # Skip container modules that don't have their own parameters
            if len(list(module.children())) == 0:  # Leaf modules only
                # Skip very basic modules
                if not any(skip_type in str(type(module)) for skip_type in 
                          ['Dropout', 'Identity', 'ReLU', 'GELU', 'SiLU']):
                    layers_to_monitor.append(name)
        
        print(f"Monitoring {len(layers_to_monitor)} layers out of {len(list(self.base_model.named_modules()))} total modules")
        return layers_to_monitor
    
    def compare_activations(self) -> Dict[str, Any]:
        """Compare model activations on GSM8K dataset"""
        print("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main", split="test")
        
        if self.config.max_samples:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
        
        results = {
            "samples": [],
            "layer_statistics": {}
        }
        
        # Get ALL layers to monitor
        layers_to_monitor = self._get_all_layers_to_monitor()
        
        for sample_idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            question = sample['question']
            answer = sample['answer']
            
            # Tokenize
            inputs = self.tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)
            
            # Capture activations
            base_activations = {}
            ft_activations = {}
            base_hooks = []
            ft_hooks = []
            
            # Add hooks for ALL layers
            for name, module in self.base_model.named_modules():
                if name in layers_to_monitor:
                    hook = module.register_forward_hook(
                        self._get_activation_hook(base_activations, name)
                    )
                    base_hooks.append(hook)
            
            for name, module in self.finetuned_model.named_modules():
                if name in layers_to_monitor:
                    hook = module.register_forward_hook(
                        self._get_activation_hook(ft_activations, name)
                    )
                    ft_hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                base_outputs = self.base_model(**inputs)
                ft_outputs = self.finetuned_model(**inputs)
            
            # Remove hooks
            for hook in base_hooks + ft_hooks:
                hook.remove()
            
            # Compare activations
            sample_comparison = {
                "sample_idx": sample_idx,
                "question": question,
                "answer": answer,
                "layer_comparisons": {}
            }
            
            for layer_name in base_activations.keys():
                if layer_name not in ft_activations:
                    continue
                
                base_act = base_activations[layer_name]
                ft_act = ft_activations[layer_name]
                
                if base_act.shape != ft_act.shape:
                    print(f"Warning: Activation shape mismatch for {layer_name} - trimming")
                    ft_act = ft_act[0:base_act.shape[0], ...]
                
                # Calculate differences and distribution stats
                diff = ft_act - base_act
                dist_stats = calculate_distribution_stats(diff)
                
                # Cosine similarity
                cosine_sim = calculate_cosine_similarity(base_act, ft_act)
                
                layer_stats = {
                    "cosine_similarity": cosine_sim,
                    **dist_stats  # Include all distribution statistics
                }
                
                sample_comparison["layer_comparisons"][layer_name] = layer_stats
                
                # Update layer statistics
                if layer_name not in results["layer_statistics"]:
                    results["layer_statistics"][layer_name] = {
                        stat_name: [] for stat_name in dist_stats.keys()
                    }
                    results["layer_statistics"][layer_name]["cosine_similarities"] = []
                
                # Collect all statistics for aggregation
                for stat_name, value in dist_stats.items():
                    results["layer_statistics"][layer_name][stat_name].append(value)
                results["layer_statistics"][layer_name]["cosine_similarities"].append(cosine_sim)
            
            results["samples"].append(sample_comparison)
            
            # Clear memory
            del base_activations, ft_activations
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate aggregate statistics
        for layer_name, stats in results["layer_statistics"].items():
            for stat_name in list(stats.keys()):
                if stat_name == "cosine_similarities":
                    stats["mean_cosine_similarity"] = np.mean(stats[stat_name])
                    stats["std_cosine_similarity"] = np.std(stats[stat_name])
                else:
                    stats[f"mean_{stat_name}"] = np.mean(stats[stat_name])
                    stats[f"std_{stat_name}"] = np.std(stats[stat_name])
        
        # Save results
        output_file = self.output_dir / "activation_comparison.json"
        save_results = {
            "layer_statistics": results["layer_statistics"],
            "num_samples": len(results["samples"])
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        samples_file = self.output_dir / "activation_samples.json"
        with open(samples_file, 'w') as f:
            json.dump(results["samples"], f, indent=2)
        
        print(f"Activation comparison saved to {output_file}")
        print(f"Sample details saved to {samples_file}")
        
        return results