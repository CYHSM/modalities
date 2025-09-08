"""
Model Comparison Library
Compare weights and activations between base and fine-tuned models
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import gc

from utils import calculate_distribution_stats, calculate_cosine_similarity

@dataclass
class ComparisonConfig:
    """Configuration for model comparison"""
    base_model_path: str
    finetuned_model_path: str
    output_dir: str = "./comparison_results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    max_samples: Optional[int] = None
    max_length: int = 512
    compare_weights: bool = True
    compare_activations: bool = True
    
class ModelComparator:
    """Main class for comparing two models"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
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
    
    def compare_weights(self) -> Dict[str, Any]:
        """Compare weights between base and finetuned models"""
        print("Comparing model weights...")
        results = {
            "layer_comparisons": {},
            "summary_stats": {}
        }
        
        base_state = self.base_model.state_dict()
        ft_state = self.finetuned_model.state_dict()
        
        total_params = 0
        
        for name in tqdm(base_state.keys(), desc="Comparing parameters"):
            if name not in ft_state:
                print(f"Warning: {name} not in finetuned model - skipping")
                continue
            
            base_param = base_state[name].float()
            ft_param = ft_state[name].float()

            # Handle size mismatch (tokenizer expansion)
            if base_param.numel() != ft_param.numel():
                print(f"Warning: Parameter size mismatch for {name}: "
                      f"base={base_param.shape}, ft={ft_param.shape} - trimming ft")
                ft_param = ft_param[0:base_param.shape[0], ...]
            
            # Calculate differences and distribution stats
            diff = ft_param - base_param
            dist_stats = calculate_distribution_stats(diff)
            
            # Cosine similarity
            cosine_sim = calculate_cosine_similarity(base_param, ft_param)
            
            # Store comprehensive statistics
            param_stats = {
                "shape": list(base_param.shape),
                "num_params": base_param.numel(),
                "cosine_similarity": cosine_sim,
                **dist_stats  # Unpack all distribution statistics
            }
            
            results["layer_comparisons"][name] = param_stats
            total_params += base_param.numel()
        
        # Summary statistics
        num_layers = len(results["layer_comparisons"])
        results["summary_stats"] = {
            "total_parameters": total_params,
            "num_layers_compared": num_layers,
        }
        
        # Save results
        output_file = self.output_dir / "weight_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Weight comparison saved to {output_file}")
        
        return results
    
    def _get_activation_hook(self, activations_dict: Dict, name: str):
        """Create a hook to capture activations"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations_dict[name] = output.detach().cpu()
        return hook
    
    def compare_activations_on_gsm8k(self) -> Dict[str, Any]:
        """Compare model activations on GSM8K dataset"""
        print("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main", split="test")
        
        if self.config.max_samples:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
        
        results = {
            "samples": [],
            "layer_statistics": {}
        }
        
        # Identify layers to monitor
        layers_to_monitor = []
        for name, module in self.base_model.named_modules():
            print(f"Checking layer: {name}")
            if 'layers' in name and name.endswith(('self_attn', 'mlp', 'norm', 'layernorm', 'lm_head')):
                print(f"Monitoring layer: {name}")
                layers_to_monitor.append(name)
        
        print(f"Monitoring {len(layers_to_monitor)} of {len(list(self.base_model.named_modules()))} total layers")
        
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
            
            # Add hooks
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
                    print(f"Warning: Activation shape mismatch for {layer_name} - skipping")
                    continue
                
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
    
    def run_full_comparison(self):
        """Run complete comparison pipeline"""
        results = {}
        
        if self.config.compare_weights:
            results["weights"] = self.compare_weights()
        
        if self.config.compare_activations:
            results["activations"] = self.compare_activations_on_gsm8k()
        
        # Save summary
        summary_file = self.output_dir / "comparison_summary.json"
        summary = {
            "config": asdict(self.config),
            "weights_compared": self.config.compare_weights,
            "activations_compared": self.config.compare_activations,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Comparison complete! Results saved to {self.output_dir}")
        return results


def compare_models(
    base_model_path: str,
    finetuned_model_path: str,
    output_dir: str = "./comparison_results",
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to compare two models"""
    config = ComparisonConfig(
        base_model_path=base_model_path,
        finetuned_model_path=finetuned_model_path,
        output_dir=output_dir,
        **kwargs
    )
    
    comparator = ModelComparator(config)
    return comparator.run_full_comparison()