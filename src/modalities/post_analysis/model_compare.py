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

@dataclass
class ComparisonConfig:
    """Configuration for model comparison"""
    base_model_path: str
    finetuned_model_path: str
    output_dir: str = "./comparison_results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    max_samples: Optional[int] = None  # Limit samples for testing
    max_length: int = 512
    compare_weights: bool = True
    compare_activations: bool = True
    activation_layers: List[str] = None  # None means all layers
    
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
        
        # Load tokenizer (assuming same for both)
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
            "summary_stats": {},
            "skipped_layers": []
        }
        
        base_state = self.base_model.state_dict()
        ft_state = self.finetuned_model.state_dict()
        
        total_params = 0
        total_diff = 0
        skipped_count = 0
        
        for name in tqdm(base_state.keys(), desc="Comparing parameters"):
            if name not in ft_state:
                print(f"Warning: {name} not in finetuned model - skipping")
                results["skipped_layers"].append({"name": name, "reason": "not_in_finetuned"})
                skipped_count += 1
                continue
            
            base_param = base_state[name].float()
            ft_param = ft_state[name].float()

            # Better method, base is [250880, 4096] and ft is [250882, 4096] so we just pad the base
            if base_param.numel() != ft_param.numel():
                print(f"Warning: Parameter size mismatch for {name}: "
                      f"base={base_param.shape}, ft={ft_param.shape} - attempting to pad")
                ft_param = ft_param[0:base_param.shape[0], ...] # hack due to tokenizer adding extra tokens
                print(f"After hack: base={base_param.shape}, ft={ft_param.shape}")

            # Calculate metrics
            diff = ft_param - base_param
            l2_distance = torch.norm(diff).item()
            relative_change = l2_distance / (torch.norm(base_param).item() + 1e-8)
            
            # Cosine similarity
            base_flat = base_param.flatten()
            ft_flat = ft_param.flatten()
            cosine_sim = torch.nn.functional.cosine_similarity(
                base_flat.unsqueeze(0), 
                ft_flat.unsqueeze(0)
            ).item()
            
            # Statistics
            param_stats = {
                "shape": list(base_param.shape),
                "num_params": base_param.numel(),
                "l2_distance": l2_distance,
                "relative_change": relative_change,
                "cosine_similarity": cosine_sim,
                "base_mean": base_param.mean().item(),
                "base_std": base_param.std().item(),
                "ft_mean": ft_param.mean().item(),
                "ft_std": ft_param.std().item(),
                "max_abs_diff": diff.abs().max().item(),
            }
            
            results["layer_comparisons"][name] = param_stats
            total_params += base_param.numel()
            total_diff += l2_distance
        
        # Summary statistics
        results["summary_stats"] = {
            "total_parameters": total_params,
            "average_l2_distance": total_diff / max(len(results["layer_comparisons"]), 1),
            "num_layers_compared": len(results["layer_comparisons"]),
            "num_layers_skipped": skipped_count,
            "total_layers": len(base_state)
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
            # Store on CPU to save GPU memory
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
        
        # Register hooks for both models
        base_hooks = []
        ft_hooks = []
        
        # Identify layers to monitor
        layers_to_monitor = []
        for name, module in self.base_model.named_modules():
            if 'layers' in name and name.endswith(('self_attn', 'mlp', 'norm', 'layernorm')):
                layers_to_monitor.append(name)
        
        print(f"Monitoring {len(layers_to_monitor)} layers")
        
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
            for hook in base_hooks:
                hook.remove()
            for hook in ft_hooks:
                hook.remove()
            base_hooks.clear()
            ft_hooks.clear()
            
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
                
                # Check shape compatibility
                if base_act.shape != ft_act.shape:
                    print(f"Warning: Activation shape mismatch for {layer_name}: "
                          f"base={base_act.shape}, ft={ft_act.shape} - skipping")
                    continue
                
                # Calculate differences
                diff = ft_act - base_act
                layer_stats = {
                    "l2_distance": torch.norm(diff).item(),
                    "cosine_similarity": torch.nn.functional.cosine_similarity(
                        base_act.flatten().unsqueeze(0),
                        ft_act.flatten().unsqueeze(0)
                    ).item(),
                    "base_mean": base_act.mean().item(),
                    "base_std": base_act.std().item(),
                    "ft_mean": ft_act.mean().item(),
                    "ft_std": ft_act.std().item(),
                    "max_abs_diff": diff.abs().max().item(),
                    "mean_abs_diff": diff.abs().mean().item(),
                }
                sample_comparison["layer_comparisons"][layer_name] = layer_stats
                
                # Update layer statistics
                if layer_name not in results["layer_statistics"]:
                    results["layer_statistics"][layer_name] = {
                        "l2_distances": [],
                        "cosine_similarities": [],
                        "mean_abs_diffs": []
                    }
                results["layer_statistics"][layer_name]["l2_distances"].append(
                    layer_stats["l2_distance"]
                )
                results["layer_statistics"][layer_name]["cosine_similarities"].append(
                    layer_stats["cosine_similarity"]
                )
                results["layer_statistics"][layer_name]["mean_abs_diffs"].append(
                    layer_stats["mean_abs_diff"]
                )
            
            results["samples"].append(sample_comparison)
            
            # Save intermediate results every 10 samples
            if (sample_idx + 1) % 10 == 0:
                self._save_activation_results(results, intermediate=True)
            
            # Clear memory
            del base_activations, ft_activations
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate aggregate statistics
        for layer_name, stats in results["layer_statistics"].items():
            stats["mean_l2_distance"] = np.mean(stats["l2_distances"])
            stats["std_l2_distance"] = np.std(stats["l2_distances"])
            stats["mean_cosine_similarity"] = np.mean(stats["cosine_similarities"])
            stats["std_cosine_similarity"] = np.std(stats["cosine_similarities"])
        
        self._save_activation_results(results, intermediate=False)
        return results
    
    def _save_activation_results(self, results: Dict, intermediate: bool = False):
        """Save activation comparison results"""
        suffix = "_intermediate" if intermediate else ""
        
        # Save detailed results
        output_file = self.output_dir / f"activation_comparison{suffix}.json"
        
        # Create a serializable version (exclude raw activation tensors)
        save_results = {
            "layer_statistics": results["layer_statistics"],
            "num_samples": len(results["samples"]),
            "sample_summaries": [
                {
                    "sample_idx": s["sample_idx"],
                    "question_preview": s["question"][:100],
                    "num_layers_compared": len(s["layer_comparisons"])
                }
                for s in results["samples"]
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # Save per-sample details separately
        samples_file = self.output_dir / f"activation_samples{suffix}.json"
        with open(samples_file, 'w') as f:
            json.dump(results["samples"], f, indent=2)
        
        if not intermediate:
            print(f"Activation comparison saved to {output_file}")
            print(f"Sample details saved to {samples_file}")
    
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