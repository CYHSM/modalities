"""
Weight comparison between base and fine-tuned models
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from typing import Dict, Any
from utils import calculate_distribution_stats, calculate_cosine_similarity


class WeightComparator:
    """Compare weights between base and fine-tuned models"""
    
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