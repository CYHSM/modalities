from collections import OrderedDict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from modalities.post_inference.core.capture import ActivationCapture
from modalities.post_inference.core.stats import ActivationStats
from modalities.post_inference.utils.h5_utils import H5Store


class MathCaptureExperiment:
    def __init__(self, *, model_path="gpt2", device="cuda"):
        self.capture = ActivationCapture(model_path, device)
        self.model_path = model_path

    def load_math_prompts(self, *, n_samples=100):
        dataset = load_dataset("gsm8k", "main", split="train")
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
        prompts = [f"Question: {ex['question']}\nAnswer:" for ex in dataset]
        return prompts

    def load_nonmath_prompts(self, *, n_samples=100):
        dataset = load_dataset("cais/mmlu", "all", split="test")
        
        nonmath_subjects = [
            "high_school_geography", "high_school_government_and_politics", 
            "high_school_european_history", "high_school_us_history",
            "high_school_world_history", "prehistory", "world_religions",
            "professional_psychology", "human_sexuality", "moral_scenarios",
            "sociology", "philosophy", "professional_law"
        ]
        
        filtered = dataset.filter(lambda x: x["subject"] in nonmath_subjects)
        filtered = filtered.shuffle(seed=42).select(range(min(n_samples, len(filtered))))
        
        prompts = []
        for ex in filtered:
            choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(ex["choices"])])
            prompt = f"{ex['question']}\n{choices}\nAnswer:"
            prompts.append(prompt)
        
        return prompts

    def flatten_activations_by_layer(self, *, activation_list):
        layer_activations = OrderedDict()
        
        for sample_acts in activation_list:
            for step_acts in sample_acts:
                for name, act in step_acts.items():
                    if name not in layer_activations:
                        layer_activations[name] = []
                    
                    if hasattr(act, "shape"):
                        if len(act.shape) == 3:
                            act = act[:, -1, :]
                        elif len(act.shape) == 2:
                            act = act[-1, :]
                        layer_activations[name].append(act.flatten())
                    else:
                        layer_activations[name].append(np.array(act).flatten())
        
        for name in layer_activations:
            min_len = min(len(a) for a in layer_activations[name])
            layer_activations[name] = [a[:min_len] for a in layer_activations[name]]
        
        return layer_activations

    def compute_statistics_with_summary(self, *, math_activations, nonmath_activations, test="welch", correction_method="fdr_bh"):
        math_by_layer = self.flatten_activations_by_layer(activation_list=math_activations)
        nonmath_by_layer = self.flatten_activations_by_layer(activation_list=nonmath_activations)
        
        common_layers = set(math_by_layer.keys()) & set(nonmath_by_layer.keys())
        
        layer_statistics = {}
        
        print(f"Computing statistics for {len(common_layers)} layers (test={test}, correction={correction_method})...")
        
        for layer in tqdm(sorted(common_layers), desc="Computing stats"):
            math_vals = math_by_layer[layer]
            nonmath_vals = nonmath_by_layer[layer]
            
            contrast = ActivationStats.compute_contrast(math_vals, nonmath_vals, test=test)
            correction = ActivationStats.multiple_comparison_correction(
                contrast["p_values"], method=correction_method, alpha=0.05
            )
            
            n1, n2 = contrast["n1"], contrast["n2"]
            std1, std2 = contrast["std1"], contrast["std2"]
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            math_array = np.array(math_vals)
            nonmath_array = np.array(nonmath_vals)
            
            layer_statistics[layer] = {
                "t_stats": contrast["t_stats"],
                "p_values": contrast["p_values"],
                "p_corrected": correction["p_corrected"],
                "significant": correction["significant"],
                "mean_diff": contrast["mean_diff"],
                "cohens_d": contrast["cohens_d"],
                "pooled_std": pooled_std,
                "math_mean": np.mean(math_array, axis=0),
                "math_std": np.std(math_array, axis=0, ddof=1),
                "math_median": np.median(math_array, axis=0),
                "math_min": np.min(math_array, axis=0),
                "math_max": np.max(math_array, axis=0),
                "nonmath_mean": np.mean(nonmath_array, axis=0),
                "nonmath_std": np.std(nonmath_array, axis=0, ddof=1),
                "nonmath_median": np.median(nonmath_array, axis=0),
                "nonmath_min": np.min(nonmath_array, axis=0),
                "nonmath_max": np.max(nonmath_array, axis=0),
                "n_math": len(math_vals),
                "n_nonmath": len(nonmath_vals)
            }
            
            n_sig = correction["n_significant"]
            n_total = correction["n_total"]
            print(f"  {layer}: {n_sig}/{n_total} significant after correction ({100*n_sig/n_total:.2f}%)")
        
        return layer_statistics

    def run(self, *, n_samples=100, max_new_tokens=1, test="welch", correction_method="fdr_bh"):
        output_path = Path("/raid/s3/opengptx/mfrey/activation/experiments") / f"{self.model_path.replace('/', '_')}_math_vs_nonmath"
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Loading {n_samples} math prompts from GSM8K...")
        math_prompts = self.load_math_prompts(n_samples=n_samples)
        
        print(f"Loading {n_samples} non-math prompts from MMLU (humanities/social sciences)...")
        nonmath_prompts = self.load_nonmath_prompts(n_samples=n_samples)

        print(f"\nCapturing MATH activations (max_new_tokens={max_new_tokens})...")
        math_activations, math_texts = self.capture.capture_prompts(math_prompts, max_new_tokens=max_new_tokens)

        print(f"\nCapturing NON-MATH activations (max_new_tokens={max_new_tokens})...")
        nonmath_activations, nonmath_texts = self.capture.capture_prompts(
            nonmath_prompts, max_new_tokens=max_new_tokens
        )

        print("\nComputing statistics with multiple comparison correction...")
        layer_statistics = self.compute_statistics_with_summary(
            math_activations=math_activations,
            nonmath_activations=nonmath_activations,
            test=test,
            correction_method=correction_method
        )

        print("\nSaving to H5 (including statistics)...")
        h5_path, stats_path = H5Store.save_activations(
            activations_dict={"math": math_activations, "nonmath": nonmath_activations},
            texts_dict={"math": math_texts, "nonmath": nonmath_texts},
            prompts_dict={"math": math_prompts, "nonmath": nonmath_prompts},
            output_path=output_path,
            model_name=self.model_path,
            statistics=layer_statistics
        )

        n_significant_corr = sum(np.sum(layer_statistics[layer]["significant"]) 
                                for layer in layer_statistics)
        total_dims = sum(len(layer_statistics[layer]["p_values"].flatten()) 
                        for layer in layer_statistics)

        print(f"\n✓ Data capture complete! All outputs in: {output_path}")
        print(f"Activations: {h5_path}")
        print(f"Statistics: {stats_path}")
        print(f"FDR-corrected significant (p<0.05): {n_significant_corr}/{total_dims} ({100*n_significant_corr/total_dims:.2f}%)")
        
        return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--test", type=str, default="welch", choices=["ttest", "welch", "mannwhitney"])
    parser.add_argument("--correction", type=str, default="fdr_bh", 
                       choices=["fdr_bh", "bonferroni", "fdr_by"])
    args = parser.parse_args()

    exp = MathCaptureExperiment(model_path=args.model)
    exp.run(
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens, 
        test=args.test, 
        correction_method=args.correction
    )