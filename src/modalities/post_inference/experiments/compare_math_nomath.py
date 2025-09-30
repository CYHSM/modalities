from collections import OrderedDict
from pathlib import Path

import numpy as np
from datasets import load_dataset

from modalities.post_inference.core.capture import ActivationCapture
from modalities.post_inference.core.stats import ActivationStats
from modalities.post_inference.plot.visualize import visualize_step
from modalities.post_inference.plot.plot_stats import create_interactive_viewer
from modalities.post_inference.utils.h5_utils import H5Store


class MathComparison:
    def __init__(self, *, model_path="gpt2", device="cuda"):
        self.capture = ActivationCapture(model_path, device)
        self.model_path = model_path

    def load_math_prompts(self, *, n_samples=100):
        dataset = load_dataset("gsm8k", "main", split="test")
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

    def compute_statistics(self, *, math_activations, nonmath_activations, test="welch", correction_method="fdr_bh"):
        math_by_layer = self.flatten_activations_by_layer(activation_list=math_activations)
        nonmath_by_layer = self.flatten_activations_by_layer(activation_list=nonmath_activations)
        
        common_layers = set(math_by_layer.keys()) & set(nonmath_by_layer.keys())
        
        stats_results = OrderedDict()
        for metric in ["mean_diff", "t_stats", "p_values", "p_corrected", "significant", "cohens_d"]:
            stats_results[metric] = OrderedDict()
        
        print(f"Computing statistics for {len(common_layers)} layers (test={test}, correction={correction_method})...")
        
        for layer in sorted(common_layers):
            contrast = ActivationStats.compute_contrast(
                math_by_layer[layer],
                nonmath_by_layer[layer],
                test=test
            )
            
            correction = ActivationStats.multiple_comparison_correction(
                contrast["p_values"],
                method=correction_method,
                alpha=0.05
            )
            
            stats_results["mean_diff"][layer] = contrast["mean_diff"]
            stats_results["t_stats"][layer] = contrast["t_stats"]
            stats_results["p_values"][layer] = contrast["p_values"]
            stats_results["p_corrected"][layer] = correction["p_corrected"]
            stats_results["significant"][layer] = correction["significant"]
            stats_results["cohens_d"][layer] = contrast["cohens_d"]
            
            n_sig = correction["n_significant"]
            n_total = correction["n_total"]
            print(f"  {layer}: {n_sig}/{n_total} significant after correction ({100*n_sig/n_total:.2f}%)")
        
        return stats_results

    def compute_average_activations_all_steps(self, *, activation_list):
        avg_activations = OrderedDict()
        
        all_keys = set()
        for sample_acts in activation_list:
            for step_acts in sample_acts:
                all_keys.update(step_acts.keys())
        
        for key in sorted(all_keys):
            all_values = []
            for sample_acts in activation_list:
                for step_acts in sample_acts:
                    if key in step_acts:
                        act = step_acts[key]
                        if hasattr(act, "shape"):
                            if len(act.shape) == 3:
                                act = act[:, -1, :]
                            elif len(act.shape) == 2:
                                act = act[-1, :]
                            all_values.append(act.flatten())
                        else:
                            all_values.append(np.array(act).flatten())
            
            if all_values:
                min_len = min(len(v) for v in all_values)
                all_values = [v[:min_len] for v in all_values]
                avg_activations[key] = np.stack(all_values).mean(axis=0)
        
        return avg_activations

    def compute_difference_activations(self, *, math_avg, nonmath_avg):
        diff_activations = OrderedDict()
        
        common_keys = set(math_avg.keys()) & set(nonmath_avg.keys())
        
        for key in sorted(common_keys):
            math_vals = math_avg[key]
            nonmath_vals = nonmath_avg[key]
            
            min_len = min(len(math_vals), len(nonmath_vals))
            diff_activations[key] = math_vals[:min_len] - nonmath_vals[:min_len]
        
        return diff_activations

    def run(self, *, n_samples=100, max_new_tokens=1, test="welch", correction_method="fdr_bh"):
        output_path = Path("/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests/experiments") / f"{self.model_path.replace('/', '_')}_math_vs_nonmath"
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

        print("\nSaving to H5...")
        h5_path = H5Store.save_activations(
            activations_dict={"math": math_activations, "nonmath": nonmath_activations},
            texts_dict={"math": math_texts, "nonmath": nonmath_texts},
            prompts_dict={"math": math_prompts, "nonmath": nonmath_prompts},
            output_path=output_path,
            model_name=self.model_path
        )

        print("\nComputing statistics with multiple comparison correction...")
        stats_results = self.compute_statistics(
            math_activations=math_activations,
            nonmath_activations=nonmath_activations,
            test=test,
            correction_method=correction_method
        )

        print("\nComputing averages across all steps...")
        math_avg_all_steps = self.compute_average_activations_all_steps(activation_list=math_activations)
        nonmath_avg_all_steps = self.compute_average_activations_all_steps(activation_list=nonmath_activations)
        
        print("Computing differences (math - nonmath)...")
        diff_activations = self.compute_difference_activations(
            math_avg=math_avg_all_steps, 
            nonmath_avg=nonmath_avg_all_steps
        )

        print("Creating visualizations...")
        
        math_img = visualize_step(math_avg_all_steps, 1, "MATH PROMPTS (averaged across all steps)")
        nonmath_img = visualize_step(nonmath_avg_all_steps, 2, "NON-MATH PROMPTS (averaged across all steps)")
        diff_img = visualize_step(diff_activations, 3, "DIFFERENCE (Math - Non-Math)")

        math_img.save(output_path / "math_activations_avg_all_steps.png")
        nonmath_img.save(output_path / "nonmath_activations_avg_all_steps.png")
        diff_img.save(output_path / "difference_math_minus_nonmath.png")

        print("\nCreating interactive viewer with corrected p-values...")
        stats_results_corrected = {
            "mean_diff": stats_results["mean_diff"],
            "t_stats": stats_results["t_stats"],
            "p_values": stats_results["p_corrected"],
            "cohens_d": stats_results["cohens_d"]
        }
        
        html_path = create_interactive_viewer(stats_results=stats_results_corrected, output_path=output_path)

        n_significant_corr = sum(np.sum(stats_results["significant"][layer]) 
                                for layer in stats_results["significant"])
        total_dims = sum(len(p.flatten()) for p in stats_results["p_values"].values())

        print(f"\n✓ Complete! All outputs in: {output_path}")
        print(f"H5 file: {h5_path}")
        print(f"Interactive HTML: {html_path}")
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

    exp = MathComparison(model_path=args.model)
    exp.run(
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens, 
        test=args.test, 
        correction_method=args.correction
    )