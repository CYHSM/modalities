from collections import OrderedDict
from pathlib import Path

import numpy as np

from modalities.post_inference.core.capture import ActivationCapture
from modalities.post_inference.core.stats import ActivationStats
from modalities.post_inference.plot.visualize import visualize_step
from modalities.post_inference.plot.plot_stats import create_interactive_viewer
from modalities.post_inference.utils.h5_utils import H5Store

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
    def __init__(self, *, model_path="gpt2", device="cuda"):
        self.capture = ActivationCapture(model_path, device)
        self.model_path = model_path

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

    def run(self, *, max_new_tokens=1, test="welch", correction_method="fdr_bh"):
        output_path = Path("/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests/experiments") / "math_comparison"
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Capturing MATH activations (max_new_tokens={max_new_tokens})...")
        math_activations, math_texts = self.capture.capture_prompts(MATH_PROMPTS, max_new_tokens=max_new_tokens)

        print(f"\nCapturing NON-MATH activations (max_new_tokens={max_new_tokens})...")
        nonmath_activations, nonmath_texts = self.capture.capture_prompts(
            NONMATH_PROMPTS, max_new_tokens=max_new_tokens
        )

        print("\nSaving to H5...")
        h5_path = H5Store.save_activations(
            activations_dict={"math": math_activations, "nonmath": nonmath_activations},
            texts_dict={"math": math_texts, "nonmath": nonmath_texts},
            prompts_dict={"math": MATH_PROMPTS, "nonmath": NONMATH_PROMPTS},
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
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--test", type=str, default="welch", choices=["ttest", "welch", "mannwhitney"])
    parser.add_argument("--correction", type=str, default="fdr_bh", 
                       choices=["fdr_bh", "bonferroni", "fdr_by"])
    args = parser.parse_args()

    exp = MathComparison(model_path=args.model)
    exp.run(max_new_tokens=args.max_new_tokens, test=args.test, correction_method=args.correction)