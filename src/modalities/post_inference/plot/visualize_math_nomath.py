from collections import OrderedDict
from pathlib import Path

import numpy as np

from modalities.post_inference.plot.visualize import visualize_step
from modalities.post_inference.plot.plot_stats import create_interactive_viewer
from modalities.post_inference.utils.h5_utils import H5Store


class MathVisualizationExperiment:
    def __init__(self, *, experiment_path, stats_only=False):
        self.experiment_path = Path(experiment_path)
        self.h5_path = self.experiment_path / "activations.h5"
        self.stats_path = self.experiment_path / "statistics.h5"
        self.stats_only = stats_only
        
        if not stats_only and not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")
        if not self.stats_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {self.stats_path}")

    def load_data(self):
        if self.stats_only:
            print(f"Loading statistics only from {self.stats_path}")
            statistics = H5Store.load_statistics(self.stats_path)
            return None, None, statistics
        else:
            print(f"Loading activations from {self.h5_path}")
            activations_dict, texts_dict, prompts_dict = H5Store.load_activations(self.h5_path)
            
            print(f"Loading statistics from {self.stats_path}")
            statistics = H5Store.load_statistics(self.stats_path)
            
            return activations_dict, texts_dict, statistics

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

    def run(self, *, downsample_resolution=64):
        activations_dict, texts_dict, statistics = self.load_data()
        
        if not self.stats_only:
            math_activations = activations_dict["math"]
            nonmath_activations = activations_dict["nonmath"]

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

            math_img.save(self.experiment_path / "math_activations_avg_all_steps.png")
            nonmath_img.save(self.experiment_path / "nonmath_activations_avg_all_steps.png")
            diff_img.save(self.experiment_path / "difference_math_minus_nonmath.png")
        else:
            print("\nSkipping PNG visualizations (stats_only mode)")

        print(f"\nCreating interactive viewer (downsampled to {downsample_resolution}x{downsample_resolution})...")
        
        stats_for_html = {
            "t_stats": {},
            "p_values": {},
            "cohens_d": {},
            "mean_diff": {},
            "pooled_std": {}
        }
        
        for layer in statistics:
            stats_for_html["t_stats"][layer] = statistics[layer]["t_stats"]
            stats_for_html["p_values"][layer] = statistics[layer]["p_corrected"]
            stats_for_html["cohens_d"][layer] = statistics[layer]["cohens_d"]
            stats_for_html["mean_diff"][layer] = statistics[layer]["mean_diff"]
            stats_for_html["pooled_std"][layer] = statistics[layer]["pooled_std"]
        
        html_path = create_interactive_viewer(
            stats_results=stats_for_html, 
            output_path=self.experiment_path,
            downsample_resolution=downsample_resolution
        )

        n_significant = sum(np.sum(statistics[layer]["significant"]) for layer in statistics)
        total_dims = sum(len(statistics[layer]["p_values"].flatten()) for layer in statistics)

        print(f"\n✓ Visualization complete!")
        if self.stats_only:
            print(f"Mode: Stats only (activations.h5 not loaded)")
        print(f"Interactive HTML: {html_path}")
        print(f"FDR-corrected significant: {n_significant}/{total_dims} ({100*n_significant/total_dims:.2f}%)")
        
        return self.experiment_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", type=str, required=True,
                       help="Path to experiment directory with activations.h5 and statistics.h5")
    parser.add_argument("--downsample", type=int, default=64,
                       help="Resolution for HTML visualization (64, 128, or 256)")
    parser.add_argument("--stats_only", action="store_true",
                       help="Only generate HTML from statistics.h5, skip loading activations.h5 and PNG generation")
    args = parser.parse_args()

    exp = MathVisualizationExperiment(experiment_path=args.experiment_path, stats_only=args.stats_only)
    exp.run(downsample_resolution=args.downsample)