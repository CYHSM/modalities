import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from modalities.post_inference.utils.h5_utils import H5Store


class PatchedHFLM(HFLM):
    def __init__(self, *, pretrained, patcher, device="cuda", **kwargs):
        super().__init__(pretrained=pretrained, device=device, **kwargs)
        self.patcher = patcher


class ActivationPatcher:
    def __init__(self, *, model_path, device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.hooks = []
        self.patch_specs = {}

    def identify_significant_neurons(self, *, h5_path, p_threshold=0.01, t_threshold=0.0):
        activations_dict, _, _ = H5Store.load_activations(h5_path)
        
        math_activations = activations_dict["math"]
        nonmath_activations = activations_dict["nonmath"]
        
        layer_stats = {}
        all_keys = set()
        for sample_acts in math_activations + nonmath_activations:
            for step_acts in sample_acts:
                all_keys.update(step_acts.keys())
        
        for layer_name in sorted(all_keys):
            math_values = []
            nonmath_values = []
            
            for sample_acts in math_activations:
                for step_acts in sample_acts:
                    if layer_name in step_acts:
                        act = step_acts[layer_name]
                        math_values.append(act.flatten())
            
            for sample_acts in nonmath_activations:
                for step_acts in sample_acts:
                    if layer_name in step_acts:
                        act = step_acts[layer_name]
                        nonmath_values.append(act.flatten())
            
            if math_values and nonmath_values:
                min_len = min(min(len(v) for v in math_values), min(len(v) for v in nonmath_values))
                math_values = np.array([v[:min_len] for v in math_values])
                nonmath_values = np.array([v[:min_len] for v in nonmath_values])
                
                n1, n2 = len(math_values), len(nonmath_values)
                mean1 = np.mean(math_values, axis=0)
                mean2 = np.mean(nonmath_values, axis=0)
                var1 = np.var(math_values, axis=0, ddof=1)
                var2 = np.var(nonmath_values, axis=0, ddof=1)
                
                mean_diff = mean1 - mean2
                se = np.sqrt(var1/n1 + var2/n2 + 1e-8)
                t_stats = mean_diff / se
                
                df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1) + 1e-8)
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
                
                layer_stats[layer_name] = {
                    "t_stats": t_stats,
                    "p_values": p_values,
                    "mean_diff": mean_diff
                }
        
        patch_targets = {}
        significant_info = []
        
        for layer_name, stats_dict in layer_stats.items():
            t_stats = stats_dict["t_stats"]
            p_values = stats_dict["p_values"]
            mean_diff = stats_dict["mean_diff"]
            
            significant_mask = (p_values < p_threshold) & (np.abs(t_stats) > t_threshold)
            significant_indices = np.where(significant_mask)[0]
            
            if len(significant_indices) > 0:
                patch_targets[layer_name] = significant_indices.tolist()
                
                for idx in significant_indices:
                    significant_info.append({
                        "layer": layer_name,
                        "neuron": int(idx),
                        "t_stat": float(t_stats[idx]),
                        "p_value": float(p_values[idx]),
                        "mean_diff": float(mean_diff[idx])
                    })
        
        significant_info = sorted(significant_info, key=lambda x: abs(x["t_stat"]), reverse=True)
        
        return patch_targets, significant_info

    def setup_patches(self, *, patch_targets, scale_factor=0.0):
        self.clear_patches()
        self.patch_specs = {layer: (neurons, scale_factor) for layer, neurons in patch_targets.items()}
        
        def make_hook(layer_name, neuron_indices, scale):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    out = output
                elif isinstance(output, tuple):
                    out = output[0]
                else:
                    return output
                
                for idx in neuron_indices:
                    if len(out.shape) == 3:
                        out[:, :, idx] *= scale
                    elif len(out.shape) == 2:
                        out[:, idx] *= scale
                
                return out if isinstance(output, torch.Tensor) else (out,) + output[1:]
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.patch_specs:
                neurons, scale = self.patch_specs[name]
                hook = module.register_forward_hook(make_hook(name, neurons, scale))
                self.hooks.append(hook)

    def clear_patches(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.patch_specs = {}

    def evaluate_with_harness(self, *, tasks, num_fewshot=None, limit=None):
        lm = PatchedHFLM(
            pretrained=self.model,
            patcher=self,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        task_config = {}
        for task in tasks:
            config = {}
            if limit is not None:
                config["limit"] = limit
            if num_fewshot is not None and task in num_fewshot:
                config["num_fewshot"] = num_fewshot[task]
            if config:
                task_config[task] = config
        
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            **task_config
        )
        
        return results["results"]


class PatchingExperiment:
    def __init__(self, *, model_path, h5_path, device="cuda"):
        self.patcher = ActivationPatcher(model_path=model_path, device=device)
        self.h5_path = h5_path
        self.model_path = model_path

    def run(self, *, scale_factors=[0.0, 0.5, 1.0, 1.5, 2.0],
            t_thresholds=[0, 5, 10, 15, 20], p_threshold=0.01,
            tasks=["gsm8k", "hellaswag"], num_fewshot={"gsm8k": 3},
            limit=100):
        
        output_path = Path("/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests/experiments") / "patching_results"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\nBaseline evaluation (no patching)")
        baseline_results = self.patcher.evaluate_with_harness(
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit
        )
        
        baseline_scores = {}
        for task in tasks:
            score = baseline_results[task].get("acc,none", baseline_results[task].get("acc_norm,none", 0))
            baseline_scores[task] = score
            print(f"  {task}: {score:.3f}")
        
        all_results = {
            "baseline": baseline_scores,
            "experiments": []
        }
        
        for t_threshold in t_thresholds:
            print(f"\n{'='*60}")
            print(f"T-STATISTIC THRESHOLD: {t_threshold}")
            print(f"{'='*60}")
            
            patch_targets, significant_neurons = self.patcher.identify_significant_neurons(
                h5_path=self.h5_path,
                p_threshold=p_threshold,
                t_threshold=t_threshold
            )
            
            n_neurons = sum(len(neurons) for neurons in patch_targets.values())
            print(f"Found {n_neurons} significant neurons (p<{p_threshold}, |t|>{t_threshold}) across {len(patch_targets)} layers")
            
            if n_neurons == 0:
                print("  No neurons meet criteria, skipping...")
                continue
            
            threshold_results = {
                "t_threshold": t_threshold,
                "p_threshold": p_threshold,
                "n_neurons": n_neurons,
                "n_layers": len(patch_targets),
                "scale_results": []
            }
            
            for scale in scale_factors:
                if scale == 1.0:
                    continue
                
                print(f"\n  Patching with scale={scale}")
                self.patcher.setup_patches(patch_targets=patch_targets, scale_factor=scale)
                
                eval_results = self.patcher.evaluate_with_harness(
                    tasks=tasks,
                    num_fewshot=num_fewshot,
                    limit=limit
                )
                
                scale_result = {"scale_factor": scale}
                
                for task in tasks:
                    score = eval_results[task].get("acc,none", eval_results[task].get("acc_norm,none", 0))
                    delta = score - baseline_scores[task]
                    scale_result[task] = score
                    scale_result[f"{task}_delta"] = delta
                    print(f"    {task}: {score:.3f} (Δ={delta:+.3f})")
                
                threshold_results["scale_results"].append(scale_result)
                
                self.patcher.clear_patches()
            
            all_results["experiments"].append(threshold_results)
            
            with open(output_path / f"neurons_t{t_threshold}.json", "w") as f:
                json.dump({
                    "patch_targets": {k: v for k, v in patch_targets.items()},
                    "top_neurons": significant_neurons[:100]
                }, f, indent=2)
        
        with open(output_path / "all_results.json", "w") as f:
            json.dump({
                "model": self.model_path,
                "p_threshold": p_threshold,
                "t_thresholds": t_thresholds,
                "scale_factors": scale_factors,
                "tasks": tasks,
                "num_fewshot": num_fewshot,
                "limit": limit,
                "results": all_results
            }, f, indent=2)
        
        summary = self._create_summary_table(all_results, tasks)
        with open(output_path / "summary.txt", "w") as f:
            f.write(summary)
        print(f"\n{summary}")
        
        print(f"\n✓ Results saved to {output_path}")
        return all_results

    def _create_summary_table(self, all_results, tasks):
        lines = []
        lines.append("ACTIVATION PATCHING SUMMARY")
        lines.append("=" * 80)
        
        for task in tasks:
            lines.append(f"Baseline {task}: {all_results['baseline'][task]:.3f}")
        lines.append("")
        
        for exp in all_results["experiments"]:
            lines.append(f"\nT-threshold={exp['t_threshold']} ({exp['n_neurons']} neurons, {exp['n_layers']} layers):")
            lines.append("-" * 80)
            
            header = f"{'Scale':>6}"
            for task in tasks:
                header += f" | {task[:8]:>8} | {f'Δ {task[:6]}':>10}"
            lines.append(header)
            
            for result in exp["scale_results"]:
                row = f"{result['scale_factor']:.1f:>6}"
                for task in tasks:
                    score = f"{result[task]:.3f}"
                    delta = f"{result[f'{task}_delta']:+.3f}"
                    row += f" | {score:>8} | {delta:>10}"
                lines.append(row)
        
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run activation patching experiment with lm_harness")
    parser.add_argument("--model", type=str, 
                       default="/raid/s3/opengptx/mfrey/instruct/hf_model",
                       help="Path to model")
    parser.add_argument("--h5_path", type=str, 
                       default="/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests/experiments/math_comparison/activations.h5",
                       help="Path to H5 file with activation data")
    parser.add_argument("--scales", type=float, nargs="+", 
                       default=[0.0, 0.5, 1.0, 1.5, 2.0],
                       help="Scale factors to test (0=prune, 1=baseline, >1=amplify)")
    parser.add_argument("--t_thresholds", type=float, nargs="+",
                       default=[0, 5, 10, 15, 20],
                       help="T-statistic thresholds for neuron selection")
    parser.add_argument("--p_threshold", type=float, default=0.01,
                       help="P-value threshold for statistical significance")
    parser.add_argument("--tasks", type=str, nargs="+",
                       default=["gsm8k", "hellaswag"],
                       help="LM harness tasks to evaluate")
    parser.add_argument("--limit", type=int, default=100,
                       help="Number of evaluation samples per benchmark")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ACTIVATION PATCHING EXPERIMENT (LM HARNESS)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"H5 file: {args.h5_path}")
    print(f"Scale factors: {args.scales}")
    print(f"T-stat thresholds: {args.t_thresholds}")
    print(f"P-value threshold: {args.p_threshold}")
    print(f"Tasks: {args.tasks}")
    print(f"Eval limit: {args.limit} per task")
    print("=" * 60)
    
    exp = PatchingExperiment(model_path=args.model, h5_path=args.h5_path)
    results = exp.run(
        scale_factors=args.scales,
        t_thresholds=args.t_thresholds,
        p_threshold=args.p_threshold,
        tasks=args.tasks,
        num_fewshot={"gsm8k": 3},
        limit=args.limit
    )