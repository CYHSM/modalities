import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from modalities.post_inference.experiments.patch import ActivationPatcher


class LayerAttributionExperiment:
    def __init__(self, *, model_path, h5_path, device="cuda"):
        self.patcher = ActivationPatcher(model_path=model_path, device=device)
        self.h5_path = h5_path
        self.model_path = model_path

    def get_layers_by_component(self, *, patch_targets):
        layer_types = defaultdict(list)
        
        for layer_name in patch_targets.keys():
            if ".layers." in layer_name:
                parts = layer_name.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            if i + 2 < len(parts):
                                component = parts[i + 2]
                                
                                if "attn" in component or "attention" in component:
                                    group_name = f"layer_{layer_idx}_attention"
                                elif "mlp" in component or "ffn" in component or "feed_forward" in component:
                                    group_name = f"layer_{layer_idx}_mlp"
                                else:
                                    group_name = f"layer_{layer_idx}_{component}"
                                
                                layer_types[group_name].append(layer_name)
                            else:
                                layer_types[f"layer_{layer_idx}"].append(layer_name)
                            break
                        except ValueError:
                            layer_types["other"].append(layer_name)
                            break
            else:
                if "embed" in layer_name.lower():
                    layer_types["embedding"].append(layer_name)
                elif "lm_head" in layer_name.lower() or "output" in layer_name.lower():
                    layer_types["lm_head"].append(layer_name)
                elif "norm" in layer_name.lower():
                    if "layers." in layer_name:
                        parts = layer_name.split(".")
                        for i, part in enumerate(parts):
                            if part == "layers" and i + 1 < len(parts):
                                try:
                                    layer_idx = int(parts[i + 1])
                                    layer_types[f"layer_{layer_idx}_norm"].append(layer_name)
                                    break
                                except ValueError:
                                    pass
                    else:
                        layer_types["final_norm"].append(layer_name)
                elif "rope" in layer_name.lower() or "rotary" in layer_name.lower():
                    layer_types["rope"].append(layer_name)
                else:
                    layer_types["other"].append(layer_name)
        
        return dict(layer_types)
    
    def group_by_layer_type(self, *, patch_targets):
        groups = {
            "all_transformer_layers": [],
            "embedding": [],
            "lm_head": [],
            "final_norm": [],
            "rope": [],
            "other": []
        }
        
        for layer_name in patch_targets.keys():
            if ".layers." in layer_name:
                groups["all_transformer_layers"].append(layer_name)
            elif "embed" in layer_name.lower() and "pos" not in layer_name.lower():
                groups["embedding"].append(layer_name)
            elif "lm_head" in layer_name.lower() or ("output" in layer_name.lower() and "layer" not in layer_name.lower()):
                groups["lm_head"].append(layer_name)
            elif "rope" in layer_name.lower() or "rotary" in layer_name.lower():
                groups["rope"].append(layer_name)
            elif "norm" in layer_name.lower() and "layers." not in layer_name:
                groups["final_norm"].append(layer_name)
            else:
                groups["other"].append(layer_name)
        
        return {k: v for k, v in groups.items() if v}

    def run_layer_by_layer(self, *, scale_factor, d_threshold, std_threshold, 
                          n_eval_samples=100, group_by_component=False, group_by_layer=False):
        
        output_path = Path(self.h5_path).parent / "layer_attribution"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nLoading statistics and identifying neurons...")
        self.patcher.load_statistics(h5_path=self.h5_path)
        
        patch_targets, significant_neurons = self.patcher.identify_significant_neurons(
            d_threshold=d_threshold,
            std_threshold=std_threshold
        )
        
        n_neurons = sum(len(neurons) for neurons in patch_targets.values())
        n_activations = len(patch_targets)
        
        has_lm_head = any("lm_head" in name.lower() or "output" in name.lower() 
                         for name in patch_targets.keys())
        
        print(f"Found {n_activations} activations with {n_neurons} significant neurons")
        print(f"✓ lm_head/output layer: {'INCLUDED' if has_lm_head else 'NOT FOUND'}")
        
        sample_layers = list(patch_targets.keys())[:5]
        print(f"\nSample activation names:")
        for name in sample_layers:
            print(f"  - {name} ({len(patch_targets[name])} neurons)")
        if n_activations > 5:
            print(f"  ... and {n_activations - 5} more")
        
        if n_neurons == 0:
            print("No neurons meet criteria, exiting...")
            return None
        
        print(f"\nBaseline evaluation (no patching)...")
        baseline_gsm8k = self.patcher.evaluate_gsm8k(n_samples=n_eval_samples)
        baseline_hellaswag = self.patcher.evaluate_hellaswag(n_samples=n_eval_samples)
        print(f"  GSM8K: {baseline_gsm8k:.3f}")
        print(f"  HellaSwag: {baseline_hellaswag:.3f}")
        
        if group_by_layer:
            layer_groups = self.group_by_layer_type(patch_targets=patch_targets)
            print(f"\nGrouped by layer type ({len(layer_groups)} groups):")
            
            order = ["all_transformer_layers", "lm_head", "embedding", "final_norm", "rope", "other"]
            for key in order:
                if key in layer_groups:
                    n_acts = len(layer_groups[key])
                    print(f"  {key}: {n_acts} activations")
            
            sorted_keys = [k for k in order if k in layer_groups]
            
        elif group_by_component:
            layer_groups = self.get_layers_by_component(patch_targets=patch_targets)
            print(f"\nGrouped into {len(layer_groups)} component groups:")
            
            for key in sorted(layer_groups.keys())[:10]:
                n_acts = len(layer_groups[key])
                print(f"  {key}: {n_acts} activations")
            if len(layer_groups) > 10:
                print(f"  ... and {len(layer_groups) - 10} more groups")
            
            sorted_keys = []
            for key in sorted(layer_groups.keys()):
                if key.startswith("layer_"):
                    sorted_keys.append(key)
            for key in sorted(layer_groups.keys()):
                if not key.startswith("layer_"):
                    sorted_keys.append(key)
        else:
            layer_groups = {name: [name] for name in patch_targets.keys()}
            sorted_keys = sorted(layer_groups.keys())
            print(f"\nEvaluating {len(sorted_keys)} individual activations")
        
        results = {
            "baseline": {
                "gsm8k": baseline_gsm8k,
                "hellaswag": baseline_hellaswag
            },
            "config": {
                "scale_factor": scale_factor,
                "d_threshold": d_threshold,
                "std_threshold": std_threshold,
                "n_neurons": n_neurons,
                "n_eval_samples": n_eval_samples,
                "group_by_component": group_by_component,
                "group_by_layer": group_by_layer
            },
            "layer_results": []
        }
        
        print(f"\n{'='*60}")
        print(f"Evaluating {len(sorted_keys)} layer groups with scale={scale_factor}")
        print(f"{'='*60}")
        
        for idx, group_name in enumerate(sorted_keys, 1):
            layer_names = layer_groups[group_name]
            
            single_layer_targets = {
                name: patch_targets[name] 
                for name in layer_names 
                if name in patch_targets
            }
            
            if not single_layer_targets:
                continue
            
            n_neurons_in_group = sum(len(neurons) for neurons in single_layer_targets.values())
            n_activations = len(single_layer_targets)
            
            print(f"\n[{idx}/{len(sorted_keys)}] {group_name}")
            print(f"  Activations: {n_activations}, Neurons: {n_neurons_in_group}")
            if n_activations <= 5:
                for name in single_layer_targets.keys():
                    print(f"    - {name}")
            
            self.patcher.setup_patches(
                patch_targets=single_layer_targets,
                scale_factor=scale_factor
            )
            
            print(f"  Evaluating GSM8K...", end=" ", flush=True)
            gsm8k_score = self.patcher.evaluate_gsm8k(n_samples=n_eval_samples)
            print(f"{gsm8k_score:.3f} (Δ={gsm8k_score - baseline_gsm8k:+.3f})")
            
            print(f"  Evaluating HellaSwag...", end=" ", flush=True)
            hellaswag_score = self.patcher.evaluate_hellaswag(n_samples=n_eval_samples)
            print(f"{hellaswag_score:.3f} (Δ={hellaswag_score - baseline_hellaswag:+.3f})")
            
            result = {
                "group_name": group_name,
                "layer_names": layer_names,
                "n_activations": n_activations,
                "n_neurons": n_neurons_in_group,
                "gsm8k": gsm8k_score,
                "hellaswag": hellaswag_score,
                "gsm8k_delta": gsm8k_score - baseline_gsm8k,
                "hellaswag_delta": hellaswag_score - baseline_hellaswag
            }
            
            results["layer_results"].append(result)
            
            self.patcher.clear_patches()
        
        results["layer_results"].sort(
            key=lambda x: abs(x["gsm8k_delta"]), 
            reverse=True
        )
        
        print(f"\n{'='*60}")
        print("TOP 10 LAYERS BY IMPACT:")
        print(f"{'='*60}")
        for i, result in enumerate(results["layer_results"][:10], 1):
            print(f"{i}. {result['group_name']}")
            print(f"   GSM8K: {result['gsm8k']:.3f} (Δ={result['gsm8k_delta']:+.3f})")
            print(f"   HellaSwag: {result['hellaswag']:.3f} (Δ={result['hellaswag_delta']:+.3f})")
            print(f"   Neurons: {result['n_neurons']}, Activations: {result['n_activations']}")
        
        grouping_suffix = ""
        if group_by_layer:
            grouping_suffix = "_bylayer"
        elif group_by_component:
            grouping_suffix = "_bycomponent"
        
        results_path = output_path / f"attribution_scale{scale_factor}_d{d_threshold}_std{std_threshold}{grouping_suffix}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        summary = self._create_summary(results)
        summary_path = output_path / f"summary_scale{scale_factor}_d{d_threshold}_std{std_threshold}{grouping_suffix}.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
        
        print(f"\n{summary}")
        print(f"\n✓ Results saved to {output_path}")
        
        return results

    def run_cumulative(self, *, scale_factor, d_threshold, std_threshold,
                      n_eval_samples=100, order="importance"):
        
        output_path = Path(self.h5_path).parent / "layer_attribution"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nLoading statistics and identifying neurons...")
        self.patcher.load_statistics(h5_path=self.h5_path)
        
        patch_targets, significant_neurons = self.patcher.identify_significant_neurons(
            d_threshold=d_threshold,
            std_threshold=std_threshold
        )
        
        n_neurons = sum(len(neurons) for neurons in patch_targets.values())
        n_activations = len(patch_targets)
        
        has_lm_head = any("lm_head" in name.lower() or "output" in name.lower() 
                         for name in patch_targets.keys())
        
        print(f"Found {n_activations} activations with {n_neurons} significant neurons")
        print(f"✓ lm_head/output layer: {'INCLUDED' if has_lm_head else 'NOT FOUND'}")
        
        print(f"\nBaseline evaluation...")
        baseline_gsm8k = self.patcher.evaluate_gsm8k(n_samples=n_eval_samples)
        baseline_hellaswag = self.patcher.evaluate_hellaswag(n_samples=n_eval_samples)
        
        if order == "importance":
            layer_importance = {}
            print(f"\nComputing layer importance (this takes {len(patch_targets)} forward passes)...")
            for idx, layer_name in enumerate(patch_targets.keys(), 1):
                print(f"  [{idx}/{len(patch_targets)}] {layer_name}...", end=" ", flush=True)
                single_target = {layer_name: patch_targets[layer_name]}
                self.patcher.setup_patches(
                    patch_targets=single_target,
                    scale_factor=scale_factor
                )
                
                gsm8k_score = self.patcher.evaluate_gsm8k(n_samples=n_eval_samples)
                layer_importance[layer_name] = abs(gsm8k_score - baseline_gsm8k)
                print(f"Δ={gsm8k_score - baseline_gsm8k:+.3f}")
                
                self.patcher.clear_patches()
            
            sorted_layers = sorted(
                layer_importance.keys(),
                key=lambda x: layer_importance[x],
                reverse=True
            )
            
            print(f"\nTop 10 most important layers:")
            for i, layer in enumerate(sorted_layers[:10], 1):
                print(f"  {i}. {layer}: Δ={layer_importance[layer]:+.3f}")
        else:
            sorted_layers = sorted(patch_targets.keys())
        
        results = {
            "baseline": {
                "gsm8k": baseline_gsm8k,
                "hellaswag": baseline_hellaswag
            },
            "config": {
                "scale_factor": scale_factor,
                "d_threshold": d_threshold,
                "std_threshold": std_threshold,
                "order": order
            },
            "cumulative_results": []
        }
        
        print(f"\n{'='*60}")
        print(f"Cumulative evaluation (order: {order})")
        print(f"{'='*60}")
        
        cumulative_targets = {}
        for idx, layer_name in enumerate(sorted_layers, 1):
            cumulative_targets[layer_name] = patch_targets[layer_name]
            
            print(f"\n[{idx}/{len(sorted_layers)}] Adding {layer_name}")
            print(f"  Total layers: {len(cumulative_targets)}")
            
            self.patcher.setup_patches(
                patch_targets=cumulative_targets,
                scale_factor=scale_factor
            )
            
            print(f"  Evaluating GSM8K...", end=" ", flush=True)
            gsm8k_score = self.patcher.evaluate_gsm8k(n_samples=n_eval_samples)
            print(f"{gsm8k_score:.3f} (Δ={gsm8k_score - baseline_gsm8k:+.3f})")
            
            print(f"  Evaluating HellaSwag...", end=" ", flush=True)
            hellaswag_score = self.patcher.evaluate_hellaswag(n_samples=n_eval_samples)
            print(f"{hellaswag_score:.3f} (Δ={hellaswag_score - baseline_hellaswag:+.3f})")
            
            result = {
                "n_layers": len(cumulative_targets),
                "last_layer_added": layer_name,
                "layers_included": list(cumulative_targets.keys()),
                "gsm8k": gsm8k_score,
                "hellaswag": hellaswag_score,
                "gsm8k_delta": gsm8k_score - baseline_gsm8k,
                "hellaswag_delta": hellaswag_score - baseline_hellaswag
            }
            
            results["cumulative_results"].append(result)
            
            self.patcher.clear_patches()
        
        results_path = output_path / f"cumulative_{order}_scale{scale_factor}_d{d_threshold}_std{std_threshold}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")
        
        return results

    def _create_summary(self, results):
        lines = []
        lines.append("LAYER ATTRIBUTION ANALYSIS")
        lines.append("=" * 100)
        lines.append(f"Baseline GSM8K: {results['baseline']['gsm8k']:.3f}")
        lines.append(f"Baseline HellaSwag: {results['baseline']['hellaswag']:.3f}")
        lines.append(f"Scale factor: {results['config']['scale_factor']}")
        lines.append(f"Cohen's d threshold: {results['config']['d_threshold']}")
        lines.append(f"Pooled std threshold: {results['config']['std_threshold']}")
        lines.append(f"Total neurons: {results['config']['n_neurons']}")
        
        if results['config'].get('group_by_layer'):
            lines.append(f"Grouping: By layer type (all_transformer_layers vs special components)")
        elif results['config'].get('group_by_component'):
            lines.append(f"Grouping: By component (attention/mlp per layer)")
        else:
            lines.append(f"Grouping: Individual activations")
        
        lines.append("")
        lines.append("TOP 20 LAYERS BY GSM8K IMPACT:")
        lines.append("-" * 100)
        lines.append(f"{'Layer/Group':<40} | {'Acts':>5} | {'Neurons':>8} | {'GSM8K':>8} | {'Δ GSM8K':>10} | {'HellaSwag':>10} | {'Δ Hella':>10}")
        lines.append("-" * 100)
        
        for result in results["layer_results"][:20]:
            name = result["group_name"][:40]
            n_acts = result.get("n_activations", 1)
            neurons = result["n_neurons"]
            gsm8k = result["gsm8k"]
            hellaswag = result["hellaswag"]
            delta_gsm = result["gsm8k_delta"]
            delta_hella = result["hellaswag_delta"]
            
            lines.append(
                f"{name:<40} | {n_acts:>5} | {neurons:>8} | {gsm8k:>8.3f} | {delta_gsm:>+10.3f} | {hellaswag:>10.3f} | {delta_hella:>+10.3f}"
            )
        
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--d_threshold", type=float, default=1.0)
    parser.add_argument("--std_threshold", type=float, default=0.1)
    parser.add_argument("--n_eval_samples", type=int, default=100)
    parser.add_argument("--mode", type=str, default="layer_by_layer",
                       choices=["layer_by_layer", "cumulative_importance", "cumulative_sequential"])
    parser.add_argument("--group_by_component", action="store_true",
                       help="Group by component type (attention, mlp, norm per layer)")
    parser.add_argument("--group_by_layer", action="store_true",
                       help="Group all transformer layers together, isolate lm_head/embedding/etc")
    
    args = parser.parse_args()
    
    if args.group_by_component and args.group_by_layer:
        parser.error("Cannot use both --group_by_component and --group_by_layer")
    
    print("=" * 60)
    print("LAYER ATTRIBUTION EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"H5 file: {args.h5_path}")
    print(f"Mode: {args.mode}")
    print(f"Scale: {args.scale}")
    print(f"d threshold: {args.d_threshold}")
    print(f"std threshold: {args.std_threshold}")
    if args.group_by_layer:
        print(f"Grouping: By layer type (all layers vs special components)")
    elif args.group_by_component:
        print(f"Grouping: By component (attention/mlp per layer)")
    else:
        print(f"Grouping: Individual activations")
    print("=" * 60)
    
    exp = LayerAttributionExperiment(
        model_path=args.model,
        h5_path=args.h5_path
    )
    
    if args.mode == "layer_by_layer":
        results = exp.run_layer_by_layer(
            scale_factor=args.scale,
            d_threshold=args.d_threshold,
            std_threshold=args.std_threshold,
            n_eval_samples=args.n_eval_samples,
            group_by_component=args.group_by_component,
            group_by_layer=args.group_by_layer
        )
    elif args.mode.startswith("cumulative"):
        order = "importance" if "importance" in args.mode else "sequential"
        results = exp.run_cumulative(
            scale_factor=args.scale,
            d_threshold=args.d_threshold,
            std_threshold=args.std_threshold,
            n_eval_samples=args.n_eval_samples,
            order=order
        )