import json
from pathlib import Path

import re
import numpy as np
import torch
from datasets import load_dataset
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from modalities.post_inference.utils.h5_utils import H5Store


class ActivationPatcher:
    def __init__(self, *, model_path, device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if "Qwen" in model_path:
            self.tokenizer.padding_side = "left"
        self.device = device
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.hooks = []
        self.patch_specs = {}
        self.layer_stats = None

    def load_statistics(self, *, h5_path):
        stats_path = Path(h5_path).parent / "statistics.h5"
        self.layer_stats = H5Store.load_statistics(stats_path)
        if self.layer_stats is None:
            raise ValueError(f"No statistics found at {stats_path}. Run compare_math_nomath first.")
        print(f"Loaded statistics for {len(self.layer_stats)} layers from {stats_path}")
        return self.layer_stats

    def identify_significant_neurons(self, *, p_threshold=0.01, t_threshold=0.0):
        if self.layer_stats is None:
            raise ValueError("Must call load_statistics first")
        
        patch_targets = {}
        significant_info = []
        
        for layer_name, stats_dict in self.layer_stats.items():
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

    def evaluate_gsm8k(self, *, n_samples=100, batch_size=64):
        def extract_answer(text):
            text = text.strip()
            
            if "\\boxed{" in text:
                match = re.search(r'\\boxed\{([^}]+)\}', text)
                if match:
                    return clean_number(match.group(1))
            
            if "####" in text:
                parts = text.split("####")
                if len(parts) > 1:
                    return clean_number(parts[-1])
            
            numbers = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
            if numbers:
                last_num = numbers[-1]
                return clean_number(last_num[0] if last_num[0] else last_num[1])
            
            return None

        def clean_number(text):
            text = text.strip()
            for char in [',', '$', '%', 'g', '.']:
                text = text.replace(char, '')
            try:
                return str(int(text))
            except:
                return text

        dataset = load_dataset("gsm8k", "main", split="test")
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
        
        correct = 0
        total = 0
        
        questions = [ex["question"] for ex in dataset]
        answers = [ex["answer"].split("####")[-1].strip() for ex in dataset]
        prompts = [f"Question: {q}\nLet's solve this step by step and put your answer in \\boxed{{}}.\nAnswer:" for q in questions]
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="GSM8K"):
            batch_prompts = prompts[i:i+batch_size]
            batch_answers = answers[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(
                    output[inputs["input_ids"][j].shape[0]:],
                    skip_special_tokens=True
                )
                
                predicted = extract_answer(response)
                expected = clean_number(batch_answers[j])
                
                if predicted == expected:
                    correct += 1
                
                total += 1
        
        return correct / total if total > 0 else 0

    def evaluate_hellaswag(self, *, n_samples=100):
        dataset = load_dataset("hellaswag", split="validation")
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
        
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc="HellaSwag"):
            context = example["ctx"]
            endings = example["endings"]
            label = example["label"]
            
            scores = []
            for ending in endings:
                prompt = context + ending
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][:, 1:].contiguous()
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="mean"
                    )
                    scores.append(-loss.item())
            
            predicted = np.argmax(scores)
            if predicted == int(label):
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0


class PatchingExperiment:
    def __init__(self, *, model_path, h5_path, device="cuda"):
        self.patcher = ActivationPatcher(model_path=model_path, device=device)
        self.h5_path = h5_path
        self.model_path = model_path

    def run(self, *, scale_factors=[0.0, 0.5, 1.0, 1.5, 2.0],
            t_thresholds=[0, 5, 10, 15, 20], p_threshold=0.01,
            n_eval_samples=100):
        
        output_path = Path("/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests/experiments") / f"patching_results{p_threshold}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\nLoading precomputed statistics from H5 file...")
        self.patcher.load_statistics(h5_path=self.h5_path)
        
        print("\nBaseline evaluation (no patching)")
        baseline_gsm8k = self.patcher.evaluate_gsm8k(n_samples=n_eval_samples)
        baseline_hellaswag = self.patcher.evaluate_hellaswag(n_samples=n_eval_samples)
        print(f"  GSM8K: {baseline_gsm8k:.3f}")
        print(f"  HellaSwag: {baseline_hellaswag:.3f}")
        
        all_results = {
            "baseline": {
                "gsm8k": baseline_gsm8k,
                "hellaswag": baseline_hellaswag
            },
            "experiments": []
        }
        
        for t_threshold in t_thresholds:
            print(f"\n{'='*60}")
            print(f"T-STATISTIC THRESHOLD: {t_threshold}")
            print(f"{'='*60}")
            
            patch_targets, significant_neurons = self.patcher.identify_significant_neurons(
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
                
                gsm8k_score = self.patcher.evaluate_gsm8k(n_samples=n_eval_samples)
                hellaswag_score = self.patcher.evaluate_hellaswag(n_samples=n_eval_samples)
                
                print(f"    GSM8K: {gsm8k_score:.3f} (Δ={gsm8k_score - baseline_gsm8k:+.3f})")
                print(f"    HellaSwag: {hellaswag_score:.3f} (Δ={hellaswag_score - baseline_hellaswag:+.3f})")
                
                threshold_results["scale_results"].append({
                    "scale_factor": scale,
                    "gsm8k": gsm8k_score,
                    "hellaswag": hellaswag_score,
                    "gsm8k_delta": gsm8k_score - baseline_gsm8k,
                    "hellaswag_delta": hellaswag_score - baseline_hellaswag
                })
                
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
                "n_eval_samples": n_eval_samples,
                "results": all_results
            }, f, indent=2)
        
        summary = self._create_summary_table(all_results)
        with open(output_path / "summary.txt", "w") as f:
            f.write(summary)
        print(f"\n{summary}")
        
        print(f"\n✓ Results saved to {output_path}")
        return all_results

    def _create_summary_table(self, all_results):
        lines = []
        lines.append("ACTIVATION PATCHING SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Baseline GSM8K: {all_results['baseline']['gsm8k']:.3f}")
        lines.append(f"Baseline HellaSwag: {all_results['baseline']['hellaswag']:.3f}")
        lines.append("")
        
        for exp in all_results["experiments"]:
            lines.append(f"\nT-threshold={exp['t_threshold']} ({exp['n_neurons']} neurons, {exp['n_layers']} layers):")
            lines.append("-" * 80)
            lines.append(f"{'Scale':>6} | {'GSM8K':>8} | {'Δ GSM8K':>10} | {'HellaSwag':>10} | {'Δ HellaSwag':>12}")
            
            for result in exp["scale_results"]:
                scale = f"{result['scale_factor']:.1f}"
                gsm8k = f"{result['gsm8k']:.3f}"
                hellaswag = f"{result['hellaswag']:.3f}"
                delta_gsm = f"{result['gsm8k_delta']:+.3f}"
                delta_hella = f"{result['hellaswag_delta']:+.3f}"
                
                lines.append(f"{scale:>6} | {gsm8k:>8} | {delta_gsm:>10} | {hellaswag:>10} | {delta_hella:>12}")
        
        return "\n".join(lines)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run activation patching experiment")
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
    parser.add_argument("--n_eval_samples", type=int, default=100,
                       help="Number of evaluation samples per benchmark")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ACTIVATION PATCHING EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"H5 file: {args.h5_path}")
    print(f"Scale factors: {args.scales}")
    print(f"T-stat thresholds: {args.t_thresholds}")
    print(f"P-value threshold: {args.p_threshold}")
    print(f"Eval samples: {args.n_eval_samples} per benchmark")
    print("=" * 60)
    
    exp = PatchingExperiment(model_path=args.model, h5_path=args.h5_path)
    results = exp.run(
        scale_factors=args.scales,
        t_thresholds=args.t_thresholds,
        p_threshold=args.p_threshold,
        n_eval_samples=args.n_eval_samples
    )