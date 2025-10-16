import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from modalities.post_inference.experiments.patch import ActivationPatcher


class PatchingInspector:
    def __init__(self, *, model_path, h5_path, device="cuda"):
        self.patcher = ActivationPatcher(model_path=model_path, device=device)
        self.h5_path = h5_path
        self.model_path = model_path
        self.few_shot_examples = None

    def extract_answer(self, text):
        text = text.strip()
        
        first_question_idx = text.find("Question:")
        if first_question_idx > 0:
            text = text[:first_question_idx]
        
        if "\\boxed{" in text:
            match = re.search(r'\\boxed\{([^}]+)\}', text)
            if match:
                return self.clean_number(match.group(1))
        
        if "####" in text:
            parts = text.split("####")
            if len(parts) > 1:
                answer_part = parts[1].split()[0] if parts[1].strip() else ""
                return self.clean_number(answer_part)
        
        numbers = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
        if numbers:
            last_num = numbers[-1]
            return self.clean_number(last_num[0] if last_num[0] else last_num[1])
        
        return None

    def clean_number(self, text):
        text = text.strip()
        for char in [',', '$', '%', 'g', '.']:
            text = text.replace(char, '')
        try:
            return str(int(text))
        except:
            return text

    def load_few_shot_examples(self, *, n_shots=3):
        if n_shots == 0:
            self.few_shot_examples = []
            return
        
        dataset = load_dataset("gsm8k", "main", split="train")
        dataset = dataset.shuffle(seed=123).select(range(n_shots))
        
        self.few_shot_examples = []
        for ex in dataset:
            self.few_shot_examples.append({
                "question": ex["question"],
                "answer": ex["answer"]
            })
        
        print(f"\nLoaded {n_shots} few-shot examples:")
        for i, ex in enumerate(self.few_shot_examples, 1):
            print(f"  {i}. {ex['question'][:60]}...")
    
    def create_prompt(self, *, question):
        if not self.few_shot_examples:
            return f"Question: {question}\nLet's solve this step by step and put your answer in \\boxed{{}}.\nAnswer:"
        
        prompt_parts = []
        for ex in self.few_shot_examples:
            prompt_parts.append(f"Question: {ex['question']}")
            prompt_parts.append(f"Answer: {ex['answer']}\n")
        
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Let's solve this step by step and put your answer in \\boxed{}.")
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)

    def generate_with_patching(self, *, prompt, patch_targets, scale_factor, max_new_tokens=256):
        self.patcher.setup_patches(patch_targets=patch_targets, scale_factor=scale_factor)
        
        inputs = self.patcher.tokenizer(prompt, return_tensors="pt").to(self.patcher.device)
        
        with torch.no_grad():
            outputs = self.patcher.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.patcher.tokenizer.pad_token_id
            )
        
        response = self.patcher.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        self.patcher.clear_patches()
        
        return response

    def inspect_layer(self, *, layer_pattern, scale_factors=[0.0, 1.0, 1.2], 
                     d_threshold=1.0, std_threshold=0.1, n_samples=20, max_new_tokens=256,
                     n_shots=0):
        
        output_path = Path(self.h5_path).parent / "inspection"
        output_path.mkdir(parents=True, exist_ok=True)
        
        if n_shots > 0:
            print(f"\nLoading {n_shots}-shot examples...")
            self.load_few_shot_examples(n_shots=n_shots)
        
        print(f"Loading statistics...")
        self.patcher.load_statistics(h5_path=self.h5_path)
        
        print(f"Identifying significant neurons...")
        patch_targets, significant_neurons = self.patcher.identify_significant_neurons(
            d_threshold=d_threshold,
            std_threshold=std_threshold
        )
        
        matching_layers = {k: v for k, v in patch_targets.items() 
                          if layer_pattern.lower() in k.lower()}
        
        if not matching_layers:
            print(f"No layers matching '{layer_pattern}' found!")
            print(f"Available layers: {list(patch_targets.keys())[:10]}...")
            return None
        
        print(f"\nFound {len(matching_layers)} layers matching '{layer_pattern}':")
        for name, neurons in matching_layers.items():
            print(f"  {name}: {len(neurons)} neurons")
        
        total_neurons = sum(len(neurons) for neurons in matching_layers.values())
        print(f"\nTotal neurons to patch: {total_neurons}")
        
        print(f"\nLoading GSM8K test samples...")
        dataset = load_dataset("gsm8k", "main", split="test")
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
        
        results = {
            "config": {
                "layer_pattern": layer_pattern,
                "matching_layers": list(matching_layers.keys()),
                "total_neurons": total_neurons,
                "scale_factors": scale_factors,
                "d_threshold": d_threshold,
                "std_threshold": std_threshold,
                "n_samples": n_samples,
                "max_new_tokens": max_new_tokens,
                "n_shots": n_shots
            },
            "examples": []
        }
        
        for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
            question = example["question"]
            expected_answer = example["answer"].split("####")[-1].strip()
            expected_answer = self.clean_number(expected_answer)
            
            prompt = self.create_prompt(question=question)
            
            example_result = {
                "question": question,
                "expected_answer": expected_answer,
                "generations": {}
            }
            
            for scale in scale_factors:
                generation = self.generate_with_patching(
                    prompt=prompt,
                    patch_targets=matching_layers,
                    scale_factor=scale,
                    max_new_tokens=max_new_tokens
                )
                
                predicted_answer = self.extract_answer(generation)
                is_correct = predicted_answer == expected_answer
                
                example_result["generations"][f"scale_{scale}"] = {
                    "scale": scale,
                    "generation": generation,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct
                }
            
            results["examples"].append(example_result)
        
        accuracy_by_scale = {}
        for scale in scale_factors:
            correct = sum(1 for ex in results["examples"] 
                         if ex["generations"][f"scale_{scale}"]["is_correct"])
            accuracy_by_scale[scale] = correct / len(results["examples"])
        
        results["summary"] = {
            "accuracy_by_scale": accuracy_by_scale
        }
        
        output_file = output_path / f"inspect_{layer_pattern.replace('.', '_')}_d{d_threshold}_std{std_threshold}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"INSPECTION SUMMARY: {layer_pattern}")
        print(f"{'='*80}")
        print(f"Layers matched: {len(matching_layers)}")
        print(f"Total neurons: {total_neurons}")
        prompting_mode = f"{n_shots}-shot" if n_shots > 0 else "Zero-shot"
        print(f"Prompting: {prompting_mode}")
        print(f"\nAccuracy by scale factor:")
        for scale, acc in accuracy_by_scale.items():
            print(f"  Scale {scale}: {acc:.1%} ({int(acc * n_samples)}/{n_samples})")
        
        print(f"\n{'='*80}")
        print("EXAMPLE COMPARISONS (first 3 examples)")
        print(f"{'='*80}")
        
        for i, ex in enumerate(results["examples"][:3]):
            print(f"\nExample {i+1}:")
            print(f"Question: {ex['question']}")
            print(f"Expected: {ex['expected_answer']}")
            print()
            
            for scale in scale_factors:
                gen = ex["generations"][f"scale_{scale}"]
                status = "✓" if gen["is_correct"] else "✗"
                print(f"Scale {scale} {status} (predicted: {gen['predicted_answer']}):")
                print(f"  {gen['generation'][:200]}...")
                print()
        
        self._create_detailed_html_report(results, output_path, layer_pattern, d_threshold, std_threshold)
        
        print(f"\n✓ Full results saved to: {output_file}")
        print(f"✓ HTML report saved to: {output_path / 'report.html'}")
        
        return results

    def compare_scales_sidebyside(self, *, layer_pattern, scale_a, scale_b,
                                  d_threshold=1.0, std_threshold=0.1, 
                                  n_samples=50, max_new_tokens=256, n_shots=0):
        
        output_path = Path(self.h5_path).parent / "inspection"
        output_path.mkdir(parents=True, exist_ok=True)
        
        if n_shots > 0:
            print(f"\nLoading {n_shots}-shot examples...")
            self.load_few_shot_examples(n_shots=n_shots)
        
        print(f"Loading statistics...")
        self.patcher.load_statistics(h5_path=self.h5_path)
        
        patch_targets, _ = self.patcher.identify_significant_neurons(
            d_threshold=d_threshold,
            std_threshold=std_threshold
        )
        
        matching_layers = {k: v for k, v in patch_targets.items() 
                          if layer_pattern.lower() in k.lower()}
        
        if not matching_layers:
            print(f"No layers matching '{layer_pattern}' found!")
            return None
        
        print(f"\nComparing scale {scale_a} vs {scale_b}")
        print(f"Matching layers: {len(matching_layers)}")
        print(f"Total neurons: {sum(len(n) for n in matching_layers.values())}")
        
        dataset = load_dataset("gsm8k", "main", split="test")
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
        
        comparisons = []
        
        for example in tqdm(dataset, desc="Comparing"):
            question = example["question"]
            expected = self.clean_number(example["answer"].split("####")[-1].strip())
            prompt = self.create_prompt(question=question)
            
            gen_a = self.generate_with_patching(
                prompt=prompt,
                patch_targets=matching_layers,
                scale_factor=scale_a,
                max_new_tokens=max_new_tokens
            )
            
            gen_b = self.generate_with_patching(
                prompt=prompt,
                patch_targets=matching_layers,
                scale_factor=scale_b,
                max_new_tokens=max_new_tokens
            )
            
            pred_a = self.extract_answer(gen_a)
            pred_b = self.extract_answer(gen_b)
            
            correct_a = pred_a == expected
            correct_b = pred_b == expected
            
            status = "both_correct" if (correct_a and correct_b) else \
                     "both_wrong" if (not correct_a and not correct_b) else \
                     "a_correct" if correct_a else "b_correct"
            
            comparisons.append({
                "question": question,
                "expected": expected,
                "scale_a": {"scale": scale_a, "generation": gen_a, "predicted": pred_a, "correct": correct_a},
                "scale_b": {"scale": scale_b, "generation": gen_b, "predicted": pred_b, "correct": correct_b},
                "status": status
            })
        
        interesting_cases = [c for c in comparisons if c["status"] in ["a_correct", "b_correct"]]
        
        print(f"\n{'='*80}")
        print(f"COMPARISON RESULTS: Scale {scale_a} vs {scale_b}")
        print(f"{'='*80}")
        
        status_counts = {}
        for c in comparisons:
            status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1
        
        print(f"Both correct: {status_counts.get('both_correct', 0)}")
        print(f"Both wrong: {status_counts.get('both_wrong', 0)}")
        print(f"Only scale {scale_a} correct: {status_counts.get('a_correct', 0)}")
        print(f"Only scale {scale_b} correct: {status_counts.get('b_correct', 0)}")
        
        print(f"\n{'='*80}")
        print(f"INTERESTING CASES (different outcomes): {len(interesting_cases)}")
        print(f"{'='*80}")
        
        for i, case in enumerate(interesting_cases[:5]):
            print(f"\nCase {i+1}: {case['status']}")
            print(f"Question: {case['question']}")
            print(f"Expected: {case['expected']}")
            print(f"\nScale {scale_a} ({'✓' if case['scale_a']['correct'] else '✗'}, predicted: {case['scale_a']['predicted']}):")
            print(f"  {case['scale_a']['generation'][:250]}...")
            print(f"\nScale {scale_b} ({'✓' if case['scale_b']['correct'] else '✗'}, predicted: {case['scale_b']['predicted']}):")
            print(f"  {case['scale_b']['generation'][:250]}...")
            print()
        
        output_file = output_path / f"compare_{scale_a}_vs_{scale_b}_{layer_pattern.replace('.', '_')}.json"
        with open(output_file, "w") as f:
            json.dump({
                "config": {
                    "layer_pattern": layer_pattern,
                    "scale_a": scale_a,
                    "scale_b": scale_b,
                    "matching_layers": list(matching_layers.keys()),
                    "total_neurons": sum(len(n) for n in matching_layers.values()),
                    "n_shots": n_shots
                },
                "status_counts": status_counts,
                "comparisons": comparisons
            }, f, indent=2)
        
        print(f"\n✓ Comparison saved to: {output_file}")
        
        return comparisons

    def _create_detailed_html_report(self, results, output_path, layer_pattern, d_threshold, std_threshold):
        import html
        
        n_shots = results['config'].get('n_shots', 0)
        prompting_mode = f"{n_shots}-shot" if n_shots > 0 else "Zero-shot"
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Patching Inspection: {html.escape(layer_pattern)}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .summary {{ background: white; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .example {{ background: white; padding: 20px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #3498db; }}
        .question {{ font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
        .expected {{ color: #27ae60; font-weight: bold; }}
        .generation {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 3px; max-height: 400px; overflow-y: auto; }}
        .generation pre {{ white-space: pre-wrap; word-wrap: break-word; font-family: monospace; margin: 0; }}
        .correct {{ border-left: 4px solid #27ae60; }}
        .incorrect {{ border-left: 4px solid #e74c3c; }}
        .scale-header {{ font-weight: bold; color: #34495e; margin-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Activation Patching Inspection</h1>
        <p>Layer Pattern: {html.escape(layer_pattern)}</p>
        <p>Cohen's d threshold: {d_threshold} | Pooled std threshold: {std_threshold}</p>
        <p>Prompting: {prompting_mode}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Layers matched: {len(results['config']['matching_layers'])}</p>
        <p>Total neurons: {results['config']['total_neurons']}</p>
        <table>
            <tr><th>Scale Factor</th><th>Accuracy</th><th>Correct / Total</th></tr>
"""
        
        for scale, acc in results["summary"]["accuracy_by_scale"].items():
            n_correct = int(acc * len(results["examples"]))
            n_total = len(results["examples"])
            html_content += f"<tr><td>{scale}</td><td>{acc:.1%}</td><td>{n_correct} / {n_total}</td></tr>\n"
        
        html_content += """        </table>
    </div>
    
    <h2>Detailed Examples</h2>
"""
        
        for i, ex in enumerate(results["examples"]):
            html_content += f"""
    <div class="example">
        <div class="question">Example {i+1}: {html.escape(ex['question'])}</div>
        <div class="expected">Expected Answer: {html.escape(str(ex['expected_answer']))}</div>
"""
            
            for scale_key, gen in ex["generations"].items():
                status_class = "correct" if gen["is_correct"] else "incorrect"
                status_symbol = "✓" if gen["is_correct"] else "✗"
                pred_answer = html.escape(str(gen['predicted_answer'])) if gen['predicted_answer'] else "None"
                generation_text = html.escape(gen['generation'])
                
                html_content += f"""
        <div class="generation {status_class}">
            <div class="scale-header">{status_symbol} Scale {gen['scale']} (Predicted: {pred_answer})</div>
            <pre>{generation_text}</pre>
        </div>
"""
            
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>"""
        
        html_path = output_path / f"d_threshold{d_threshold}_std_threshold{std_threshold}_report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--layer_pattern", type=str, required=True,
                       help="Pattern to match layer names (e.g., 'lm_head', 'layers.5', 'mlp')")
    parser.add_argument("--mode", type=str, default="inspect",
                       choices=["inspect", "compare"])
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0, 1.2],
                       help="Scale factors to test (inspect mode)")
    parser.add_argument("--scale_a", type=float, default=0.0,
                       help="First scale for comparison (compare mode)")
    parser.add_argument("--scale_b", type=float, default=1.2,
                       help="Second scale for comparison (compare mode)")
    parser.add_argument("--d_threshold", type=float, default=1.0)
    parser.add_argument("--std_threshold", type=float, default=0.1)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--n_shots", type=int, default=0,
                       help="Number of few-shot examples (0=zero-shot, 3=three-shot, 5=five-shot)")
    
    args = parser.parse_args()
    
    inspector = PatchingInspector(
        model_path=args.model,
        h5_path=args.h5_path
    )
    
    if args.mode == "inspect":
        prompting_mode = f"{args.n_shots}-shot" if args.n_shots > 0 else "Zero-shot"
        print("=" * 80)
        print("PATCHING OUTPUT INSPECTOR")
        print("=" * 80)
        print(f"Mode: Inspect multiple scales")
        print(f"Layer pattern: {args.layer_pattern}")
        print(f"Scales: {args.scales}")
        print(f"Prompting: {prompting_mode}")
        print("=" * 80)
        
        inspector.inspect_layer(
            layer_pattern=args.layer_pattern,
            scale_factors=args.scales,
            d_threshold=args.d_threshold,
            std_threshold=args.std_threshold,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            n_shots=args.n_shots
        )
    
    else:
        prompting_mode = f"{args.n_shots}-shot" if args.n_shots > 0 else "Zero-shot"
        print("=" * 80)
        print("PATCHING OUTPUT COMPARISON")
        print("=" * 80)
        print(f"Mode: Compare two scales")
        print(f"Layer pattern: {args.layer_pattern}")
        print(f"Scale A: {args.scale_a}")
        print(f"Scale B: {args.scale_b}")
        print(f"Prompting: {prompting_mode}")
        print("=" * 80)
        
        inspector.compare_scales_sidebyside(
            layer_pattern=args.layer_pattern,
            scale_a=args.scale_a,
            scale_b=args.scale_b,
            d_threshold=args.d_threshold,
            std_threshold=args.std_threshold,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            n_shots=args.n_shots
        )