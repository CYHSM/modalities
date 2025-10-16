import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from modalities.post_inference.core.capture import ActivationCapture
from modalities.post_inference.core.stats import ActivationStats
from modalities.post_inference.utils.h5_utils import H5Store


class CorrectIncorrectMathExperiment:
    def __init__(self, *, model_path="gpt2", device="cuda"):
        self.capture = ActivationCapture(model_path, device)
        self.model_path = model_path

    def extract_answer(self, text):
        text = text.strip()
        
        if "\\boxed{" in text:
            match = re.search(r'\\boxed\{([^}]+)\}', text)
            if match:
                return self.clean_number(match.group(1))
        
        if "####" in text:
            parts = text.split("####")
            if len(parts) > 1:
                return self.clean_number(parts[-1])
        
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

    def load_math_prompts(self, *, n_samples=100, split="train"):
        dataset = load_dataset("gsm8k", "main", split=split)
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
        
        prompts = []
        expected_answers = []
        
        for ex in dataset:
            prompt = f"Question: {ex['question']}\nLet's solve this step by step and put your answer in \\boxed{{}}.\nAnswer:"
            expected = self.clean_number(ex['answer'].split("####")[-1].strip())
            
            prompts.append(prompt)
            expected_answers.append(expected)
        
        return prompts, expected_answers

    def capture_with_generation(self, *, prompts, expected_answers, max_new_tokens=256, batch_size=8):
        correct_activations = []
        incorrect_activations = []
        correct_texts = []
        incorrect_texts = []
        correct_prompts = []
        incorrect_prompts = []
        
        print(f"Generating and capturing activations for {len(prompts)} prompts...")
        
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            batch_expected = expected_answers[i:i+batch_size]
            
            for prompt, expected in zip(batch_prompts, batch_expected):
                inputs = self.capture.tokenizer(prompt, return_tensors="pt").to(self.capture.device)
                
                with torch.no_grad():
                    outputs = self.capture.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=self.capture.tokenizer.pad_token_id
                    )
                
                response = self.capture.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                
                predicted = self.extract_answer(response)
                is_correct = predicted == expected
                
                activations, texts = self.capture.capture_generation(
                    prompt,
                    max_new_tokens=1,
                    layers_to_capture=None
                )
                
                if is_correct:
                    correct_activations.append(activations)
                    correct_texts.append(texts)
                    correct_prompts.append(prompt)
                else:
                    incorrect_activations.append(activations)
                    incorrect_texts.append(texts)
                    incorrect_prompts.append(prompt)
        
        return (correct_activations, correct_texts, correct_prompts,
                incorrect_activations, incorrect_texts, incorrect_prompts)

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

    def compute_statistics_with_summary(self, *, correct_activations, incorrect_activations, 
                                       test="welch", correction_method="fdr_bh"):
        correct_by_layer = self.flatten_activations_by_layer(activation_list=correct_activations)
        incorrect_by_layer = self.flatten_activations_by_layer(activation_list=incorrect_activations)
        
        common_layers = set(correct_by_layer.keys()) & set(incorrect_by_layer.keys())
        
        layer_statistics = {}
        
        print(f"Computing statistics for {len(common_layers)} layers (test={test}, correction={correction_method})...")
        
        for layer in tqdm(sorted(common_layers), desc="Computing stats"):
            correct_vals = correct_by_layer[layer]
            incorrect_vals = incorrect_by_layer[layer]
            
            contrast = ActivationStats.compute_contrast(correct_vals, incorrect_vals, test=test)
            correction = ActivationStats.multiple_comparison_correction(
                contrast["p_values"], method=correction_method, alpha=0.05
            )
            
            n1, n2 = contrast["n1"], contrast["n2"]
            std1, std2 = contrast["std1"], contrast["std2"]
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            correct_array = np.array(correct_vals)
            incorrect_array = np.array(incorrect_vals)
            
            layer_statistics[layer] = {
                "t_stats": contrast["t_stats"],
                "p_values": contrast["p_values"],
                "p_corrected": correction["p_corrected"],
                "significant": correction["significant"],
                "mean_diff": contrast["mean_diff"],
                "cohens_d": contrast["cohens_d"],
                "pooled_std": pooled_std,
                "correct_mean": np.mean(correct_array, axis=0),
                "correct_std": np.std(correct_array, axis=0, ddof=1),
                "correct_median": np.median(correct_array, axis=0),
                "correct_min": np.min(correct_array, axis=0),
                "correct_max": np.max(correct_array, axis=0),
                "incorrect_mean": np.mean(incorrect_array, axis=0),
                "incorrect_std": np.std(incorrect_array, axis=0, ddof=1),
                "incorrect_median": np.median(incorrect_array, axis=0),
                "incorrect_min": np.min(incorrect_array, axis=0),
                "incorrect_max": np.max(incorrect_array, axis=0),
                "n_correct": len(correct_vals),
                "n_incorrect": len(incorrect_vals)
            }
            
            n_sig = correction["n_significant"]
            n_total = correction["n_total"]
            print(f"  {layer}: {n_sig}/{n_total} significant after correction ({100*n_sig/n_total:.2f}%)")
        
        return layer_statistics

    def run(self, *, n_samples=500, max_new_tokens=256, batch_size=8, 
            test="welch", correction_method="fdr_bh"):
        output_path = Path("/raid/s3/opengptx/mfrey/activation/experiments") / f"{self.model_path.replace('/', '_')}_correct_vs_incorrect"
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Loading {n_samples} math prompts from GSM8K...")
        prompts, expected_answers = self.load_math_prompts(n_samples=n_samples, split="train")

        print(f"\nGenerating answers and capturing activations...")
        (correct_acts, correct_texts, correct_prompts,
         incorrect_acts, incorrect_texts, incorrect_prompts) = self.capture_with_generation(
            prompts=prompts,
            expected_answers=expected_answers,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size
        )

        n_correct = len(correct_acts)
        n_incorrect = len(incorrect_acts)
        accuracy = n_correct / (n_correct + n_incorrect) if (n_correct + n_incorrect) > 0 else 0

        print(f"\nResults:")
        print(f"  Correct: {n_correct}")
        print(f"  Incorrect: {n_incorrect}")
        print(f"  Accuracy: {accuracy:.2%}")

        if n_correct == 0 or n_incorrect == 0:
            print("\nError: Need both correct and incorrect samples. Exiting.")
            return None

        print("\nComputing statistics with multiple comparison correction...")
        layer_statistics = self.compute_statistics_with_summary(
            correct_activations=correct_acts,
            incorrect_activations=incorrect_acts,
            test=test,
            correction_method=correction_method
        )

        print("\nSaving to H5 (including statistics)...")
        h5_path, stats_path = H5Store.save_activations(
            activations_dict={"correct": correct_acts, "incorrect": incorrect_acts},
            texts_dict={"correct": correct_texts, "incorrect": incorrect_texts},
            prompts_dict={"correct": correct_prompts, "incorrect": incorrect_prompts},
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
        print(f"Model accuracy: {accuracy:.2%} ({n_correct} correct, {n_incorrect} incorrect)")
        print(f"FDR-corrected significant (p<0.05): {n_significant_corr}/{total_dims} ({100*n_significant_corr/total_dims:.2f}%)")
        
        return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test", type=str, default="welch", choices=["ttest", "welch", "mannwhitney"])
    parser.add_argument("--correction", type=str, default="fdr_bh", 
                       choices=["fdr_bh", "bonferroni", "fdr_by"])
    args = parser.parse_args()

    exp = CorrectIncorrectMathExperiment(model_path=args.model)
    exp.run(
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        test=args.test,
        correction_method=args.correction
    )