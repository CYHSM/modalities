from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

from modalities.post_inference.core.capture import ActivationCapture, DataStore
from modalities.post_inference.plot.visualize import visualize_step

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

    def compute_std_activations_all_steps(self, *, activation_list):
        std_activations = OrderedDict()
        
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
                std_activations[key] = np.stack(all_values).std(axis=0)
        
        return std_activations

    def compute_difference_activations(self, *, math_avg, nonmath_avg):
        diff_activations = OrderedDict()
        
        common_keys = set(math_avg.keys()) & set(nonmath_avg.keys())
        
        for key in sorted(common_keys):
            math_vals = math_avg[key]
            nonmath_vals = nonmath_avg[key]
            
            min_len = min(len(math_vals), len(nonmath_vals))
            diff_activations[key] = math_vals[:min_len] - nonmath_vals[:min_len]
        
        return diff_activations

    def compute_average_activations(self, *, activation_list):
        avg_activations = OrderedDict()

        all_keys = set()
        for acts in activation_list:
            act_dict = acts.get("activations", acts)
            all_keys.update(act_dict.keys())

        for key in sorted(all_keys):
            values = []
            for acts in activation_list:
                act_dict = acts.get("activations", acts)
                if key in act_dict:
                    act = act_dict[key]
                    if hasattr(act, "shape"):
                        if len(act.shape) == 3:
                            act = act[:, -1, :]
                        elif len(act.shape) == 2:
                            act = act[-1, :]
                        values.append(act.flatten())
                    else:
                        values.append(np.array(act).flatten())

            if values:
                min_len = min(len(v) for v in values)
                values = [v[:min_len] for v in values]
                avg_activations[key] = np.stack(values).mean(axis=0)

        return avg_activations

    def save_to_h5(self, *, math_activations, nonmath_activations, math_texts, nonmath_texts, output_path):
        h5_path = output_path / "activations.h5"

        with h5py.File(h5_path, "w") as f:
            f.attrs["model"] = self.model_path
            f.attrs["n_math_samples"] = len(math_activations)
            f.attrs["n_nonmath_samples"] = len(nonmath_activations)
            f.attrs["max_generation_steps"] = max(len(acts) for acts in math_activations + nonmath_activations)

            self._save_group_to_h5(f, math_activations, math_texts, "math", MATH_PROMPTS)
            self._save_group_to_h5(f, nonmath_activations, nonmath_texts, "nonmath", NONMATH_PROMPTS)

        return h5_path

    def _save_group_to_h5(self, f, activations_list, texts_list, group_name, prompts):
        group = f.create_group(group_name)

        for i, (sample_activations, sample_texts) in enumerate(zip(activations_list, texts_list)):
            sample_grp = group.create_group(f"sample_{i:03d}")
            
            prompt = prompts[i] if i < len(prompts) else ""
            sample_grp.attrs["prompt"] = prompt
            sample_grp.attrs["n_steps"] = len(sample_activations)
            
            text_grp = sample_grp.create_group("texts")
            for step_idx, text in enumerate(sample_texts):
                text_grp.attrs[f"step_{step_idx}"] = text
                if step_idx == 0:
                    text_grp.attrs["initial_prompt"] = text
                else:
                    continuation = text[len(sample_texts[0]):] if text.startswith(sample_texts[0]) else text
                    text_grp.attrs[f"continuation_step_{step_idx}"] = continuation

            for step_idx, step_activations in enumerate(sample_activations):
                step_grp = sample_grp.create_group(f"step_{step_idx}")
                step_grp.attrs["step_number"] = step_idx
                step_grp.attrs["text"] = sample_texts[step_idx] if step_idx < len(sample_texts) else ""

                for name, values in step_activations.items():
                    clean_name = name.replace(".", "_")
                    if hasattr(values, "shape"):
                        if len(values.shape) == 3:
                            values = values[:, -1, :]
                        elif len(values.shape) == 2:
                            values = values[-1, :]
                        data_array = values.flatten()
                    else:
                        data_array = np.array(values).flatten()

                    step_grp.create_dataset(clean_name, data=data_array, compression="gzip")

    def run(self, *, max_new_tokens=1):
        output_path = Path("/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests/experiments") / "math_comparison"
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Capturing MATH activations (max_new_tokens={max_new_tokens})...")
        math_activations, math_texts = self.capture.capture_prompts(MATH_PROMPTS, max_new_tokens=max_new_tokens)

        print(f"\nCapturing NON-MATH activations (max_new_tokens={max_new_tokens})...")
        nonmath_activations, nonmath_texts = self.capture.capture_prompts(
            NONMATH_PROMPTS, max_new_tokens=max_new_tokens
        )

        print("\nSaving to H5 with full generation data...")
        self.save_to_h5(
            math_activations=math_activations,
            nonmath_activations=nonmath_activations, 
            math_texts=math_texts,
            nonmath_texts=nonmath_texts,
            output_path=output_path
        )

        print("\nComputing averages across all steps...")
        math_avg_all_steps = self.compute_average_activations_all_steps(activation_list=math_activations)
        nonmath_avg_all_steps = self.compute_average_activations_all_steps(activation_list=nonmath_activations)
        
        print("Computing standard deviations across all steps...")
        math_std_all_steps = self.compute_std_activations_all_steps(activation_list=math_activations)
        nonmath_std_all_steps = self.compute_std_activations_all_steps(activation_list=nonmath_activations)
        
        print("Computing differences (math - nonmath)...")
        diff_activations = self.compute_difference_activations(
            math_avg=math_avg_all_steps, 
            nonmath_avg=nonmath_avg_all_steps
        )

        print("Creating visualizations...")
        
        math_img = visualize_step(math_avg_all_steps, 1, "MATH PROMPTS (averaged across all steps)")
        nonmath_img = visualize_step(nonmath_avg_all_steps, 2, "NON-MATH PROMPTS (averaged across all steps)")
        diff_img = visualize_step(diff_activations, 3, "DIFFERENCE (Math - Non-Math)")
        math_std_img = visualize_step(math_std_all_steps, 4, "MATH PROMPTS (standard deviation)")
        nonmath_std_img = visualize_step(nonmath_std_all_steps, 5, "NON-MATH PROMPTS (standard deviation)")

        math_img.save(output_path / "math_activations_avg_all_steps.png")
        nonmath_img.save(output_path / "nonmath_activations_avg_all_steps.png")
        diff_img.save(output_path / "difference_math_minus_nonmath.png")
        math_std_img.save(output_path / "math_activations_std.png")
        nonmath_std_img.save(output_path / "nonmath_activations_std.png")

        print("Computing last-step averages for comparison...")
        math_data = [{"activations": acts[-1]} for acts in math_activations]
        nonmath_data = [{"activations": acts[-1]} for acts in nonmath_activations]

        math_avg_last = self.compute_average_activations(activation_list=math_data)
        nonmath_avg_last = self.compute_average_activations(activation_list=nonmath_data)

        math_last_img = visualize_step(math_avg_last, 6, "MATH PROMPTS (last step only)")
        nonmath_last_img = visualize_step(nonmath_avg_last, 7, "NON-MATH PROMPTS (last step only)")

        math_last_img.save(output_path / "math_activations_last_step.png")
        nonmath_last_img.save(output_path / "nonmath_activations_last_step.png")

        with open(output_path / "prompts.txt", "w") as f:
            f.write("MATH PROMPTS:\n")
            for i, p in enumerate(MATH_PROMPTS):
                f.write(f"{i+1}. {p}\n")
            f.write("\nNON-MATH PROMPTS:\n")
            for i, p in enumerate(NONMATH_PROMPTS):
                f.write(f"{i+1}. {p}\n")

        with open(output_path / "analysis_summary.txt", "w") as f:
            f.write("ANALYSIS SUMMARY\n")
            f.write("================\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Max new tokens: {max_new_tokens}\n")
            f.write(f"Math samples: {len(math_activations)}\n")
            f.write(f"Non-math samples: {len(nonmath_activations)}\n\n")
            f.write("Generated visualizations:\n")
            f.write("- math_activations_avg_all_steps.png: Math prompts averaged across all generation steps\n")
            f.write("- nonmath_activations_avg_all_steps.png: Non-math prompts averaged across all generation steps\n")
            f.write("- difference_math_minus_nonmath.png: Difference between math and non-math averages\n")
            f.write("- math_activations_std.png: Standard deviation of math activations\n")
            f.write("- nonmath_activations_std.png: Standard deviation of non-math activations\n")
            f.write("- math_activations_last_step.png: Math prompts (last step only, for comparison)\n")
            f.write("- nonmath_activations_last_step.png: Non-math prompts (last step only, for comparison)\n")

        print(f"\n✓ Complete! All outputs in: {output_path}")
        print(f"H5 file: {output_path / 'activations.h5'}")
        print(f"Analysis summary: {output_path / 'analysis_summary.txt'}")
        return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--max_new_tokens", type=int, default=1)
    args = parser.parse_args()

    exp = MathComparison(model_path=args.model)
    exp.run(max_new_tokens=args.max_new_tokens)