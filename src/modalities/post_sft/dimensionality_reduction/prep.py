import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_benchmark_filename(filename):
    step_match = re.match(r"(full_\w+)_step_(\d+)_.*\.json", filename)
    if step_match:
        return step_match.group(1), int(step_match.group(2))

    base_match = re.match(r"full_basemodel_.*\.json", filename)
    if base_match:
        return "full_basemodel", 0

    return None, None


def extract_benchmark_scores(benchmark_path, save_path):
    benchmark_path = Path(benchmark_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    metrics = {
        "gsm8k_math": ("gsm8k_cot", "exact_match,flexible-extract"),
        "hellaswag_reasoning": ("hellaswag", "acc_norm,none"),
        "humaneval_coding": ("humaneval_instruct", "pass@10,create_test"),
        "wmt16_translation": ("wmt16-de-en", "bleu,none"),
    }

    data = []

    for json_file in benchmark_path.glob("*.json"):
        dataset, step = parse_benchmark_filename(json_file.name)
        if dataset is None:
            continue

        with open(json_file, "r") as f:
            results = json.load(f)["results"]

        if dataset == "full_basemodel":
            datasets_to_add = ["full_code", "full_general", "full_math", "full_mix"]
            for base_dataset in datasets_to_add:
                row = {"dataset": base_dataset, "step": step}
                for metric_name, (task_name, metric_key) in metrics.items():
                    if task_name in results and metric_key in results[task_name]:
                        row[metric_name] = results[task_name][metric_key]
                    else:
                        row[metric_name] = np.nan
                data.append(row)
        else:
            row = {"dataset": dataset, "step": step}
            for metric_name, (task_name, metric_key) in metrics.items():
                if task_name in results and metric_key in results[task_name]:
                    row[metric_name] = results[task_name][metric_key]
                else:
                    row[metric_name] = np.nan
            data.append(row)

    df = pd.DataFrame(data).sort_values(["dataset", "step"]).reset_index(drop=True)

    df.to_csv(save_path / "benchmark_scores.csv", index=False)

    datasets = ["full_code", "full_general", "full_math", "full_mix"]
    steps = [0] + list(range(5000, 105000, 5000))

    for metric in ["gsm8k_math", "hellaswag_reasoning", "humaneval_coding", "wmt16_translation"]:
        scores = []
        for dataset in datasets:
            for step in steps:
                row = df[(df["dataset"] == dataset) & (df["step"] == step)]
                if len(row) > 0:
                    scores.append(row[metric].iloc[0])
                else:
                    scores.append(np.nan)

        np.save(save_path / f"{metric}_scores.npy", np.array(scores))

    print(f"Saved benchmark data to {save_path}")
    print(f"Data shape: {len(df)} records")
    return df


def main():
    benchmark_path = "/raid/s3/opengptx/mfrey/cp_analysis/benchmark_cps"
    save_path = "/raid/s3/opengptx/mfrey/cp_analysis/dim_reduction"
    return extract_benchmark_scores(benchmark_path, save_path)


if __name__ == "__main__":
    main()
