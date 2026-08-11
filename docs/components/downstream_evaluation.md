# Downstream Evaluation & Generic Model Adapter

This document describes the design and configuration of the automated downstream evaluation pipeline in Modalities.

---

## Overview

The evaluation pipeline converts Modalities checkpoints to Hugging Face format and runs downstream evaluation benchmarks via [OLMES](https://github.com/allenai/olmes) or `lm-evaluation-harness`, syncing evaluation scores automatically to Weights & Biases (`eval/{task}`).

### Generic Model Adapter (`auto_map`)
To eliminate the need to write custom conversion scripts and HF model class definitions every time model architectures are adapted (e.g. DualPath, Looped models, MoE, custom gates), Modalities provides a **Generic HF Adapter** utilizing Hugging Face's `auto_map` feature.

When `AutoModelForCausalLM.from_pretrained(hf_dir)` is called by OLMES or `lm-eval`:
1. Hugging Face reads `config.json` and loads `ModalitiesHFAdapter` via `auto_map`.
2. `ModalitiesHFAdapter` dynamically instantiates the **native Modalities model** using Modalities `ModelFactory` and the embedded Modalities YAML config.
3. Weights are loaded directly in native Modalities format without requiring parameter key renaming or custom modeling scripts.

---

## 1. Conversion Callback (`ModelConverter`)

**Location:** `src/modalities/conversion/model_converter.py`

### Behavior
- Triggered if `num_train_steps_done % eval_interval == 0` on `global_rank == 0`.
- Reads `last_checkpoint_info.json` from the checkpoint directory.
- Checks if `{checkpoint_path}/hf_checkpoint` already exists (skips if present).
- Runs `command_template` via subprocess.

### YAML Configuration
```yaml
model_converter:
  component_key: model_converter
  variant_key: default
  config:
    command_template: "python -m modalities.conversion {modalities_config} {output_dir} --checkpoint_path {checkpoint_path}"
    checkpoint_dir: ${settings.paths.experiments_root_path}/${settings.experiment_id}
    global_rank: ${settings.cuda_env.global_rank}
    eval_interval: 1000
```

---

## 2. Downstream Evaluation Callback (`DownstreamEvaluator`)

**Location:** `src/modalities/evaluator.py`

### Behavior
- Triggered if `num_train_steps_done % eval_interval == 0` on `global_rank == 0`.
- Launches evaluation subprocess via `olmes_command_template`.
- Tracks active subprocesses asynchronously.
- At post-training exit (`wait_for_evaluations()`), waits for all background evaluation jobs to complete and parses `metrics-all.jsonl` to log `eval/{task}` metrics to W&B.

### YAML Configuration
```yaml
downstream_evaluator:
  component_key: downstream_evaluator
  variant_key: default
  config:
    tokenizer:
      instance_key: tokenizer
      pass_type: BY_REFERENCE
    tasks:
      - "arc_challenge:rc::olmes:full"
      - "hellaswag:rc::olmes:full"
      - "gsm8k::olmes"
    eval_interval: 1000
    checkpoint_dir: ${settings.paths.experiments_root_path}/${settings.experiment_id}
    global_rank: ${settings.cuda_env.global_rank}
    olmes_command_template: "bash scripts/evaluation/run_olmes_sbatch.sh {hf_model_dir} '{tasks}' {step} 128 1"
```

---

## 3. HPC / SLURM Environment Isolation

When submitting nested evaluation jobs via `sbatch` on HPC clusters (e.g. Leonardo Booster), use `scripts/evaluation/run_olmes_sbatch.sh`.

> [!IMPORTANT]
> The evaluation wrapper script uses `#SBATCH --export=NONE` to prevent SLURM environment variable leakage (such as CUDA variables, `RANK`, `WORLD_SIZE`) from the parent training job.
