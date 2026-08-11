# Model Ablation Test Configurations

This directory contains test configurations and instructions for running automated training, model conversion, and downstream evaluation.

---

## Configurations Included

1. **[`base_gpt2_test.yaml`](file:///leonardo_work/EUHPC_D21_101/mfrey/modalities/config_files/model_tests/base_gpt2_test.yaml)**
   - Standard GPT-2 baseline model configured for a fast test run (10 steps).
   - Automatically saves checkpoint at step 10, converts via generic HF adapter, and runs OLMES evaluation.

2. **[`dualpath_test.yaml`](file:///leonardo_work/EUHPC_D21_101/mfrey/modalities/config_files/model_tests/dualpath_test.yaml)**
   - DualPath adaptive model (loops + deep/wide gating) configured for a fast test run (10 steps).
   - Automatically saves checkpoint at step 10, converts via generic HF adapter, and runs OLMES evaluation.

---

## How to Extend for Ablations

To create a new ablation variant:
1. Copy `dualpath_test.yaml` or `base_gpt2_test.yaml` to a new filename (e.g. `dualpath_ablation_nocross_alpha75.yaml`).
2. Adjust model hyperparameters under `model_raw.config`:
   - `adaptive_config`: `max_loops`, `wide_ffn_hidden`, `gate_mode`, `use_cross`, etc.
   - `evaluation_subscriber.config.experiment_id`: Unique identifier for W&B tracking.
3. To scale up training steps for production runs:
   - Update `intervals.checkpointing_interval_in_steps` and `intervals.evaluation_interval_in_steps` (e.g., `1836`).
   - Update `training_target.num_target_tokens` to your target total tokens.

---

## Running a Test Run

You can run test configurations using the helper script:

```bash
# Run DualPath test
bash config_files/model_tests/run_tests.sh config_files/model_tests/dualpath_test.yaml

# Run GPT-2 baseline test
bash config_files/model_tests/run_tests.sh config_files/model_tests/base_gpt2_test.yaml
```
