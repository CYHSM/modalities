# GOLD Training

Minimal implementation for training language models using General Online Logit Distillation (GOLD).

## Setup

```bash
pip install transformers trl datasets wandb simple-parsing filelock lighteval
```

## Quick Start

### OpenR1-Math Dataset (default)

```bash
python gold_train.py \
  --model.student-model meta-llama/Llama-3.2-1B-Instruct \
  --model.teacher-model Qwen/Qwen2.5-0.5B-Instruct \
  --model.teacher-gpu 7 \
  --training.output-dir ./output/gold-llama-1b \
  --data.dataset HuggingFaceTB/OpenR1-Math-220k-default-verified \
  --data.subset all \
  --data.split "train[:1024]" \
  --training.num-train-epochs 1 \
  --training.learning-rate 3e-5 \
  --evaluation.eval-enabled true \
  --evaluation.eval-gpu 6
```

### OpenMathInstruct-2 Dataset

```bash
python gold_train.py \
  --model.student-model meta-llama/Llama-3.2-1B-Instruct \
  --model.teacher-model Qwen/Qwen3-8B-Instruct \
  --model.teacher-gpu 7 \
  --training.output-dir ./output/gold-llama-qwen3 \
  --data.dataset nvidia/OpenMathInstruct-2 \
  --data.subset train \
  --data.split "train[:10000]" \
  --training.num-train-epochs 1 \
  --training.per-device-train-batch-size 2 \
  --training.gradient-accumulation-steps 8 \
  --evaluation.eval-enabled true
```

## Configuration

All parameters can be set via command line using `--section.parameter value` format:

### Model
- `--model.student-model`: Student model path/name
- `--model.teacher-model`: Teacher model path/name  
- `--model.teacher-gpu`: GPU for teacher model (default: 7)

### Data
- `--data.dataset`: Dataset name
- `--data.subset`: Dataset subset (e.g., "all", "train")
- `--data.split`: Dataset split (e.g., "train[:1024]")
- `--data.eval-ratio`: Evaluation split ratio (default: 0.001)

### Training
- `--training.output-dir`: Output directory (required)
- `--training.learning-rate`: Learning rate (default: 3e-5)
- `--training.per-device-train-batch-size`: Batch size (default: 1)
- `--training.gradient-accumulation-steps`: Gradient accumulation (default: 4)
- `--training.save-steps`: Save checkpoint every N steps (default: 500)
- `--training.temperature`: Sampling temperature (default: 0.9)
- `--training.lmbda`: Student data fraction (default: 0.5)
- `--training.beta`: JSD interpolation coefficient (default: 0.5)

### Evaluation  
- `--evaluation.eval-enabled`: Enable evaluation (default: true)
- `--evaluation.eval-gpu`: GPU for evaluation (default: 6)
- `--evaluation.eval-tasks`: Benchmark tasks
- `--evaluation.eval-max-samples`: Max samples per task (default: 50)

### WandB
- `--wandb.project`: Project name (default: "gold-distillation")
- `--wandb.name`: Run name (auto-generated if not set)

## Features

- ✅ Cross-tokenizer distillation via ULD loss
- ✅ Hybrid loss for partial vocabulary overlap
- ✅ Async evaluation during training
- ✅ WandB logging with out-of-order metrics
- ✅ Teacher model on separate GPU
- ✅ Support for OpenR1-Math and OpenMathInstruct-2
- ✅ Minimal codebase (~400 lines)

## File Structure

```
gold_config.py      # Configuration classes
gold_data.py        # Dataset loading
gold_evaluation.py  # Async evaluation with LightEval
gold_train.py       # Main training script
```