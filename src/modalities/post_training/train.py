import argparse
import os
import sys

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from data.dataset_loader import DatasetLoader
from evaluation.benchmark_runner import BenchmarkEvaluator
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from training.trainer import InstructionTuningTrainer

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: dict):
    """Load model and tokenizer"""
    model_config = config["model"]

    # Load custom configuration if needed
    if model_config.get("trust_remote_code"):
        config_obj = AutoConfig.from_pretrained(model_config["model_path"], trust_remote_code=True)
        # Ensure use_cache is False for gradient checkpointing
        config_obj.use_cache = False

    # Determine device map
    if torch.cuda.device_count() == 1:
        device_map = {"": 6}  # Single GPU
    else:
        device_map = "auto"  # Multi-GPU

    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_path"],
        torch_dtype=torch.bfloat16,  # Force bfloat16 for memory efficiency
        trust_remote_code=model_config.get("trust_remote_code", False),
        # use_flash_attention_2=model_config.get('use_flash_attention', False),
        device_map=device_map,
        # use_cache=False,  # Disable cache for gradient checkpointing
        # low_cpu_mem_usage=True,
        # config=config_obj if model_config.get('trust_remote_code') else None
    )

    # Enable gradient checkpointing if specified
    if model_config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        # Also set use_cache to False explicitly
        model.config.use_cache = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_path"], trust_remote_code=model_config.get("trust_remote_code", False), use_fast=True
    )

    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Print model size for debugging
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count / 1e9:.2f}B parameters")
    print(f"Model dtype: {model.dtype}")
    print(f"Model device: {next(model.parameters()).device}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Instruction Tuning Training Script")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to configuration file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    # Override arguments
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--wandb_project", type=str, help="Override WandB project name")
    parser.add_argument("--eval_steps", type=int, help="Override evaluation steps")
    parser.add_argument("--save_steps", type=int, help="Override save steps")
    parser.add_argument("--max_samples", type=int, help="Override max samples per dataset")
    parser.add_argument("--min_instruction_length", type=int, help="Override min instruction length")
    parser.add_argument("--max_instruction_length", type=int, help="Override max instruction length")
    parser.add_argument("--dataset", type=str, help="Use single dataset")

    args = parser.parse_args()

    # Print GPU information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

        # Print memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {gpu_memory:.1f} GB")

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_epochs:
        config["training"]["num_train_epochs"] = args.num_epochs
    if args.batch_size:
        config["training"]["per_device_train_batch_size"] = args.batch_size
        print(f"Overriding batch size to: {args.batch_size}")
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.wandb_project:
        config["wandb"]["project"] = args.wandb_project
    if args.eval_steps:
        config["training"]["eval_steps"] = args.eval_steps
    if args.save_steps:
        config["training"]["save_steps"] = args.save_steps
    if args.max_samples:
        config["data"]["max_samples_per_dataset"] = args.max_samples

    # Override dataset filters
    if args.min_instruction_length or args.max_instruction_length:
        for dataset_cfg in config["data"]["datasets"]:
            if args.min_instruction_length:
                dataset_cfg["filters"]["min_instruction_tokens"] = args.min_instruction_length
            if args.max_instruction_length:
                dataset_cfg["filters"]["max_instruction_tokens"] = args.max_instruction_length

    # Use single dataset if specified
    if args.dataset:
        print(f"Using single dataset: {args.dataset}")
        config["data"]["datasets"] = [
            {
                "name": args.dataset,
                "split": "train",
                "weight": 1.0,
                "filters": config["data"]["datasets"][0].get("filters", {}),
            }
        ]

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="bf16" if config["training"].get("bf16") else "no",
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        log_with="wandb" if config.get("wandb") else None,
        project_dir=config["training"].get("output_dir", "./outputs"),
    )

    # Set seed for reproducibility
    set_seed(config["training"].get("seed", 42))

    # Load model and tokenizer
    print("=" * 50)
    print("Loading model and tokenizer...")
    print(f"Model path: {config['model']['model_path']}")

    try:
        model, tokenizer = setup_model_and_tokenizer(config)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model path is correct and the model files exist.")
        raise

    # Load and process datasets
    print("=" * 50)
    print("Loading and processing datasets...")
    dataset_loader = DatasetLoader(config, tokenizer)
    train_dataset = dataset_loader.load_and_process_datasets()

    # Create eval dataset (10% of train data or separate eval set)
    eval_size = min(1000, len(train_dataset) // 10)
    eval_indices = torch.randperm(len(train_dataset))[:eval_size].tolist()
    train_indices = [i for i in range(len(train_dataset)) if i not in eval_indices]

    eval_dataset = torch.utils.data.Subset(train_dataset, eval_indices)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Batch size: {config['training']['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {config['training'].get('gradient_accumulation_steps', 1)}")
    print(
        f"Effective batch size: {config['training']['per_device_train_batch_size'] * config['training'].get('gradient_accumulation_steps', 1)}"  # noqa: E501
    )

    # Initialize trainer
    print("=" * 50)
    print("Initializing trainer...")
    trainer = InstructionTuningTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        accelerator=accelerator,
    )

    # Start training
    print("=" * 50)
    print("Starting training...")
    print(f"Output directory: {config['training']['output_dir']}")

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        print("=" * 50)
        print("ERROR: Out of GPU memory!")
        print("Try reducing batch_size or max_length in the config.")
        print(f"Current batch_size: {config['training']['per_device_train_batch_size']}")
        print(f"Current max_length: {config['data']['prompt']['max_length']}")
        print("=" * 50)
        raise e

    # Run final evaluation
    if accelerator.is_main_process and config["evaluation"].get("eval_during_training"):
        print("Running final benchmark evaluation...")
        evaluator = BenchmarkEvaluator(model, tokenizer, config)
        results = evaluator.run_all_benchmarks()

        print("\nFinal Benchmark Results:")
        for benchmark, score in results.items():
            print(f"{benchmark}: {score:.2f}")

        # Log to wandb
        if config.get("wandb"):
            evaluator.log_results_to_wandb(results)

    print("=" * 50)
    print("Training complete!")
    print(f"Model saved to: {config['training']['output_dir']}")


if __name__ == "__main__":
    main()
