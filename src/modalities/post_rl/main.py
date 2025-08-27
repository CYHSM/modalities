#!/usr/bin/env python3
"""Main script for GRPO training on GSM8K."""
import argparse
import logging

import wandb
from grpo_trainer import setup_grpo_trainer
from model_utils import load_model_and_tokenizer

from config import Config, DataConfig, LoRAConfig, ModelConfig, TrainingConfig
from data import load_gsm8k_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training on GSM8K")

    # Model arguments
    parser.add_argument("--model", required=True, help="Model path or HF hub ID")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    # Memory optimization options (choose one)
    parser.add_argument("--use-qlora", action="store_true", help="Use QLoRA (4-bit quantization + LoRA)")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA (without quantization)")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization (no LoRA)")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization (no LoRA)")

    # LoRA hyperparameters
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Data arguments
    parser.add_argument("--train-size", type=int, help="Limit training samples")
    parser.add_argument("--eval-size", type=int, default=50)

    # Other arguments
    parser.add_argument("--wandb-project", default="grpo-runs")
    parser.add_argument("--no-wandb", action="store_true")

    return parser.parse_args()


def get_memory_optimization_config(args) -> tuple[ModelConfig, str]:
    """Determine memory optimization strategy."""

    # Create base LoRA config
    lora_config = LoRAConfig(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    if args.use_qlora:
        logger.info("Strategy: QLoRA (4-bit + LoRA) - Most memory efficient")
        lora_config.use_lora = True
        lora_config.use_qlora = True
        model_config = ModelConfig(
            model_path=args.model,
            load_in_4bit=True,
            gradient_checkpointing=True,
            lora=lora_config,
        )
        strategy = "qlora"

    elif args.use_lora:
        logger.info("Strategy: LoRA (bfloat16) - Moderate memory usage")
        lora_config.use_lora = True
        model_config = ModelConfig(
            model_path=args.model,
            torch_dtype="bfloat16",
            gradient_checkpointing=True,
            lora=lora_config,
        )
        strategy = "lora"

    elif args.use_8bit:
        logger.info("Strategy: 8-bit quantization - Good for medium models")
        model_config = ModelConfig(
            model_path=args.model,
            load_in_8bit=True,
            gradient_checkpointing=True,
            lora=lora_config,
        )
        strategy = "8bit"

    elif args.use_4bit:
        logger.info("Strategy: 4-bit quantization (no LoRA) - Memory efficient")
        model_config = ModelConfig(
            model_path=args.model,
            load_in_4bit=True,
            gradient_checkpointing=True,
            lora=lora_config,
        )
        strategy = "4bit"

    else:
        logger.info("Strategy: Default (bfloat16) - Requires most memory")
        model_config = ModelConfig(
            model_path=args.model,
            torch_dtype="bfloat16",
            gradient_checkpointing=True,
            lora=lora_config,
        )
        strategy = "default"

    return model_config, strategy


def main():
    args = parse_args()

    # Get memory optimization config
    model_config, strategy = get_memory_optimization_config(args)

    # Create config
    config = Config(
        model=model_config,
        data=DataConfig(
            train_size=args.train_size,
            eval_size=args.eval_size,
        ),
        training=TrainingConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_generations=args.num_generations,
            temperature=args.temperature,
        ),
    )

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"grpo-{strategy}-b{args.batch_size}-g{args.num_generations}-lr{args.lr}-t{args.temperature}-lr{args.lora_r}-la{args.lora_alpha}",
            config=vars(args),
        )

    logger.info("=" * 60)
    logger.info("🚀 Starting GRPO Training")
    logger.info(f"📊 Model: {args.model}")
    logger.info(f"💾 Strategy: {strategy}")
    logger.info(f"📦 Effective batch size: {args.batch_size * args.grad_accum}")
    logger.info("=" * 60)

    try:
        # Load model and tokenizer
        logger.info("Loading model...")
        model, tokenizer = load_model_and_tokenizer(config.model)

        # Load dataset
        logger.info("Loading GSM8K dataset...")
        dataset = load_gsm8k_dataset(
            train_size=config.data.train_size,
            eval_size=config.data.eval_size,
            seed=config.data.seed,
        )

        # Setup trainer
        logger.info("Setting up GRPO trainer...")
        trainer = setup_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            training_config=config.training,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()

        logger.info("✅ Training completed successfully!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
