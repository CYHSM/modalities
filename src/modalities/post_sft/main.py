#!/usr/bin/env python3
"""Main CLI script for model training with SFT and GRPO support."""
import argparse
import logging
import sys

import wandb
from model_utils import get_model_info, load_model_and_tokenizer
from trainer import setup_trainer

from config import Config, DataConfig, EvaluationConfig, LoRAConfig, ModelConfig, TrainingConfig, WandBConfig
from data import load_and_format_dataset, load_and_shuffle_multiple_datasets

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune language model with TRL (SFT or GRPO)")

    # Model arguments
    parser.add_argument("--model-path", required=True, help="Path to the base model")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--trainable-layers", help="Layers to train: '20' or 'last_5' or '18,19,20'")

    # LoRA arguments
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=128, help="LoRA rank (default: 128 for full-FT performance)")
    parser.add_argument("--lora-alpha", type=int, default=256, help="LoRA alpha (default: 256)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    parser.add_argument(
        "--lora-target-modules", nargs="+", default=None, help="LoRA target modules (default: all linear layers)"
    )
    parser.add_argument("--lora-bias", default="none", choices=["none", "all", "lora_only"], help="LoRA bias strategy")
    parser.add_argument("--use-dora", action="store_true", help="Use DoRA (Weight-Decomposed LoRA)")
    parser.add_argument("--use-rslora", action="store_true", default=True, help="Use RSLoRA (Rank-Stabilized LoRA)")

    # Data arguments
    parser.add_argument("--dataset", default="nvidia/OpenMathInstruct-2", help="Dataset name")
    parser.add_argument("--dataset-split", default="train", help="Dataset split")
    parser.add_argument("--test-size", type=float, default=0.0001, help="Test split size")
    parser.add_argument("--max-length", type=int, help="Maximum sequence length filter")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to use (for testing)")

    # Training arguments
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Evaluate every N steps")
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.01, help="Warmup ratio for learning rate scheduler")

    # GRPO specific arguments
    parser.add_argument("--num-generations", type=int, default=4, help="Number of generations per prompt for GRPO")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for generation in GRPO")
    parser.add_argument("--vllm-server-host", default="localhost", help="vLLM server host")
    parser.add_argument("--vllm-server-port", type=int, default=8000, help="vLLM server port")

    # Evaluation arguments
    parser.add_argument("--eval-gpu", type=int, default=7, help="GPU for lighteval evaluation")
    parser.add_argument("--eval-samples", type=int, default=100, help="Max samples for evaluation")

    # Environment arguments
    parser.add_argument("--hf-home", default="/raid/s3/opengptx/mfrey/huggingface", help="HF cache dir")

    # WandB arguments
    parser.add_argument("--wandb-project", default="sft-Aug2025", help="WandB project name")
    parser.add_argument("--wandb-name", help="WandB run name")
    parser.add_argument("--wandb-tags", nargs="*", default=[], help="WandB tags")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")

    # Other arguments
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation during training")
    parser.add_argument("--resume-from", help="Resume training from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Setup only, don't train")

    return parser.parse_args()


def create_config_from_args(args) -> Config:
    """Create configuration from command line arguments."""
    trainable_layers = None
    if args.trainable_layers:
        if args.trainable_layers == "all" or args.trainable_layers.startswith("last_"):
            trainable_layers = args.trainable_layers
        else:
            trainable_layers = [int(x.strip()) for x in args.trainable_layers.split(",")]

    # Create LoRA config
    lora_target_modules = args.lora_target_modules
    if lora_target_modules is None:
        # Default to all linear layers for maximum adaptation
        lora_target_modules = [
            "embed_tokens",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]

    lora_config = LoRAConfig(
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias=args.lora_bias,
        use_dora=args.use_dora,
        use_rslora=args.use_rslora,
    )

    model_config = ModelConfig(
        model_path=args.model_path, torch_dtype=args.torch_dtype, trainable_layers=trainable_layers, lora=lora_config
    )

    data_config = DataConfig(
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        test_size=args.test_size,
        max_length=args.max_length,
    )

    # Adjust learning rate for LoRA if needed
    learning_rate = args.learning_rate
    if args.use_lora:
        # LoRA can typically use higher learning rates
        if learning_rate == 3e-5:  # If using default
            learning_rate = 1e-4  # Increase for LoRA
            logger.info(f"Using higher learning rate for LoRA: {learning_rate}")

    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * args.grad_accum,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        learning_rate=learning_rate,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to="wandb" if not args.no_wandb else "none",
    )

    eval_config = EvaluationConfig(gpu_id=args.eval_gpu, max_samples=args.eval_samples)

    wandb_tags = args.wandb_tags
    if args.use_lora:
        wandb_tags.append("lora")

    wandb_config = WandBConfig(
        project=args.wandb_project,
        name=f"{args.dataset.split('/')[-1]}-ga{args.grad_accum}-wr{args.warmup_ratio}-lr{learning_rate}",
        tags=wandb_tags,
    )

    config = Config(
        model=model_config,
        data=data_config,
        training=training_config,
        evaluation=eval_config,
        wandb=wandb_config,
        hf_home=args.hf_home,
    )

    return config


def setup_wandb(config: Config):
    """Setup Weights & Biases logging."""
    if config.training.report_to == "wandb":
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            tags=config.wandb.tags,
            notes=config.wandb.notes,
            config={
                "model": config.model.__dict__,
                "data": config.data.__dict__,
                "training": config.training.__dict__,
                "evaluation": config.evaluation.__dict__,
                "lora": config.model.lora.__dict__,
            },
        )
        logger.info(f"WandB initialized: {wandb.run.name}")
    else:
        logger.info("WandB logging disabled")


def main():
    """Main training function."""
    args = parse_args()
    config = create_config_from_args(args)

    # Setup environment
    config.setup_environment()

    logger.info(f"Model: {config.model.model_path}")
    logger.info(f"Dataset: {config.data.dataset_name}")
    logger.info(f"Output: {config.training.output_dir}")
    if args.use_lora:
        logger.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        logger.info(f"LoRA targets: {config.model.lora.target_modules}")

    print(config)
    try:
        # Setup WandB
        setup_wandb(config)

        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(
            config.model.model_path,
            torch_dtype=config.model.get_torch_dtype(),
            trust_remote_code=config.model.trust_remote_code,
            device_map=config.model.device_map,
            trainable_layers=config.model.trainable_layers,
            lora_config=config.model.lora,
        )

        # Log model info
        model_info = get_model_info(model, tokenizer)
        logger.info(f"Model info: {model_info}")
        if wandb.run:
            wandb.log({"model_info": model_info})

        # Load dataset based on training mode
        logger.info("Loading dataset for SFT training...")
        # dataset = load_and_format_dataset(
        #     config.data.dataset_name,
        #     config.data.dataset_split,
        #     config.data.test_size,
        #     config.data.seed,
        #     config.data.max_length,
        # )

        # Shuffled
        dataset = load_and_shuffle_multiple_datasets(config.data.dataset_name, eval_ratio=config.data.test_size, total_samples=21e6)
        logger.info(f"SFT dataset loaded: {len(dataset['train'])} training samples")

        if args.dry_run:
            logger.info("Dry run completed successfully")
            return

        # Setup trainer based on mode
        logger.info("Setting up SFT trainer...")
        trainer = setup_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            source_model_path=config.model.model_path,
            training_config=config.training,
            eval_config=config.evaluation if not args.no_eval else None,
            hf_home=config.hf_home,
        )

        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"Resuming training from: {args.resume_from}")
            trainer.train(resume_from_checkpoint=args.resume_from)
        else:
            # Start training
            logger.info(f"🚀 Starting training...")
            trainer.train()

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()

        logger.info(f"✨ training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Clean up WandB
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
