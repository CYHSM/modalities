#!/usr/bin/env python3
"""Main CLI script for model training."""
import argparse
import logging
import sys

import wandb
from model_utils import get_model_info, load_model_and_tokenizer
from trainer import setup_trainer

from config import Config, DataConfig, EvaluationConfig, ModelConfig, TrainingConfig, WandBConfig
from data import load_and_format_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune language model with TRL")

    # Model arguments
    parser.add_argument("--model-path", required=True, help="Path to the base model")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    # Data arguments
    parser.add_argument("--dataset", default="meta-math/MetaMathQA", help="Dataset name")
    parser.add_argument("--dataset-split", default="train", help="Dataset split")
    parser.add_argument("--test-size", type=float, default=0.01, help="Test split size")
    parser.add_argument("--max-length", type=int, help="Maximum sequence length filter")

    # Training arguments
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Evaluate every N steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")

    # Evaluation arguments
    parser.add_argument("--eval-gpu", type=int, default=7, help="GPU for evaluation")
    parser.add_argument("--eval-samples", type=int, default=100, help="Max samples for evaluation")

    # Environment arguments
    parser.add_argument("--hf-home", default="/raid/s3/opengptx/mfrey/huggingface", help="HF cache dir")

    # WandB arguments
    parser.add_argument("--wandb-project", default="trl-training", help="WandB project name")
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
    model_config = ModelConfig(model_path=args.model_path, torch_dtype=args.torch_dtype)

    data_config = DataConfig(
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        test_size=args.test_size,
        max_length=args.max_length,
    )

    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to="wandb" if not args.no_wandb else "none",
    )

    eval_config = EvaluationConfig(gpu_id=args.eval_gpu, max_samples=args.eval_samples)

    wandb_config = WandBConfig(project=args.wandb_project, name=args.wandb_name, tags=args.wandb_tags)

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
            },
        )
        logger.info(f"WandB initialized: {wandb.run.name}")
    else:
        logger.info("WandB logging disabled")


def main():
    """Main training function."""
    args = parse_args()
    config = create_config_from_args(args)
    print(config)

    # Setup environment
    config.setup_environment()

    logger.info("Starting model training pipeline")
    logger.info(f"Model: {config.model.model_path}")
    logger.info(f"Dataset: {config.data.dataset_name}")
    logger.info(f"Output: {config.training.output_dir}")

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
        )

        # Log model info
        model_info = get_model_info(model, tokenizer)
        logger.info(f"Model info: {model_info}")
        if wandb.run:
            wandb.log({"model_info": model_info})

        # Load and format dataset
        logger.info("Loading and formatting dataset...")
        dataset = load_and_format_dataset(
            config.data.dataset_name,
            config.data.dataset_split,
            config.data.test_size,
            config.data.seed,
            config.data.max_length,
        )

        if args.dry_run:
            logger.info("Dry run completed successfully")
            return

        # Setup trainer
        logger.info("Setting up trainer...")
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
            logger.info("🚀 Starting training...")
            trainer.train()

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()

        logger.info("✨ Training completed successfully!")

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
