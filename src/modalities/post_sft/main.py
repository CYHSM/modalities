import logging
import sys

import wandb
from simple_parsing import parse
from model_utils import get_model_info, load_model_and_tokenizer
from trainer import setup_trainer
from config import Config
from data import load_datasets

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_wandb(config: Config):
    """Setup Weights & Biases logging."""
    if config.training.report_to == "wandb":
        # Auto-generate name if not provided
        wandb_name = config.wandb.name
        if not wandb_name:
            dataset_name = config.data.dataset.split(':')[0].split('/')[-1]
            wandb_name = f"{dataset_name}-ga{config.training.gradient_accumulation_steps}-lr{config.training.learning_rate}"
            if config.model.lora.use_lora:
                wandb_name += f"-lora{config.model.lora.lora_r}"
        
        wandb_tags = list(config.wandb.tags)
        if config.model.lora.use_lora:
            wandb_tags.append("lora")
        
        wandb.init(
            project=config.wandb.project,
            name=wandb_name,
            tags=wandb_tags,
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
    # Parse config from command line - automatically generates CLI from dataclass!
    config: Config = parse(Config, description="Fine-tune language model with SFT")
    
    # Validate required fields
    if not config.model.model_path:
        logger.error("--model.model-path is required")
        sys.exit(1)
    if not config.training.output_dir:
        logger.error("--training.output-dir is required")
        sys.exit(1)
    
    # Setup environment
    config.setup_environment()
    
    logger.info(f"Model: {config.model.model_path}")
    logger.info(f"Dataset: {config.data.dataset}")
    logger.info(f"Output: {config.training.output_dir}")
    if config.model.lora.use_lora:
        logger.info(f"LoRA: r={config.model.lora.lora_r}, alpha={config.model.lora.lora_alpha}")
    
    try:
        # Setup WandB
        setup_wandb(config)
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(
            config.model.model_path,
            device=config.model.device,
            torch_dtype=config.model.torch_dtype,
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
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_datasets(
            config.data.dataset,
            total_samples=config.data.total_samples,
            eval_ratio=config.data.test_size,
            seed=config.data.seed
        )
        logger.info(f"Dataset loaded: {len(dataset['train'])} training samples")
        
        # Setup trainer
        logger.info("Setting up SFT trainer...")
        trainer = setup_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            source_model_path=config.model.model_path,
            training_config=config.training,
            eval_config=config.evaluation,
            hf_home=config.hf_home,
        )
        
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
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()