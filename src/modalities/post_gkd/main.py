"""Main script for GKD (Generalized Knowledge Distillation) training."""
import logging
import sys

import wandb
from simple_parsing import parse
from modalities.post_gkd.model_utils import get_gkd_model_info, load_student_and_teacher
from modalities.post_gkd.trainer import setup_gkd_trainer
from modalities.post_gkd.config import Config
from modalities.post_gkd.data import load_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_wandb(config: Config):
    """Setup Weights & Biases logging for GKD."""
    if config.training.report_to == "wandb":
        wandb_name = config.wandb.name
        if not wandb_name:
            dataset_name = config.data.dataset.split(':')[0].split('/')[-1]
            student_size = config.model.model_path.split('-')[-1] if '-' in config.model.model_path else "8B"
            teacher_size = config.model.teacher_model_path.split('-')[-1] if '-' in config.model.teacher_model_path else "Teacher"
            
            wandb_name = f"gkd-{student_size}←{teacher_size}-{dataset_name}-λ{config.gkd.lmbda}-β{config.gkd.beta}-lr{config.training.learning_rate}"
            if config.model.lora.use_lora:
                wandb_name += f"-lora{config.model.lora.lora_r}"
        
        wandb_tags = list(config.wandb.tags)
        wandb_tags.extend([f"lambda_{config.gkd.lmbda}", f"beta_{config.gkd.beta}"])
        if config.model.lora.use_lora:
            wandb_tags.append("lora")
        
        wandb.init(
            project=config.wandb.project,
            name=wandb_name,
            tags=wandb_tags,
            notes=config.wandb.notes,
            config={
                "model": config.model.__dict__,
                "gkd": config.gkd.__dict__,
                "data": config.data.__dict__,
                "training": config.training.__dict__,
                "evaluation": config.evaluation.__dict__,
            },
        )
        logger.info(f"WandB initialized for GKD: {wandb.run.name}")
    else:
        logger.info("WandB logging disabled")


def main():
    """Main GKD training function."""
    config: Config = parse(Config, description="Fine-tune language model with GKD (Generalized Knowledge Distillation)")
    
    # Validate required fields
    if not config.model.model_path:
        logger.error("--model.model-path is required (student model)")
        sys.exit(1)
    if not config.model.teacher_model_path:
        logger.error("--model.teacher-model-path is required for GKD")
        sys.exit(1)
    if not config.training.output_dir:
        logger.error("--training.output-dir is required")
        sys.exit(1)
    
    config.setup_environment()
    
    logger.info("=== GKD Training Configuration ===")
    logger.info(f"Student Model: {config.model.model_path}")
    logger.info(f"Teacher Model: {config.model.teacher_model_path}")
    logger.info(f"Dataset: {config.data.dataset}")
    logger.info(f"Output: {config.training.output_dir}")
    logger.info(f"GKD Lambda (student fraction): {config.gkd.lmbda}")
    logger.info(f"GKD Beta (JSD interpolation): {config.gkd.beta}")
    logger.info(f"Temperature: {config.gkd.temperature}")
    if config.model.lora.use_lora:
        logger.info(f"LoRA: r={config.model.lora.lora_r}, alpha={config.model.lora.lora_alpha}")
    
    try:
        setup_wandb(config)
        
        logger.info("Loading student and teacher models...")
        student_model, teacher_model, tokenizer = load_student_and_teacher(
            student_path=config.model.model_path,
            teacher_path=config.model.teacher_model_path,
            device=config.model.device,
            torch_dtype=config.model.torch_dtype,
            trust_remote_code=config.model.trust_remote_code,
            device_map=config.model.device_map,
            trainable_layers=config.model.trainable_layers,
            lora_config=config.model.lora,
            vocab_alignment_method=config.model.vocab_alignment_method,
        )
        print("=== Student Model ===")
        print(student_model)
        print("=== Teacher Model ===")
        print(teacher_model)
        model_info = get_gkd_model_info(student_model, teacher_model, tokenizer)
        logger.info(f"Student info: {model_info['student']['model_size']} params, {model_info['student']['num_trainable_parameters']} trainable")
        logger.info(f"Teacher info: {model_info['teacher']['model_size']} params")
        logger.info(f"Compression ratio: {model_info['compression_ratio']:.1f}x")
        
        if wandb.run:
            wandb.log({"model_info": model_info})
        
        logger.info("Loading dataset...")
        dataset = load_datasets(
            config.data.dataset,
            total_samples=config.data.total_samples,
            eval_ratio=config.data.test_size,
            seed=config.data.seed
        )
        logger.info(f"Dataset loaded: {len(dataset['train'])} training samples")
        
        logger.info("Setting up GKD trainer...")
        trainer = setup_gkd_trainer(
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            source_model_path=config.model.model_path,
            training_config=config.training,
            gkd_config=config.gkd,
            eval_config=config.evaluation,
            hf_home=config.hf_home,
        )
        
        logger.info("Starting GKD training...")
        trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
        
        logger.info("Saving final GKD model...")
        trainer.save_model()
        
        logger.info("GKD training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("GKD training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"GKD training failed: {e}")
        raise
    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()