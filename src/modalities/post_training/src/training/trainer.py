import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


class InstructionTuningTrainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset,
        eval_dataset,
        config: Dict[str, Any],
        accelerator: Optional[Accelerator] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config

        # Initialize accelerator with proper settings
        if accelerator is None:
            accelerator = Accelerator(
                mixed_precision="bf16" if config["training"].get("bf16") else "no",
                gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
            )
        self.accelerator = accelerator

        # Initialize training components
        self.setup_training()

    def setup_training(self):
        """Setup training components"""
        training_config = self.config["training"]

        # Create data loaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=training_config["per_device_train_batch_size"],
            shuffle=True,
            num_workers=training_config.get("dataloader_num_workers", 4),
            pin_memory=training_config.get("dataloader_pin_memory", True),
            drop_last=True,
        )

        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=training_config["per_device_eval_batch_size"],
                shuffle=False,
                num_workers=training_config.get("dataloader_num_workers", 4),
                pin_memory=training_config.get("dataloader_pin_memory", True),
            )

        # Setup optimizer
        self.setup_optimizer()

        # Setup scheduler
        self.setup_scheduler()

        # Setup mixed precision if needed
        self.scaler = GradScaler() if training_config.get("fp16") else None

        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

        if self.eval_dataset:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

    def setup_optimizer(self):
        """Setup optimizer with weight decay"""
        training_config = self.config["training"]

        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_config.get("weight_decay", 0.01),
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=training_config["learning_rate"], betas=(0.9, 0.999), eps=1e-8
        )

    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        training_config = self.config["training"]

        num_training_steps = (len(self.train_dataloader) * training_config["num_train_epochs"]) // training_config.get(
            "gradient_accumulation_steps", 1
        )

        num_warmup_steps = int(num_training_steps * training_config.get("warmup_ratio", 0.03))

        scheduler_type = training_config.get("lr_scheduler_type", "cosine")

        if scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )

    def train(self):
        """Main training loop"""
        training_config = self.config["training"]

        # Initialize wandb
        if self.accelerator.is_main_process:
            wandb.init(
                project=self.config["wandb"]["project"],
                entity=self.config["wandb"].get("entity"),
                name=self.config["wandb"].get("name"),
                tags=self.config["wandb"].get("tags", []),
                config=self.config,
            )

        global_step = 0
        best_eval_loss = float("inf")

        for epoch in range(training_config["num_train_epochs"]):
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{training_config['num_train_epochs']}",
                disable=not self.accelerator.is_local_main_process,
            )

            for step, batch in enumerate(progress_bar):
                # Forward pass
                with self.accelerator.autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
                    )
                    loss = outputs.loss / training_config.get("gradient_accumulation_steps", 1)

                # Backward pass
                self.accelerator.backward(loss)

                # Gradient accumulation
                if (step + 1) % training_config.get("gradient_accumulation_steps", 1) == 0:
                    # Gradient clipping
                    if training_config.get("max_grad_norm"):
                        self.accelerator.clip_grad_norm_(self.model.parameters(), training_config["max_grad_norm"])

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1

                    # Logging
                    if global_step % 10 == 0 and self.accelerator.is_main_process:
                        wandb.log(
                            {
                                "train/loss": loss.item() * training_config.get("gradient_accumulation_steps", 1),
                                "train/learning_rate": self.scheduler.get_last_lr()[0],
                                "train/epoch": epoch,
                                "train/global_step": global_step,
                            }
                        )

                    # Evaluation
                    if global_step % training_config.get("eval_steps", 500) == 0:
                        eval_loss = self.evaluate()

                        if self.accelerator.is_main_process:
                            wandb.log(
                                {
                                    "eval/loss": eval_loss,
                                    "eval/perplexity": np.exp(eval_loss),
                                    "eval/global_step": global_step,
                                }
                            )

                            # Save best model
                            if eval_loss < best_eval_loss:
                                best_eval_loss = eval_loss
                                self.save_model(os.path.join(training_config["output_dir"], "best_model"))

                    # Save checkpoint
                    if global_step % training_config.get("save_steps", 1000) == 0:
                        self.save_model(os.path.join(training_config["output_dir"], f"checkpoint-{global_step}"))

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item() * training_config.get("gradient_accumulation_steps", 1)})

            # End of epoch evaluation
            eval_loss = self.evaluate()
            if self.accelerator.is_main_process:
                print(
                    f"Epoch {epoch + 1} - Train Loss: {epoch_loss/len(self.train_dataloader):.4f}, \
                      Eval Loss: {eval_loss:.4f}"
                )

        # Save final model
        self.save_model(os.path.join(training_config["output_dir"], "final_model"))

        if self.accelerator.is_main_process:
            wandb.finish()

    def evaluate(self):
        """Evaluation loop"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(
                self.eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_local_main_process
            ):
                with self.accelerator.autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
                    )
                    loss = outputs.loss

                total_loss += loss.item()

        avg_loss = total_loss / len(self.eval_dataloader)
        self.model.train()

        return avg_loss

    def save_model(self, output_dir: str):
        """Save model and tokenizer"""
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

            # Unwrap model if using FSDP or DDP
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            # Save model
            unwrapped_model.save_pretrained(
                output_dir, save_function=self.accelerator.save, state_dict=self.accelerator.get_state_dict(self.model)
            )

            # Save tokenizer
            self.tokenizer.save_pretrained(output_dir)

            print(f"Model saved to {output_dir}")
