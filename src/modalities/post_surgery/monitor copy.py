import h5py
import time
import numpy as np
from pathlib import Path
from transformers import TrainerCallback
import torch

class Monitor(TrainerCallback):
    def __init__(self, output_dir: str, tokenizer, max_steps: int = 10):
        self.output_dir = Path(output_dir) / "dynamics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.previous_weights = {}
        self.step_count = 0
        self.max_steps = max_steps
        self.current_batch_tokens = None
        self.current_batch_text = None
        self.current_gradients = None
        
        # Create the HDF5 file
        self.h5_file = self.output_dir / "training_dynamics.h5"
        
    def on_train_batch_start(self, args, state, control, **kwargs):
        """Capture batch data at the start of training step"""
        if self.step_count >= self.max_steps:
            return
            
        model = kwargs.get("model")
        inputs = kwargs.get("inputs", {})
        
        # Store weights before step
        self.previous_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        
        # Capture current batch
        print(inputs)
        if "input_ids" in inputs:
            self.current_batch_tokens = inputs["input_ids"].cpu().clone()
            try:
                self.current_batch_text = self.tokenizer.decode(
                    inputs["input_ids"][0], skip_special_tokens=True
                )
            except:
                self.current_batch_text = None
    
    def on_train_batch_end(self, args, state, control, **kwargs):
        """Capture gradients right after backward pass but before optimizer step"""
        if self.step_count >= self.max_steps:
            return
            
        model = kwargs.get("model")
        
        # Capture gradients
        self.current_gradients = {
            name: param.grad.clone() if param.grad is not None else None
            for name, param in model.named_parameters()
        }

        # Print for debugging
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Grad norm {name}: {param.grad.norm().item():.2e}")
            else:
                print(f"Grad {name}: None")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Save all captured data to HDF5 after the training step"""
        if self.step_count >= self.max_steps:
            return
            
        model = kwargs.get("model")
        logs = kwargs.get("logs", {})
        
        # Save to HDF5
        with h5py.File(self.h5_file, "a") as f:
            # Create group for this step
            step_group = f.create_group(f"step_{state.global_step:04d}")
            
            # Save metadata as attributes
            step_group.attrs["step"] = state.global_step
            step_group.attrs["loss"] = logs.get("train_loss", 0)
            step_group.attrs["timestamp"] = time.time()
            
            # Save batch text
            if self.current_batch_text:
                step_group.create_dataset("batch_text", data=self.current_batch_text)
            
            # Save batch tokens
            if self.current_batch_tokens is not None:
                step_group.create_dataset("batch_tokens", 
                                        data=self.current_batch_tokens.numpy(),
                                        compression="gzip")
            
            # Save gradients
            if self.current_gradients:
                grad_group = step_group.create_group("gradients")
                for name, grad in self.current_gradients.items():
                    if grad is not None:
                        grad_group.create_dataset(
                            name.replace(".", "/"),  # HDF5 uses / as separator
                            data=grad.cpu().numpy(),
                            compression="gzip"
                        )
            
            # Save weight deltas
            delta_group = step_group.create_group("weight_deltas")
            for name, param in model.named_parameters():
                if name in self.previous_weights:
                    delta = param.data - self.previous_weights[name]
                    delta_group.create_dataset(
                        name.replace(".", "/"),
                        data=delta.cpu().numpy(),
                        compression="gzip"
                    )
            
            # Save current weights (optional, takes more space)
            # weights_group = step_group.create_group("weights")
            # for name, param in model.named_parameters():
            #     weights_group.create_dataset(
            #         name.replace(".", "/"),
            #         data=param.data.cpu().numpy(),
            #         compression="gzip"
            #     )
        
        print(f"Saved dynamics for step {state.global_step} to HDF5")
        self.step_count += 1
        
        # Clean up to save memory
        self.current_gradients = None
        self.current_batch_tokens = None
        self.current_batch_text = None
        
        # Stop after max_steps steps
        if self.step_count >= self.max_steps:
            control.should_training_stop = True