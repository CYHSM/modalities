import h5py
import time
import numpy as np
from pathlib import Path
from transformers import TrainerCallback
import torch
import gc
import re

class Monitor(TrainerCallback):
    def __init__(self, output_dir: str, tokenizer, max_steps: int = 10, 
                 track_layers=None, memory_efficient=True):
        print("🔍 Monitor callback initialized!")
        self.output_dir = Path(output_dir) / "dynamics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.step_count = 0
        self.max_steps = max_steps
        self.memory_efficient = memory_efficient
        
        # Only track specific layers to save memory
        self.track_layers = track_layers or [
            "embed_tokens", "lm_head", 
            "layers.0.", "layers.1.", "layers.2."  # First few layers
        ]
        
        # Storage for captured data
        self.current_batch_tokens = None
        self.current_batch_text = None
        self.captured_gradients = {}
        self.prev_step_weights = {}
        self.loss_value = 0.0
        
        print(f"📁 Will save to: {self.output_dir}")
        print(f"🎯 Tracking layers: {self.track_layers}")
        
    def _should_track_param(self, name):
        """Check if we should track this parameter"""
        if not self.track_layers:
            return True
        return any(layer in name for layer in self.track_layers)
    
    def _clean_name_for_path(self, name):
        """Convert parameter name to filesystem-safe path"""
        # Replace dots with underscores, remove special chars
        clean_name = re.sub(r'[^\w\-_.]', '_', name)
        clean_name = clean_name.replace('.', '_')
        return clean_name
    
    def _group_params_by_layer(self, param_dict):
        """Group parameters by their layer/component"""
        layer_groups = {}
        
        for param_name, param_data in param_dict.items():
            # Extract the main component/layer name
            if "embed_tokens" in param_name:
                layer_key = "embed_tokens"
            elif "lm_head" in param_name:
                layer_key = "lm_head"
            elif "layers." in param_name:
                # Extract layer number: layers.0.attention.q_proj.weight -> layers_0_attention
                parts = param_name.split('.')
                if len(parts) >= 3:
                    layer_key = f"{parts[0]}_{parts[1]}_{parts[2]}"  # e.g., "layers_0_attention"
                else:
                    layer_key = f"{parts[0]}_{parts[1]}"  # e.g., "layers_0"
            else:
                # For any other components, use the first part
                layer_key = param_name.split('.')[0]
            
            if layer_key not in layer_groups:
                layer_groups[layer_key] = {}
            layer_groups[layer_key][param_name] = param_data
        
        return layer_groups
        
    def on_train_begin(self, args, state, control, **kwargs):
        print("🚀 Training started - callback working!")
        
        model = kwargs.get("model")
        if model is not None:
            # Register backward hooks to capture gradients BEFORE they're cleared
            self.hook_handles = []
            
            for name, param in model.named_parameters():
                if param.requires_grad and self._should_track_param(name):
                    def make_grad_hook(param_name):
                        def grad_hook(grad):
                            if grad is not None:
                                # Store gradient immediately when computed
                                self.captured_gradients[param_name] = grad.clone().detach()
                            return grad
                        return grad_hook
                    
                    handle = param.register_hook(make_grad_hook(name))
                    self.hook_handles.append(handle)
            
            print(f"🎯 Registered gradient hooks for {len(self.hook_handles)} parameters")
            
            # Forward hook to capture batch data
            def forward_hook(module, input_data, output):
                try:
                    if len(input_data) > 0 and hasattr(input_data[0], 'shape'):
                        input_tensor = input_data[0]
                        if len(input_tensor.shape) >= 2:
                            # Only store first item to save memory
                            self.current_batch_tokens = input_tensor[0:1].cpu().clone()
                            try:
                                self.current_batch_text = self.tokenizer.decode(
                                    input_tensor[0].cpu(), skip_special_tokens=True
                                )
                            except:
                                self.current_batch_text = None
                except Exception as e:
                    print(f"⚠️ Forward hook error: {e}")
            
            if hasattr(model, 'model'):
                model.model.register_forward_hook(forward_hook)
            else:
                model.register_forward_hook(forward_hook)
        
    def on_step_begin(self, args, state, control, **kwargs):
        if self.step_count >= self.max_steps:
            return
            
        model = kwargs.get("model")
        
        # Store only tracked weights to save memory
        if self.memory_efficient:
            self.prev_step_weights = {}
            for name, param in model.named_parameters():
                if self._should_track_param(name):
                    self.prev_step_weights[name] = param.data.clone().cpu()  # Move to CPU
        else:
            self.prev_step_weights = {
                name: param.data.clone()
                for name, param in model.named_parameters()
                if self._should_track_param(name)
            }
        
        print(f"💾 Stored {len(self.prev_step_weights)} parameter tensors")
        
        # Clear previous gradients
        self.captured_gradients = {}
        
        # Force garbage collection
        if self.memory_efficient:
            gc.collect()
            torch.cuda.empty_cache()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture loss from logs"""
        if logs:
            # Try different loss keys
            self.loss_value = (
                logs.get("train_loss") or
                logs.get("loss") or  
                logs.get("train/loss") or
                logs.get("training_loss") or
                0.0
            )
            print(f"📈 Captured loss: {self.loss_value:.4f}")

    def on_step_end(self, args, state, control, **kwargs):
        """Save all captured data to separate HDF5 files organized by step and layer"""
        if self.step_count >= self.max_steps:
            return
            
        model = kwargs.get("model")
        
        print(f"💾 Saving step {state.global_step}")
        print(f"🎯 Gradients captured: {len(self.captured_gradients)}")
        
        try:
            # Create step directory
            step_dir = self.output_dir / f"step_{state.global_step:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate weight deltas
            weight_deltas = {}
            if self.prev_step_weights:
                for name, param in model.named_parameters():
                    if name in self.prev_step_weights:
                        prev_weight = self.prev_step_weights[name]
                        if self.memory_efficient:
                            prev_weight = prev_weight.to(param.device)
                        
                        # Ensure both tensors have same dtype for subtraction
                        current_weight = param.data
                        if current_weight.dtype == torch.bfloat16:
                            current_weight = current_weight.float()
                        if prev_weight.dtype == torch.bfloat16:
                            prev_weight = prev_weight.float()
                            
                        weight_deltas[name] = (current_weight - prev_weight).cpu()
                print(f"📈 Calculated {len(weight_deltas)} weight deltas")
            
            # Save step metadata and batch data
            metadata_file = step_dir / "metadata.h5"
            with h5py.File(metadata_file, "w") as f:
                f.attrs["step"] = state.global_step
                f.attrs["loss"] = float(self.loss_value)
                f.attrs["timestamp"] = time.time()
                f.attrs["n_gradients"] = len(self.captured_gradients)
                f.attrs["n_weight_deltas"] = len(weight_deltas)
                
                # Save batch data (first item only)
                if self.current_batch_text:
                    f.create_dataset("batch_text", data=self.current_batch_text)
                if self.current_batch_tokens is not None:
                    f.create_dataset("batch_tokens", 
                                    data=self.current_batch_tokens.numpy(),
                                    compression="gzip")
            
            # Group gradients by layer and save
            if self.captured_gradients:
                gradient_groups = self._group_params_by_layer(self.captured_gradients)
                
                for layer_name, layer_grads in gradient_groups.items():
                    layer_dir = step_dir / self._clean_name_for_path(layer_name)
                    layer_dir.mkdir(parents=True, exist_ok=True)
                    
                    grad_file = layer_dir / "gradients.h5"
                    with h5py.File(grad_file, "w") as f:
                        f.attrs["layer_name"] = layer_name
                        f.attrs["n_parameters"] = len(layer_grads)
                        
                        for param_name, grad in layer_grads.items():
                            dataset_name = param_name.replace(".", "/")
                            f.create_dataset(
                                dataset_name,
                                data=grad.cpu().to(torch.float32).numpy(),
                                compression="gzip"
                            )
                            # Add parameter metadata
                            f[dataset_name].attrs["original_name"] = param_name
                            f[dataset_name].attrs["shape"] = grad.shape
            
            # Group weight deltas by layer and save
            if weight_deltas:
                delta_groups = self._group_params_by_layer(weight_deltas)
                
                for layer_name, layer_deltas in delta_groups.items():
                    layer_dir = step_dir / self._clean_name_for_path(layer_name)
                    layer_dir.mkdir(parents=True, exist_ok=True)
                    
                    delta_file = layer_dir / "weight_deltas.h5"
                    with h5py.File(delta_file, "w") as f:
                        f.attrs["layer_name"] = layer_name
                        f.attrs["n_parameters"] = len(layer_deltas)
                        
                        for param_name, delta in layer_deltas.items():
                            dataset_name = param_name.replace(".", "/")
                            f.create_dataset(
                                dataset_name,
                                data=delta.cpu().to(torch.float32).numpy(),
                                compression="gzip"
                            )
                            # Add parameter metadata
                            f[dataset_name].attrs["original_name"] = param_name
                            f[dataset_name].attrs["shape"] = delta.shape
            
            print(f"✅ Saved step {state.global_step} to {step_dir}")
            print(f"   📁 Created {len(gradient_groups) if self.captured_gradients else 0} gradient files")
            print(f"   📁 Created {len(delta_groups) if weight_deltas else 0} weight delta files")
            
        except Exception as e:
            print(f"❌ Error saving step {state.global_step}: {e}")
            import traceback
            traceback.print_exc()
        
        self.step_count += 1
        
        # Memory cleanup
        self.captured_gradients = {}
        self.current_batch_tokens = None
        self.current_batch_text = None
        
        if self.memory_efficient:
            gc.collect()
            torch.cuda.empty_cache()
        
        # Stop after max_steps
        if self.step_count >= self.max_steps:
            print(f"🏁 Stopping training after {self.max_steps} steps")
            control.should_training_stop = True
    
    def on_train_end(self, args, state, control, **kwargs):
        """Cleanup hooks"""
        print("🧹 Cleaning up hooks")
        if hasattr(self, 'hook_handles'):
            for handle in self.hook_handles:
                handle.remove()

# Utility functions to explore the split file structure
def explore_dynamics_directory(dynamics_dir):
    """Print structure of the split dynamics files"""
    dynamics_path = Path(dynamics_dir)
    
    if not dynamics_path.exists():
        print(f"Directory not found: {dynamics_path}")
        return
    
    print(f"Training Dynamics Directory: {dynamics_path}")
    print("=" * 50)
    
    # List all step directories
    step_dirs = sorted([d for d in dynamics_path.iterdir() if d.is_dir() and d.name.startswith('step_')])
    
    for step_dir in step_dirs:
        print(f"\n📁 {step_dir.name}/")
        
        # Show metadata
        metadata_file = step_dir / "metadata.h5"
        if metadata_file.exists():
            with h5py.File(metadata_file, "r") as f:
                print(f"  📊 metadata.h5")
                print(f"     Step: {f.attrs.get('step', 'N/A')}")
                print(f"     Loss: {f.attrs.get('loss', 'N/A'):.4f}")
                print(f"     Gradients: {f.attrs.get('n_gradients', 'N/A')}")
                print(f"     Weight Deltas: {f.attrs.get('n_weight_deltas', 'N/A')}")
                
                if 'batch_tokens' in f:
                    print(f"     Batch tokens shape: {f['batch_tokens'].shape}")
                if 'batch_text' in f:
                    text_preview = f['batch_text'][()].decode() if isinstance(f['batch_text'][()], bytes) else str(f['batch_text'][()])
                    print(f"     Batch text: {text_preview[:50]}...")
        
        # Show layer directories
        layer_dirs = sorted([d for d in step_dir.iterdir() if d.is_dir()])
        for layer_dir in layer_dirs:
            print(f"  📁 {layer_dir.name}/")
            
            # Show files in layer directory
            for file_path in sorted(layer_dir.glob("*.h5")):
                with h5py.File(file_path, "r") as f:
                    print(f"     📄 {file_path.name}")
                    print(f"        Layer: {f.attrs.get('layer_name', 'N/A')}")
                    print(f"        Parameters: {f.attrs.get('n_parameters', 'N/A')}")
                    
                    # Show first few datasets
                    datasets = list(f.keys())[:3]  # Show first 3
                    for ds_name in datasets:
                        ds = f[ds_name]
                        print(f"        - {ds_name}: {ds.shape} {ds.dtype}")
                    
                    if len(f.keys()) > 3:
                        print(f"        - ... and {len(f.keys()) - 3} more")

def load_layer_data(dynamics_dir, step, layer_name, data_type="gradients"):
    """
    Load specific layer data from split files
    
    Args:
        dynamics_dir: Path to dynamics directory
        step: Step number
        layer_name: Layer name (e.g., 'layers_0_attention')  
        data_type: Either 'gradients' or 'weight_deltas'
    """
    dynamics_path = Path(dynamics_dir)
    step_dir = dynamics_path / f"step_{step:04d}"
    layer_dir = step_dir / layer_name
    
    data_file = layer_dir / f"{data_type}.h5"
    
    if not data_file.exists():
        print(f"File not found: {data_file}")
        return None
    
    data = {}
    with h5py.File(data_file, "r") as f:
        print(f"Loading {data_type} for {layer_name} at step {step}")
        print(f"Layer: {f.attrs.get('layer_name', 'N/A')}")
        print(f"Parameters: {f.attrs.get('n_parameters', 'N/A')}")
        
        def load_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                original_name = obj.attrs.get('original_name', name.replace('/', '.'))
                data[original_name] = obj[()]
                print(f"  Loaded {original_name}: {obj.shape}")
        
        f.visititems(load_dataset)
    
    return data

def get_step_summary(dynamics_dir, step):
    """Get summary information for a specific step"""
    dynamics_path = Path(dynamics_dir)
    step_dir = dynamics_path / f"step_{step:04d}"
    metadata_file = step_dir / "metadata.h5"
    
    if not metadata_file.exists():
        print(f"Metadata file not found for step {step}")
        return None
    
    summary = {}
    with h5py.File(metadata_file, "r") as f:
        summary['step'] = f.attrs.get('step', step)
        summary['loss'] = f.attrs.get('loss', 0.0)
        summary['timestamp'] = f.attrs.get('timestamp', 0)
        summary['n_gradients'] = f.attrs.get('n_gradients', 0)
        summary['n_weight_deltas'] = f.attrs.get('n_weight_deltas', 0)
        
        if 'batch_text' in f:
            text = f['batch_text'][()]
            if isinstance(text, bytes):
                text = text.decode()
            summary['batch_text'] = text
        
        if 'batch_tokens' in f:
            summary['batch_tokens'] = f['batch_tokens'][()]
    
    # Count layer directories
    layer_dirs = [d for d in step_dir.iterdir() if d.is_dir()]
    summary['n_layers'] = len(layer_dirs)
    summary['layers'] = [d.name for d in layer_dirs]
    
    return summary