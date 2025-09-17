import os

import torch

from modalities.post_sft.config import LoRAConfig
from modalities.post_sft.model_utils import load_model_and_tokenizer, save_model_with_custom_code

# Your exact config from command line
model_path = "/raid/s3/opengptx/mfrey/instruct/hf_model"
output_dir = "/raid/s3/opengptx/mfrey/instruct/checkpoints_sft/full_baseline"
checkpoint_0_path = os.path.join(output_dir, "checkpoint-0")

print(f"Loading model from: {model_path}")
model, tokenizer = load_model_and_tokenizer(
    model_path,
    device="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    trainable_layers=None,
    lora_config=LoRAConfig(),  # Default LoRA config (disabled)
)

print(f"Saving checkpoint-0 to: {checkpoint_0_path}")
os.makedirs(checkpoint_0_path, exist_ok=True)

# Save model and tokenizer
model.save_pretrained(checkpoint_0_path)
tokenizer.save_pretrained(checkpoint_0_path)

# Save custom code files
save_model_with_custom_code(model, checkpoint_0_path, model_path)

print("✅ Checkpoint-0 saved successfully!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
