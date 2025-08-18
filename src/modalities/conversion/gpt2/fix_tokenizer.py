import json
import sys
from pathlib import Path


def fix_tokenizer_special_tokens(model_dir: str) -> None:
    """
    Fixes the tokenizer special tokens after modalities conversion.

    Args:
        model_dir: Path to the converted model directory
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"

    with open(config_path, "r") as f:
        model_config = json.load(f)
    with open(tokenizer_config_path, "r") as f:
        tokenizer_config = json.load(f)

    # Get token IDs from model config
    bos_id = model_config.get("bos_token_id")
    eos_id = model_config.get("eos_token_id")
    pad_id = model_config.get("pad_token_id")

    # Update tokenizer config
    updated = False
    if bos_id is not None:
        tokenizer_config["bos_token_id"] = bos_id
        updated = True
        print(f"Set bos_token_id = {bos_id}")
    if eos_id is not None:
        tokenizer_config["eos_token_id"] = eos_id
        updated = True
        print(f"Set eos_token_id = {eos_id}")
    if pad_id is not None:
        tokenizer_config["pad_token_id"] = pad_id
        updated = True
        print(f"Set pad_token_id = {pad_id}")

    if updated:
        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        print(f"✓ Updated tokenizer config at {tokenizer_config_path}")
    else:
        print("No updates needed - special tokens already set or not found in model config")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_tokenizer_special_tokens.py /path/to/converted/model")
        sys.exit(1)

    model_path = sys.argv[1]
    fix_tokenizer_special_tokens(model_path)
