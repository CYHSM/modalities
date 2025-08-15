#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

import yaml

# Configuration
languages = ["deu_Latn", "eng_Latn", "fra_Latn", "ita_Latn", "spa_Latn", "nob_Latn"]
splits = ["train", "val"]
base_data_path = Path("/raid/s3/opengptx/mfrey/fineweb-30B")
base_index_path = Path("data/preprocessed")
temp_config_path = Path("configs/tokenization_config_temp.yaml")


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"  Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {description}")
        print(f"  Error: {e.stderr}")
        sys.exit(1)


def create_index(lang, split):
    """Create raw index if it doesn't exist"""
    index_path = base_index_path / f"{lang}_{split}.idx"
    data_path = base_data_path / f"{lang}_{split}.jsonl"

    if index_path.exists():
        print("  Index exists - skipping")
        return

    cmd = ["modalities", "data", "create_raw_index", "--index_path", str(index_path), str(data_path)]
    run_command(cmd, f"Created index for {lang}_{split}")


def pack_data(lang, split):
    """Pack encoded data if output doesn't exist"""
    dst_path = base_data_path / f"packed/{lang}_{split}.pbin"

    if dst_path.exists():
        print("  Packed data exists - skipping")
        return

    # Create config
    config = {
        "settings": {
            "src_path": str(base_data_path / f"{lang}_{split}.jsonl"),
            "dst_path": str(dst_path),
            "index_path": str(base_index_path / f"{lang}_{split}.idx"),
            "jq_pattern": ".text",
            "num_cpus": "${node_env:num_cpus}",
            "eod_token": "</s>",
            "processing_batch_size": 10,
            "raw_samples_queue_size": 300,
            "processed_samples_queue_size": 300,
        },
        "tokenizer": {
            "component_key": "tokenizer",
            "variant_key": "pretrained_sp_tokenizer",
            "config": {
                "tokenizer_model_file": "/home/markus_frey/Github/modalities/tutorials/2b_fineweb/tokenizer/eurolingua_tokenizer.model"  # noqa: E501
            },
        },
    }

    # Write config and run
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    cmd = ["modalities", "data", "pack_encoded_data", str(temp_config_path)]
    run_command(cmd, f"Packed data for {lang}_{split}")


def main():
    for lang in languages:
        for split in splits:
            print(f"Processing {lang}_{split}...")
            create_index(lang, split)
            pack_data(lang, split)

    # Cleanup
    if temp_config_path.exists():
        temp_config_path.unlink()
    print("All processing complete!")


if __name__ == "__main__":
    main()
