#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

def prepare_new_seed(new_seed: int):
    config_list_path = Path("budapest_configs.txt")
    if not config_list_path.exists():
        print(f"Error: {config_list_path} not found.")
        sys.exit(1)

    # Read original configs
    with open(config_list_path, "r") as f:
        configs = [line.strip() for line in f if line.strip()]

    new_configs = []
    for cfg_path_str in configs:
        cfg_path = Path(cfg_path_str)
        if not cfg_path.exists():
            print(f"Warning: {cfg_path} does not exist. Skipping.")
            continue

        # Determine new config file path
        # e.g., final_budapest/foo.yaml -> final_budapest_seed{new_seed}/foo.yaml
        parent_dir = cfg_path.parent
        new_parent_dir = parent_dir.with_name(f"{parent_dir.name}_seed{new_seed}")
        new_parent_dir.mkdir(parents=True, exist_ok=True)
        new_cfg_path = new_parent_dir / cfg_path.name

        # Read, modify seed, and write
        content = cfg_path.read_text()
        # Replace 'seed: 42' with 'seed: new_seed'
        modified_content = re.sub(r'seed:\s*42', f'seed: {new_seed}', content)
        new_cfg_path.write_text(modified_content)

        new_configs.append(str(new_cfg_path.resolve()))
        print(f"Created: {new_cfg_path}")

    # Write new config list file
    new_config_list_path = Path(f"budapest_configs_seed{new_seed}.txt")
    with open(new_config_list_path, "w") as f:
        for cfg in new_configs:
            f.write(cfg + "\n")
    print(f"\nSuccessfully created new config list file: {new_config_list_path}")
    print(f"To run the array with this new seed, use:")
    print(f"sbatch --export=ALL,CONFIG_LIST={new_config_list_path.name} --array=1-14 run_array_16nodes.sh")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 prepare_new_seed.py <new_seed_integer>")
        sys.exit(1)
    try:
        seed = int(sys.argv[1])
    except ValueError:
        print("Error: Seed must be an integer.")
        sys.exit(1)
    prepare_new_seed(seed)
