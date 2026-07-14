#!/usr/bin/env python3
import sys
import yaml

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 create_warmstart_config.py <input_config.yaml> <output_config.yaml>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)
        
    # 1. Add warmstart_checkpoint_paths to settings
    if "settings" not in data:
        data["settings"] = {}
    data["settings"]["warmstart_checkpoint_paths"] = "${warmstart_env:checkpoint_paths}"
    
    # 2. Update training_progress in settings
    if "training_progress" not in data["settings"]:
        data["settings"]["training_progress"] = {}
        
    data["settings"]["training_progress"]["global_num_seen_tokens"] = {
        "component_key": "number_conversion",
        "variant_key": "global_num_seen_tokens_from_checkpoint_path",
        "config": {
            "checkpoint_path": "${settings.warmstart_checkpoint_paths.checkpoint_folder_path}"
        }
    }
    data["settings"]["training_progress"]["num_seen_steps"] = {
        "component_key": "number_conversion",
        "variant_key": "num_seen_steps_from_checkpoint_path",
        "config": {
            "checkpoint_path": "${settings.warmstart_checkpoint_paths.checkpoint_folder_path}"
        }
    }
    data["settings"]["training_progress"]["num_seen_samples"] = {
        "component_key": "number_conversion",
        "variant_key": "num_samples_from_num_tokens",
        "config": {
            "num_tokens": "${settings.training_progress.global_num_seen_tokens}",
            "sequence_length": "${settings.step_profile.sequence_length}"
        }
    }
    data["settings"]["training_progress"]["last_step"] = {
        "component_key": "number_conversion",
        "variant_key": "last_step_from_checkpoint_path",
        "config": {
            "checkpoint_path": "${settings.warmstart_checkpoint_paths.checkpoint_folder_path}"
        }
    }
    
    # 3. Add checkpoint_loading component
    data["checkpoint_loading"] = {
        "component_key": "checkpoint_loading",
        "variant_key": "dcp",
        "config": {
            "global_rank": "${settings.cuda_env.global_rank}"
        }
    }
    
    # 4. Rename and configure app_state to load checkpoint
    if "app_state" in data:
        data["app_state_raw"] = data["app_state"]
        
    data["app_state"] = {
        "component_key": "app_state",
        "variant_key": "dcp",
        "config": {
            "raw_app_state": {
                "instance_key": "app_state_raw",
                "pass_type": "BY_REFERENCE"
            },
            "checkpoint_dir_path": "${settings.warmstart_checkpoint_paths.checkpoint_folder_path}"
        }
    }
    
    # 5. Write output
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
    print(f"Warmstart config successfully written to {output_path}")

if __name__ == "__main__":
    main()
