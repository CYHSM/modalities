import os
import sys
from ruamel.yaml import YAML

# Initialize YAML handler
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

# Paths configuration
BASE_YAML_PATH = '/capstor/store/cscs/swissai/a0164/markusfrey/modalities/config_files/neurips/midapril2026_cscs/configs_d768_loop3/d768_dense_isoflop_60L.yaml' 
TARGET_DIRECTORIES = [
    '/capstor/store/cscs/swissai/a0164/markusfrey/modalities/config_files/neurips/midapril2026_cscs/configs_d768_loop3',
    '/capstor/store/cscs/swissai/a0164/markusfrey/modalities/config_files/neurips/midapril2026_cscs/configs_d768_loop5',
    '/capstor/store/cscs/swissai/a0164/markusfrey/modalities/config_files/neurips/midapril2026_cscs/configs_d1280_loop3',
    '/capstor/store/cscs/swissai/a0164/markusfrey/modalities/config_files/neurips/midapril2026_cscs/configs_d1280_loop5'
]

def update_yamls():
    # 1. Load the Base Data
    if not os.path.exists(BASE_YAML_PATH):
        print(f"Error: Base file {BASE_YAML_PATH} not found.")
        return

    with open(BASE_YAML_PATH, 'r') as f:
        base_data = yaml.load(f)

    # 2. Extract the specific parts you need
    updates = {
        "saving_path": base_data['settings']['paths']['checkpoint_saving_path'],
        "root_path": base_data['settings']['paths']['experiments_root_path'],
        "grad_acc": base_data['settings']['step_profile']['gradient_accumulation_steps'],
        "batch_size": base_data['settings']['step_profile']['local_train_micro_batch_size'],
        "train_dataset": base_data['train_dataset'],
        "eval_dataset": base_data['eval_dataset'],
        "tokenizer": base_data['tokenizer']
    }

    # 3. Iterate through directories
    for directory in TARGET_DIRECTORIES:
        if not os.path.exists(directory):
            print(f"Skipping {directory}, path does not exist.")
            continue

        for filename in os.listdir(directory):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                # Avoid overwriting the base if it's in the same folder
                file_path = os.path.join(directory, filename)
                if os.path.abspath(file_path) == os.path.abspath(BASE_YAML_PATH):
                    continue

                print(f"Updating: {file_path}")
                
                with open(file_path, 'r') as f:
                    target_data = yaml.load(f)

                # Apply updates to the nested structure
                target_data['settings']['paths']['checkpoint_saving_path'] = updates['saving_path']
                target_data['settings']['paths']['experiments_root_path'] = updates['root_path']
                target_data['settings']['step_profile']['gradient_accumulation_steps'] = updates['grad_acc']
                target_data['settings']['step_profile']['local_train_micro_batch_size'] = updates['batch_size']
                
                # Full replacements for top-level keys
                target_data['train_dataset'] = updates['train_dataset']
                target_data['eval_dataset'] = updates['eval_dataset']
                target_data['tokenizer'] = updates['tokenizer']

                # Save the file back
                with open(file_path, 'w') as f:
                    yaml.dump(target_data, f)

if __name__ == "__main__":
    update_yamls()
    print("\nDone! All YAML files updated.")