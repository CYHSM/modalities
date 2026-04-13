import os
import re

def update_yaml_configs(config_list_file="configs.txt"):
    # The base path to prepend to each file in the text file
    base_path = "/leonardo_work/EUHPC_D21_101/mfrey/modalities/"
    
    # 1. Read the list of YAML files from the text file
    if not os.path.exists(config_list_file):
        print(f"❌ Error: Could not find the file '{config_list_file}'")
        return

    with open(config_list_file, 'r', encoding='utf-8') as f:
        # Read lines, strip whitespace, ignore empty lines, and construct the full path
        yaml_files = []
        for line in f:
            clean_line = line.strip()
            if clean_line:
                # This safely joins the base path and the relative path
                full_path = os.path.join(base_path, clean_line)
                yaml_files.append(full_path)

    if not yaml_files:
        print(f"⚠️ '{config_list_file}' is empty. Nothing to do.")
        return

    print(f"Found {len(yaml_files)} files listed in {config_list_file}. Starting update...\n")

    # 2. Process each file
    success_count = 0
    missing_files = []

    for filepath in yaml_files:
        # Check if file exists before trying to read it
        if not os.path.exists(filepath):
            missing_files.append(filepath)
            continue

        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()

        # Use regex to find the keys and replace whatever number is currently there 
        # with the new desired values (4 and 2), preserving the original spacing.
        content = re.sub(
            r'(gradient_accumulation_steps:\s*)\d+', 
            r'\g<1>4', 
            content
        )
        content = re.sub(
            r'(local_train_micro_batch_size:\s*)\d+', 
            r'\g<1>2', 
            content
        )

        # Write the updated content back to the file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"✅ Updated: {filepath}")
        success_count += 1

    # 3. Print the results
    print("\n--- Summary ---")
    print(f"Successfully updated {success_count} files.")
    
    if missing_files:
        print(f"⚠️ Could not find {len(missing_files)} files on disk:")
        for missing in missing_files:
            print(f"  - {missing}")

if __name__ == "__main__":
    update_yaml_configs("configs_list_full.txt")