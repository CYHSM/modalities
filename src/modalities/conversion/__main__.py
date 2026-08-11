import argparse
import sys
from pathlib import Path

from modalities.conversion.generic_adapter import export_generic_hf_adapter


def main():
    parser = argparse.ArgumentParser(description="Convert Modalities checkpoint to Hugging Face format.")
    parser.add_argument("modalities_config", type=str, help="Path to Modalities YAML config file.")
    parser.add_argument("output_dir", type=str, help="Output directory for Hugging Face checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to Modalities checkpoint directory/file.")
    parser.add_argument(
        "--mode",
        type=str,
        default="generic",
        choices=["generic", "gpt2"],
        help="Conversion mode: 'generic' uses HF auto_map adapter (works for all models), 'gpt2' converts to native HF GPT2LMHeadModel.",
    )
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Directory containing tokenizer files.")

    args = parser.parse_args()

    modalities_config_path = Path(args.modalities_config)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint_path)

    if args.mode == "generic":
        print(f"Exporting checkpoint via generic HF adapter to {output_dir}...")
        export_generic_hf_adapter(
            checkpoint_file_path=checkpoint_path,
            modalities_config_path=modalities_config_path,
            output_dir=output_dir,
            tokenizer_dir=args.tokenizer_dir,
        )
        print("Export completed successfully.")
    elif args.mode == "gpt2":
        from modalities.conversion.gpt2.convert_gpt2 import convert_checkpoint
        print(f"Exporting standard GPT-2 checkpoint to {output_dir}...")
        convert_checkpoint(
            modalities_config_path=modalities_config_path,
            checkpoint_file_path=checkpoint_path,
            output_dir=output_dir,
        )
        print("Export completed successfully.")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
