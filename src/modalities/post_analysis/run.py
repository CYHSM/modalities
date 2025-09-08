#!/usr/bin/env python3
"""
Main script to run model comparison and analysis
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from weight import WeightComparator
from activation import ActivationComparator
from plot import ComparisonPlotter


@dataclass
class ComparisonConfig:
    """Configuration for model comparison"""
    base_model_path: str
    finetuned_model_path: str
    output_dir: str = "./comparison_results"
    device: str = "cuda"
    batch_size: int = 1
    max_samples: int = 100
    max_length: int = 512
    compare_weights: bool = True
    compare_activations: bool = True


def main():
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned models")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to base model")
    parser.add_argument("--finetuned_model", type=str, required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="./comparison_results",
                       help="Directory to save results")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of GSM8K samples to process")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--skip_weights", action="store_true",
                       help="Skip weight comparison")
    parser.add_argument("--skip_activations", action="store_true",
                       help="Skip activation comparison")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only run analysis on existing results")
    parser.add_argument("--plot_only", action="store_true",
                       help="Only generate plots from existing results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = ComparisonConfig(
        base_model_path=args.base_model,
        finetuned_model_path=args.finetuned_model,
        output_dir=args.output_dir,
        device=args.device,
        max_samples=args.max_samples,
        compare_weights=not args.skip_weights,
        compare_activations=not args.skip_activations
    )
    
    if not args.analyze_only and not args.plot_only:
        # Run comparison
        print("="*60)
        print("🔍 STARTING MODEL COMPARISON")
        print("="*60)
        print(f"Base model: {args.base_model}")
        print(f"Fine-tuned model: {args.finetuned_model}")
        print(f"Output directory: {args.output_dir}")
        print(f"Max samples: {args.max_samples}")
        print("="*60)
        
        # Run weight comparison
        if config.compare_weights:
            print("\n📊 Running weight comparison...")
            weight_comparator = WeightComparator(config)
            weight_comparator.compare_weights()
        
        # Run activation comparison  
        if config.compare_activations:
            print("\n📊 Running activation comparison...")
            activation_comparator = ActivationComparator(config)
            activation_comparator.compare_activations()
        
        print("\n" + "="*60)
        print("✅ COMPARISON COMPLETE")
        print("="*60)
    
    # Generate plots and analysis
    if output_dir.exists():
        print("\n" + "="*60)
        print("📈 GENERATING PLOTS AND ANALYSIS")
        print("="*60)
        
        plotter = ComparisonPlotter(args.output_dir)
        plotter.plot_all_analyses()
    else:
        print(f"❌ No results found in {args.output_dir}")


if __name__ == "__main__":
    main()