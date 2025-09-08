#!/usr/bin/env python3
"""
Script to run model comparison and analysis
"""

from model_compare import ModelComparator, ComparisonConfig
from analyze_results import ComparisonAnalyzer
import argparse
from pathlib import Path

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
    
    args = parser.parse_args()
    
    if not args.analyze_only:
        # Run comparison
        print("="*60)
        print("🔍 STARTING MODEL COMPARISON")
        print("="*60)
        print(f"Base model: {args.base_model}")
        print(f"Fine-tuned model: {args.finetuned_model}")
        print(f"Output directory: {args.output_dir}")
        print(f"Max samples: {args.max_samples}")
        print("="*60)
        
        config = ComparisonConfig(
            base_model_path=args.base_model,
            finetuned_model_path=args.finetuned_model,
            output_dir=args.output_dir,
            device=args.device,
            max_samples=args.max_samples,
            compare_weights=not args.skip_weights,
            compare_activations=not args.skip_activations
        )
        
        comparator = ModelComparator(config)
        results = comparator.run_full_comparison()
        
        print("\n" + "="*60)
        print("✅ COMPARISON COMPLETE")
        print("="*60)
    
    # Run analysis
    if Path(args.output_dir).exists():
        print("\n" + "="*60)
        print("📊 RUNNING ANALYSIS")
        print("="*60)
        
        analyzer = ComparisonAnalyzer(args.output_dir)
        analyzer.run_full_analysis()
    else:
        print(f"❌ No results found in {args.output_dir}")

if __name__ == "__main__":
    main()