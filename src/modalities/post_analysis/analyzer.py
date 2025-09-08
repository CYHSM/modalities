"""
Main analysis coordinator for model comparison results
"""

from pathlib import Path
from weight_analyzer import WeightAnalyzer
from activation_analyzer import ActivationAnalyzer


class ComparisonAnalyzer:
    """Main analyzer that coordinates weight and activation analysis"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
        # Initialize sub-analyzers
        self.weight_analyzer = WeightAnalyzer(results_dir)
        self.activation_analyzer = ActivationAnalyzer(results_dir)
        
        print(f"Initialized analyzer for results in: {self.results_dir}")
    
    def analyze_weights(self):
        """Run weight analysis"""
        return self.weight_analyzer.analyze_weights()
    
    def analyze_activations(self):
        """Run activation analysis"""
        return self.activation_analyzer.analyze_activations()
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting comprehensive model comparison analysis...\n")
        
        weight_df = None
        activation_df = None
        
        # Run weight analysis if data exists
        if self.weight_analyzer.weight_results:
            print("📊 Running weight analysis...")
            weight_df = self.analyze_weights()
        else:
            print("⚠️  No weight comparison data found - skipping weight analysis")
        
        print()  # spacing
        
        # Run activation analysis if data exists
        if self.activation_analyzer.activation_results:
            print("📊 Running activation analysis...")
            activation_df = self.analyze_activations()
        else:
            print("⚠️  No activation comparison data found - skipping activation analysis")
        
        # Print overall summary
        self._print_overall_summary(weight_df, activation_df)
        
        print("\n✅ Analysis complete!")
        
        return {
            'weight_analysis': weight_df,
            'activation_analysis': activation_df
        }
    
    def _print_overall_summary(self, weight_df, activation_df):
        """Print overall comparison summary"""
        print("\n" + "="*60)
        print("OVERALL COMPARISON SUMMARY")
        print("="*60)
        
        if weight_df is not None and not weight_df.empty:
            print(f"Weight Analysis:")
            print(f"  • {len(weight_df)} layers analyzed")
            print(f"  • Median cosine similarity: {weight_df['cosine_similarity'].median():.4f}")
            print(f"  • Average parameter change: {weight_df['abs_mean_diff'].mean():.6f}")
            
            # Most changed layer types
            type_changes = weight_df.groupby('layer_type')['abs_mean_diff'].mean().sort_values(ascending=False)
            print(f"  • Most changed layer type: {type_changes.index[0]} ({type_changes.iloc[0]:.6f})")
        
        if activation_df is not None and not activation_df.empty:
            print(f"\nActivation Analysis:")
            print(f"  • {len(activation_df)} layers analyzed")
            print(f"  • Median cosine similarity: {activation_df['cosine_similarity'].median():.4f}")
            print(f"  • Average activation change: {activation_df['abs_mean_diff'].mean():.6f}")
            
            # Most changed layer types
            type_changes = activation_df.groupby('layer_type')['abs_mean_diff'].mean().sort_values(ascending=False)
            print(f"  • Most changed layer type: {type_changes.index[0]} ({type_changes.iloc[0]:.6f})")
        
        if weight_df is not None and activation_df is not None and not weight_df.empty and not activation_df.empty:
            print(f"\nCross-Analysis Insights:")
            # Find layers that changed a lot in both weights and activations
            common_layers = set(weight_df['layer']) & set(activation_df['layer'])
            if common_layers:
                weight_changes = weight_df[weight_df['layer'].isin(common_layers)].set_index('layer')['abs_mean_diff']
                activation_changes = activation_df[activation_df['layer'].isin(common_layers)].set_index('layer')['abs_mean_diff']
                
                # Correlation between weight and activation changes
                correlation = weight_changes.corr(activation_changes)
                print(f"  • Correlation between weight and activation changes: {correlation:.3f}")
                
                if correlation > 0.3:
                    print("    → Strong positive correlation - weight changes drive activation changes")
                elif correlation < -0.3:
                    print("    → Strong negative correlation - interesting inverse relationship")
                else:
                    print("    → Weak correlation - weight and activation changes are largely independent")


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model comparison results")
    parser.add_argument("--results_dir", type=str, default="./comparison_results",
                       help="Directory containing comparison results")
    
    args = parser.parse_args()
    
    analyzer = ComparisonAnalyzer(args.results_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()