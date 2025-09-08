"""
Analysis and Visualization for Model Comparison Results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

class ComparisonAnalyzer:
    """Analyze and visualize model comparison results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.weight_results = None
        self.activation_results = None
        self.sample_results = None
        
        # Load results
        self._load_results()
        
        # Set modern style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams.update({
            'figure.figsize': (16, 12),
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10,
            'figure.titlesize': 15
        })
    
    def _load_results(self):
        """Load saved results from files"""
        weight_file = self.results_dir / "weight_comparison.json"
        if weight_file.exists():
            with open(weight_file, 'r') as f:
                self.weight_results = json.load(f)
            print(f"✓ Loaded weight comparison from {weight_file}")
        
        activation_file = self.results_dir / "activation_comparison.json"
        if activation_file.exists():
            with open(activation_file, 'r') as f:
                self.activation_results = json.load(f)
            print(f"✓ Loaded activation comparison from {activation_file}")
        
        samples_file = self.results_dir / "activation_samples.json"
        if samples_file.exists():
            with open(samples_file, 'r') as f:
                self.sample_results = json.load(f)
            print(f"✓ Loaded sample details from {samples_file}")
    
    def _shorten_layer_name(self, name):
        """Create shorter, readable layer names"""
        # Extract key parts
        parts = name.split('.')
        
        # Find layer number
        layer_num = None
        for part in parts:
            if part.isdigit():
                layer_num = part
                break
        
        # Identify component type
        if 'self_attn' in name:
            if 'q_proj' in name:
                component = 'Q'
            elif 'k_proj' in name:
                component = 'K'
            elif 'v_proj' in name:
                component = 'V'
            elif 'o_proj' in name:
                component = 'O'
            else:
                component = 'Attn'
        elif 'mlp' in name:
            if 'gate_proj' in name:
                component = 'Gate'
            elif 'up_proj' in name:
                component = 'Up'
            elif 'down_proj' in name:
                component = 'Down'
            else:
                component = 'MLP'
        elif any(norm in name for norm in ['norm', 'layernorm']):
            component = 'Norm'
        elif 'embed' in name:
            component = 'Embed'
        elif 'lm_head' in name:
            component = 'Head'
        else:
            component = parts[-1][:8]  # Last part, truncated
        
        if layer_num:
            return f"L{layer_num}.{component}"
        else:
            return component
    
    def _categorize_layers(self, layer_names):
        """Categorize layers by type"""
        layer_types = []
        for name in layer_names:
            if 'self_attn' in name:
                layer_types.append('Attention')
            elif 'mlp' in name:
                layer_types.append('MLP')
            elif any(norm in name for norm in ['norm', 'layernorm']):
                layer_types.append('Normalization')
            elif 'embed' in name:
                layer_types.append('Embedding')
            elif 'lm_head' in name:
                layer_types.append('LM Head')
            else:
                layer_types.append('Other')
        return layer_types
    
    def _get_layer_number(self, name):
        """Extract layer number from layer name"""
        parts = name.split('.')
        for part in parts:
            if part.isdigit():
                return int(part)
        return -1
    
    def _create_unified_plots(self, data_dict, title_prefix, output_filename, top_k=15):
        """Create unified 4-plot analysis for weights or activations"""
        
        # Prepare data
        if 'layer_comparisons' in data_dict:
            # Weight data format
            layer_data = []
            for name, stats in data_dict['layer_comparisons'].items():
                layer_data.append({
                    'layer': name,
                    'short_name': self._shorten_layer_name(name),
                    'l1_distance': stats.get('l1_distance', 0),
                    'cosine_similarity': stats['cosine_similarity'],
                    'layer_number': self._get_layer_number(name)
                })
        else:
            # Activation data format
            layer_data = []
            for name, stats in data_dict['layer_statistics'].items():
                layer_data.append({
                    'layer': name,
                    'short_name': self._shorten_layer_name(name),
                    'l1_distance': stats.get('mean_l1_distance', 0),
                    'cosine_similarity': stats['mean_cosine_similarity'],
                    'layer_number': self._get_layer_number(name)
                })
        
        df = pd.DataFrame(layer_data)
        
        # Sort by layer number for the line plots
        df_sorted = df[df['layer_number'] >= 0].sort_values('layer_number')
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{title_prefix} Comparison Analysis', fontsize=15, fontweight='bold')
        
        # 1. L1 distance across layers (top left)
        ax = axes[0, 0]
        if len(df_sorted) > 0:
            x_indices = range(len(df_sorted))
            ax.plot(x_indices, df_sorted['l1_distance'].values, marker='o', markersize=6, 
                    linewidth=2, color='steelblue', markerfacecolor='white', 
                    markeredgewidth=2)
            
            # Add trend line
            if len(df_sorted) > 1:
                z = np.polyfit(x_indices, df_sorted['l1_distance'].values, 1)
                p = np.poly1d(z)
                ax.plot(x_indices, p(x_indices), "--", alpha=0.7, color='red', 
                        label=f'Trend: {z[0]:.2e}')
                ax.legend()
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('L1 Distance')
        ax.set_title('L1 Distance Across Layers', fontweight='bold')
        ax.grid(True, alpha=0.4)
        ax.set_facecolor('white')
        
        # 2. Cosine similarity across layers (top right)
        ax = axes[0, 1]
        if len(df_sorted) > 0:
            x_indices = range(len(df_sorted))
            ax.plot(x_indices, df_sorted['cosine_similarity'].values, marker='s', markersize=6, 
                    linewidth=2, color='forestgreen', markerfacecolor='white', 
                    markeredgewidth=2)
            
            # Add trend line
            if len(df_sorted) > 1:
                z = np.polyfit(x_indices, df_sorted['cosine_similarity'].values, 1)
                p = np.poly1d(z)
                ax.plot(x_indices, p(x_indices), "--", alpha=0.7, color='red',
                        label=f'Trend: {z[0]:.2e}')
                ax.legend()
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Cosine Similarity Across Layers', fontweight='bold')
        ax.grid(True, alpha=0.4)
        ax.set_facecolor('white')
        
        # 3. Cosine similarity by layer type (bottom left)
        ax = axes[1, 0]
        layer_types = self._categorize_layers(df['layer'].values)
        df['layer_type'] = layer_types
        
        type_stats = df.groupby('layer_type')['cosine_similarity'].agg(['mean', 'std'])
        x_pos = np.arange(len(type_stats))
        
        bars = ax.bar(x_pos, type_stats['mean'], yerr=type_stats['std'], 
                     capsize=5, alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Layer Type')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Cosine Similarity by Layer Type', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(type_stats.index, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, type_stats['mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Top layers by lowest cosine similarity (bottom right)
        ax = axes[1, 1]
        # Get layers with lowest cosine similarity (most different)
        bottom_layers = df.nsmallest(top_k, 'cosine_similarity')
        
        y_pos = range(len(bottom_layers))
        bars = ax.barh(y_pos, bottom_layers['cosine_similarity'].values, 
                       color='orange', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bottom_layers['short_name'].values, fontsize=9)
        ax.set_xlabel('Cosine Similarity')
        ax.set_title(f'Top {len(bottom_layers)} Most Different Layers\n(Lowest Cosine Similarity)', 
                     fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, bottom_layers['cosine_similarity'].values)):
            ax.text(val + 0.001, i, f'{val:.3f}', 
                   va='center', fontsize=8)
        
        plt.tight_layout()
        output_file = self.results_dir / output_filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"{title_prefix} analysis saved to {output_file}")
        plt.show()
        
        return df
    
    def analyze_weight_changes(self, top_k: int = 15):
        """Analyze and visualize weight changes"""
        if not self.weight_results:
            print("No weight comparison results found")
            return
        
        df = self._create_unified_plots(
            self.weight_results, 
            "Weight", 
            "weight_analysis.png", 
            top_k
        )
        
        # Print summary
        print("\n" + "="*60)
        print("WEIGHT COMPARISON SUMMARY")
        print("="*60)
        print(f"Total parameters compared: {self.weight_results['summary_stats']['total_parameters']:,}")
        print(f"Average L1 distance: {self.weight_results['summary_stats']['average_l1_distance']:.4f}")
        print(f"Average L2 distance: {self.weight_results['summary_stats']['average_l2_distance']:.4f}")
        print(f"Median cosine similarity: {df['cosine_similarity'].median():.4f}")
        
        print(f"\nMost different layers (lowest cosine similarity):")
        for _, row in df.nsmallest(5, 'cosine_similarity').iterrows():
            print(f"  {row['short_name']}: {row['cosine_similarity']:.4f}")
    
    def analyze_activation_differences(self, top_k: int = 15):
        """Analyze activation differences"""
        if not self.activation_results:
            print("No activation comparison results found")
            return
        
        df = self._create_unified_plots(
            self.activation_results, 
            "Activation", 
            "activation_analysis.png", 
            top_k
        )
        
        # Print summary
        print("\n" + "="*60)
        print("ACTIVATION COMPARISON SUMMARY")
        print("="*60)
        print(f"Number of samples analyzed: {self.activation_results['num_samples']}")
        print(f"Number of layers compared: {len(df)}")
        print(f"Average L1 distance: {df['l1_distance'].mean():.4f}")
        print(f"Average cosine similarity: {df['cosine_similarity'].mean():.4f}")
        
        print(f"\nMost different layers (lowest cosine similarity):")
        for _, row in df.nsmallest(5, 'cosine_similarity').iterrows():
            print(f"  {row['short_name']}: {row['cosine_similarity']:.4f}")
    
    def run_full_analysis(self):
        """Run all analyses"""
        print("Starting analysis...\n")
        
        if self.weight_results:
            self.analyze_weight_changes()
        
        if self.activation_results:
            self.analyze_activation_differences()
        
        print("\n✓ Analysis complete!")


def main():
    """Main function to run analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model comparison results")
    parser.add_argument("--results_dir", type=str, default="./comparison_results",
                       help="Directory containing comparison results")
    parser.add_argument("--top_k", type=int, default=15,
                       help="Number of top layers to show in analysis")
    
    args = parser.parse_args()
    
    analyzer = ComparisonAnalyzer(args.results_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()