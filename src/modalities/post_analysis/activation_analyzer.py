"""
Activation-specific analysis for model comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from utils import (
    shorten_layer_name, categorize_layers, get_layer_number, 
    get_layer_type_colors, group_layers_by_actual_layer
)


class ActivationAnalyzer:
    """Analyze activation differences between models"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.activation_results = None
        self.sample_results = None
        self._load_results()
        self._setup_plotting()
    
    def _load_results(self):
        """Load activation comparison results"""
        activation_file = self.results_dir / "activation_comparison.json"
        if activation_file.exists():
            with open(activation_file, 'r') as f:
                self.activation_results = json.load(f)
            print(f"✓ Loaded activation comparison from {activation_file}")
        else:
            print(f"❌ No activation comparison results found in {activation_file}")
        
        samples_file = self.results_dir / "activation_samples.json"
        if samples_file.exists():
            with open(samples_file, 'r') as f:
                self.sample_results = json.load(f)
            print(f"✓ Loaded sample details from {samples_file}")
    
    def _setup_plotting(self):
        """Setup modern plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'figure.figsize': (14, 10),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def _prepare_data(self) -> pd.DataFrame:
        """Prepare activation data for analysis"""
        if not self.activation_results:
            return pd.DataFrame()
        
        layer_data = []
        for name, stats in self.activation_results['layer_statistics'].items():
            layer_data.append({
                'layer': name,
                'short_name': shorten_layer_name(name),
                'mean_diff': stats.get('mean_mean_diff', 0),
                'std_diff': stats.get('mean_std_diff', 0), 
                'abs_mean_diff': stats.get('mean_abs_mean_diff', 0),
                'cosine_similarity': stats['mean_cosine_similarity'],
                'layer_number': get_layer_number(name),
                'layer_type': categorize_layers([name])[0]
            })
        
        return pd.DataFrame(layer_data)
    
    def create_activation_analysis_plot(self) -> Optional[pd.DataFrame]:
        """Create simplified activation analysis plots"""
        if not self.activation_results:
            print("No activation comparison results found")
            return None
        
        df = self._prepare_data()
        if df.empty:
            return None
        
        # Get layer grouping for x-axis
        layer_data_list = df.to_dict('records')
        x_positions, x_labels, layer_types = group_layers_by_actual_layer(layer_data_list)
        
        # Get colors for layer types
        colors = get_layer_type_colors()
        point_colors = [colors.get(lt, colors['Other']) for lt in layer_types]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Activation Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Mean absolute difference across layers (top)
        values = [df[df['layer_number'] >= 0].iloc[i]['abs_mean_diff'] for i in range(len(x_positions))]
        
        scatter = ax1.scatter(x_positions, values, c=point_colors, alpha=0.8, s=60, edgecolors='white', linewidth=1)
        
        # Add trend line
        if len(x_positions) > 1:
            z = np.polyfit(x_positions, values, 1)
            p = np.poly1d(z)
            ax1.plot(x_positions, p(x_positions), "--", alpha=0.7, color='red', 
                    linewidth=2, label=f'Trend: {z[0]:.2e}')
            ax1.legend()
        
        ax1.set_xlabel('Layer Number')
        ax1.set_ylabel('Mean |Δ| per Activation')
        ax1.set_title('Average Activation Change Magnitude', fontweight='bold')
        ax1.grid(True, alpha=0.4)
        
        # Set x-axis ticks to show layer numbers properly
        unique_positions = sorted(set(x_positions))
        unique_labels = [x_labels[x_positions.index(pos)] for pos in unique_positions]
        ax1.set_xticks(unique_positions)
        ax1.set_xticklabels(unique_labels, rotation=45, ha='right')
        
        # 2. Cosine similarity across layers (bottom)
        values = [df[df['layer_number'] >= 0].iloc[i]['cosine_similarity'] for i in range(len(x_positions))]
        
        scatter = ax2.scatter(x_positions, values, c=point_colors, alpha=0.8, s=60, edgecolors='white', linewidth=1)
        
        # Add trend line
        if len(x_positions) > 1:
            z = np.polyfit(x_positions, values, 1)
            p = np.poly1d(z)
            ax2.plot(x_positions, p(x_positions), "--", alpha=0.7, color='red',
                    linewidth=2, label=f'Trend: {z[0]:.2e}')
            ax2.legend()
        
        ax2.set_xlabel('Layer Number')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Cosine Similarity Across Layers', fontweight='bold')
        ax2.grid(True, alpha=0.4)
        
        # Set x-axis ticks
        ax2.set_xticks(unique_positions)
        ax2.set_xticklabels(unique_labels, rotation=45, ha='right')
        
        # Add legend for layer types
        unique_types = list(set(layer_types))
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=colors.get(lt, colors['Other']), 
                                    markersize=8, label=lt) for lt in unique_types]
        ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        output_file = self.results_dir / "activation_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Activation analysis saved to {output_file}")
        plt.show()
        
        return df
    
    def print_activation_summary(self, df: pd.DataFrame):
        """Print activation comparison summary"""
        if df.empty:
            return
        
        print("\n" + "="*60)
        print("ACTIVATION COMPARISON SUMMARY")
        print("="*60)
        print(f"Number of samples analyzed: {self.activation_results.get('num_samples', 'N/A')}")
        print(f"Number of layers compared: {len(df)}")
        print(f"Average activation difference: {df['abs_mean_diff'].mean():.6f}")
        print(f"Average cosine similarity: {df['cosine_similarity'].mean():.4f}")
        
        print(f"\nLayers with largest activation differences:")
        for _, row in df.nlargest(5, 'abs_mean_diff').iterrows():
            print(f"  {row['short_name']}: {row['abs_mean_diff']:.6f}")
        
        print(f"\nMost different layers (lowest cosine similarity):")
        for _, row in df.nsmallest(5, 'cosine_similarity').iterrows():
            print(f"  {row['short_name']}: {row['cosine_similarity']:.4f}")
        
        # Layer type analysis
        print(f"\nAnalysis by layer type:")
        type_stats = df.groupby('layer_type').agg({
            'abs_mean_diff': ['mean', 'std'],
            'cosine_similarity': ['mean', 'std']
        }).round(6)
        
        for layer_type in type_stats.index:
            mean_change = type_stats.loc[layer_type, ('abs_mean_diff', 'mean')]
            mean_sim = type_stats.loc[layer_type, ('cosine_similarity', 'mean')]
            print(f"  {layer_type}: avg_change={mean_change:.6f}, avg_similarity={mean_sim:.4f}")
    
    def analyze_activations(self):
        """Run complete activation analysis"""
        df = self.create_activation_analysis_plot()
        if df is not None:
            self.print_activation_summary(df)
        return df