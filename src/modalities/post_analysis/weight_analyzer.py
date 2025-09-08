"""
Weight-specific analysis for model comparison
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


class WeightAnalyzer:
    """Analyze weight differences between models"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.weight_results = None
        self._load_results()
        self._setup_plotting()
    
    def _load_results(self):
        """Load weight comparison results"""
        weight_file = self.results_dir / "weight_comparison.json"
        if weight_file.exists():
            with open(weight_file, 'r') as f:
                self.weight_results = json.load(f)
            print(f"✓ Loaded weight comparison from {weight_file}")
        else:
            print(f"❌ No weight comparison results found in {weight_file}")
    
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
        """Prepare weight data for analysis"""
        if not self.weight_results:
            return pd.DataFrame()
        
        layer_data = []
        for name, stats in self.weight_results['layer_comparisons'].items():
            layer_data.append({
                'layer': name,
                'short_name': shorten_layer_name(name),
                'mean_diff': stats.get('mean_diff', 0),
                'std_diff': stats.get('std_diff', 0),
                'abs_mean_diff': stats.get('abs_mean_diff', 0),
                'cosine_similarity': stats['cosine_similarity'],
                'layer_number': get_layer_number(name),
                'layer_type': categorize_layers([name])[0]
            })
        
        return pd.DataFrame(layer_data)
    
    def create_weight_analysis_plot(self) -> Optional[pd.DataFrame]:
        """Create simplified weight analysis plots"""
        if not self.weight_results:
            print("No weight comparison results found")
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
        fig.suptitle('Weight Comparison Analysis', fontsize=16, fontweight='bold')
        
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
        ax1.set_ylabel('Mean |Δ| per Parameter')
        ax1.set_title('Average Parameter Change Magnitude', fontweight='bold')
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
        output_file = self.results_dir / "weight_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Weight analysis saved to {output_file}")
        plt.show()
        
        return df
    
    def print_weight_summary(self, df: pd.DataFrame):
        """Print weight comparison summary"""
        if df.empty:
            return
        
        print("\n" + "="*60)
        print("WEIGHT COMPARISON SUMMARY")
        print("="*60)
        print(f"Total layers compared: {len(df)}")
        print(f"Average |change| per parameter: {df['abs_mean_diff'].mean():.6f}")
        print(f"Average change variability (std): {df['std_diff'].mean():.6f}")
        print(f"Median cosine similarity: {df['cosine_similarity'].median():.4f}")
        
        print(f"\nLayers with largest parameter changes:")
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
    
    def analyze_weights(self):
        """Run complete weight analysis"""
        df = self.create_weight_analysis_plot()
        if df is not None:
            self.print_weight_summary(df)
        return df