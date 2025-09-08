"""
Plotting utilities for model comparison results
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


class ComparisonPlotter:
    """Generate plots for model comparison results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.weight_results = None
        self.activation_results = None
        self._load_results()
        self._setup_plotting()
    
    def _load_results(self):
        """Load comparison results from JSON files"""
        # Load weight results
        weight_file = self.results_dir / "weight_comparison.json"
        if weight_file.exists():
            with open(weight_file, 'r') as f:
                self.weight_results = json.load(f)
            print(f"✓ Loaded weight comparison from {weight_file}")
        else:
            print(f"⚠️  No weight comparison results found in {weight_file}")
        
        # Load activation results
        activation_file = self.results_dir / "activation_comparison.json"
        if activation_file.exists():
            with open(activation_file, 'r') as f:
                self.activation_results = json.load(f)
            print(f"✓ Loaded activation comparison from {activation_file}")
        else:
            print(f"⚠️  No activation comparison results found in {activation_file}")
    
    def _setup_plotting(self):
        """Setup modern plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'figure.figsize': (16, 12),
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
    
    def _prepare_weight_data(self) -> pd.DataFrame:
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
    
    def _prepare_activation_data(self) -> pd.DataFrame:
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
    
    def _plot_with_type_lines(self, ax, x_positions, values, layer_types, title, ylabel):
        """Plot data with lines connecting points of the same type"""
        colors = get_layer_type_colors()
        
        # Group data by layer type
        type_data = {}
        for i, (x, val, layer_type) in enumerate(zip(x_positions, values, layer_types)):
            if layer_type not in type_data:
                type_data[layer_type] = {'x': [], 'y': []}
            type_data[layer_type]['x'].append(x)
            type_data[layer_type]['y'].append(val)
        
        # Plot lines and points for each type
        for layer_type, data in type_data.items():
            color = colors.get(layer_type, colors['Other'])
            
            # Sort by x position for proper line connections
            sorted_data = sorted(zip(data['x'], data['y']))
            x_sorted, y_sorted = zip(*sorted_data) if sorted_data else ([], [])
            
            # Plot line
            ax.plot(x_sorted, y_sorted, color=color, alpha=0.7, linewidth=2, 
                   label=layer_type, marker='o', markersize=6, markeredgecolor='white', 
                   markeredgewidth=1)
        
        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    def plot_weight_analysis(self) -> Optional[pd.DataFrame]:
        """Create weight analysis plots with connected lines by type"""
        if not self.weight_results:
            print("No weight comparison results found")
            return None
        
        df = self._prepare_weight_data()
        if df.empty:
            return None
        
        # Get layer grouping for x-axis
        layer_data_list = df.to_dict('records')
        x_positions, x_labels, layer_types = group_layers_by_actual_layer(layer_data_list)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        # fig.suptitle('Weight Comparison Analysis', fontsize=20, fontweight='bold')
        
        # 1. Mean absolute difference across layers (top)
        values = [df[df['layer_number'] >= 0].iloc[i]['abs_mean_diff'] for i in range(len(x_positions))]
        self._plot_with_type_lines(ax1, x_positions, values, layer_types,
                                  'Average Parameter Change Magnitude',
                                  'Mean |Δ| per Parameter')
        
        # 2. Cosine similarity across layers (bottom)
        values = [df[df['layer_number'] >= 0].iloc[i]['cosine_similarity'] for i in range(len(x_positions))]
        self._plot_with_type_lines(ax2, x_positions, values, layer_types,
                                  'Cosine Similarity Across Layers',
                                  'Cosine Similarity')
        
        # Set x-axis ticks for both plots
        unique_positions = sorted(set(x_positions))
        unique_labels = [x_labels[x_positions.index(pos)] for pos in unique_positions]
        
        for ax in [ax1, ax2]:
            ax.set_xticks(unique_positions[::max(1, len(unique_positions)//20)])  # Limit tick count
            ax.set_xticklabels([unique_labels[i] for i in range(0, len(unique_labels), 
                               max(1, len(unique_labels)//20))], rotation=45, ha='right')
            ax.set_xlabel('Layer Number', fontsize=14)
        
        plt.tight_layout()
        output_file = self.results_dir / "weight_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Weight analysis saved to {output_file}")
        plt.show()
        
        # Print summary
        self._print_weight_summary(df)
        
        return df
    
    def plot_activation_analysis(self) -> Optional[pd.DataFrame]:
        """Create activation analysis plots with connected lines by type"""
        if not self.activation_results:
            print("No activation comparison results found")
            return None
        
        df = self._prepare_activation_data()
        if df.empty:
            return None
        
        # Get layer grouping for x-axis
        layer_data_list = df.to_dict('records')
        x_positions, x_labels, layer_types = group_layers_by_actual_layer(layer_data_list)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        # fig.suptitle('Activation Comparison Analysis', fontsize=20, fontweight='bold')
        
        # 1. Mean absolute difference across layers (top)
        values = [df[df['layer_number'] >= 0].iloc[i]['abs_mean_diff'] for i in range(len(x_positions))]
        self._plot_with_type_lines(ax1, x_positions, values, layer_types,
                                  'Average Activation Change Magnitude',
                                  'Mean |Δ| per Activation')
        
        # 2. Cosine similarity across layers (bottom)
        values = [df[df['layer_number'] >= 0].iloc[i]['cosine_similarity'] for i in range(len(x_positions))]
        self._plot_with_type_lines(ax2, x_positions, values, layer_types,
                                  'Cosine Similarity Across Layers',
                                  'Cosine Similarity')
        
        # Set x-axis ticks for both plots
        unique_positions = sorted(set(x_positions))
        unique_labels = [x_labels[x_positions.index(pos)] for pos in unique_positions]
        
        for ax in [ax1, ax2]:
            ax.set_xticks(unique_positions[::max(1, len(unique_positions)//20)])  # Limit tick count
            ax.set_xticklabels([unique_labels[i] for i in range(0, len(unique_labels), 
                               max(1, len(unique_labels)//20))], rotation=45, ha='right')
            ax.set_xlabel('Layer Number', fontsize=14)
        
        plt.tight_layout()
        output_file = self.results_dir / "activation_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Activation analysis saved to {output_file}")
        plt.show()
        
        # Print summary
        self._print_activation_summary(df)
        
        return df
    
    def _print_weight_summary(self, df: pd.DataFrame):
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
    
    def _print_activation_summary(self, df: pd.DataFrame):
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
    
    def plot_all_analyses(self):
        """Generate all available plots and analyses"""
        weight_df = None
        activation_df = None
        
        # Plot weight analysis if data exists
        if self.weight_results:
            print("📊 Generating weight analysis plots...")
            weight_df = self.plot_weight_analysis()
        else:
            print("⚠️  No weight comparison data found - skipping weight plots")
        
        print()  # spacing
        
        # Plot activation analysis if data exists
        if self.activation_results:
            print("📊 Generating activation analysis plots...")
            activation_df = self.plot_activation_analysis()
        else:
            print("⚠️  No activation comparison data found - skipping activation plots")
        
        # Print overall summary
        self._print_overall_summary(weight_df, activation_df)
        
        print("\n✅ All plots generated successfully!")
        
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