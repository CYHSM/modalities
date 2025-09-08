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
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def _load_results(self):
        """Load saved results from files"""
        # Load weight comparison
        weight_file = self.results_dir / "weight_comparison.json"
        if weight_file.exists():
            with open(weight_file, 'r') as f:
                self.weight_results = json.load(f)
            print(f"Loaded weight comparison from {weight_file}")
        
        # Load activation comparison
        activation_file = self.results_dir / "activation_comparison.json"
        if activation_file.exists():
            with open(activation_file, 'r') as f:
                self.activation_results = json.load(f)
            print(f"Loaded activation comparison from {activation_file}")
        
        # Load sample details
        samples_file = self.results_dir / "activation_samples.json"
        if samples_file.exists():
            with open(samples_file, 'r') as f:
                self.sample_results = json.load(f)
            print(f"Loaded sample details from {samples_file}")
    
    def analyze_weight_changes(self, top_k: int = 20):
        """Analyze and visualize weight changes"""
        if not self.weight_results:
            print("No weight comparison results found")
            return
        
        # Convert to DataFrame for easier analysis
        layer_data = []
        for name, stats in self.weight_results["layer_comparisons"].items():
            layer_data.append({
                "layer": name,
                "l2_distance": stats["l2_distance"],
                "relative_change": stats["relative_change"],
                "cosine_similarity": stats["cosine_similarity"],
                "num_params": stats["num_params"],
                "max_abs_diff": stats["max_abs_diff"]
            })
        
        df = pd.DataFrame(layer_data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Top changed layers by L2 distance
        ax = axes[0, 0]
        top_layers = df.nlargest(top_k, 'l2_distance')
        ax.barh(range(len(top_layers)), top_layers['l2_distance'].values)
        ax.set_yticks(range(len(top_layers)))
        ax.set_yticklabels([name.split('.')[-1] if '.' in name else name 
                            for name in top_layers['layer'].values], fontsize=8)
        ax.set_xlabel('L2 Distance')
        ax.set_title(f'Top {top_k} Layers by L2 Distance')
        ax.invert_yaxis()
        
        # 2. Relative change distribution
        ax = axes[0, 1]
        ax.hist(df['relative_change'].values, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Relative Change')
        ax.set_ylabel('Number of Layers')
        ax.set_title('Distribution of Relative Changes')
        ax.axvline(df['relative_change'].median(), color='red', 
                  linestyle='--', label=f'Median: {df["relative_change"].median():.3f}')
        ax.legend()
        
        # 3. Cosine similarity distribution
        ax = axes[1, 0]
        ax.hist(df['cosine_similarity'].values, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Number of Layers')
        ax.set_title('Cosine Similarity Distribution')
        ax.axvline(df['cosine_similarity'].mean(), color='red', 
                  linestyle='--', label=f'Mean: {df["cosine_similarity"].mean():.3f}')
        ax.legend()
        
        # 4. Layer type analysis
        ax = axes[1, 1]
        layer_types = []
        for name in df['layer'].values:
            if 'self_attn' in name:
                if 'q_proj' in name:
                    layer_types.append('Q Projection')
                elif 'k_proj' in name:
                    layer_types.append('K Projection')
                elif 'v_proj' in name:
                    layer_types.append('V Projection')
                elif 'o_proj' in name:
                    layer_types.append('Output Projection')
                else:
                    layer_types.append('Attention Other')
            elif 'mlp' in name:
                if 'gate_proj' in name:
                    layer_types.append('MLP Gate')
                elif 'up_proj' in name:
                    layer_types.append('MLP Up')
                elif 'down_proj' in name:
                    layer_types.append('MLP Down')
                else:
                    layer_types.append('MLP Other')
            elif 'norm' in name or 'layernorm' in name:
                layer_types.append('Normalization')
            elif 'embed' in name:
                layer_types.append('Embedding')
            elif 'lm_head' in name:
                layer_types.append('LM Head')
            else:
                layer_types.append('Other')
        
        df['layer_type'] = layer_types
        type_stats = df.groupby('layer_type')['relative_change'].agg(['mean', 'std'])
        type_stats.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('Layer Type')
        ax.set_ylabel('Relative Change')
        ax.set_title('Average Relative Change by Layer Type')
        ax.legend(['Mean', 'Std Dev'])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        output_file = self.results_dir / "weight_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Weight analysis saved to {output_file}")
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("WEIGHT COMPARISON SUMMARY")
        print("="*50)
        print(f"Total parameters compared: {self.weight_results['summary_stats']['total_parameters']:,}")
        print(f"Average L2 distance: {self.weight_results['summary_stats']['average_l2_distance']:.4f}")
        print(f"Median relative change: {df['relative_change'].median():.4f}")
        print(f"Mean cosine similarity: {df['cosine_similarity'].mean():.4f}")
        print(f"\nMost changed layers:")
        for _, row in df.nlargest(5, 'relative_change').iterrows():
            print(f"  {row['layer']}: {row['relative_change']:.4f} relative change")
    
    def analyze_activation_differences(self):
        """Analyze activation differences across layers"""
        if not self.activation_results:
            print("No activation comparison results found")
            return
        
        # Prepare data for visualization
        layer_names = []
        mean_l2_distances = []
        mean_cosine_sims = []
        
        for layer_name, stats in self.activation_results["layer_statistics"].items():
            if "mean_l2_distance" in stats:
                layer_names.append(layer_name)
                mean_l2_distances.append(stats["mean_l2_distance"])
                mean_cosine_sims.append(stats["mean_cosine_similarity"])
        
        # Extract layer numbers for ordering
        def get_layer_number(name):
            parts = name.split('.')
            for part in parts:
                if part.startswith('layers'):
                    continue
                if part.isdigit():
                    return int(part)
            return -1
        
        layer_numbers = [get_layer_number(name) for name in layer_names]
        
        # Sort by layer number
        sorted_indices = np.argsort(layer_numbers)
        layer_names = [layer_names[i] for i in sorted_indices]
        mean_l2_distances = [mean_l2_distances[i] for i in sorted_indices]
        mean_cosine_sims = [mean_cosine_sims[i] for i in sorted_indices]
        layer_numbers = [layer_numbers[i] for i in sorted_indices]
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. L2 distance across layers
        ax = axes[0, 0]
        ax.plot(range(len(layer_names)), mean_l2_distances, marker='o', markersize=4)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Mean L2 Distance')
        ax.set_title('Activation L2 Distance Across Layers')
        ax.grid(True, alpha=0.3)
        
        # 2. Cosine similarity across layers
        ax = axes[0, 1]
        ax.plot(range(len(layer_names)), mean_cosine_sims, marker='o', markersize=4, color='green')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Mean Cosine Similarity')
        ax.set_title('Activation Cosine Similarity Across Layers')
        ax.grid(True, alpha=0.3)
        
        # 3. Layer type comparison
        ax = axes[1, 0]
        layer_types = []
        type_l2_distances = {}
        
        for name, l2_dist in zip(layer_names, mean_l2_distances):
            if 'self_attn' in name:
                layer_type = 'Attention'
            elif 'mlp' in name:
                layer_type = 'MLP'
            elif 'norm' in name or 'layernorm' in name:
                layer_type = 'Normalization'
            else:
                layer_type = 'Other'
            
            if layer_type not in type_l2_distances:
                type_l2_distances[layer_type] = []
            type_l2_distances[layer_type].append(l2_dist)
        
        # Box plot for layer types
        data_for_box = []
        labels_for_box = []
        for layer_type, distances in type_l2_distances.items():
            data_for_box.append(distances)
            labels_for_box.append(f"{layer_type}\n(n={len(distances)})")
        
        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_ylabel('L2 Distance')
        ax.set_title('Activation Differences by Layer Type')
        
        # 4. Depth analysis - group by depth thirds
        ax = axes[1, 1]
        unique_layer_nums = sorted(set([n for n in layer_numbers if n >= 0]))
        if unique_layer_nums:
            max_layer = max(unique_layer_nums)
            
            early_layers = []
            middle_layers = []
            late_layers = []
            
            for num, l2_dist in zip(layer_numbers, mean_l2_distances):
                if num < 0:
                    continue
                if num < max_layer / 3:
                    early_layers.append(l2_dist)
                elif num < 2 * max_layer / 3:
                    middle_layers.append(l2_dist)
                else:
                    late_layers.append(l2_dist)
            
            depth_data = [early_layers, middle_layers, late_layers]
            depth_labels = ['Early\nLayers', 'Middle\nLayers', 'Late\nLayers']
            
            bp = ax.boxplot(depth_data, labels=depth_labels, patch_artist=True)
            colors = ['lightgreen', 'lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            ax.set_ylabel('L2 Distance')
            ax.set_title('Activation Differences by Model Depth')
        
        plt.tight_layout()
        output_file = self.results_dir / "activation_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Activation analysis saved to {output_file}")
        plt.show()
        
        # Print summary
        print("\n" + "="*50)
        print("ACTIVATION COMPARISON SUMMARY")
        print("="*50)
        print(f"Number of samples analyzed: {self.activation_results['num_samples']}")
        print(f"Number of layers compared: {len(layer_names)}")
        print(f"Average L2 distance: {np.mean(mean_l2_distances):.4f}")
        print(f"Average cosine similarity: {np.mean(mean_cosine_sims):.4f}")
        
        # Find layers with biggest differences
        top_diff_indices = np.argsort(mean_l2_distances)[-5:]
        print(f"\nLayers with largest activation differences:")
        for idx in reversed(top_diff_indices):
            print(f"  {layer_names[idx]}: L2={mean_l2_distances[idx]:.4f}, "
                  f"Cosine={mean_cosine_sims[idx]:.4f}")
    
    def analyze_sample_variations(self, num_samples: int = 10):
        """Analyze how activation differences vary across samples"""
        if not self.sample_results:
            print("No sample results found")
            return
        
        # Analyze variance across samples
        sample_l2_distances = []
        
        for sample in self.sample_results[:num_samples]:
            total_l2 = 0
            count = 0
            for layer_name, stats in sample["layer_comparisons"].items():
                total_l2 += stats["l2_distance"]
                count += 1
            if count > 0:
                sample_l2_distances.append(total_l2 / count)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.bar(range(len(sample_l2_distances)), sample_l2_distances, color='steelblue', alpha=0.7)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Average L2 Distance')
        ax.set_title(f'Activation Differences Across First {num_samples} GSM8K Samples')
        ax.axhline(np.mean(sample_l2_distances), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(sample_l2_distances):.4f}')
        ax.legend()
        
        output_file = self.results_dir / "sample_variation_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Sample variation analysis saved to {output_file}")
        plt.show()
        
        # Find samples with highest differences
        print("\n" + "="*50)
        print("SAMPLE VARIATION ANALYSIS")
        print("="*50)
        
        sorted_indices = np.argsort(sample_l2_distances)
        print("\nSamples with highest activation differences:")
        for idx in reversed(sorted_indices[-3:]):
            sample = self.sample_results[idx]
            print(f"\nSample {idx}:")
            print(f"  Question: {sample['question'][:100]}...")
            print(f"  Average L2 distance: {sample_l2_distances[idx]:.4f}")
    
    def run_full_analysis(self):
        """Run all analyses"""
        print("Starting full analysis...\n")
        
        if self.weight_results:
            self.analyze_weight_changes()
        
        if self.activation_results:
            self.analyze_activation_differences()
        
        if self.sample_results:
            self.analyze_sample_variations()
        
        print("\nAnalysis complete!")


def main():
    """Main function to run analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model comparison results")
    parser.add_argument("--results_dir", type=str, default="./comparison_results",
                       help="Directory containing comparison results")
    parser.add_argument("--top_k", type=int, default=20,
                       help="Number of top layers to show in weight analysis")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to analyze for variation")
    
    args = parser.parse_args()
    
    analyzer = ComparisonAnalyzer(args.results_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
