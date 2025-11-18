import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from metrics import compute_pass_at_k
import glob
import os

def load_results(result_files):
    all_results = {}
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
            if len(data) > 0:
                temp = data[0]['temperature']
                all_results[temp] = data
    return all_results

def plot_pass_at_k(results_dict, k_values, output_file='pass_at_k.png', power_labels=False):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict)))
    
    for (temp, results), color in zip(sorted(results_dict.items()), colors):
        pass_k_values = compute_pass_at_k(results, k_values)
        k_list = sorted(pass_k_values.keys())
        pass_list = [pass_k_values[k] for k in k_list]
        plt.plot(k_list, pass_list, marker='o', label=f'T={temp}', 
                color=color, linewidth=2, markersize=6)
    
    plt.xlabel('Number of Samples (k)', fontsize=12)
    plt.ylabel('pass@k', fontsize=12)
    plt.title('pass@k Performance on GSM8K', fontsize=14)
    plt.xscale('log', base=2)
    plt.ylim([0, 1])
    
    ax = plt.gca()
    ax.set_xticks(k_values)
    if power_labels:
        ax.set_xticklabels([f'$2^{{{int(np.log2(k))}}}$' for k in k_values])
    else:
        ax.set_xticklabels([str(k) for k in k_values])
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

def print_results_table(results_dict, k_values):
    print("\n" + "="*80)
    print("pass@k Results Summary")
    print("="*80)
    
    for temp in sorted(results_dict.keys()):
        results = results_dict[temp]
        pass_k_values = compute_pass_at_k(results, k_values)
        print(f"\nTemperature: {temp}")
        print("-" * 40)
        for k in k_values:
            print(f"  pass@{k:4d}: {pass_k_values[k]:.4f} ({pass_k_values[k]*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Plot pass@k results')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--output_file', type=str, default='pass_at_k.png')
    parser.add_argument('--k_values', type=int, nargs='+', 
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument('--power_labels', action='store_true')
    args = parser.parse_args()
    
    result_files = glob.glob(os.path.join(args.results_dir, 'results_*.json'))
    
    if not result_files:
        print(f"No result files found in {args.results_dir}")
        return
    
    print(f"Found {len(result_files)} result files")
    results_dict = load_results(result_files)
    print_results_table(results_dict, args.k_values)
    plot_pass_at_k(results_dict, args.k_values, args.output_file, args.power_labels)

if __name__ == "__main__":
    main()