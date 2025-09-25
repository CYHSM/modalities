from capture import capture_activations, save_activations
from visualize import load_activations, process_all_steps
from animate import create_gif, create_comparison_grid

def run_pipeline(
    model_path="/raid/s3/opengptx/mfrey/instruct/hf_model",
    input_text="Janet's ducks laid 16 eggs. She eats 3 for breakfast every morning. How many eggs are left after three days? Think step by step.",
    max_new_tokens=80,
    normalizations=['percentile', 'zscore'],
    colormaps=[None, 'viridis', 'inferno', 'plasma']
):
    
    print("1. Capturing activations...")
    tokens, activations = capture_activations(model_path, input_text, max_new_tokens)
    save_activations(activations, "/raid/s3/opengptx/mfrey/cp_analysis/inference_viz/activations.pkl")
    
    print("\n2. Creating visualizations...")
    viz_dirs = []
    
    for norm in normalizations:
        for cmap in colormaps:
            suffix = f"{norm}_{cmap if cmap else 'gray'}"
            output_dir = f"/raid/s3/opengptx/mfrey/cp_analysis/inference_viz/viz_{suffix}"
            
            print(f"   Processing: {suffix}")
            process_all_steps(activations, output_dir, norm, cmap)
            viz_dirs.append(output_dir)
            
            gif_path = f"/raid/s3/opengptx/mfrey/cp_analysis/inference_viz/activation_{suffix}.gif"
            create_gif(output_dir, gif_path, duration=300)
    
    print("\n3. Creating comparison grid...")
    create_comparison_grid(viz_dirs[:3], "/raid/s3/opengptx/mfrey/cp_analysis/inference_viz/activation_comparison.gif", duration=300)
    
    print("\nDone!")

if __name__ == "__main__":
    run_pipeline()