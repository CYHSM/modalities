from capture import capture_activations, save_activations
from visualize import load_activations, process_all_steps
from animate import create_gif, create_comparison_grid

def run_pipeline(
    model_path= "Qwen/Qwen3-8B", #"/raid/s3/opengptx/mfrey/instruct/hf_model",
    input_text="Janet's ducks laid 16 eggs. She eats 3 for breakfast every morning. How many eggs are left after three days? Think step by step.",
    max_new_tokens=80,
    normalizations= ['nonorm'], #['percentile', 'zscore'],
    colormaps=['inferno'],
):
    
    print("1. Capturing activations and text...")
    tokens, activations, step_texts = capture_activations(model_path, input_text, max_new_tokens)
    save_activations(activations, step_texts, "/raid/s3/opengptx/mfrey/cp_analysis/inference_qwen/activations.pkl")
    
    print(f"   Generated {len(step_texts)} text steps")
    print(f"   Final text: {step_texts[-1]}")
    
    print("\n2. Creating enhanced visualizations...")
    viz_dirs = []
    
    for norm in normalizations:
        for cmap in colormaps:
            suffix = f"{norm}_{cmap if cmap else 'gray'}"
            output_dir = f"/raid/s3/opengptx/mfrey/cp_analysis/inference_qwen/viz_{suffix}"
            
            print(f"   Processing: {suffix}")
            process_all_steps(activations, step_texts, output_dir, norm, cmap)
            viz_dirs.append(output_dir)
            
            gif_path = f"/raid/s3/opengptx/mfrey/cp_analysis/inference_qwen/activation_{suffix}.gif"
            create_gif(output_dir, gif_path, duration=800)
    
    print("\n3. Creating comparison grid...")
    create_comparison_grid(viz_dirs[:3], "/raid/s3/opengptx/mfrey/cp_analysis/inference_qwen/activation_comparison.gif", duration=800)
    
    print("\nDone! Enhanced visualizations with step counters and text overlays created.")

if __name__ == "__main__":
    run_pipeline()