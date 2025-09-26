import argparse
from pathlib import Path

from capture import capture_activations, save_data, load_data
from visualize import process_all_steps
from animate import create_gif

def main():
    parser = argparse.ArgumentParser(description='Visualize LLM activations during inference')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model name or path (default: gpt2)')
    parser.add_argument('--prompt', type=str, default='The quick brown fox',
                       help='Input prompt text')
    parser.add_argument('--tokens', type=int, default=10,
                       help='Number of tokens to generate (default: 10)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for all files (default: output)')
    parser.add_argument('--fps', type=int, default=2,
                       help='Frames per second (default: 2)')
    parser.add_argument('--load', action='store_true',
                       help='Load existing activations instead of capturing')
    
    args = parser.parse_args()
    
    if args.load:
        print(f"Loading activations from {args.output}/activations.pkl...")
        activations, step_texts = load_data(args.output)
    else:
        print(f"Model: {args.model}")
        print(f"Prompt: '{args.prompt}'")
        print(f"Generating {args.tokens} tokens...\n")
        
        activations, step_texts = capture_activations(
            args.model, 
            args.prompt, 
            args.tokens
        )
        
        save_data(activations, step_texts, args.output)
        print(f"\nSaved to {args.output}/")
    
    print(f"\nCreating visualizations...")
    process_all_steps(activations, step_texts, args.output)
    
    print(f"\nGenerating animation...")
    gif_path = create_gif(args.output, args.fps)
    
    print(f"\n✓ Complete! Output in: {args.output}/")
    print(f"  - animation.gif")
    print(f"  - activations.pkl") 
    print(f"  - step_*.png frames")

if __name__ == "__main__":
    main()