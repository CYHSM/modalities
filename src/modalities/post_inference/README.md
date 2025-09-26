# LLM Activation Visualizer

Visualize neural network activations during text generation in real-time. Watch your language model think as it generates each token.

![Demo](demo.gif)

## Features

- 🧠 Captures all layer activations during inference
- 🎨 Creates beautiful heatmap visualizations 
- 📝 Shows generated text with token-by-token progression
- 🎬 Exports as animated GIF
- 🚀 Works with any Hugging Face transformer model

## Installation

```bash
git clone https://github.com/yourusername/llm-activation-viz
cd llm-activation-viz
pip install -r requirements.txt
```

## Quick Start

```bash
# Visualize GPT-2 generating text (outputs to 'output/' folder)
python main.py --model gpt2 --prompt "Once upon a time" --tokens 20

# Use a custom output folder
python main.py --model meta-llama/Llama-2-7b-hf --prompt "The meaning of life is" --output my_viz

# Load and re-visualize existing activations
python main.py --load --output my_viz
```

## Output Structure

All outputs are saved to a single folder (default: `output/`):

```
output/
├── animation.gif      # Final animated visualization
├── activations.pkl    # Saved activation data
├── step_000.png      # Individual frames
├── step_001.png
└── ...
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Hugging Face model name or path | `gpt2` |
| `--prompt` | Input text to continue | `The quick brown fox` |
| `--tokens` | Number of tokens to generate | `10` |
| `--output` | Output directory for all files | `output` |
| `--fps` | Animation frames per second | `2` |
| `--load` | Load existing activations from output dir | False |

## How It Works

1. **Capture**: Hooks into model layers during generation to extract activations
2. **Visualize**: Converts activations to 256x256 heatmaps per layer
3. **Animate**: Combines frames into an animated GIF showing the generation process

Each frame shows:
- All model layers arranged in a grid
- Current generation step number
- Generated text with new tokens highlighted

## Python API

```python
from capture import capture_activations, save_data
from visualize import process_all_steps
from animate import create_gif

# Capture activations
activations, texts = capture_activations("gpt2", "Hello world", max_new_tokens=10)

# Save everything to one folder
output_dir = "my_visualization"
save_data(activations, texts, output_dir)
process_all_steps(activations, texts, output_dir)
create_gif(output_dir, fps=3)
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Pillow
- NumPy
- Matplotlib

## Examples

### GPT-2 Story Generation
```bash
python main.py --model gpt2 --prompt "In a galaxy far, far away" --tokens 30 --output story_viz
```

### Code Generation
```bash
python main.py --model codeparrot/codeparrot-small --prompt "def fibonacci(n):" --tokens 25 --output code_viz
```

### Fast Animation (5 fps)
```bash
python main.py --prompt "Once upon a time" --fps 5 --output fast_viz
```

## License

MIT

## Contributing

Pull requests welcome! Feel free to:
- Add support for more model architectures
- Improve visualization layouts
- Add new color schemes
- Optimize performance

## Citation

If you use this in your research, please cite:

```bibtex
@software{llm_activation_viz,
  title = {LLM Activation Visualizer},
  year = {2024},
  url = {https://github.com/yourusername/llm-activation-viz}
}
```