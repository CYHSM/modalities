# Activation Visualization Pipeline

Clean, modular code for capturing and visualizing model activations during text generation.

## Structure

- **capture.py** - Run inference and save activations to disk
- **visualize.py** - Load saved activations and create visualizations  
- **animate.py** - Generate GIFs/videos from visualization frames
- **explore.py** - Analyze activation statistics and dynamics
- **main.py** - Run the complete pipeline

## Quick Start

```bash
# Run everything with defaults
python main.py

# Or step by step:
python capture.py         # Saves to data/activations.pkl
python visualize.py       # Creates viz_* directories
python animate.py         # Creates GIF animations
```

## Customization

### Normalization Methods
- `minmax` - Simple min-max scaling
- `percentile` - Robust to outliers (1st-99th percentile)
- `zscore` - Z-score with tanh squashing
- `log` - Logarithmic scaling

### Color Maps
- `None` - Grayscale
- `viridis`, `inferno`, `plasma` - Matplotlib colormaps

### Example: Custom Pipeline

```python
from capture import capture_activations, save_activations
from visualize import load_activations, process_all_steps
from animate import create_gif

# Capture with more tokens
tokens, acts = capture_activations(model_path, "Once upon a time", max_new_tokens=20)
save_activations(acts, "data/story.pkl")

# Visualize with custom settings
acts = load_activations("data/story.pkl")
process_all_steps(acts, "viz_custom", normalization='log', colormap='plasma')

# Animate
create_gif("viz_custom", "story.gif", duration=200)
```

## Features

✓ Separates inference from visualization (save to file)  
✓ Includes all layers (lm_head, embed_tokens, rotary_emb)  
✓ Multiple normalization strategies  
✓ Color mapping options  
✓ GIF/video generation  
✓ Minimal, clean code