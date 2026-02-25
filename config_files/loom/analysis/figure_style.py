"""
Publication-quality figure style for adaptive computation analysis.

Usage:
    from figure_style import setup_style, COLORS, plot_with_band, get_layer_colors, despine
    setup_style(use_latex=True)  # Set False if LaTeX not installed
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colormaps
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ============================================================================
# COLOR PALETTE (colorblind-friendly)
# ============================================================================

COLORS = {
    'blue': '#0077BB',
    'cyan': '#33BBEE', 
    'teal': '#009988',
    'orange': '#EE7733',
    'red': '#CC3311',
    'magenta': '#EE3377',
    'grey': '#BBBBBB',
    'darkgrey': '#666666',
    'black': '#000000',
}

# def get_layer_colors(n):
#     """Get a sequential colormap for n layers."""
#     cmap = plt.cm.YlOrRd
#     return [cmap(0.2 + 0.7 * i / (n - 1)) for i in range(n)]

def get_layer_colors(n):
    cmap = colormaps['cividis_r']
    return [cmap(0.1 + 0.8 * i / (n - 1)) if n > 1 else [cmap(0.5)] for i in range(n)]


# def get_layer_colors(n, map_name='cividis_r'):
#     """
#     Generates n colors from a colormap, avoiding the absolute extremes.
    
#     Args:
#         n (int): Number of layers.
#         map_name (str): Name of the matplotlib colormap.
#     """
#     # Use the modern registry (plt.cm.get_cmap is deprecated)
#     cmap = colormaps[map_name]
    
#     # 0.1 to 0.9 keeps the colors distinguishable but not too washed out/dark
#     return [cmap(0.1 + 0.8 * i / (n - 1)) if n > 1 else cmap(0.5) for i in range(n)]

# ============================================================================
# STYLE SETUP
# ============================================================================

def setup_style(use_latex=True):
    """
    Configure matplotlib for publication-quality figures.
    
    Args:
        use_latex: If True, use LaTeX for text rendering (requires LaTeX installation).
                   If False, use mathtext which looks similar but doesn't require LaTeX.
    
    Returns:
        bool: Whether LaTeX is actually being used
    """
    
    style_dict = {
        # Font sizes
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        
        # Spines - only left and bottom (we'll add offset in despine())
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        
        # Line widths
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
        
        # No grid
        'axes.grid': False,
        
        # Ticks - outward
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        
        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Legend
        'legend.frameon': False,
        'legend.borderpad': 0.4,
    }
    
    # Try LaTeX, fall back gracefully
    latex_available = False
    if use_latex:
        try:
            plt.rcParams['text.usetex'] = True
            style_dict.update({
                'text.usetex': True,
                'font.family': 'serif',
                'font.serif': ['Times', 'Times New Roman', 'Computer Modern Roman'],
            })
            # Quick test
            fig_test, ax_test = plt.subplots(figsize=(0.5, 0.5))
            ax_test.set_title(r'$\alpha$')
            plt.close(fig_test)
            latex_available = True
            print("✓ LaTeX rendering enabled")
        except:
            print("⚠ LaTeX not available, falling back to mathtext")
    
    if not latex_available:
        style_dict.update({
            'text.usetex': False,
            'font.family': 'serif',
            'mathtext.fontset': 'cm',  # Computer Modern (LaTeX-like)
        })
        if not use_latex:
            print("✓ Mathtext mode (no LaTeX required)")
    
    plt.rcParams.update(style_dict)
    
    return latex_available


def despine(ax, offset=10):
    """
    Remove top/right spines and offset left/bottom spines to create gap at origin.
    
    Args:
        ax: Matplotlib axis (or list/array of axes)
        offset: Distance in points to offset spines from data area
    """
    # Handle arrays/lists of axes
    if hasattr(ax, 'flat'):
        for a in ax.flat:
            despine(a, offset)
        return
    elif hasattr(ax, '__iter__') and not isinstance(ax, plt.Axes):
        for a in ax:
            despine(a, offset)
        return
    
    # Hide top and right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Offset left and bottom (creates the gap at origin)
    ax.spines['left'].set_position(('outward', offset))
    ax.spines['bottom'].set_position(('outward', offset))
    
    # Move ticks accordingly
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def smooth(y, sigma=5):
    """Gaussian smoothing for cleaner plots."""
    y = np.array(y, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return y
    # Interpolate NaNs for smoothing
    y_filled = np.interp(np.arange(len(y)), np.where(mask)[0], y[mask])
    return gaussian_filter1d(y_filled, sigma=sigma)


def plot_with_band(ax, x, y, color, label=None, smooth_sigma=None, no_band=False,
                   alpha_line=1.0, alpha_fill=0.2, linewidth=2, linestyle='-'):
    """
    Plot line with optional smoothing and confidence band.
    
    Args:
        ax: Matplotlib axis
        x: x values
        y: y values
        color: Line and fill color
        label: Legend label
        smooth_sigma: If set, apply Gaussian smoothing and show ±1 std band
        alpha_line: Line opacity
        alpha_fill: Fill opacity for confidence band
        linewidth: Line width
        linestyle: Line style ('-', '--', ':', etc.)
    
    Returns:
        line: The Line2D object (or None if insufficient data)
    """
    y = np.array(y)
    x = np.array(x)
    mask = ~np.isnan(y)
    
    if mask.sum() < 3:
        return None
    
    x_valid, y_valid = x[mask], y[mask]

    if smooth_sigma:
        y_smooth = smooth(y_valid, sigma=smooth_sigma)
    
    if not no_band:
        # Estimate band from residuals
        residuals = y_valid - y_smooth
        std = np.std(residuals)
        
        ax.fill_between(x_valid, y_smooth - std, y_smooth + std, 
                       color=color, alpha=alpha_fill, linewidth=0)
        line, = ax.plot(x_valid, y_smooth, color=color, label=label, 
                       alpha=alpha_line, linewidth=linewidth, linestyle=linestyle)
    elif smooth_sigma:
        line, = ax.plot(x_valid, y_smooth, color=color, label=label, 
                       alpha=alpha_line, linewidth=linewidth, linestyle=linestyle)
    else:
        line, = ax.plot(x_valid, y_valid, color=color, label=label, 
                       alpha=alpha_line, linewidth=linewidth, linestyle=linestyle)
    
    return line


# ============================================================================
# FIGURE HELPERS
# ============================================================================

def save_figure(fig, name, output_dir='./figures', formats=['pdf', 'png']):
    """
    Save figure in multiple formats.
    
    Args:
        fig: Matplotlib figure
        name: Base filename (without extension)
        output_dir: Output directory
        formats: List of formats to save
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for fmt in formats:
        fig.savefig(output_dir / f'{name}.{fmt}')
    
    print(f"  → Saved: {name}.{{{', '.join(formats)}}}")


# ============================================================================
# EXAMPLE / TEST
# ============================================================================

if __name__ == "__main__":
    # Test the style
    setup_style(use_latex=False)  # Set True if you have LaTeX
    
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    
    # Left: Line plot with bands
    ax = axes[0]
    x = np.linspace(0, 10, 200)
    y1 = np.sin(x) + np.random.normal(0, 0.15, len(x))
    y2 = np.cos(x) + np.random.normal(0, 0.15, len(x))
    
    plot_with_band(ax, x, y1, COLORS['blue'], label=r'$\sin(x)$', smooth_sigma=5)
    plot_with_band(ax, x, y2, COLORS['orange'], label=r'$\cos(x)$', smooth_sigma=5, linestyle='--')
    
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.set_title(r'\textbf{A}  Line plot', loc='left')
    ax.legend()
    
    # Right: Bar plot
    ax = axes[1]
    x_bar = np.arange(6)
    heights = np.random.rand(6) * 0.5 + 0.5
    colors = get_layer_colors(6)
    
    ax.bar(x_bar, heights, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Category')
    ax.set_ylabel('Value')
    ax.set_title(r'\textbf{B}  Bar plot', loc='left')
    
    # Apply despine to all axes (creates gap at origin)
    despine(axes)
    
    plt.tight_layout()
    save_figure(fig, 'style_test')
    plt.show()