import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def group_by_layer(step_activations):
    layer_groups = {}
    
    for name, activation in step_activations.items():
        if ".layers." in name:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        if layer_idx not in layer_groups:
                            layer_groups[layer_idx] = {}
                        layer_groups[layer_idx][name] = activation
                        break
                    except ValueError:
                        continue
        else:
            key = name.split(".")[0] if "." in name else name
            if key not in layer_groups:
                layer_groups[key] = {}
            layer_groups[key][name] = activation
    
    return layer_groups


def normalize(values, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        p1, p99 = np.percentile(values, [1, 99])
        vmin, vmax = p1, p99
    clipped = np.clip(values, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin + 1e-8)


def combine_layer_activations(layer_dict):
    all_values = []
    index_map = []
    for name in sorted(layer_dict.keys()):
        values = layer_dict[name].flatten()
        all_values.extend(values)
        for i in range(len(values)):
            index_map.append((name, i))
    return np.array(all_values), index_map


def to_256x256(values):
    target_size = 256 * 256
    if len(values) < target_size:
        padded = np.zeros(target_size)
        padded[:len(values)] = values
        values = padded
    else:
        values = values[:target_size]
    return values.reshape(256, 256)


def apply_colormap(image, cmap_name="inferno"):
    import matplotlib.pyplot as plt
    
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(image)
    return (colored[:, :, :3] * 255).astype(np.uint8)


def create_grid(layer_images):
    n_layers = len(layer_images)
    grid_cols = int(math.ceil(math.sqrt(n_layers)))
    grid_rows = int(math.ceil(n_layers / grid_cols))
    
    height = grid_rows * 256
    width = grid_cols * 256
    
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    
    for idx, (_, img, cmap) in enumerate(layer_images):
        row = idx // grid_cols
        col = idx % grid_cols
        
        y_start = row * 256
        y_end = y_start + 256
        x_start = col * 256
        x_end = x_start + 256
        
        colored = apply_colormap(img, cmap)
        grid[y_start:y_end, x_start:x_end] = colored
    
    return grid


def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Arial.ttf", size)
        except:
            return ImageFont.load_default()


def add_text_overlay(image, title, stats_info=None):
    img_array = np.array(image)
    overlay_height = 150 if stats_info else 100
    new_height = img_array.shape[0] + overlay_height
    new_img = np.ones((new_height, img_array.shape[1], 3), dtype=np.uint8) * 30
    new_img[:img_array.shape[0]] = img_array
    result = Image.fromarray(new_img)
    
    draw = ImageDraw.Draw(result)
    font_large = get_font(28)
    font_small = get_font(16)
    
    y_start = img_array.shape[0] + 20
    draw.text((20, y_start), title, fill=(255, 255, 255), font=font_large)
    
    if stats_info:
        y_stats = y_start + 50
        draw.text((20, y_stats), stats_info, fill=(200, 200, 200), font=font_small)
    
    return result


def visualize_stats(*, stats_results, metric="t_stats", p_threshold=0.05, title=None):
    layer_groups_mean = group_by_layer(stats_results["mean_diff"])
    layer_groups_stat = group_by_layer(stats_results[metric])
    layer_groups_pval = group_by_layer(stats_results["p_values"])
    
    int_keys = [k for k in layer_groups_mean.keys() if isinstance(k, int)]
    str_keys = [k for k in layer_groups_mean.keys() if isinstance(k, str)]
    all_keys = sorted(int_keys) + sorted(str_keys)
    
    layer_images = []
    n_significant_total = 0
    n_total = 0
    
    for layer_id in all_keys:
        mean_dict = layer_groups_mean[layer_id]
        stat_dict = layer_groups_stat[layer_id]
        pval_dict = layer_groups_pval[layer_id]
        
        combined_stat, _ = combine_layer_activations(stat_dict)
        combined_pval, _ = combine_layer_activations(pval_dict)
        
        significant_mask = combined_pval < p_threshold
        n_significant_total += np.sum(significant_mask)
        n_total += len(combined_pval)
        
        masked_stat = np.where(significant_mask, combined_stat, 0)
        
        normalized = normalize(masked_stat, vmin=-5, vmax=5)
        img_array = to_256x256(normalized)
        
        layer_name = f"Layer_{layer_id}" if isinstance(layer_id, int) else str(layer_id)
        layer_images.append((layer_name, img_array, "RdBu_r"))
    
    grid = create_grid(layer_images)
    base_image = Image.fromarray(grid, mode="RGB")
    
    if title is None:
        title = f"Statistical Map ({metric})"
    
    pct_sig = 100 * n_significant_total / n_total if n_total > 0 else 0
    stats_info = f"p < {p_threshold}: {n_significant_total:,}/{n_total:,} dims ({pct_sig:.2f}%)"
    
    final_image = add_text_overlay(base_image, title, stats_info)
    
    return final_image, {"n_significant": n_significant_total, "n_total": n_total, "pct_significant": pct_sig}


def create_interactive_viewer(*, stats_results, output_path):
    import json
    
    layer_groups_stat = group_by_layer(stats_results["t_stats"])
    layer_groups_pval = group_by_layer(stats_results["p_values"])
    
    int_keys = [k for k in layer_groups_stat.keys() if isinstance(k, int)]
    str_keys = [k for k in layer_groups_stat.keys() if isinstance(k, str)]
    all_keys = sorted(int_keys) + sorted(str_keys)
    
    layers_data = []
    for layer_id in all_keys:
        stat_dict = layer_groups_stat[layer_id]
        pval_dict = layer_groups_pval[layer_id]
        
        combined_stat, index_map = combine_layer_activations(stat_dict)
        combined_pval, _ = combine_layer_activations(pval_dict)
        
        img_array = to_256x256(combined_stat)
        pval_array = to_256x256(combined_pval)
        
        index_map_256 = [None] * (256 * 256)
        for i, (name, idx) in enumerate(index_map[:256*256]):
            index_map_256[i] = {"name": name, "idx": idx}
        
        layer_name = f"Layer {layer_id}" if isinstance(layer_id, int) else str(layer_id)
        layers_data.append({
            "name": layer_name,
            "stats": img_array.tolist(),
            "pvals": pval_array.tolist(),
            "index_map": index_map_256
        })
    
    html_path = Path(output_path) / "interactive_stats.html"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Interactive Statistical Map</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }}
        .controls {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .control-group {{
            margin: 15px 0;
        }}
        label {{
            display: inline-block;
            width: 150px;
            font-weight: bold;
        }}
        input[type="range"] {{
            width: 300px;
            vertical-align: middle;
        }}
        .value-display {{
            display: inline-block;
            width: 100px;
            text-align: right;
            font-family: monospace;
        }}
        .stats {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .click-info {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            min-height: 80px;
        }}
        .click-info.active {{
            background: #3a4a3a;
            border: 2px solid #5a7a5a;
        }}
        .info-row {{
            margin: 8px 0;
            font-family: monospace;
        }}
        .info-label {{
            display: inline-block;
            width: 120px;
            color: #aaa;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(256px, 1fr));
            gap: 20px;
        }}
        .layer {{
            background: #2a2a2a;
            padding: 10px;
            border-radius: 8px;
        }}
        .layer-title {{
            text-align: center;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        canvas {{
            display: block;
            width: 100%;
            height: auto;
            image-rendering: pixelated;
            cursor: crosshair;
        }}
    </style>
</head>
<body>
    <h1>Interactive Statistical Significance Map</h1>
    
    <div class="controls">
        <div class="control-group">
            <label>P-value threshold:</label>
            <input type="range" id="pThreshold" min="0.001" max="0.1" step="0.001" value="0.05">
            <span class="value-display" id="pValue">0.050</span>
        </div>
        <div class="control-group">
            <label>T-stat threshold:</label>
            <input type="range" id="tThreshold" min="0" max="10" step="0.1" value="0">
            <span class="value-display" id="tValue">0.0</span>
        </div>
        <div class="control-group">
            <label>Colormap range:</label>
            <input type="range" id="colorRange" min="1" max="10" step="0.5" value="5">
            <span class="value-display" id="rangeValue">±5.0</span>
        </div>
    </div>
    
    <div class="stats">
        <strong>Significant dimensions:</strong> <span id="sigCount">-</span> / <span id="totalCount">-</span> (<span id="sigPercent">-</span>%)
    </div>
    
    <div class="click-info" id="clickInfo">
        <div style="color: #888; text-align: center;">Click on any pixel to see activation details</div>
    </div>
    
    <div class="grid" id="layerGrid"></div>
    
    <script>
        const layersData = {json.dumps(layers_data)};
        
        function applyColormap(value, vmin, vmax) {{
            const normalized = Math.max(0, Math.min(1, (value - vmin) / (vmax - vmin)));
            
            const r = Math.min(255, Math.max(0, 
                normalized < 0.5 
                    ? 255 * (1 - 2 * normalized)
                    : 255 * (2 * normalized - 1)
            ));
            const b = Math.min(255, Math.max(0,
                normalized < 0.5
                    ? 255 * (2 * normalized)
                    : 255 * (2 - 2 * normalized)
            ));
            const g = Math.min(255, Math.max(0, 255 * (1 - Math.abs(2 * normalized - 1))));
            
            return [r, g, b];
        }}
        
        function renderLayers() {{
            const pThreshold = parseFloat(document.getElementById('pThreshold').value);
            const tThreshold = parseFloat(document.getElementById('tThreshold').value);
            const colorRange = parseFloat(document.getElementById('colorRange').value);
            
            document.getElementById('pValue').textContent = pThreshold.toFixed(3);
            document.getElementById('tValue').textContent = tThreshold.toFixed(1);
            document.getElementById('rangeValue').textContent = `±${{colorRange.toFixed(1)}}`;
            
            let totalSig = 0;
            let totalDims = 0;
            
            const grid = document.getElementById('layerGrid');
            grid.innerHTML = '';
            
            layersData.forEach(layer => {{
                const div = document.createElement('div');
                div.className = 'layer';
                
                const title = document.createElement('div');
                title.className = 'layer-title';
                title.textContent = layer.name;
                div.appendChild(title);
                
                const canvas = document.createElement('canvas');
                canvas.width = 256;
                canvas.height = 256;
                div.appendChild(canvas);
                
                const ctx = canvas.getContext('2d');
                const imageData = ctx.createImageData(256, 256);
                
                let layerSig = 0;
                for (let i = 0; i < 256; i++) {{
                    for (let j = 0; j < 256; j++) {{
                        const idx = i * 256 + j;
                        const stat = layer.stats[i][j];
                        const pval = layer.pvals[i][j];
                        
                        const significant = pval < pThreshold && Math.abs(stat) > tThreshold;
                        if (significant) {{
                            layerSig++;
                            const [r, g, b] = applyColormap(stat, -colorRange, colorRange);
                            const pixelIdx = idx * 4;
                            imageData.data[pixelIdx] = r;
                            imageData.data[pixelIdx + 1] = g;
                            imageData.data[pixelIdx + 2] = b;
                            imageData.data[pixelIdx + 3] = 255;
                        }} else {{
                            const pixelIdx = idx * 4;
                            imageData.data[pixelIdx] = 30;
                            imageData.data[pixelIdx + 1] = 30;
                            imageData.data[pixelIdx + 2] = 30;
                            imageData.data[pixelIdx + 3] = 255;
                        }}
                    }}
                }}
                
                totalSig += layerSig;
                totalDims += 256 * 256;
                
                ctx.putImageData(imageData, 0, 0);
                
                canvas.addEventListener('click', (e) => {{
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    const x = Math.floor((e.clientX - rect.left) * scaleX);
                    const y = Math.floor((e.clientY - rect.top) * scaleY);
                    
                    const pixelIdx = y * 256 + x;
                    const stat = layer.stats[y][x];
                    const pval = layer.pvals[y][x];
                    const mapping = layer.index_map[pixelIdx];
                    
                    const clickInfo = document.getElementById('clickInfo');
                    clickInfo.className = 'click-info active';
                    
                    if (mapping && mapping.name) {{
                        clickInfo.innerHTML = `
                            <div class="info-row"><span class="info-label">Layer:</span> ${{layer.name}}</div>
                            <div class="info-row"><span class="info-label">Activation:</span> ${{mapping.name}}</div>
                            <div class="info-row"><span class="info-label">Index:</span> ${{mapping.idx}}</div>
                            <div class="info-row"><span class="info-label">T-statistic:</span> ${{stat.toFixed(4)}}</div>
                            <div class="info-row"><span class="info-label">P-value:</span> ${{pval.toExponential(4)}}</div>
                            <div class="info-row"><span class="info-label">Significant:</span> ${{pval < pThreshold && Math.abs(stat) > tThreshold ? 'Yes' : 'No'}}</div>
                        `;
                    }} else {{
                        clickInfo.innerHTML = `
                            <div class="info-row"><span class="info-label">Layer:</span> ${{layer.name}}</div>
                            <div class="info-row"><span class="info-label">Pixel:</span> (${{x}}, ${{y}})</div>
                            <div class="info-row"><span class="info-label">T-statistic:</span> ${{stat.toFixed(4)}}</div>
                            <div class="info-row"><span class="info-label">P-value:</span> ${{pval.toExponential(4)}}</div>
                            <div class="info-row" style="color: #888;">(Padding region - no activation mapped)</div>
                        `;
                    }}
                }});
                
                grid.appendChild(div);
            }});
            
            document.getElementById('sigCount').textContent = totalSig.toLocaleString();
            document.getElementById('totalCount').textContent = totalDims.toLocaleString();
            document.getElementById('sigPercent').textContent = ((100 * totalSig / totalDims).toFixed(2));
        }}
        
        document.getElementById('pThreshold').addEventListener('input', renderLayers);
        document.getElementById('tThreshold').addEventListener('input', renderLayers);
        document.getElementById('colorRange').addEventListener('input', renderLayers);
        
        renderLayers();
    </script>
</body>
</html>"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path