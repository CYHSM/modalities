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
            key = name
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


def create_activation_map(index_map, target_resolution):
    if len(index_map) == 0:
        return [[None] * target_resolution for _ in range(target_resolution)]
    
    total_elements = len(index_map)
    current_size = int(np.sqrt(total_elements))
    
    if current_size * current_size != total_elements:
        source_resolution = 256
        padded_map = index_map + [(None, -1)] * (256 * 256 - total_elements)
    else:
        source_resolution = current_size
        padded_map = index_map
    
    map_2d = []
    for i in range(source_resolution):
        row = []
        for j in range(source_resolution):
            idx = i * source_resolution + j
            row.append(padded_map[idx][0] if idx < len(padded_map) and padded_map[idx][0] is not None else None)
        map_2d.append(row)
    
    if source_resolution == target_resolution:
        return map_2d
    
    downsampled_map = []
    scale = source_resolution / target_resolution
    
    for i in range(target_resolution):
        row = []
        for j in range(target_resolution):
            src_i_start = int(i * scale)
            src_i_end = int((i + 1) * scale)
            src_j_start = int(j * scale)
            src_j_end = int((j + 1) * scale)
            
            names_in_region = []
            for si in range(src_i_start, min(src_i_end, source_resolution)):
                for sj in range(src_j_start, min(src_j_end, source_resolution)):
                    name = map_2d[si][sj]
                    if name is not None:
                        names_in_region.append(name)
            
            if names_in_region:
                most_common = max(set(names_in_region), key=names_in_region.count)
                row.append(most_common)
            else:
                row.append(None)
        downsampled_map.append(row)
    
    return downsampled_map


def to_resolution(values, resolution):
    target_size = resolution * resolution
    if len(values) < target_size:
        padded = np.zeros(target_size)
        padded[:len(values)] = values
        values = padded
    else:
        values = values[:target_size]
    return values.reshape(resolution, resolution)


def downsample_array(arr, target_resolution):
    if len(arr) == 0:
        return np.zeros((target_resolution, target_resolution))
    
    current_size = int(np.sqrt(len(arr)))
    if current_size * current_size != len(arr):
        arr_2d = to_resolution(arr, 256)
        current_size = 256
    else:
        arr_2d = arr.reshape(current_size, current_size)
    
    if current_size == target_resolution:
        return arr_2d
    
    from scipy.ndimage import zoom
    scale = target_resolution / current_size
    return zoom(arr_2d, scale, order=1)


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
    
    input_layers = []
    output_layers = []
    other_layers = []
    
    for k in str_keys:
        k_lower = k.lower().replace("_", "").replace(".", "")
        k_orig = k
        if 'embed' in k_lower or 'rotary' in k_lower:
            input_layers.append(k_orig)
        elif 'lm' in k_lower and 'head' in k_lower:
            output_layers.append(k_orig)
        elif 'norm' in k_lower:
            output_layers.append(k_orig)
        else:
            other_layers.append(k_orig)
    
    input_layers.sort()
    output_layers.sort()
    other_layers.sort()
    
    all_keys = input_layers + sorted(int_keys) + other_layers + output_layers
    
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
        img_array = to_resolution(normalized, 256)
        
        layer_images.append((str(layer_id), img_array, "RdBu_r"))
    
    grid = create_grid(layer_images)
    base_image = Image.fromarray(grid, mode="RGB")
    
    if title is None:
        title = f"Statistical Map ({metric})"
    
    pct_sig = 100 * n_significant_total / n_total if n_total > 0 else 0
    stats_info = f"p < {p_threshold}: {n_significant_total:,}/{n_total:,} dims ({pct_sig:.2f}%)"
    
    final_image = add_text_overlay(base_image, title, stats_info)
    
    return final_image, {"n_significant": n_significant_total, "n_total": n_total, "pct_significant": pct_sig}


def create_interactive_viewer(*, stats_results, output_path, downsample_resolution=64):
    import json
    from scipy.ndimage import zoom
    
    layer_groups_stat = group_by_layer(stats_results["t_stats"])
    layer_groups_pval = group_by_layer(stats_results["p_values"])
    layer_groups_cohens = group_by_layer(stats_results["cohens_d"])
    layer_groups_mean = group_by_layer(stats_results["mean_diff"])
    layer_groups_pooled = group_by_layer(stats_results["pooled_std"])
    
    int_keys = [k for k in layer_groups_stat.keys() if isinstance(k, int)]
    str_keys = [k for k in layer_groups_stat.keys() if isinstance(k, str)]
    
    input_layers = []
    output_layers = []
    other_layers = []
    
    for k in str_keys:
        k_lower = k.lower().replace("_", "").replace(".", "")
        k_orig = k
        if 'embed' in k_lower or 'rotary' in k_lower:
            input_layers.append(k_orig)
        elif 'lm' in k_lower and 'head' in k_lower:
            output_layers.append(k_orig)
        elif 'norm' in k_lower:
            output_layers.append(k_orig)
        else:
            other_layers.append(k_orig)
    
    input_layers.sort()
    output_layers.sort()
    other_layers.sort()
    
    all_keys = input_layers + sorted(int_keys) + other_layers + output_layers
    
    layers_data = []
    for layer_id in all_keys:
        stat_dict = layer_groups_stat[layer_id]
        pval_dict = layer_groups_pval[layer_id]
        cohens_dict = layer_groups_cohens[layer_id]
        mean_dict = layer_groups_mean[layer_id]
        pooled_dict = layer_groups_pooled[layer_id]
        
        combined_stat, index_map = combine_layer_activations(stat_dict)
        combined_pval, _ = combine_layer_activations(pval_dict)
        combined_cohens, _ = combine_layer_activations(cohens_dict)
        combined_mean, _ = combine_layer_activations(mean_dict)
        combined_pooled, _ = combine_layer_activations(pooled_dict)
        
        stat_2d = downsample_array(combined_stat, downsample_resolution)
        pval_2d = downsample_array(combined_pval, downsample_resolution)
        cohens_2d = downsample_array(combined_cohens, downsample_resolution)
        mean_2d = downsample_array(combined_mean, downsample_resolution)
        pooled_2d = downsample_array(combined_pooled, downsample_resolution)
        
        activation_map = create_activation_map(index_map, downsample_resolution)
        
        layers_data.append({
            "name": str(layer_id),
            "stats": stat_2d.tolist(),
            "pvals": pval_2d.tolist(),
            "cohens": cohens_2d.tolist(),
            "mean_diff": mean_2d.tolist(),
            "pooled_std": pooled_2d.tolist(),
            "activation_map": activation_map
        })
    
    print(f"Creating interactive HTML with {len(layers_data)} layers...")
    
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
            width: 180px;
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
            width: 140px;
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
    <h1>Interactive Statistical Significance Map (Resolution: {downsample_resolution}x{downsample_resolution})</h1>
    
    <div class="controls">
        <div class="control-group">
            <label>P-value threshold:</label>
            <input type="range" id="pThreshold" min="0.001" max="1.001" step="0.001" value="1.001">
            <span class="value-display" id="pValue">0.050</span>
        </div>
        <div class="control-group">
            <label>|T-stat| threshold:</label>
            <input type="range" id="tThreshold" min="0" max="100" step="0.1" value="0">
            <span class="value-display" id="tValue">0.0</span>
        </div>
        <div class="control-group">
            <label>|Cohen's d| threshold:</label>
            <input type="range" id="cohensThreshold" min="0" max="10" step="0.01" value="0">
            <span class="value-display" id="cohensValue">0.00</span>
        </div>
        <div class="control-group">
            <label>|Mean diff| threshold:</label>
            <input type="range" id="meanThreshold" min="0" max="10" step="0.01" value="0">
            <span class="value-display" id="meanValue">0.00</span>
        </div>
        <div class="control-group">
            <label>Pooled std threshold:</label>
            <input type="range" id="pooledThreshold" min="0" max="10" step="0.01" value="0">
            <span class="value-display" id="pooledValue">0.00</span>
        </div>
        <div class="control-group">
            <label>Colormap range:</label>
            <input type="range" id="colorRange" min="1" max="100" step="1" value="5">
            <span class="value-display" id="rangeValue">±5.0</span>
        </div>
    </div>
    
    <div class="stats">
        <strong>Surviving dimensions:</strong> <span id="sigCount">-</span> / <span id="totalCount">-</span> (<span id="sigPercent">-</span>%)
    </div>
    
    <div class="click-info" id="clickInfo">
        <div style="color: #888; text-align: center;">Click on any pixel to see activation details</div>
    </div>
    
    <div class="grid" id="layerGrid"></div>
    
    <script>
        const layersData = {json.dumps(layers_data)};
        const resolution = {downsample_resolution};
        
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
            const cohensThreshold = parseFloat(document.getElementById('cohensThreshold').value);
            const meanThreshold = parseFloat(document.getElementById('meanThreshold').value);
            const pooledThreshold = parseFloat(document.getElementById('pooledThreshold').value);
            const colorRange = parseFloat(document.getElementById('colorRange').value);
            
            document.getElementById('pValue').textContent = pThreshold.toFixed(3);
            document.getElementById('tValue').textContent = tThreshold.toFixed(1);
            document.getElementById('cohensValue').textContent = cohensThreshold.toFixed(2);
            document.getElementById('meanValue').textContent = meanThreshold.toFixed(2);
            document.getElementById('pooledValue').textContent = pooledThreshold.toFixed(2);
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
                canvas.width = resolution;
                canvas.height = resolution;
                div.appendChild(canvas);
                
                const ctx = canvas.getContext('2d');
                const imageData = ctx.createImageData(resolution, resolution);
                
                let layerSig = 0;
                for (let i = 0; i < resolution; i++) {{
                    for (let j = 0; j < resolution; j++) {{
                        const idx = i * resolution + j;
                        const stat = layer.stats[i][j];
                        const pval = layer.pvals[i][j];
                        const cohens = layer.cohens[i][j];
                        const mean_diff = layer.mean_diff[i][j];
                        const pooled_std = layer.pooled_std[i][j];
                        
                        const significant = pval < pThreshold && 
                                          Math.abs(stat) > tThreshold && 
                                          Math.abs(cohens) > cohensThreshold &&
                                          Math.abs(mean_diff) > meanThreshold &&
                                          pooled_std > pooledThreshold;
                        
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
                totalDims += resolution * resolution;
                
                ctx.putImageData(imageData, 0, 0);
                
                canvas.addEventListener('click', (e) => {{
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    const x = Math.floor((e.clientX - rect.left) * scaleX);
                    const y = Math.floor((e.clientY - rect.top) * scaleY);
                    
                    const stat = layer.stats[y][x];
                    const pval = layer.pvals[y][x];
                    const cohens = layer.cohens[y][x];
                    const mean_diff = layer.mean_diff[y][x];
                    const pooled_std = layer.pooled_std[y][x];
                    const activation_name = layer.activation_map[y][x];
                    
                    const clickInfo = document.getElementById('clickInfo');
                    clickInfo.className = 'click-info active';
                    
                    const isSig = pval < pThreshold && 
                                Math.abs(stat) > tThreshold && 
                                Math.abs(cohens) > cohensThreshold &&
                                Math.abs(mean_diff) > meanThreshold &&
                                pooled_std > pooledThreshold;
                    
                    let activationInfo = '';
                    if (activation_name) {{
                        activationInfo = `<div class="info-row"><span class="info-label">Activation:</span> ${{activation_name}}</div>`;
                    }} else {{
                        activationInfo = `<div class="info-row"><span class="info-label">Activation:</span> (padding region)</div>`;
                    }}
                    
                    clickInfo.innerHTML = `
                        <div class="info-row"><span class="info-label">Layer group:</span> ${{layer.name}}</div>
                        ${{activationInfo}}
                        <div class="info-row"><span class="info-label">Pixel position:</span> (${{x}}, ${{y}}) [downsampled {downsample_resolution}x{downsample_resolution}]</div>
                        <div class="info-row"><span class="info-label">T-statistic:</span> ${{stat.toFixed(4)}}</div>
                        <div class="info-row"><span class="info-label">P-value:</span> ${{pval.toExponential(4)}}</div>
                        <div class="info-row"><span class="info-label">Cohen's d:</span> ${{cohens.toFixed(4)}}</div>
                        <div class="info-row"><span class="info-label">Mean diff:</span> ${{mean_diff.toFixed(4)}}</div>
                        <div class="info-row"><span class="info-label">Pooled std:</span> ${{pooled_std.toFixed(4)}}</div>
                        <div class="info-row"><span class="info-label">Survives filters:</span> ${{isSig ? 'Yes' : 'No'}}</div>
                    `;
                }});
                
                grid.appendChild(div);
            }});
            
            document.getElementById('sigCount').textContent = totalSig.toLocaleString();
            document.getElementById('totalCount').textContent = totalDims.toLocaleString();
            document.getElementById('sigPercent').textContent = ((100 * totalSig / totalDims).toFixed(2));
        }}
        
        document.getElementById('pThreshold').addEventListener('input', renderLayers);
        document.getElementById('tThreshold').addEventListener('input', renderLayers);
        document.getElementById('cohensThreshold').addEventListener('input', renderLayers);
        document.getElementById('meanThreshold').addEventListener('input', renderLayers);
        document.getElementById('pooledThreshold').addEventListener('input', renderLayers);
        document.getElementById('colorRange').addEventListener('input', renderLayers);
        
        renderLayers();
    </script>
</body>
</html>"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path