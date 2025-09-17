from pathlib import Path

import numpy as np
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def extract_weights_from_checkpoint(checkpoint_path, window=1000, normalize=True):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    all_weights = []
    for param in model.parameters():
        all_weights.append(param.detach().cpu().flatten())

    all_weights = torch.cat(all_weights).numpy()
    print(f"Extracted {all_weights.shape} weights from {checkpoint_path}")
    sampled_weights = all_weights[::window].copy()

    if normalize:
        sampled_weights = (sampled_weights - sampled_weights.mean()) / (sampled_weights.std() + 1e-8)

    del all_weights, model
    torch.cuda.empty_cache()

    return sampled_weights


def create_weight_matrix(base_paths, base_model_path, checkpoint_numbers, window=1000, normalize=True):
    all_weights = []

    for base_path in tqdm(base_paths, desc="Processing datasets"):
        base_weights = extract_weights_from_checkpoint(base_model_path, window=window, normalize=normalize)
        all_weights.append(base_weights)

        for cp_num in checkpoint_numbers[1:]:
            cp_path = f"{base_path}/checkpoint-{cp_num}"
            weights = extract_weights_from_checkpoint(cp_path, window=window, normalize=normalize)
            all_weights.append(weights)

    return np.stack(all_weights)


def reduce_dimensions(weight_matrix, method="pca", save_path=None):
    if method == "pca":
        reducer = PCA(n_components=10, random_state=42)
        coords_2d = reducer.fit_transform(weight_matrix)
    elif method == "pca+tsne":
        pca = PCA(n_components=50, random_state=42)
        data_reduced = pca.fit_transform(weight_matrix)
        tsne = TSNE(n_components=2, random_state=42)
        coords_2d = tsne.fit_transform(data_reduced)
    elif method == "umap":
        reducer = umap.UMAP(n_components=10, random_state=42)
        coords_2d = reducer.fit_transform(weight_matrix)
    elif method == "pca+umap":
        pca = PCA(n_components=50, random_state=42)
        data_reduced = pca.fit_transform(weight_matrix)
        reducer = umap.UMAP(n_components=10, random_state=42)
        coords_2d = reducer.fit_transform(data_reduced)

    if save_path:
        np.save(save_path, coords_2d)
        print(f"Saved {method} coordinates to {save_path}")

    return coords_2d


def main(window=1000, method="pca", normalize=True, force_recompute=False):
    base_paths = [
        "/raid/s3/opengptx/mfrey/instruct/checkpoints_sft/full_code",
        "/raid/s3/opengptx/mfrey/instruct/checkpoints_sft/full_general",
        "/raid/s3/opengptx/mfrey/instruct/checkpoints_sft/full_math",
        "/raid/s3/opengptx/mfrey/instruct/checkpoints_sft/full_mix",
    ]

    base_model_path = "/raid/s3/opengptx/mfrey/instruct/checkpoints_sft/full_baseline/checkpoint-0"
    checkpoint_numbers = [0] + list(range(5000, 105000, 5000))
    save_dir = Path("/raid/s3/opengptx/mfrey/cp_analysis/dim_reduction")
    save_dir.mkdir(parents=True, exist_ok=True)

    weight_matrix_path = save_dir / f"weight_matrix_window{window}_norm{normalize}.npy"
    coords_path = save_dir / f"coords_2d_{method}_window{window}_norm{normalize}.npy"

    if weight_matrix_path.exists() and not force_recompute:
        print(f"Loading existing weight matrix from {weight_matrix_path}")
        weight_matrix = np.load(weight_matrix_path)
    else:
        print("Creating weight matrix...")
        weight_matrix = create_weight_matrix(base_paths, base_model_path, checkpoint_numbers, window, normalize)
        np.save(weight_matrix_path, weight_matrix)
        print(f"Saved weight matrix to {weight_matrix_path}")

    print(f"Weight matrix shape: {weight_matrix.shape}")

    if coords_path.exists() and not force_recompute:
        print(f"Loading existing coordinates from {coords_path}")
        coords_2d = np.load(coords_path)
    else:
        print(f"Applying {method} dimensionality reduction...")
        coords_2d = reduce_dimensions(weight_matrix, method, coords_path)

    print(f"2D coordinates shape: {coords_2d.shape}")
    return weight_matrix, coords_2d


if __name__ == "__main__":
    window = 100
    method = "pca"
    force_recompute = True
    normalize = False

    weight_matrix, coords_2d = main(window, method, normalize, force_recompute)
