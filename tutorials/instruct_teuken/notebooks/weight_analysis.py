import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
from plotly.subplots import make_subplots
from transformers import AutoModel


class LLMWeightAnalyzer:
    """
    A comprehensive framework for analyzing weight differences between base and fine-tuned language models.

    This tool helps understand:
    - Which layers change most during fine-tuning
    - How different components (attention, MLP, embeddings) adapt
    - The magnitude and distribution of weight changes
    - Insights into where mathematical reasoning improvements occur
    """

    def __init__(
        self, base_model_path: str, finetuned_model_path: str, device: str = "auto", allow_vocab_mismatch: bool = True
    ):
        """
        Initialize the analyzer with base and fine-tuned models.

        Args:
            base_model_path: Path or HuggingFace model ID for base model
            finetuned_model_path: Path or HuggingFace model ID for fine-tuned model
            device: Device to load models on ("auto", "cpu", "cuda")
            allow_vocab_mismatch: Whether to allow vocabulary size differences (common in fine-tuning)
        """
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.device = self._get_device(device)
        self.allow_vocab_mismatch = allow_vocab_mismatch

        # Load models
        print("Loading base model...")
        self.base_model = AutoModel.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        print("Loading fine-tuned model...")
        self.finetuned_model = AutoModel.from_pretrained(
            finetuned_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

        # Move to device
        self.base_model = self.base_model.to(self.device)
        self.finetuned_model = self.finetuned_model.to(self.device)

        # Storage for analysis results
        self.weight_diffs = {}
        self.layer_analysis = {}
        self.component_analysis = {}
        self.summary_stats = {}
        self.skipped_params = []

        # Verify models are compatible
        self._verify_model_compatibility()

        print(f"✅ Models loaded successfully on {self.device}")
        print(f"📊 Model has {self._count_parameters():,} parameters")
        if self.skipped_params:
            print(f"⚠️ Skipped {len(self.skipped_params)} parameters due to shape mismatches (likely vocab changes)")

    def _get_device(self, device: str) -> torch.device:
        """Determine the appropriate device for computation."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _count_parameters(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.base_model.parameters())

    def _verify_model_compatibility(self):
        """Verify that both models have compatible architectures."""
        base_params = dict(self.base_model.named_parameters())
        ft_params = dict(self.finetuned_model.named_parameters())

        if set(base_params.keys()) != set(ft_params.keys()):
            missing_in_ft = set(base_params.keys()) - set(ft_params.keys())
            missing_in_base = set(ft_params.keys()) - set(base_params.keys())

            if missing_in_ft:
                print(f"⚠️ Parameters in base but not fine-tuned: {missing_in_ft}")
            if missing_in_base:
                print(f"⚠️ Parameters in fine-tuned but not base: {missing_in_base}")

            raise ValueError("Models have different parameter names - incompatible architectures")

        # Check for shape mismatches
        for name in base_params:
            base_shape = base_params[name].shape
            ft_shape = ft_params[name].shape

            if base_shape != ft_shape:
                # Check if this is a vocabulary/embedding mismatch
                is_vocab_param = any(
                    vocab_key in name.lower()
                    for vocab_key in ["embed", "lm_head", "word_embeddings", "token_embeddings"]
                )

                if is_vocab_param and self.allow_vocab_mismatch:
                    print(f"📝 Vocabulary size change detected in '{name}': {base_shape} → {ft_shape}")
                    self.skipped_params.append(name)
                    continue
                else:
                    raise ValueError(
                        f"Parameter {name} has different shapes in the two models - "
                        f"Base: {base_shape}, Fine-tuned: {ft_shape}"
                    )

    def _handle_vocab_mismatch(self, base_weight: torch.Tensor, ft_weight: torch.Tensor, param_name: str) -> Dict:
        """
        Handle vocabulary size mismatches by comparing the overlapping portion.

        Args:
            base_weight: Base model weight tensor
            ft_weight: Fine-tuned model weight tensor
            param_name: Name of the parameter

        Returns:
            Dictionary of computed metrics for the overlapping portion
        """
        # Determine the overlapping dimensions
        min_shape = tuple(min(b, f) for b, f in zip(base_weight.shape, ft_weight.shape))

        # Extract overlapping portions
        if len(base_weight.shape) == 2:  # Most embedding matrices are 2D
            base_overlap = base_weight[: min_shape[0], : min_shape[1]]
            ft_overlap = ft_weight[: min_shape[0], : min_shape[1]]
        elif len(base_weight.shape) == 1:  # Bias vectors
            base_overlap = base_weight[: min_shape[0]]
            ft_overlap = ft_weight[: min_shape[0]]
        else:
            # For other shapes, just take the overlapping hypercube
            slices = tuple(slice(0, dim) for dim in min_shape)
            base_overlap = base_weight[slices]
            ft_overlap = ft_weight[slices]

        # Compute metrics on overlapping portion
        diff_metrics = {}
        diff_metrics["l2"] = torch.norm(ft_overlap - base_overlap, p=2).item()
        diff_metrics["l2_normalized"] = (
            (diff_metrics["l2"] / torch.norm(base_overlap, p=2).item())
            if torch.norm(base_overlap, p=2).item() > 0
            else 0
        )
        diff_metrics["l1"] = torch.norm(ft_overlap - base_overlap, p=1).item()
        diff_metrics["relative_mean"] = torch.abs((ft_overlap - base_overlap) / (base_overlap + 1e-8)).mean().item()

        # Cosine similarity
        flat_base = base_overlap.flatten()
        flat_ft = ft_overlap.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(flat_base.unsqueeze(0), flat_ft.unsqueeze(0)).item()
        diff_metrics["cosine_distance"] = 1 - cosine_sim

        # Additional info about the mismatch
        diff_metrics["param_count"] = base_overlap.numel()
        diff_metrics["param_shape"] = list(base_overlap.shape)
        diff_metrics["vocab_expansion"] = True
        diff_metrics["original_base_shape"] = list(base_weight.shape)
        diff_metrics["original_ft_shape"] = list(ft_weight.shape)
        diff_metrics["added_tokens"] = ft_weight.shape[0] - base_weight.shape[0] if len(ft_weight.shape) >= 1 else 0

        return diff_metrics

    def compute_weight_differences(self, metrics: List[str] = ["l2", "l1", "cosine", "relative"]):
        """
        Compute various weight difference metrics between models.

        Args:
            metrics: List of metrics to compute ["l2", "l1", "cosine", "relative", "max"]
        """
        print("🔄 Computing weight differences...")

        base_params = dict(self.base_model.named_parameters())
        ft_params = dict(self.finetuned_model.named_parameters())

        processed_count = 0
        skipped_count = 0

        for name in base_params:
            # Skip parameters that were marked as incompatible
            if name in self.skipped_params:
                skipped_count += 1
                continue

            base_weight = base_params[name].detach()
            ft_weight = ft_params[name].detach()

            # Handle shape mismatches (vocabulary expansion)
            if base_weight.shape != ft_weight.shape:
                if self.allow_vocab_mismatch:
                    diff_metrics = self._handle_vocab_mismatch(base_weight, ft_weight, name)
                    self.weight_diffs[name] = diff_metrics
                    processed_count += 1
                    continue
                else:
                    print(f"⚠️ Skipping {name} due to shape mismatch: {base_weight.shape} vs {ft_weight.shape}")
                    skipped_count += 1
                    continue

            # Normal case: same shapes
            diff_metrics = {}

            if "l2" in metrics:
                diff_metrics["l2"] = torch.norm(ft_weight - base_weight, p=2).item()
                diff_metrics["l2_normalized"] = (
                    (diff_metrics["l2"] / torch.norm(base_weight, p=2).item())
                    if torch.norm(base_weight, p=2).item() > 0
                    else 0
                )

            if "l1" in metrics:
                diff_metrics["l1"] = torch.norm(ft_weight - base_weight, p=1).item()
                diff_metrics["l1_normalized"] = (
                    (diff_metrics["l1"] / torch.norm(base_weight, p=1).item())
                    if torch.norm(base_weight, p=1).item() > 0
                    else 0
                )

            if "cosine" in metrics:
                flat_base = base_weight.flatten()
                flat_ft = ft_weight.flatten()
                cosine_sim = torch.nn.functional.cosine_similarity(flat_base.unsqueeze(0), flat_ft.unsqueeze(0)).item()
                diff_metrics["cosine_distance"] = 1 - cosine_sim

            if "relative" in metrics:
                relative_change = torch.abs((ft_weight - base_weight) / (base_weight + 1e-8))
                diff_metrics["relative_mean"] = relative_change.mean().item()
                diff_metrics["relative_std"] = relative_change.std().item()

            if "max" in metrics:
                diff_metrics["max_change"] = torch.max(torch.abs(ft_weight - base_weight)).item()

            # Additional statistics
            diff_metrics["param_count"] = base_weight.numel()
            diff_metrics["param_shape"] = list(base_weight.shape)
            diff_metrics["vocab_expansion"] = False

            self.weight_diffs[name] = diff_metrics
            processed_count += 1

        print(f"✅ Computed differences for {processed_count} parameter groups")
        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} parameters due to shape mismatches")

        # Report vocabulary changes if any
        vocab_changes = [name for name, metrics in self.weight_diffs.items() if metrics.get("vocab_expansion", False)]
        if vocab_changes:
            print(f"📝 Analyzed {len(vocab_changes)} vocabulary parameters with size changes")

    def analyze_by_layers(self):
        """Analyze weight changes grouped by transformer layers."""
        print("🔍 Analyzing changes by layer...")

        layer_stats = defaultdict(lambda: defaultdict(list))

        for param_name, metrics in self.weight_diffs.items():
            layer_info = self._parse_parameter_name(param_name)

            if layer_info["layer_num"] is not None:
                layer_key = f"layer_{layer_info['layer_num']}"
                component_key = layer_info["component"]

                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        layer_stats[layer_key][metric_name].append(value)
                        layer_stats[f"{layer_key}_{component_key}"][metric_name].append(value)

        # Aggregate statistics
        self.layer_analysis = {}
        for layer_component, metrics in layer_stats.items():
            self.layer_analysis[layer_component] = {}
            for metric_name, values in metrics.items():
                if values:
                    self.layer_analysis[layer_component][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "total": np.sum(values),
                    }

    def analyze_by_components(self):
        """Analyze weight changes grouped by component types."""
        print("🔍 Analyzing changes by component type...")

        component_stats = defaultdict(lambda: defaultdict(list))

        for param_name, metrics in self.weight_diffs.items():
            layer_info = self._parse_parameter_name(param_name)
            component = layer_info["component"]

            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    component_stats[component][metric_name].append(value)

        # Aggregate statistics
        self.component_analysis = {}
        for component, metrics in component_stats.items():
            self.component_analysis[component] = {}
            for metric_name, values in metrics.items():
                if values:
                    self.component_analysis[component][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "total": np.sum(values),
                        "count": len(values),
                    }

    def _parse_parameter_name(self, param_name: str) -> Dict:
        """Parse parameter name to extract layer and component information."""
        layer_info = {"layer_num": None, "component": "other", "subcomponent": None}

        # Extract layer number
        layer_match = re.search(r"layer[s]?\.(\d+)", param_name)
        if layer_match:
            layer_info["layer_num"] = int(layer_match.group(1))

        # Determine component type
        if "embed" in param_name.lower():
            layer_info["component"] = "embedding"
        elif "attention" in param_name.lower() or "attn" in param_name.lower():
            layer_info["component"] = "attention"
            if "query" in param_name or "q_proj" in param_name:
                layer_info["subcomponent"] = "query"
            elif "key" in param_name or "k_proj" in param_name:
                layer_info["subcomponent"] = "key"
            elif "value" in param_name or "v_proj" in param_name:
                layer_info["subcomponent"] = "value"
            elif "output" in param_name or "o_proj" in param_name:
                layer_info["subcomponent"] = "output"
        elif "mlp" in param_name.lower() or "feed_forward" in param_name.lower() or "ffn" in param_name.lower():
            layer_info["component"] = "mlp"
            if "gate" in param_name or "up" in param_name:
                layer_info["subcomponent"] = "up"
            elif "down" in param_name:
                layer_info["subcomponent"] = "down"
        elif "norm" in param_name.lower() or "ln" in param_name.lower():
            layer_info["component"] = "norm"
        elif "lm_head" in param_name.lower() or "classifier" in param_name.lower():
            layer_info["component"] = "head"

        return layer_info

    def generate_summary_stats(self):
        """Generate overall summary statistics."""
        print("📈 Generating summary statistics...")

        all_l2_diffs = [metrics.get("l2", 0) for metrics in self.weight_diffs.values()]
        all_relative_diffs = [metrics.get("relative_mean", 0) for metrics in self.weight_diffs.values()]
        all_param_counts = [metrics.get("param_count", 0) for metrics in self.weight_diffs.values()]

        self.summary_stats = {
            "total_parameters": sum(all_param_counts),
            "total_parameter_groups": len(self.weight_diffs),
            "l2_change_stats": {
                "mean": np.mean(all_l2_diffs),
                "std": np.std(all_l2_diffs),
                "min": np.min(all_l2_diffs),
                "max": np.max(all_l2_diffs),
                "total": np.sum(all_l2_diffs),
            },
            "relative_change_stats": {
                "mean": np.mean(all_relative_diffs),
                "std": np.std(all_relative_diffs),
                "min": np.min(all_relative_diffs),
                "max": np.max(all_relative_diffs),
            },
        }

    def plot_layer_wise_changes(self, metric: str = "l2", save_path: Optional[str] = None):
        """Plot weight changes by layer."""
        if not self.layer_analysis:
            self.analyze_by_layers()

        # Extract layer-wise data
        layer_data = []
        for key, stats in self.layer_analysis.items():
            if key.startswith("layer_") and "_" not in key[6:]:  # Pure layer keys
                layer_num = int(key.split("_")[1])
                if metric in stats:
                    layer_data.append(
                        {"layer": layer_num, "change": stats[metric]["mean"], "std": stats[metric]["std"]}
                    )

        layer_data = sorted(layer_data, key=lambda x: x["layer"])

        if not layer_data:
            print(f"⚠️ No layer data found for metric '{metric}'")
            return

        plt.figure(figsize=(12, 6))

        layers = [d["layer"] for d in layer_data]
        changes = [d["change"] for d in layer_data]
        stds = [d["std"] for d in layer_data]

        plt.errorbar(layers, changes, yerr=stds, marker="o", capsize=5, capthick=2)
        plt.xlabel("Layer Number")
        plt.ylabel(f"{metric.upper()} Change")
        plt.title(f"Weight Changes by Layer ({metric.upper()} metric)")
        plt.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(layers, changes, 1)
        p = np.poly1d(z)
        plt.plot(layers, p(layers), "--", alpha=0.7, color="red", label=f"Trend (slope: {z[0]:.2e})")
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_component_comparison(self, metric: str = "l2", save_path: Optional[str] = None):
        """Plot weight changes by component type."""
        if not self.component_analysis:
            self.analyze_by_components()

        components = []
        changes = []
        stds = []

        for component, stats in self.component_analysis.items():
            if metric in stats:
                components.append(component)
                changes.append(stats[metric]["mean"])
                stds.append(stats[metric]["std"])

        if not components:
            print(f"⚠️ No component data found for metric '{metric}'")
            return

        plt.figure(figsize=(10, 6))

        bars = plt.bar(components, changes, yerr=stds, capsize=5, alpha=0.7)
        plt.xlabel("Component Type")
        plt.ylabel(f"{metric.upper()} Change")
        plt.title(f"Weight Changes by Component ({metric.upper()} metric)")
        plt.xticks(rotation=45)

        # Color bars by magnitude
        colors = plt.cm.viridis([c / max(changes) for c in changes])
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_parameter_heatmap(self, metric: str = "l2_normalized", top_n: int = 50, save_path: Optional[str] = None):
        """Plot heatmap of parameter changes."""
        # Get top N changed parameters
        param_changes = [(name, metrics.get(metric, 0)) for name, metrics in self.weight_diffs.items()]
        param_changes = sorted(param_changes, key=lambda x: x[1], reverse=True)[:top_n]

        names = [name for name, _ in param_changes]
        values = [value for _, value in param_changes]

        # Create heatmap data
        heatmap_data = np.array(values).reshape(-1, 1)

        plt.figure(figsize=(15, max(8, len(names) * 0.3)))

        sns.heatmap(
            heatmap_data,
            yticklabels=[name.replace(".", ".\n") for name in names],
            xticklabels=[f"{metric.upper()}"],
            annot=True,
            fmt=".2e",
            cmap="viridis",
            cbar_kws={"label": f"{metric.upper()} Change"},
        )

        plt.title(f"Top {top_n} Parameter Changes ({metric.upper()} metric)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_interactive_analysis(self, save_path: Optional[str] = None):
        """Create an interactive Plotly dashboard."""
        if not self.layer_analysis or not self.component_analysis:
            self.analyze_by_layers()
            self.analyze_by_components()

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Layer-wise L2 Changes",
                "Component-wise L2 Changes",
                "Parameter Count by Component",
                "Change Distribution",
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]],
        )

        # Layer-wise plot
        layer_data = []
        for key, stats in self.layer_analysis.items():
            if key.startswith("layer_") and "_" not in key[6:]:
                layer_num = int(key.split("_")[1])
                if "l2" in stats:
                    layer_data.append((layer_num, stats["l2"]["mean"]))

        layer_data = sorted(layer_data)
        if layer_data:
            layers, l2_changes = zip(*layer_data)
            fig.add_trace(go.Scatter(x=layers, y=l2_changes, mode="lines+markers", name="L2 Change"), row=1, col=1)

        # Component-wise plot
        components = []
        component_changes = []
        param_counts = []

        for component, stats in self.component_analysis.items():
            if "l2" in stats:
                components.append(component)
                component_changes.append(stats["l2"]["mean"])
                param_counts.append(stats["l2"]["count"])

        if components:
            fig.add_trace(go.Bar(x=components, y=component_changes, name="L2 Change"), row=1, col=2)

            fig.add_trace(go.Bar(x=components, y=param_counts, name="Parameter Count"), row=2, col=1)

        # Distribution plot
        all_l2_changes = [metrics.get("l2", 0) for metrics in self.weight_diffs.values()]
        fig.add_trace(go.Histogram(x=all_l2_changes, nbinsx=50, name="L2 Distribution"), row=2, col=2)

        fig.update_layout(height=800, showlegend=False, title_text="LLM Weight Change Analysis Dashboard")

        if save_path:
            fig.write_html(save_path)

        fig.show()

    def get_insights(self) -> Dict:
        """Generate automated insights from the analysis."""
        if not self.summary_stats:
            self.generate_summary_stats()

        insights = {
            "top_changed_layers": [],
            "top_changed_components": [],
            "vocabulary_changes": [],
            "recommendations": [],
            "key_findings": [],
        }

        # Check for vocabulary changes
        vocab_changes = []
        for name, metrics in self.weight_diffs.items():
            if metrics.get("vocab_expansion", False):
                vocab_changes.append(
                    {
                        "parameter": name,
                        "added_tokens": metrics.get("added_tokens", 0),
                        "change_magnitude": metrics.get("l2", 0),
                        "original_shape": metrics.get("original_base_shape"),
                        "new_shape": metrics.get("original_ft_shape"),
                    }
                )
        insights["vocabulary_changes"] = vocab_changes

        # Top changed layers
        if self.layer_analysis:
            layer_changes = []
            for key, stats in self.layer_analysis.items():
                if key.startswith("layer_") and "_" not in key[6:] and "l2" in stats:
                    layer_num = int(key.split("_")[1])
                    layer_changes.append((layer_num, stats["l2"]["mean"]))

            layer_changes = sorted(layer_changes, key=lambda x: x[1], reverse=True)
            insights["top_changed_layers"] = layer_changes[:5]

        # Top changed components
        if self.component_analysis:
            component_changes = []
            for component, stats in self.component_analysis.items():
                if "l2" in stats:
                    component_changes.append((component, stats["l2"]["mean"]))

            component_changes = sorted(component_changes, key=lambda x: x[1], reverse=True)
            insights["top_changed_components"] = component_changes

        # Generate recommendations
        if vocab_changes:
            total_added = sum(vc["added_tokens"] for vc in vocab_changes)
            insights["recommendations"].append(
                f"Model vocabulary expanded by {total_added} tokens - analyze new token embeddings for task-specific adaptations"  # noqa: E501
            )

        if insights["top_changed_components"]:
            top_component = insights["top_changed_components"][0][0]
            insights["recommendations"].append(
                f"Focus analysis on {top_component} components as they show the highest changes"
            )

        if insights["top_changed_layers"]:
            top_layers = [str(layer) for layer, _ in insights["top_changed_layers"][:3]]
            insights["recommendations"].append(
                f"Layers {', '.join(top_layers)} show the most adaptation during fine-tuning"
            )

        # Key findings
        if vocab_changes:
            insights["key_findings"].append(f"Vocabulary expanded in {len(vocab_changes)} parameters")

        if insights["top_changed_layers"]:
            highest_layer = insights["top_changed_layers"][0][0]
            lowest_layer = insights["top_changed_layers"][-1][0]
            insights["key_findings"].append(f"Layer {highest_layer} changed most, layer {lowest_layer} changed least")

        return insights

    def export_results(self, output_dir: str = "weight_analysis_results"):
        """Export all analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Export raw data
        with open(output_path / "weight_differences.json", "w") as f:
            # Convert tensor data to serializable format
            serializable_diffs = {}
            for name, metrics in self.weight_diffs.items():
                serializable_diffs[name] = {k: v for k, v in metrics.items() if isinstance(v, (int, float, list))}
            json.dump(
                serializable_diffs, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x)
            )

        # Export analysis results
        with open(output_path / "layer_analysis.json", "w") as f:
            json.dump(
                self.layer_analysis, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x)
            )

        with open(output_path / "component_analysis.json", "w") as f:
            json.dump(
                self.component_analysis, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x)
            )

        with open(output_path / "summary_stats.json", "w") as f:
            json.dump(
                self.summary_stats, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x)
            )

        # Export insights
        insights = self.get_insights()
        with open(output_path / "insights.json", "w") as f:
            json.dump(insights, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x))

        # Generate plots
        print(f"📁 Exporting results to {output_path}")
        self.plot_layer_wise_changes(save_path=output_path / "layer_changes.png")
        self.plot_component_comparison(save_path=output_path / "component_changes.png")
        self.plot_parameter_heatmap(save_path=output_path / "parameter_heatmap.png")
        self.plot_interactive_analysis(save_path=output_path / "interactive_dashboard.html")

        print(f"✅ Results exported to {output_path}")

    def run_full_analysis(self, export: bool = True):
        """Run the complete analysis pipeline."""
        print("🚀 Starting full weight difference analysis...")

        # Run all analysis steps
        self.compute_weight_differences()
        self.analyze_by_layers()
        self.analyze_by_components()
        self.generate_summary_stats()

        # Display insights
        insights = self.get_insights()
        print("\n📊 KEY INSIGHTS:")
        print("=" * 50)

        if insights["vocabulary_changes"]:
            print("📝 Vocabulary Changes:")
            for vc in insights["vocabulary_changes"]:
                print(f"   {vc['parameter']}: +{vc['added_tokens']} tokens (L2 change: {vc['change_magnitude']:.2e})")

        if insights["top_changed_layers"]:
            print("\n🔥 Top Changed Layers:")
            for layer, change in insights["top_changed_layers"]:
                print(f"   Layer {layer}: {change:.2e}")

        if insights["top_changed_components"]:
            print("\n🧩 Component Changes (L2 norm):")
            for component, change in insights["top_changed_components"]:
                print(f"   {component}: {change:.2e}")

        if insights["key_findings"]:
            print("\n🔍 Key Findings:")
            for finding in insights["key_findings"]:
                print(f"   • {finding}")

        if insights["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in insights["recommendations"]:
                print(f"   • {rec}")

        # Generate visualizations
        print("\n📈 Generating visualizations...")
        self.plot_layer_wise_changes()
        self.plot_component_comparison()
        self.plot_interactive_analysis()

        if export:
            self.export_results()

        print("\n✅ Analysis complete!")


# Example usage
def main():
    """Example usage of the LLM Weight Analyzer."""

    # Initialize analyzer with your models
    analyzer = LLMWeightAnalyzer(
        base_model_path="microsoft/DialoGPT-small",  # Replace with your base model
        finetuned_model_path="microsoft/DialoGPT-small",  # Replace with your fine-tuned model
        device="auto",
        allow_vocab_mismatch=True,  # Allow vocabulary size differences (common in fine-tuning)
    )

    # Run complete analysis
    analyzer.run_full_analysis()

    # Or run individual components
    # analyzer.compute_weight_differences()
    # analyzer.analyze_by_layers()
    # analyzer.plot_layer_wise_changes()


if __name__ == "__main__":
    # For demo purposes, we'll show how to use it
    print(
        """
    🔬 LLM Weight Difference Analysis Framework
    ==========================================
    
    Usage:
    1. Initialize with your base and fine-tuned models:
       analyzer = LLMWeightAnalyzer("base_model_path", "finetuned_model_path", 
                                  allow_vocab_mismatch=True)
    
    2. Run full analysis:
       analyzer.run_full_analysis()
    
    3. Or run individual components:
       analyzer.compute_weight_differences()
       analyzer.analyze_by_layers()
       analyzer.plot_layer_wise_changes()
    
    Features:
    ✅ Multiple weight difference metrics (L2, L1, cosine, relative)
    ✅ Layer-wise analysis
    ✅ Component-wise analysis (attention, MLP, embeddings)
    ✅ Handles vocabulary expansion during fine-tuning
    ✅ Interactive visualizations
    ✅ Automated insights generation
    ✅ Export capabilities
    
    This framework will help you understand:
    • Which layers change most during fine-tuning
    • How different components adapt
    • Where mathematical reasoning improvements occur
    • Impact of vocabulary expansion
    """
    )
