import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import rich
import torch
import wandb
import yaml
from rich.console import Group
from rich.panel import Panel

from modalities.batch import EvaluationResultBatch, ResultItem
from modalities.config.config import WandbMode
from modalities.logging_broker.messages import Message
from modalities.logging_broker.subscriber import MessageSubscriberIF


# Keys in eval_result.metrics that are *tensor attachments* for visualization,
# not scalar metrics to log. The subscriber must skip these when building
# its wandb scalar dict, otherwise it will try to .value them as scalars.
_VIS_TENSOR_KEYS = {
    # new single-gate world:
    "eval_tokens",
    "eval_gate",
    "eval_expected_steps",
    "eval_delta_deep_norm",
    "eval_delta_wide_norm",
    # legacy names, harmless to keep skipping:
    "eval_gate_probs",
    "eval_gate_deep_probs",
    "eval_gate_wide_probs",
}


class DummyResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        pass

    def consume_dict(self, message_dict: dict[str, Any]):
        pass


class RichResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    def __init__(self, num_ranks: int) -> None:
        super().__init__()
        self.num_ranks = num_ranks

    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        eval_result = message.payload
        losses = {
            f"{eval_result.dataloader_tag} {loss_key}: {loss_values}"
            for loss_key, loss_values in eval_result.losses.items()
        }
        metrics = {
            f"{eval_result.dataloader_tag} {metric_key}: {metric_values}"
            for metric_key, metric_values in eval_result.metrics.items()
            if metric_key not in _VIS_TENSOR_KEYS
        }

        num_samples = eval_result.num_train_steps_done * self.num_ranks
        group_content = [f"[yellow]Iteration #{num_samples}:"]
        if losses:
            group_content.append("\nLosses:")
            group_content.extend(losses)
        if metrics:
            group_content.append("\nMetrics:")
            group_content.extend(metrics)
        if losses or metrics:
            rich.print(Panel(Group(*group_content)))

    def consume_dict(self, message_dict: dict[str, Any]):
        raise NotImplementedError


def _gate_color(g: float) -> str:
    """Map a single gate value in [0, 1] to a blue↔white↔red color.

    g = 0 -> pure blue     (token prefers capacity / wide FFN)
    g = 0.5 -> white       (neutral / averaged)
    g = 1 -> pure red      (token prefers compute / deep recursive)
    """
    g = max(0.0, min(1.0, float(g)))
    p = g - 0.5          # in [-0.5, 0.5]
    t = abs(p) * 2.0     # in [0, 1], saturation
    if p >= 0:
        # toward red
        return f"rgb(255, {int(255 * (1 - t))}, {int(255 * (1 - t))})"
    else:
        # toward blue
        return f"rgb({int(255 * (1 - t))}, {int(255 * (1 - t))}, 255)"


class WandBEvaluationResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    """A subscriber object for the WandBEvaluationResult observable."""

    def __init__(
        self,
        project: str,
        experiment_id: str,
        mode: WandbMode,
        logging_directory: Path,
        config_file_path: Path,
        entity: Optional[str] = None,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        with open(config_file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.run = wandb.init(
            entity=entity,
            project=project,
            name=experiment_id,
            mode=mode.value.lower(),
            dir=logging_directory,
            config=config,
            settings=wandb.Settings(init_timeout=120, console="off"),
        )

        self.run.log_artifact(config_file_path, name=f"config_{self.run.id}", type="config")

    def consume_dict(self, message_dict: dict[str, Any]):
        for k, v in message_dict.items():
            self.run.config[k] = v

    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        eval_result = message.payload

        losses = {
            f"{eval_result.dataloader_tag} {loss_key}": loss_values.value
            for loss_key, loss_values in eval_result.losses.items()
        }

        # Skip visualization-only tensor attachments when building the numeric
        # metrics dict; they are (B,T) or (L,B,T) tensors, not scalars.
        metrics = {}
        for metric_key, metric_values in eval_result.metrics.items():
            if metric_key in _VIS_TENSOR_KEYS:
                continue
            metrics[f"{eval_result.dataloader_tag} {metric_key}"] = metric_values.value

        # TODO step is not semantically correct here. Need to check if we can rename step to num_samples
        wandb.log(
            data=losses, step=eval_result.num_train_steps_done
        )
        wandb.log(
            data=metrics, step=eval_result.num_train_steps_done
        )
        throughput_metrics = {
            f"{eval_result.dataloader_tag} {metric_key}": metric_values.value
            for metric_key, metric_values in eval_result.throughput_metrics.items()
        }

        wandb.log(data=throughput_metrics, step=eval_result.num_train_steps_done)

        # ------------------------------------------------------------------
        # HTML visualization: single-gate routing per token
        # ------------------------------------------------------------------
        # The gate is a single value g ∈ [0,1] per (layer, example, token).
        # g -> 1 means the token routes to the compute/deep path.
        # g -> 0 means the token routes to the capacity/wide path.
        # Each token gets exactly one color on a blue↔white↔red scale.
        # ------------------------------------------------------------------
        has_tokens = "eval_tokens" in eval_result.metrics
        has_gate   = "eval_gate"   in eval_result.metrics
        has_steps  = "eval_expected_steps" in eval_result.metrics

        if has_tokens and has_gate and self.tokenizer is not None:
            tokens = eval_result.metrics["eval_tokens"].value          # (B, T)
            gates  = eval_result.metrics["eval_gate"].value            # (L, B, T) in [0,1]
            steps  = eval_result.metrics["eval_expected_steps"].value if has_steps else None

            num_layers = gates.shape[0]
            batch_size = tokens.shape[0]
            seq_len    = tokens.shape[1]

            master_html_parts = [
                "<div style='font-family: monospace; font-size: 14px; line-height: 1.7; "
                "color: black; background: #f9f9f9; padding: 15px;'>",
                "<h3 style='margin-top: 0;'>Gate Routing per Token</h3>",
                "<p style='margin: 5px 0;'>",
                "<span style='color:blue; font-weight: bold;'>Blue</span> = prefers capacity (wide FFN)"
                " &nbsp;&nbsp; ",
                "White = neutral (g ≈ 0.5) &nbsp;&nbsp; ",
                "<span style='color:red; font-weight: bold;'>Red</span> = prefers compute "
                "(deep / recursive)",
                "</p>",
                "<p style='margin: 5px 0; color: #555;'>"
                "Hover over a token to see its exact g value."
                "</p>",
                "<hr/>",
            ]

            for b_idx in range(batch_size):
                seq_tokens = tokens[b_idx].tolist()

                token_texts = []
                for t_idx in range(seq_len):
                    try:
                        token_text = self.tokenizer.decode([seq_tokens[t_idx]])
                    except Exception:
                        token_text = str(seq_tokens[t_idx])
                    # Whitespace-only tokens are invisible with colored background;
                    # prefix a middle-dot so the color is readable.
                    if token_text.strip() == "":
                        token_text = "·" + token_text
                    token_texts.append(token_text)

                master_html_parts.append(f"<h2>Example {b_idx + 1}</h2>")

                # --- Average across all layers ---
                master_html_parts.append(
                    "<strong>Avg across layers:</strong>"
                    "<div style='margin-bottom: 20px;'>"
                )
                for t_idx in range(seq_len):
                    g = gates[:, b_idx, t_idx].mean().item()
                    color = _gate_color(g)
                    master_html_parts.append(
                        f"<span title='g={g:.2f}' style='background-color: {color}; "
                        f"padding: 2px 3px; border-radius: 3px;'>{token_texts[t_idx]}</span>"
                    )
                master_html_parts.append("</div>")

                # --- Per-layer rows ---
                for l_idx in range(num_layers):
                    master_html_parts.append(
                        f"<strong>Layer {l_idx}:</strong>"
                        "<div style='margin-bottom: 8px;'>"
                    )
                    for t_idx in range(seq_len):
                        g = gates[l_idx, b_idx, t_idx].item()
                        color = _gate_color(g)
                        tip = f"g={g:.2f}"
                        if steps is not None:
                            tip += f", E[steps]={steps[l_idx, b_idx, t_idx].item():.2f}"
                        master_html_parts.append(
                            f"<span title='{tip}' style='background-color: {color}; "
                            f"padding: 2px 3px; border-radius: 3px;'>{token_texts[t_idx]}</span>"
                        )
                    master_html_parts.append("</div>")

                master_html_parts.append("<hr/>")

            master_html_parts.append("</div>")
            final_html = "".join(master_html_parts)

            wandb.log(
                {f"{eval_result.dataloader_tag} token_routing": wandb.Html(final_html)},
                step=eval_result.num_train_steps_done,
            )


class EvaluationResultToDiscSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    """A subscriber that writes EvaluationResultBatch messages to a JSONL file."""

    def __init__(self, output_file_path: Path) -> None:
        super().__init__()
        self.output_file_path = output_file_path
        self.output_file_path.parent.mkdir(parents=True, exist_ok=True)

    def consume_dict(self, message_dict: dict[str, Any]):
        """Optional: log config data if needed (here: no-op)."""
        pass

    @staticmethod
    def _convert_evaluation_result_batch(obj: EvaluationResultBatch) -> dict[str, Any]:
        """
        Recursively convert EvaluationResultBatch structure to JSON-serializable format.
        Handles dataclasses and torch.Tensor.
        """

        def shallow_asdict(obj):
            if not is_dataclass(obj):
                raise TypeError("shallow_asdict() should be called on dataclass instances")
            return {f.name: getattr(obj, f.name) for f in fields(obj)}

        if isinstance(obj, ResultItem):
            return obj.value.item() if obj.value.ndim == 0 else obj.value.tolist()
        elif is_dataclass(obj):
            result_dict = {}
            for k, v in shallow_asdict(obj).items():
                result_dict[k] = EvaluationResultToDiscSubscriber._convert_evaluation_result_batch(v)
            return result_dict

        elif isinstance(obj, dict):
            return {k: EvaluationResultToDiscSubscriber._convert_evaluation_result_batch(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [EvaluationResultToDiscSubscriber._convert_evaluation_result_batch(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.ndim == 0 else obj.tolist()
        else:
            return obj

    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Writes the evaluation result to the JSONL file if rank 0."""
        if torch.distributed.get_rank() == 0:
            eval_result = message.payload
            record_converted = EvaluationResultToDiscSubscriber._convert_evaluation_result_batch(eval_result)
            with self.output_file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record_converted) + "\n")