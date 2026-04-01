import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

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


def _preference_color_rgb(deep_val: float, wide_val: float) -> str:
    """Blue (wide) <-> White (neutral) <-> Red (deep).

    pref = deep - wide, clamped to [-1, 1].
    """
    pref = max(-1.0, min(1.0, deep_val - wide_val))
    if pref >= 0:
        r, g, b = 255, int(255 * (1 - pref)), int(255 * (1 - pref))
    else:
        t = -pref
        r, g, b = int(255 * (1 - t)), int(255 * (1 - t)), 255
    return f"rgb({r}, {g}, {b})"


class WandBEvaluationResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    """A subscriber object for the WandBEvaluationResult observable."""

    def __init__(
        self,
        project: str,
        experiment_id: str,
        mode: WandbMode,
        logging_directory: Path,
        config_file_path: Path,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        with open(config_file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.run = wandb.init(
            project=project,
            name=experiment_id,
            mode=mode.value.lower(),
            dir=logging_directory,
            config=config,
            settings=wandb.Settings(init_timeout=120, console="off"),
        )

        self.run.log_artifact(config_file_path, name=f"config_{wandb.run.id}", type="config")

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
        metrics = {}
        for metric_key, metric_values in eval_result.metrics.items():
            # Skip visualization-only keys so they don't break standard numeric logging
            if metric_key in [
                "eval_tokens", "eval_gate_probs",
                "eval_gate_deep_probs", "eval_gate_wide_probs",
                "eval_expected_steps",
            ]:
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

        # --- HTML VISUALIZATION ---
        has_tokens = "eval_tokens" in eval_result.metrics
        has_deep = "eval_gate_deep_probs" in eval_result.metrics
        has_wide = "eval_gate_wide_probs" in eval_result.metrics

        if has_tokens and (has_deep or has_wide) and self.tokenizer is not None:
            tokens = eval_result.metrics["eval_tokens"].value
            gates_deep = eval_result.metrics["eval_gate_deep_probs"].value if has_deep else None
            gates_wide = eval_result.metrics["eval_gate_wide_probs"].value if has_wide else None

            # Find number of layers dynamically based on whichever gate is present
            if gates_deep is not None:
                num_layers = gates_deep.shape[0]
            else:
                num_layers = gates_wide.shape[0]

            batch_size = tokens.shape[0]
            seq_len = tokens.shape[1]

            master_html_parts = [
                "<div style='font-family: monospace; font-size: 14px; line-height: 1.5; color: black; background: #f9f9f9; padding: 15px;'>",
                "<h3 style='margin-top: 0;'>Decoupled Gate Activation Heatmaps</h3>",
                "<p style='margin: 5px 0;'><strong>Deep Track:</strong> White (0.0) ➝ <span style='color:red; font-weight: bold;'>Red (1.0)</span></p>",
                "<p style='margin: 5px 0;'><strong>Wide Track:</strong> White (0.0) ➝ <span style='color:blue; font-weight: bold;'>Blue (1.0)</span></p>",
                "<p style='margin: 5px 0;'><strong>Preference:</strong> <span style='color:blue; font-weight: bold;'>Blue (wide)</span> ➝ White (neutral) ➝ <span style='color:red; font-weight: bold;'>Red (deep)</span></p>",
                "<hr/>",
            ]

            def get_deep_color(prob):
                v = int((1.0 - prob) * 255)
                return f"rgb(255, {v}, {v})"

            def get_wide_color(prob):
                v = int((1.0 - prob) * 255)
                return f"rgb({v}, {v}, 255)"

            for b_idx in range(batch_size):
                seq_tokens = tokens[b_idx].tolist()

                token_texts = []
                for t_idx in range(seq_len):
                    try:
                        token_text = self.tokenizer.decode([seq_tokens[t_idx]])
                    except Exception:
                        token_text = str(seq_tokens[t_idx])
                    token_texts.append(token_text)

                master_html_parts.append(f"<h2>Example {b_idx + 1}</h2>")

                # --- Average across all layers ---
                master_html_parts.append("<strong>Average across all layers:</strong><br/><div style='margin-bottom: 20px;'>")

                if gates_deep is not None:
                    master_html_parts.append("<div style='line-height: 2.0;'><strong>Deep: </strong>")
                    for t_idx in range(seq_len):
                        prob = gates_deep[:, b_idx, t_idx].mean().item()
                        color = get_deep_color(prob)
                        master_html_parts.append(f"<span style='background-color: {color}; padding: 2px; border-radius: 3px;'>{token_texts[t_idx]}</span>")
                    master_html_parts.append("</div>")

                if gates_wide is not None:
                    master_html_parts.append("<div style='line-height: 2.0;'><strong>Wide: </strong>")
                    for t_idx in range(seq_len):
                        prob = gates_wide[:, b_idx, t_idx].mean().item()
                        color = get_wide_color(prob)
                        master_html_parts.append(f"<span style='background-color: {color}; padding: 2px; border-radius: 3px;'>{token_texts[t_idx]}</span>")
                    master_html_parts.append("</div>")

                if gates_deep is not None and gates_wide is not None:
                    master_html_parts.append("<div style='line-height: 2.0;'><strong>Pref: </strong>")
                    for t_idx in range(seq_len):
                        deep_val = gates_deep[:, b_idx, t_idx].mean().item()
                        wide_val = gates_wide[:, b_idx, t_idx].mean().item()
                        color = _preference_color_rgb(deep_val, wide_val)
                        master_html_parts.append(f"<span style='background-color: {color}; padding: 2px; border-radius: 3px;'>{token_texts[t_idx]}</span>")
                    master_html_parts.append("</div>")

                master_html_parts.append("</div>")

                # --- Individual Layers ---
                for l_idx in range(num_layers):
                    master_html_parts.append(f"<strong>Layer {l_idx}:</strong><br/><div style='margin-bottom: 15px;'>")

                    if gates_deep is not None:
                        master_html_parts.append("<div style='line-height: 2.0;'><strong>Deep: </strong>")
                        for t_idx in range(seq_len):
                            prob = gates_deep[l_idx, b_idx, t_idx].item()
                            color = get_deep_color(prob)
                            master_html_parts.append(f"<span style='background-color: {color}; padding: 2px; border-radius: 3px;'>{token_texts[t_idx]}</span>")
                        master_html_parts.append("</div>")

                    if gates_wide is not None:
                        master_html_parts.append("<div style='line-height: 2.0;'><strong>Wide: </strong>")
                        for t_idx in range(seq_len):
                            prob = gates_wide[l_idx, b_idx, t_idx].item()
                            color = get_wide_color(prob)
                            master_html_parts.append(f"<span style='background-color: {color}; padding: 2px; border-radius: 3px;'>{token_texts[t_idx]}</span>")
                        master_html_parts.append("</div>")

                    if gates_deep is not None and gates_wide is not None:
                        master_html_parts.append("<div style='line-height: 2.0;'><strong>Pref: </strong>")
                        for t_idx in range(seq_len):
                            deep_val = gates_deep[l_idx, b_idx, t_idx].item()
                            wide_val = gates_wide[l_idx, b_idx, t_idx].item()
                            color = _preference_color_rgb(deep_val, wide_val)
                            master_html_parts.append(f"<span style='background-color: {color}; padding: 2px; border-radius: 3px;'>{token_texts[t_idx]}</span>")
                        master_html_parts.append("</div>")

                    master_html_parts.append("</div>")

                master_html_parts.append("<hr/>")

            master_html_parts.append("</div>")
            final_html = "".join(master_html_parts)

            wandb.log({f"{eval_result.dataloader_tag} token_routing": wandb.Html(final_html)}, step=eval_result.num_train_steps_done)


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