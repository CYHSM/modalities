import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from modalities.config.config import load_app_config_dict
from modalities.models.utils import ModelTypeEnum, get_model_from_config


# Attribute under which ModalitiesHFAdapter stores the native Modalities model.
# Checkpoint keys are prefixed with it so that from_pretrained matches them.
_ADAPTER_SUBMODULE_NAME = "model"


class ModalitiesHFAdapterConfig(PretrainedConfig):
    model_type = "modalities_adapter"

    def __init__(self, modalities_config: Optional[dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.modalities_config = modalities_config or {}

        # The adapter recomputes the full sequence on every forward: it has no KV cache.
        # Saying so keeps transformers' generate() from building a DynamicCache, which
        # needs the standard config fields below and crashes without them.
        self.use_cache = False

        # Mirror the Modalities model dims onto the standard HF config names. Generative
        # evaluation (e.g. gsm8k) reads these; loglikelihood-only tasks never touch them.
        model_section = self.modalities_config.get("model_raw", self.modalities_config.get("model"))
        model_config = (model_section or {}).get("config", {})
        for hf_name, modalities_name in (
            ("num_hidden_layers", "n_layer"),
            ("num_attention_heads", "n_head_q"),
            ("num_key_value_heads", "n_head_kv"),
            ("hidden_size", "n_embd"),
            ("vocab_size", "vocab_size"),
            ("max_position_embeddings", "sequence_length"),
        ):
            value = model_config.get(modalities_name)
            if value is not None and not isinstance(value, str):
                setattr(self, hf_name, value)


class ModalitiesHFAdapter(PreTrainedModel):
    config_class = ModalitiesHFAdapterConfig
    base_model_prefix = "modalities_model"

    def __init__(self, config: ModalitiesHFAdapterConfig):
        super().__init__(config)
        self.config = config
        modalities_config = config.modalities_config

        if not modalities_config:
            raise ValueError("modalities_config is missing from model config.")

        # Reconstruct the native Modalities model via the component factory.
        # The model definition in modalities_config is typically under 'model_raw' or 'model'
        if "model_raw" in modalities_config:
            model_config = modalities_config["model_raw"]
        elif "model" in modalities_config:
            model_config = modalities_config["model"]
        else:
            raise KeyError("Neither 'model_raw' nor 'model' key found in modalities_config.")

        self.model = get_model_from_config(config={"model": model_config}, model_type=ModelTypeEnum.MODEL)
        self.sample_key = getattr(self.model, "sample_key", "input_ids")
        self.prediction_key = getattr(self.model, "prediction_key", "logits")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        inputs = {self.sample_key: input_ids}
        outputs = self.model(inputs)
        logits = self._extract_logits(outputs)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    def _extract_logits(self, outputs: Any) -> torch.Tensor:
        """Pulls the logits tensor out of whatever a Modalities model returns.

        Models return either a bare tensor or a dict keyed by ``prediction_key``. Models with
        auxiliary outputs (e.g. DualPathLLM, which also returns a ponder loss and a metrics bag)
        nest one more dict under that key, so unwrap until a tensor is reached.

        Args:
            outputs: The raw return value of the wrapped Modalities model.

        Returns:
            torch.Tensor: The logits.

        Raises:
            KeyError: If no logits tensor can be found in the model output.
        """
        while isinstance(outputs, dict):
            if self.prediction_key in outputs:
                outputs = outputs[self.prediction_key]
            elif "logits" in outputs:
                outputs = outputs["logits"]
            else:
                raise KeyError(
                    f"Model output has no '{self.prediction_key}' or 'logits' entry; got keys {list(outputs)}."
                )
        return outputs

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


def _load_dcp_state_dict(checkpoint_dir: Path, tmp_dir: Path) -> dict:
    """Load a DCP (Distributed Checkpoint Protocol) checkpoint and return the model state dict.

    This handles checkpoints saved by FSDP2's DCPCheckpointSaving, which produces
    __0_0.distcp + .metadata files instead of a single model.pt file.
    Uses torch.distributed.checkpoint.format_utils.dcp_to_torch_save (PyTorch 2.3+).

    Args:
        checkpoint_dir: Path to the DCP checkpoint directory containing .distcp files.
        tmp_dir: Directory in which the intermediate consolidated file is written.

    Returns:
        dict: The model state dict, with the raw Modalities parameter names.
    """
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    # dcp_to_torch_save converts the entire DCP checkpoint to a single torch.save file
    # The resulting file contains the full state dict (e.g. {"app": {...model state...}})
    tmp_path = tmp_dir / ".tmp_dcp_converted.pt"
    try:
        dcp_to_torch_save(str(checkpoint_dir), str(tmp_path))

        # Load and extract just the model weights from the nested structure
        full_state = torch.load(tmp_path, map_location="cpu", weights_only=False)

        # DCP saves as {"app": AppState} where AppState.state_dict() contains model weights
        # The exact structure depends on how AppState serializes, try common patterns
        model_state = full_state
        if isinstance(full_state, dict):
            if "app" in full_state:
                app_state = full_state["app"]
                if isinstance(app_state, dict) and "model" in app_state:
                    model_state = app_state["model"]
                else:
                    model_state = app_state

        return model_state
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _prefix_state_dict_keys(state_dict: dict) -> dict:
    """Prefixes raw Modalities parameter names with the adapter's submodule name.

    The checkpoint stores the bare Modalities model parameters (e.g. ``transformer.wte.weight``),
    but ``ModalitiesHFAdapter`` holds that model under ``self.model``. Without the prefix,
    ``from_pretrained`` treats every weight as unexpected and silently keeps the randomly
    initialized parameters.

    Args:
        state_dict: State dict using the raw Modalities parameter names.

    Returns:
        dict: State dict whose keys match ``ModalitiesHFAdapter.state_dict()``.
    """
    if all(key.startswith(f"{_ADAPTER_SUBMODULE_NAME}.") for key in state_dict):
        return state_dict
    return {f"{_ADAPTER_SUBMODULE_NAME}.{key}": value for key, value in state_dict.items()}


def _assert_config_is_resolved(config_dict: dict, modalities_config_path: Union[str, Path]) -> None:
    """Fails if the model section still contains unresolved ``${...}`` interpolations.

    The plain config copy in the experiment folder cannot be resolved outside of the training
    process (it references e.g. ``${modalities_env:experiment_id}``). Embedding it unresolved
    produces an adapter that only blows up much later, when the evaluation tries to load it,
    so reject it here instead.

    Args:
        config_dict: The loaded modalities config.
        modalities_config_path: Path the config was loaded from, used for the error message.

    Raises:
        ValueError: If an unresolved interpolation is found in the model section.
    """
    model_section = config_dict.get("model_raw", config_dict.get("model"))
    if model_section is None:
        return

    def find_unresolved(node: Any, path: str = "") -> Optional[str]:
        if isinstance(node, str) and "${" in node:
            return f"{path}: {node}"
        if isinstance(node, dict):
            for key, value in node.items():
                hit = find_unresolved(value, f"{path}.{key}" if path else str(key))
                if hit is not None:
                    return hit
        if isinstance(node, list):
            for idx, value in enumerate(node):
                hit = find_unresolved(value, f"{path}[{idx}]")
                if hit is not None:
                    return hit
        return None

    unresolved = find_unresolved(model_section)
    if unresolved is not None:
        raise ValueError(
            f"The model config in {modalities_config_path} contains unresolved interpolations "
            f"({unresolved}). Pass the fully resolved config (the '*.yaml.resolved' dump written "
            "to the experiment folder) instead."
        )


def export_generic_hf_adapter(
    checkpoint_file_path: Union[str, Path],
    modalities_config_path: Union[str, Path],
    output_dir: Union[str, Path],
    tokenizer_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Exports a Modalities model checkpoint to a Hugging Face compatible format

    using Hugging Face's auto_map feature.

    Args:
        checkpoint_file_path: Path to model state dict (.pt file).
        modalities_config_path: Path to Modalities config YAML.
        output_dir: Directory where HF checkpoint artifacts will be saved.
        tokenizer_dir: Optional directory containing tokenizer files.

    Returns:
        Path: Output directory path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load modalities config
    try:
        config_dict = load_app_config_dict(Path(modalities_config_path))
    except Exception:
        import yaml

        with open(modalities_config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    _assert_config_is_resolved(config_dict, modalities_config_path)

    # 2. Write modalities_adapter.py code to output_dir for auto_map
    adapter_code_path = output_dir / "modalities_adapter.py"
    current_file = Path(__file__)
    if current_file.exists():
        shutil.copy(current_file, adapter_code_path)
    else:
        # Fallback if executing from an inline context
        adapter_code_path.write_text(Path(__file__).read_text())

    # 3. Create HF config with auto_map
    hf_config = {
        "architectures": ["ModalitiesHFAdapter"],
        "model_type": "modalities_adapter",
        "auto_map": {
            "AutoConfig": "modalities_adapter.ModalitiesHFAdapterConfig",
            "AutoModelForCausalLM": "modalities_adapter.ModalitiesHFAdapter",
        },
        "modalities_config": config_dict,
    }

    # default=str: the resolved config carries pathlib.Path values (e.g. settings.config_file_path)
    # that json cannot encode on its own.
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2, default=str)

    # 4. Load state dict weights and save as pytorch_model.bin
    checkpoint_file_path = Path(checkpoint_file_path)
    target_weights_path = output_dir / "pytorch_model.bin"
    if checkpoint_file_path.exists():
        if checkpoint_file_path.is_dir():
            model_pt = checkpoint_file_path / "model.pt"
            distcp_files = list(checkpoint_file_path.glob("*.distcp"))
            if model_pt.exists():
                # Legacy single-file checkpoint
                state_dict = torch.load(model_pt, map_location="cpu", weights_only=False)
            elif distcp_files:
                # DCP (Distributed Checkpoint Protocol) format
                state_dict = _load_dcp_state_dict(checkpoint_file_path, tmp_dir=output_dir)
            else:
                raise FileNotFoundError(
                    f"No model.pt or .distcp files found in checkpoint folder {checkpoint_file_path}"
                )
        else:
            state_dict = torch.load(checkpoint_file_path, map_location="cpu", weights_only=False)
        torch.save(_prefix_state_dict_keys(state_dict), target_weights_path)

    # 5. Copy tokenizer files if available
    if tokenizer_dir is not None:
        tokenizer_dir = Path(tokenizer_dir)
        if tokenizer_dir.exists():
            for item in tokenizer_dir.glob("*"):
                if item.is_file() and not (output_dir / item.name).exists():
                    shutil.copy(item, output_dir / item.name)

    return output_dir
