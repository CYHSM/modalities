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
from modalities.models.model_factory import ModelFactory


class ModalitiesHFAdapterConfig(PretrainedConfig):
    model_type = "modalities_adapter"

    def __init__(self, modalities_config: Optional[dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.modalities_config = modalities_config or {}


class ModalitiesHFAdapter(PreTrainedModel):
    config_class = ModalitiesHFAdapterConfig
    base_model_prefix = "modalities_model"

    def __init__(self, config: ModalitiesHFAdapterConfig):
        super().__init__(config)
        self.config = config
        modalities_config = config.modalities_config

        if not modalities_config:
            raise ValueError("modalities_config is missing from model config.")

        # Reconstruct native Modalities model using ModelFactory
        # The model definition in modalities_config is typically under 'model_raw' or 'model'
        if "model_raw" in modalities_config:
            model_config = modalities_config["model_raw"]
        elif "model" in modalities_config:
            model_config = modalities_config["model"]
        else:
            raise KeyError("Neither 'model_raw' nor 'model' key found in modalities_config.")

        self.model = ModelFactory.get_model(model_config)
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

        if isinstance(outputs, dict):
            logits = outputs[self.prediction_key]
        else:
            logits = outputs

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

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


def _load_dcp_and_save(checkpoint_dir: Path, target_weights_path: Path, config_dict: dict):
    """Load a DCP (Distributed Checkpoint Protocol) checkpoint and save as a single pytorch_model.bin.

    This handles checkpoints saved by FSDP2's DCPCheckpointSaving, which produces
    __0_0.distcp + .metadata files instead of a single model.pt file.
    Uses torch.distributed.checkpoint.format_utils.dcp_to_torch_save (PyTorch 2.3+).

    Args:
        checkpoint_dir: Path to the DCP checkpoint directory containing .distcp files.
        target_weights_path: Path where the consolidated pytorch_model.bin will be saved.
        config_dict: The modalities config dict (unused, kept for API compatibility).
    """
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    # dcp_to_torch_save converts the entire DCP checkpoint to a single torch.save file
    # The resulting file contains the full state dict (e.g. {"app": {...model state...}})
    tmp_path = target_weights_path.parent / ".tmp_dcp_converted.pt"
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

        torch.save(model_state, target_weights_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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

    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # 4. Load state dict weights and save as pytorch_model.bin
    checkpoint_file_path = Path(checkpoint_file_path)
    target_weights_path = output_dir / "pytorch_model.bin"
    if checkpoint_file_path.exists():
        if checkpoint_file_path.is_dir():
            model_pt = checkpoint_file_path / "model.pt"
            distcp_files = list(checkpoint_file_path.glob("*.distcp"))
            if model_pt.exists():
                # Legacy single-file checkpoint
                shutil.copy(model_pt, target_weights_path)
            elif distcp_files:
                # DCP (Distributed Checkpoint Protocol) format
                _load_dcp_and_save(checkpoint_file_path, target_weights_path, config_dict)
            else:
                raise FileNotFoundError(
                    f"No model.pt or .distcp files found in checkpoint folder {checkpoint_file_path}"
                )
        else:
            shutil.copy(checkpoint_file_path, target_weights_path)

    # 5. Copy tokenizer files if available
    if tokenizer_dir is not None:
        tokenizer_dir = Path(tokenizer_dir)
        if tokenizer_dir.exists():
            for item in tokenizer_dir.glob("*"):
                if item.is_file() and not (output_dir / item.name).exists():
                    shutil.copy(item, output_dir / item.name)

    return output_dir
