import json
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from modalities.conversion.generic_adapter import export_generic_hf_adapter
from modalities.models.gpt2.gpt2_model import GPT2ModelFactory
from modalities.models.model_factory import ModelFactory


@pytest.fixture
def dummy_modalities_config(tmp_path):
    config_dict = {
        "model_raw": {
            "component_key": "model",
            "variant_key": "gpt2",
            "config": {
                "sample_key": "input_ids",
                "prediction_key": "logits",
                "vocab_size": 128,
                "n_layer": 2,
                "n_head_q": 2,
                "n_head_kv": 2,
                "ffn_hidden": 128,
                "n_embd": 128,
                "dropout": 0.0,
                "bias": False,
                "sequence_length": 16,
                "poe_type": "NOPE",
                "activation_type": "swiglu",
                "attention_implementation": "manual",
                "attention_config": {
                    "qkv_transforms": []
                },
                "attention_norm_config": {
                    "norm_type": "pytorch_rms_norm",
                    "config": {"normalized_shape": 128, "eps": 1e-5}
                },
                "ffn_norm_config": {
                    "norm_type": "pytorch_rms_norm",
                    "config": {"normalized_shape": 128, "eps": 1e-5}
                },
                "lm_head_norm_config": {
                    "norm_type": "pytorch_rms_norm",
                    "config": {"normalized_shape": 128, "eps": 1e-5}
                },
                "use_weight_tying": False,
                "use_meta_device": False,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
    return config_path, config_dict


def test_export_generic_hf_adapter(dummy_modalities_config, tmp_path):
    config_path, config_dict = dummy_modalities_config

    # Create dummy model and save state dict
    model = GPT2ModelFactory.get_gpt2_model(**config_dict["model_raw"]["config"])
    weights_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), weights_path)

    output_dir = tmp_path / "hf_export"

    export_generic_hf_adapter(
        checkpoint_file_path=weights_path,
        modalities_config_path=config_path,
        output_dir=output_dir,
    )

    assert (output_dir / "config.json").exists()
    assert (output_dir / "modalities_adapter.py").exists()
    assert (output_dir / "pytorch_model.bin").exists()

    # Verify loading via Hugging Face AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)
    assert hf_model is not None

    input_ids = torch.randint(0, 128, (2, 10))
    output = hf_model(input_ids=input_ids)
    assert output.logits is not None
    assert output.logits.shape == (2, 10, 128)
