"""
Unified Conversion Script: Modalities (.pt or DCP) -> Hugging Face.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch

from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

from modeling_adaptive_gpt import AdaptiveGPTConfig, AdaptiveGPTForCausalLM
from transformers import AutoTokenizer

from modalities.config.config import load_app_config_dict
from modalities.conversion.gpt2.conversion_tokenizer import convert_tokenizer
from modalities.utils.env import EnvOverride

logger = logging.getLogger(__name__)


def is_dcp_checkpoint(checkpoint_path: str) -> bool:
    if not os.path.isdir(checkpoint_path): return False
    return os.path.exists(os.path.join(checkpoint_path, ".metadata")) or any(f.endswith(".distcp") for f in os.listdir(checkpoint_path))

def find_yaml_config_in_dir(directory: str) -> Optional[str]:
    if not os.path.isdir(directory) or not os.access(directory, os.R_OK): return None
    for filename in os.listdir(directory):
        if filename.endswith(".yaml") or filename.endswith(".yml"): return os.path.join(directory, filename)
    return None

def get_modalities_config(checkpoint_path: str, explicit_config_path: Optional[str]) -> dict:
    config_src = explicit_config_path
    if config_src is None and is_dcp_checkpoint(checkpoint_path):
        config_src = find_yaml_config_in_dir(checkpoint_path)
        if config_src is None: config_src = find_yaml_config_in_dir(str(Path(checkpoint_path).parent))
        if config_src: logger.info(f"Auto-discovered config at: {config_src}")
        else: raise FileNotFoundError("No YAML config found. Provide it via --modalities_config")
    elif config_src is None:
        raise ValueError("You must provide --modalities_config for standard .pt checkpoints.")

    # Critical Fix: Wrap the load in EnvOverride so OmegaConf doesn't crash 
    # when it looks for ${cuda_env:LOCAL_RANK} in the YAML.
    with EnvOverride({"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "localhost", "MASTER_PORT": "12345"}):
        return load_app_config_dict(Path(config_src), experiment_id="-1", experiments_root_path=Path("."))


def build_config_from_modalities(model_cfg: dict) -> AdaptiveGPTConfig:
    # CRITICAL FIX: Map pytorch_rms_norm directly to itself so it triggers native torch.nn.RMSNorm
    norm_type_map = {"layer_norm": "layer_norm", "rms_norm": "rms_norm", "pytorch_rms_norm": "pytorch_rms_norm"}
    ffn_norm = model_cfg["ffn_norm_config"]
    raw_norm_type = str(ffn_norm["norm_type"]).split(".")[-1]
    norm_type = norm_type_map.get(raw_norm_type, "layer_norm")
    norm_inner = ffn_norm["config"]
    
    norm_bias = False if raw_norm_type == "pytorch_rms_norm" else norm_inner.get("bias", True)

    qkv_transforms = model_cfg.get("attention_config", {}).get("qkv_transforms", [])
    use_rotary, rotary_base_freq = False, 10000
    for t in qkv_transforms:
        if str(t.get("type_hint", "")).split(".")[-1] == "RotaryTransform":
            use_rotary = True
            rotary_base_freq = t["config"].get("base_freq", 10000)
            break

    qk_norm_cfg = model_cfg.get("attention_config", {}).get("qk_norm_config")
    use_qk_norm = qk_norm_cfg is not None
    qk_norm_dim = None
    if use_qk_norm:
        qk_cfg = qk_norm_cfg.get("config", {})
        qk_norm_dim = qk_cfg.get("ndim", qk_cfg.get("normalized_shape"))

    adaptive_cfg = model_cfg.get("adaptive_config") or {}

    return AdaptiveGPTConfig(
        vocab_size=model_cfg["vocab_size"],
        sequence_length=model_cfg["sequence_length"],
        n_layer=model_cfg["n_layer"],
        n_head_q=model_cfg["n_head_q"],
        n_head_kv=model_cfg["n_head_kv"],
        n_embd=model_cfg["n_embd"],
        ffn_hidden=model_cfg["ffn_hidden"],
        dropout=model_cfg.get("dropout", 0.0),
        bias=model_cfg["bias"],
        activation_type=str(model_cfg["activation_type"]).split(".")[-1].lower(),
        enforce_swiglu_hidden_dim_multiple_of=model_cfg.get("enforce_swiglu_hidden_dim_multiple_of", 256),
        poe_type=str(model_cfg["poe_type"]).split(".")[-1].upper(),
        use_rotary=use_rotary,
        rotary_base_freq=rotary_base_freq,
        norm_type=norm_type,
        norm_eps=norm_inner.get("eps", norm_inner.get("epsilon", 1e-5)),
        norm_bias=norm_bias,
        norm_elementwise_affine=norm_inner.get("elementwise_affine", True),
        use_weight_tying=model_cfg.get("use_weight_tying", False),
        use_qk_norm=use_qk_norm,
        qk_norm_dim=qk_norm_dim,
        enable_adaptive=bool(adaptive_cfg.get("enable_adaptive", False)),
        max_loops=adaptive_cfg.get("max_loops", 10),
        ponder_penalty_weight=adaptive_cfg.get("ponder_penalty_weight", 0.0),
        wide_ffn_hidden=adaptive_cfg.get("wide_ffn_hidden", 0),
        deep_gate_init_bias=adaptive_cfg.get("deep_gate_init_bias", 0.0),
        wide_gate_init_bias=adaptive_cfg.get("wide_gate_init_bias", 0.0),
        layer_types=adaptive_cfg.get("layer_types"),
    )


def load_raw_state_dict(checkpoint_path: str) -> dict:
    if is_dcp_checkpoint(checkpoint_path):
        logger.info(f"Reading Distributed Checkpoint (DCP) from: {checkpoint_path}")
        sd = {}
        planner = _EmptyStateDictLoadPlanner(keys=["app.model"], allow_partial_load=True)
        _load_state_dict(sd, storage_reader=FileSystemReader(checkpoint_path), planner=planner, no_dist=True)
        return sd.get("app", {}).get("model", sd)
    else:
        logger.info(f"Reading standard PyTorch checkpoint from: {checkpoint_path}")
        ckpt: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict): return ckpt[key]
        return ckpt

def fixup_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("module."): k = k[len("module."):]
        if k == "transformer.lm_head.weight": k = "lm_head.weight"
        out[k] = v
    return out


def convert(checkpoint_path: str, output_dir: str, config_path: Optional[str] = None):
    logger.info("1. Resolving configuration...")
    modalities_config = get_modalities_config(checkpoint_path, config_path)
    model_cfg = modalities_config.get("model_raw", modalities_config.get("model"))["config"]
    hf_config = build_config_from_modalities(model_cfg)

    logger.info("2. Loading PyTorch Checkpoint into RAM...")
    raw_sd = load_raw_state_dict(checkpoint_path)
    clean_sd = fixup_state_dict(raw_sd)

    logger.info("3. Instantiating HF Model and Loading Weights...")
    model = AdaptiveGPTForCausalLM(hf_config)
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)

    if hf_config.use_weight_tying:
        missing = [k for k in missing if k != "lm_head.weight"]

    if missing: logger.warning(f"Missing keys: {missing}")
    if unexpected: logger.warning(f"Unexpected keys: {unexpected}")
    if not missing and not unexpected: logger.info("✅ All state dict keys matched perfectly.")

    logger.info("4. Handling Tokenizer conversion...")
    tokenizer_config_raw = None
    for key in ["tokenizer", "tokenizer_raw", "wrapped_tokenizer"]:
        if key in modalities_config and isinstance(modalities_config[key], dict):
            tokenizer_config_raw = modalities_config[key]
            break

    if tokenizer_config_raw:
        variant = tokenizer_config_raw.get("variant_key")
        if variant == "pretrained_sp_tokenizer":
            tokenizer_model = tokenizer_config_raw["config"]["tokenizer_model_file"]
            bos_id, eos_id, pad_id, _ = convert_tokenizer(tokenizer_model, output_dir)
            model.config.bos_token_id = bos_id
            model.config.eos_token_id = eos_id
            model.config.pad_token_id = pad_id
        elif variant == "pretrained_hf_tokenizer":
            hf_path = tokenizer_config_raw["config"].get("pretrained_model_name_or_path", "openai-community/gpt2")
            tokenizer = AutoTokenizer.from_pretrained(hf_path)
            tokenizer.save_pretrained(output_dir)
            model.config.bos_token_id = tokenizer.bos_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            load_default_hf_tokenizer(model, output_dir)
    else:
        load_default_hf_tokenizer(model, output_dir)

    logger.info(f"5. Saving HF model to {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    logger.info("✅ Conversion complete!")

def load_default_hf_tokenizer(model: AdaptiveGPTForCausalLM, output_dir: str):
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.save_pretrained(output_dir)
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Convert Modalities (DCP or .pt) to HF.")
    parser.add_argument("checkpoint", help="Path to the .pt file or DCP directory")
    parser.add_argument("output_dir", help="Directory to write the HF model to")
    parser.add_argument("--modalities_config", default=None, help="Path to YAML config (optional if DCP has it inside)")
    args = parser.parse_args()
    convert(args.checkpoint, args.output_dir, args.modalities_config)