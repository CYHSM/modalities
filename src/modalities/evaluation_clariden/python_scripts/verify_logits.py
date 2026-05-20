import torch
import logging
from pathlib import Path
from tqdm import tqdm

from modeling_adaptive_gpt import AdaptiveGPTForCausalLM
from modalities.config.config import load_app_config_dict
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from modalities.utils.env import EnvOverride
from convert_adaptive_gpt import load_raw_state_dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def verify_logits(modalities_config_path: str, checkpoint_path: str, hf_model_dir: str, num_testruns: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CRITICAL: Enforce bfloat16 to match training precision and mask FP32 math variations
    dtype = torch.bfloat16 
    
    logger.info(f"Using device: {device} with dtype: {dtype}")

    # 1. Load the original Modalities model
    logger.info("1. Loading uninitialized Modalities model architecture...")
    with EnvOverride({"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "localhost", "MASTER_PORT": "12345"}):
        config_dict = load_app_config_dict(Path(modalities_config_path), experiment_id="-1", experiments_root_path=Path("."))
    
    model_cfg = config_dict.get("model_raw", config_dict.get("model"))
    model_cfg["config"]["use_meta_device"] = False 
    
    mini_config = {"model": model_cfg}
    modalities_model = get_model_from_config(mini_config, model_type=ModelTypeEnum.MODEL)
    
    logger.info("2. Reading DCP weights directly into RAM and injecting into Modalities model...")
    raw_sd = load_raw_state_dict(checkpoint_path)
    clean_raw_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
    modalities_model.load_state_dict(clean_raw_sd, strict=False)
    
    # Move to GPU and cast to bfloat16
    modalities_model.to(device=device, dtype=dtype)
    modalities_model.eval()

    # 2. Load the new Hugging Face model
    logger.info(f"3. Loading converted Hugging Face model from {hf_model_dir}...")
    hf_model = AdaptiveGPTForCausalLM.from_pretrained(hf_model_dir)
    
    # Move to GPU and cast to bfloat16
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    vocab_size = hf_model.config.vocab_size
    seq_len = hf_model.config.sequence_length
    sample_key = modalities_model.sample_key
    prediction_key = modalities_model.prediction_key

    # 3. Run the head-to-head comparison
    logger.info("4. Starting logit verification...")
    max_diffs = []

    for i in tqdm(range(num_testruns), desc="Testing forward passes"):
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        modalities_inputs = {sample_key: input_ids}

        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids).logits
            modalities_outputs = modalities_model(modalities_inputs)
            modalities_logits = modalities_outputs[prediction_key] if isinstance(modalities_outputs, dict) else modalities_outputs
            if isinstance(modalities_logits, dict) and "logits" in modalities_logits:
                modalities_logits = modalities_logits["logits"]

        # 1. Check exact shapes
        assert hf_logits.shape == modalities_logits.shape, \
            f"Shape mismatch! HF: {hf_logits.shape}, Modalities: {modalities_logits.shape}"

        # 2. Check exact equality (The ultimate test)
        is_equal = torch.equal(hf_logits, modalities_logits)
        
        diff = torch.abs(hf_logits - modalities_logits).max().item()
        max_diffs.append(diff)
        
        if not is_equal:
            logger.warning(f"Run {i+1}: Tensors are not perfectly equal. Max diff: {diff}")

    overall_max_diff = max(max_diffs)
    logger.info(f"Verification complete across {num_testruns} runs.")
    logger.info(f"Maximum absolute difference between logits: {overall_max_diff:.8f}")

    if overall_max_diff == 0.0:
        logger.info("🎯 FLAWLESS VICTORY: torch.equal() passed! The models are mathematically identical in bfloat16.")
    elif overall_max_diff < 1e-4:
        logger.info("✅ SUCCESS: Models are functionally identical, but slight math backend variances remain.")
    else:
        logger.error("❌ FAILED: The logits differ significantly.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("modalities_config", type=str)
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("hf_model_dir", type=str)
    
    args = parser.parse_args()
    verify_logits(args.modalities_config, args.checkpoint_path, args.hf_model_dir)