import os
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
import re
import json
import tempfile
import shutil
import copy
import sys

from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper
from modalities.models.utils import get_model_from_config, ModelTypeEnum
from modalities.checkpointing.convert_dcp_to_torch import convert_dcp_to_torch, load_dcp_config
from modalities.config.config import load_app_config_dict
from modalities.config.component_factory import ComponentFactory
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry

from oe_eval.run_eval import load_task, compute_save_metrics, evaluate
from oe_eval.default_configs import MODEL_DEFAULTS
from oe_eval.utils import hash_dict
from oe_eval.configs.tasks import TASK_CONFIGS
from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
from oe_eval.launch import resolve_task_suite
from oe_eval.tasks.aggregate_tasks import add_aggregate_tasks


class ModalitiesOLMESWrapper:
    """OLMES-compatible wrapper for Modalities models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: TokenizerWrapper,
        max_length: int = 2048,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        eod_token: str = "<eod>",
    ):
        self.model = model
        self._tokenizer = tokenizer
        self._max_length = max_length
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eod_token = eod_token
        self.rank = 0
        
        self.model.to(self.device)
        self.model.eval()
        
        # Determine EOT token ID
        eod_tokens = self._tokenizer.tokenize(eod_token)
        self.eot_token_id = eod_tokens[0] if eod_tokens else 0
    
    @property
    def max_length(self) -> int:
        return self._max_length
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    def tok_encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self._tokenizer.tokenize(text)
    
    def tok_decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(tokens)
    
    def _get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            input_dict = {"input_ids": input_ids}
            output = self.model(input_dict)
            if isinstance(output, dict):
                logits_output = output.get("logits", output)
                if isinstance(logits_output, dict):
                    return logits_output.get("logits", logits_output)
                return logits_output
            return output
    
    def unload_model(self):
        import gc
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def loglikelihood_verbose(
        self,
        requests: List,
        override_bs: Optional[int] = None,
        disable_tqdm: bool = False,
    ) -> List[dict]:
        results = []
        
        for req in tqdm(requests, disable=disable_tqdm, desc="Running loglikelihood"):
            context = req.context
            continuation = req.continuation
            
            if context == "":
                context_enc = [self.eot_token_id]
            else:
                context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation)
            
            # Truncate context if needed, preserving continuation
            full_enc = (context_enc + continuation_enc)[-(self._max_length + 1):-1]
            ctx_len = len(context_enc)
            if len(context_enc) + len(continuation_enc) > self._max_length + 1:
                ctx_len = len(context_enc) - (len(context_enc) + len(continuation_enc) - self._max_length - 1)
            ctx_len = max(1, ctx_len)
            
            input_ids = torch.tensor([full_enc], dtype=torch.long, device=self.device)
            logits = self._get_logits(input_ids)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            
            cont_len = len(continuation_enc)
            start_idx = max(0, ctx_len - 1)
            end_idx = start_idx + cont_len
            
            # Handle boundary conditions
            if end_idx > log_probs.shape[1]:
                end_idx = log_probs.shape[1]
                cont_len = end_idx - start_idx
            
            cont_log_probs = log_probs[0, start_idx:end_idx, :]
            cont_tokens = torch.tensor(continuation_enc[:cont_len], dtype=torch.long, device=self.device)
            
            if cont_log_probs.shape[0] > 0 and cont_tokens.shape[0] > 0:
                token_log_probs = cont_log_probs.gather(1, cont_tokens.unsqueeze(1)).squeeze(1)
                sum_logits = token_log_probs.sum().item()
                greedy_tokens = cont_log_probs.argmax(dim=-1)
                is_greedy = (greedy_tokens == cont_tokens).all().item()
            else:
                sum_logits = 0.0
                is_greedy = False
            
            results.append({
                "continuation": continuation,  
                "sum_logits": sum_logits,
                "num_tokens": cont_len,
                "num_tokens_all": len(full_enc),
                "is_greedy": bool(is_greedy),
            })
        
        return results
    
    def generate_until_verbose(
        self,
        requests: List,
        disable_tqdm: bool = False,
    ) -> List[dict]:
        results = []
        
        for req in tqdm(requests, disable=disable_tqdm, desc="Running generation"):
            context = req.context
            stop_sequences = req.stop_sequences or []
            gen_kwargs = req.generation_kwargs or {}
            
            max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
            temperature = gen_kwargs.get("temperature", 0.0)
            
            token_ids = self.tok_encode(context)
            max_ctx_len = self._max_length - max_gen_toks
            if len(token_ids) > max_ctx_len:
                token_ids = token_ids[-max_ctx_len:]
            
            generated_tokens = []
            sum_logits = 0.0
            
            with torch.no_grad():
                for _ in range(max_gen_toks):
                    input_ids = torch.tensor(
                        [token_ids + generated_tokens], 
                        dtype=torch.long, 
                        device=self.device
                    )
                    
                    logits = self._get_logits(input_ids)
                    next_logits = logits[0, -1, :]
                    
                    if temperature > 0:
                        scaled_logits = next_logits / temperature
                        probs = F.softmax(scaled_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    else:
                        next_token = torch.argmax(next_logits).item()
                    
                    log_probs = F.log_softmax(next_logits.float(), dim=-1)
                    token_log_prob = log_probs[next_token].item()
                    sum_logits += token_log_prob
                    
                    generated_tokens.append(next_token)
                    
                    if next_token == self.eot_token_id:
                        break
                    
                    generated_text = self.tok_decode(generated_tokens)
                    if any(stop_seq and stop_seq in generated_text for stop_seq in stop_sequences):
                        break
            
            generated_text_raw = self.tok_decode(generated_tokens)
            generated_text = generated_text_raw
            for stop_seq in stop_sequences:
                if stop_seq and stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.find(stop_seq)]
            
            result = {
                "continuation": generated_text,
                "sum_logits": sum_logits,
                "num_tokens": len(generated_tokens),
            }
            if generated_text != generated_text_raw:
                result["continuation_raw"] = generated_text_raw
            
            results.append(result)
        
        return results
    
    def loglikelihood_rolling_verbose(
        self,
        requests: List,
        disable_tqdm: bool = False,
        verbose_token_logits: bool = False,
    ) -> List[dict]:
        results = []
        
        for req in tqdm(requests, disable=disable_tqdm, desc="Running rolling loglikelihood"):
            context = req.context
            token_ids = self.tok_encode(context)
            
            total_logits = 0.0
            total_tokens = 0
            all_tokens = []
            all_logits = []
            
            chunk_size = self._max_length
            for start in range(0, len(token_ids), chunk_size - 1):
                end = min(start + chunk_size, len(token_ids))
                chunk = token_ids[start:end]
                
                if len(chunk) < 2:
                    continue
                
                input_ids = torch.tensor([chunk], dtype=torch.long, device=self.device)
                logits = self._get_logits(input_ids)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                
                score_start = 1 if start == 0 else 0
                for i in range(score_start, len(chunk)):
                    token_id = chunk[i]
                    token_log_prob = log_probs[0, i-1, token_id].item()
                    total_logits += token_log_prob
                    total_tokens += 1
                    
                    if verbose_token_logits:
                        all_tokens.append(self.tok_decode([token_id]))
                        all_logits.append(token_log_prob)
            
            result = {
                "sum_logits": total_logits,
                "num_tokens": total_tokens,
                "num_tokens_all": len(token_ids),
                "num_chars": len(context),
                "num_words": len(re.split(r"\s+", context)),
                "num_bytes": len(context.encode("utf-8")),
            }
            
            if verbose_token_logits:
                result["tokens"] = all_tokens
                result["logits"] = all_logits
            
            results.append(result)
        
        return results
    
    def generate_until_and_loglikelihood_verbose(self, instances, requests) -> List[dict]:
        return self.loglikelihood_verbose(requests)


def is_dcp_checkpoint(checkpoint_path: str) -> bool:
    """Check if path is a distributed checkpoint (DCP) directory."""
    if not os.path.isdir(checkpoint_path):
        return False
    has_metadata = os.path.exists(os.path.join(checkpoint_path, ".metadata"))
    has_distcp = any(f.endswith(".distcp") for f in os.listdir(checkpoint_path))
    return has_metadata or has_distcp


def find_checkpoint_dir(base_path: str) -> str:
    """Find the actual checkpoint directory within a base path."""
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {base_path}")
    
    if is_dcp_checkpoint(base_path):
        return base_path
    
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and is_dcp_checkpoint(item_path):
            return item_path
    
    return base_path


def load_tokenizer_from_config(config: dict) -> TokenizerWrapper:
    """Load tokenizer from a config dict with a fallback to GPT-2."""
    from pydantic import BaseModel
    from modalities.config.pydantic_if_types import PydanticTokenizerIFType
    
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)
    
    # 1. Try to find existing tokenizer keys in the config
    tokenizer_config_raw = None
    for key in ["tokenizer", "tokenizer_raw", "wrapped_tokenizer"]:
        if key in config:
            tokenizer_config_raw = config[key]
            break
            
    # 2. If no tokenizer key is found, use the GPT-2 fallback
    if tokenizer_config_raw is None:
        print("No tokenizer found in config. Falling back to default GPT-2 config.")
        tokenizer_config_raw = {
            "component_key": "tokenizer",
            "variant_key": "pretrained_hf_tokenizer",
            "config": {
                "pretrained_model_name_or_path": "openai-community/gpt2",
                "padding": False,
                "truncation": False
            }
        }

    # 3. Build the component using the factory
    class TokenizerConfig(BaseModel):
        tokenizer: PydanticTokenizerIFType
    
    tokenizer_build_dict = {"tokenizer": tokenizer_config_raw}
    components = component_factory.build_components(
        config_dict=tokenizer_build_dict,
        components_model_type=TokenizerConfig,
    )
    return components.tokenizer


def load_modalities_model(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    model_key: str = "model_raw",
    converted_output_dir: Optional[str] = None,
) -> tuple:
    """
    Load a modalities model from either a DCP or standard checkpoint.
    """
    # Check if this is a DCP checkpoint
    if os.path.isdir(checkpoint_path):
        actual_ckpt_dir = find_checkpoint_dir(checkpoint_path)
        if is_dcp_checkpoint(actual_ckpt_dir):
            print(f"Detected distributed checkpoint (DCP) at: {actual_ckpt_dir}")
            
            # Create output directory for converted checkpoint
            use_temp_dir = converted_output_dir is None
            if use_temp_dir:
                converted_output_dir = tempfile.mkdtemp(prefix="modalities_converted_")
                print(f"Using temporary directory: {converted_output_dir}")
            else:
                os.makedirs(converted_output_dir, exist_ok=True)
            
            # Convert DCP to PyTorch format
            print(f"Converting DCP to PyTorch format...")
            converted_config_path = convert_dcp_to_torch(
                dcp_checkpoint_dir=actual_ckpt_dir,
                output_dir=converted_output_dir,
                model_key=model_key,
            )
            print(f"Converted config: {converted_config_path}")
            
            # Load model using existing utility
            converted_config = load_app_config_dict(Path(converted_config_path))
            model = get_model_from_config(converted_config, ModelTypeEnum.CHECKPOINTED_MODEL)
            
            # Load tokenizer from original config
            print("Loading tokenizer from original DCP config...")
            _, original_config = load_dcp_config(actual_ckpt_dir)
            print('Loading tokenizer...')
            tokenizer = load_tokenizer_from_config(original_config)
            
            return model, tokenizer, converted_config
    
    # Standard checkpoint loading
    if config_path is None:
        raise ValueError("config_path is required for standard (non-DCP) checkpoints")
    
    config_dict = load_app_config_dict(Path(config_path))
    model = get_model_from_config(config_dict, ModelTypeEnum.MODEL)
    tokenizer = load_tokenizer_from_config(config_dict)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    return model, tokenizer, config_dict


def resolve_tasks_and_configs(tasks: List[str]) -> List[dict]:
    """
    Resolve task names (which may include task suites) into full task configs.
    """
    # Step 1: Expand task suites into individual tasks
    all_tasks = []
    task_suite_parent: dict = {}
    for task in tasks:
        all_tasks += resolve_task_suite(task, task_suite_parent)
    
    # Step 2: Look up each task in TASK_CONFIGS
    task_configs = []
    for task in all_tasks:
        if task in TASK_CONFIGS:
            task_config = copy.deepcopy(TASK_CONFIGS[task])
            if "metadata" not in task_config:
                task_config["metadata"] = {}
            task_config["metadata"]["alias"] = task
            task_configs.append(task_config)
        elif task in task_suite_parent:
            print(f"Warning: No config found for task: {task} (from task suite {task_suite_parent[task]})")
            task_configs.append({"task_name": task})
        else:
            print(f"Warning: No config found for task: {task}, using raw task name")
            task_configs.append({"task_name": task})
    
    return task_configs


def evaluate_modalities_checkpoint(
    checkpoint_path: str,
    tasks: List[str],
    config_path: Optional[str] = None,
    output_dir: str = "./eval_results",
    max_length: int = 2048,
    batch_size: int = 1,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    model_key: str = "model_raw",
    converted_checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a modalities checkpoint using OLMES.
    """
    import datetime
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {checkpoint_path}...")
    model, tokenizer, _ = load_modalities_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        model_key=model_key,
        converted_output_dir=converted_checkpoint_dir,
    )
    
    print("Wrapping model for OLMES...")
    olmes_model = ModalitiesOLMESWrapper(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        device=torch.device(device),
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_config = {
        "model": checkpoint_path,
        "model_type": "modalities",
        "max_length": max_length,
    }
    model_hash = hash_dict(model_config, MODEL_DEFAULTS)
    
    compute_config = {
        "batch_size": batch_size,
        "output_dir": output_dir,
        "num_recorded_inputs": 3,
        "gsheet": None,
        "hf_save_dir": None,
        "save_raw_requests": True,
        "recompute_metrics": False,
        "wandb_run_path": None,
        "wandb_run_step": None,
    }
    
    # Resolve task suites and get full task configs
    print(f"Resolving {len(tasks)} task(s)/suite(s)...")
    task_configs = resolve_tasks_and_configs(tasks)
    print(f"Expanded to {len(task_configs)} individual task(s)")
    
    # Apply limit override if specified
    if limit is not None:
        for tc in task_configs:
            tc["limit"] = limit
    
    all_metrics = []
    
    for task_idx, task_config in enumerate(task_configs):
        task_name = task_config.get("metadata", {}).get("alias") or task_config.get("task_name")
        print(f"\n{'='*60}")
        print(f"Task {task_idx + 1}/{len(task_configs)}: {task_name}")
        print(f"{'='*60}")
        
        task = load_task(task_config, output_dir)
        task_hash = task._task_hash
        
        print(f"  Downloading...")
        task.download()
        task.set_model(olmes_model, model_config)
        
        print(f"  Building requests...")
        task.build_all_requests()
        
        instances = task._instances or []
        print(f"  Instances: {len(instances)}")
        
        if not instances:
            print(f"  Skipping (no instances)")
            continue
        
        print(f"  Running inference...")
        start_time = datetime.datetime.now()
        
        results_for_requests = evaluate(
            model=olmes_model,
            instances=instances,
            task_config=task.task_config,
            model_config=model_config,
        )
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        task.make_metrics()
        
        full_config = {
            "task_name": task_name,
            "task_hash": task_hash["hash"],
            "model_hash": model_hash["hash"],
            "model_config": model_config,
            "task_config": task.task_config,
            "compute_config": compute_config,
            "processing_time": processing_time,
            "current_date": datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "num_instances": 0,
            "beaker_info": {},
        }
        
        eval_requests_raw = [ins.to_dict() for ins in instances]
        metrics = compute_save_metrics(
            task_idx=task_idx,
            task=task,
            full_config=full_config,
            compute_config=compute_config,
            eval_requests_raw=eval_requests_raw,
            results_for_requests=results_for_requests,
        )
        all_metrics.append(metrics)
        
        print(f"\n  Primary score: {metrics['metrics'].get('primary_score', 'N/A')}")
    
    # Compute aggregate metrics for task suites
    print(f"\n{'='*60}")
    print("Computing aggregate metrics...")
    print(f"{'='*60}")
    aggregate_metrics = add_aggregate_tasks(all_metrics)
    
    # Combine aggregate metrics with individual metrics
    all_metrics_with_aggregates = aggregate_metrics + all_metrics
    
    # Save results
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_metrics_with_aggregates, f, indent=2, default=str)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # Print aggregate scores first
    if aggregate_metrics:
        print("\n  Aggregate scores:")
        for m in aggregate_metrics:
            score = m["metrics"].get("primary_score", "N/A")
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            print(f"    {m['task_name']}: {score_str}")
    
    # Print individual scores
    print("\n  Individual task scores:")
    for m in all_metrics:
        task_name = m["task_config"].get("metadata", {}).get("alias", m["task_name"])
        score = m["metrics"].get("primary_score", "N/A")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(f"    {task_name}: {score_str}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return {"metrics": all_metrics_with_aggregates, "output_dir": output_dir}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate modalities checkpoint with OLMES")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tasks", type=str, nargs="+", default=["arc_challenge::olmes"])
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-key", type=str, default="model_raw")
    parser.add_argument("--converted-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    evaluate_modalities_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        tasks=args.tasks,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        limit=args.limit,
        device=args.device,
        model_key=args.model_key,
        converted_checkpoint_dir=args.converted_dir,
    )