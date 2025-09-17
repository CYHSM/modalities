"""Evaluation utilities for GKD model assessment."""
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from filelock import FileLock
import wandb

from modalities.post_gkd.config import EvaluationConfig

logger = logging.getLogger(__name__)


def setup_wandb_metrics():
    """Setup WandB metrics to allow out-of-order logging for evaluation metrics."""
    if wandb.run is not None:
        wandb.define_metric("eval/*", step_metric="eval_step")
        wandb.define_metric("eval_step")
        logger.info("WandB metrics configured for out-of-order evaluation logging")


def merge_peft_model(peft_path: str, base_model_path: str, output_dir: str) -> bool:
    """Merge PEFT adapters into base model."""
    try:
        logger.info(f"Merging PEFT model: {peft_path} with base: {base_model_path}")

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

        model = PeftModel.from_pretrained(base_model, peft_path)
        merged_model = model.merge_and_unload()

        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)

        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_dir)

        for filename in ["modeling_qwen3.py", "configuration_qwen3.py"]:
            src_file = Path(base_model_path) / filename
            if src_file.exists():
                dst_file = Path(output_dir) / filename
                dst_file.write_text(src_file.read_text())
                logger.info(f"Copied {filename}")

        logger.info(f"Successfully merged GKD model to: {output_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to merge PEFT model: {e}")
        return False


def run_lighteval_cli(
    checkpoint_path: str, step: int, eval_config: EvaluationConfig, hf_home: str = "/raid/s3/opengptx/mfrey/huggingface"
) -> Optional[Dict[str, Any]]:
    """Run LightEval CLI evaluation and return results."""
    
    lock_path = f"/tmp/lighteval_gpu_{eval_config.eval_gpu}.lock"
    gpu_lock = FileLock(lock_path)

    try:
        logger.info(f"GKD evaluation process for step {step} is WAITING for GPU {eval_config.eval_gpu} lock...")
        with gpu_lock:
            logger.info(f"GKD evaluation process for step {step} has ACQUIRED lock for GPU {eval_config.eval_gpu}.")
            logger.info(f"Starting CLI evaluation for GKD model at step {step} on {checkpoint_path}")

            checkpoint_dir = Path(checkpoint_path)
            eval_model_path = checkpoint_path
            merged_dir = None

            if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
                logger.info("PEFT GKD model detected, merging with base model...")
                merged_dir = os.path.join(checkpoint_path, "merged")
                if not os.path.exists(merged_dir):
                    if not merge_peft_model(checkpoint_path, eval_config.source_model_path, merged_dir):
                        return None
                else:
                    logger.info(f"Using existing merged GKD model at {merged_dir}")
                eval_model_path = merged_dir
                checkpoint_dir = Path(merged_dir)

            model_args = f"model_name={eval_model_path},use_chat_template=True,trust_remote_code=True"
            cmd_string = (
                f"lighteval accelerate "
                f'"{model_args}" '
                f'"{eval_config.eval_tasks}" '
                f"--max-samples {eval_config.eval_max_samples} "
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(eval_config.eval_gpu)
            env["HF_HOME"] = hf_home
            logger.info(f"Running GKD evaluation: CUDA_VISIBLE_DEVICES={eval_config.eval_gpu} {cmd_string}")

            result = subprocess.run(
                cmd_string,
                shell=True,
                env=env,
                capture_output=False,
                text=True,
                check=False,
                preexec_fn=os.setsid,
                cwd=os.getcwd(),
            )

            if result.returncode != 0:
                logger.error(f"GKD evaluation failed for step {step}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return None

            logger.info(f"CLI evaluation completed for GKD model at step {step}")
            json_files = list(checkpoint_dir.glob("results_*.json"))
            if not json_files:
                logger.error(f"No results JSON file found in {checkpoint_dir}")
                return None
            
            results_file = max(json_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Reading GKD results from {results_file}")
            with open(results_file, "r") as f:
                eval_results = json.load(f)

            logger.info(f"GKD evaluation process for step {step} has RELEASED lock for GPU {eval_config.eval_gpu}.")
            return eval_results

    except Exception as e:
        logger.error(f"GKD evaluation failed for step {step}: {e}")
        return None


def parse_and_log_results(eval_results: Dict[str, Any], step: int) -> Dict[str, float]:
    """Parse LightEval results and log to WandB."""
    if not eval_results or "results" not in eval_results:
        logger.error(f"No valid GKD results to log for step {step}")
        return {}

    results_to_log = {}
    for task_name, metrics in eval_results["results"].items():
        if task_name == "all":
            continue

        clean_task_name = task_name.replace("leaderboard|", "").split("|")[0]
        for metric_name, value in metrics.items():
            log_key = f"eval/gkd_{clean_task_name}_{metric_name}"
            results_to_log[log_key] = value

    if results_to_log and wandb.run is not None:
        results_to_log["eval_step"] = step
        wandb.log(results_to_log)
        logger.info(f"Logged {len(results_to_log)} GKD metrics to WandB for eval_step {step}")
        for key, value in results_to_log.items():
            if key != "eval_step":
                logger.info(f"  {key}: {value}")
    else:
        logger.warning(f"No GKD metrics to log for step {step}")

    return results_to_log


class AsyncEvaluator:
    """Asynchronous evaluator for running GKD evaluations in background."""

    def __init__(self, eval_config: EvaluationConfig, hf_home: str = "/raid/s3/opengptx/mfrey/huggingface"):
        self.eval_config = eval_config
        self.hf_home = hf_home
        self.executor = ThreadPoolExecutor(max_workers=eval_config.eval_max_workers)
        self.futures = []

        setup_wandb_metrics()

    def submit_evaluation(self, checkpoint_path: str, step: int):
        """Submit a GKD evaluation job."""

        def eval_and_log(eval_step, eval_checkpoint_path):
            results = run_lighteval_cli(eval_checkpoint_path, eval_step, self.eval_config, self.hf_home)
            if results:
                return parse_and_log_results(results, eval_step)
            return {}

        future = self.executor.submit(eval_and_log, step, checkpoint_path)
        self.futures.append((future, step))

        self.futures = [(f, s) for f, s in self.futures if not f.done()]

        logger.info(f"GKD evaluation job submitted for step {step}")

    def wait_for_completion(self):
        """Wait for all GKD evaluations to complete."""
        if self.futures:
            logger.info("Waiting for remaining GKD evaluations to complete...")
            for future, step in self.futures:
                try:
                    future.result()
                    logger.info(f"GKD evaluation completed for step {step}")
                except Exception as e:
                    logger.error(f"GKD evaluation failed for step {step}: {e}")

        self.executor.shutdown(wait=True)
        logger.info("All GKD evaluations completed")

    def get_completed_results(self) -> List[tuple]:
        """Get results from completed GKD evaluations."""
        completed = []
        remaining = []

        for future, step in self.futures:
            if future.done():
                try:
                    result = future.result()
                    completed.append((step, result))
                except Exception as e:
                    logger.error(f"Error getting GKD result for step {step}: {e}")
                    completed.append((step, None))
            else:
                remaining.append((future, step))

        self.futures = remaining
        return completed