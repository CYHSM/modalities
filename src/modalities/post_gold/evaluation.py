"""Evaluation utilities for GOLD training."""
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import wandb

logger = logging.getLogger(__name__)


def setup_wandb_metrics():
    if wandb.run is not None:
        wandb.define_metric("eval/*", step_metric="eval_step")
        wandb.define_metric("eval_step")
        logger.info("✅ WandB metrics configured for evaluation")


def run_lighteval_cli(checkpoint_path: str, step: int, eval_gpu: int, eval_tasks: str, eval_max_samples: int, hf_home: str) -> Optional[Dict[str, Any]]:
    try:
        logger.info(f"🔄 Step {step}: Starting evaluation on GPU {eval_gpu}")
        
        model_args = f"model_name={checkpoint_path},use_chat_template=True,trust_remote_code=True"
        cmd_string = (
            f"lighteval accelerate "
            f'"{model_args}" '
            f'"{eval_tasks}" '
            f"--max-samples {eval_max_samples} "
        )
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
        env["HF_HOME"] = hf_home
        
        logger.info(f"Running: CUDA_VISIBLE_DEVICES={eval_gpu} {cmd_string}")
        
        result = subprocess.run(
            cmd_string,
            shell=True,
            env=env,
            capture_output=False,
            text=True,
            check=False,
            preexec_fn=os.setsid,
        )
        
        if result.returncode != 0:
            logger.error(f"❌ Evaluation failed for step {step} with return code {result.returncode}")
            return None
        
        checkpoint_dir = Path(checkpoint_path)
        json_files = list(checkpoint_dir.glob("results_*.json"))
        
        if not json_files:
            logger.error(f"❌ No results JSON found in {checkpoint_dir}")
            return None
        
        results_file = max(json_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"📖 Reading results from {results_file}")
        
        with open(results_file, "r") as f:
            eval_results = json.load(f)
        
        logger.info(f"✅ Step {step}: Evaluation completed successfully")
        return eval_results
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed for step {step}: {e}", exc_info=True)
        return None


def parse_and_log_results(eval_results: Dict[str, Any], step: int) -> Dict[str, float]:
    if not eval_results or "results" not in eval_results:
        logger.error(f"❌ No valid results for step {step}")
        return {}
    
    results_to_log = {}
    for task_name, metrics in eval_results["results"].items():
        if task_name == "all":
            continue
        
        clean_task_name = task_name.replace("leaderboard|", "").split("|")[0]
        for metric_name, value in metrics.items():
            log_key = f"eval/{clean_task_name}_{metric_name}"
            results_to_log[log_key] = value
    
    if results_to_log and wandb.run is not None:
        results_to_log["eval_step"] = step
        wandb.log(results_to_log)
        logger.info(f"📊 Logged {len(results_to_log)} metrics for step {step}")
    
    return results_to_log


class AsyncEvaluator:
    def __init__(self, eval_gpu: int, eval_tasks: str, eval_max_samples: int, eval_max_workers: int, hf_home: str):
        self.eval_gpu = eval_gpu
        self.eval_tasks = eval_tasks
        self.eval_max_samples = eval_max_samples
        self.hf_home = hf_home
        self.executor = ThreadPoolExecutor(max_workers=eval_max_workers)
        self.futures = []
        self.pending_count = 0
        setup_wandb_metrics()
    
    def submit_evaluation(self, checkpoint_path: str, step: int):
        def eval_and_log(eval_step, eval_checkpoint_path):
            self.pending_count -= 1
            logger.info(f"⚙️  Step {eval_step}: Evaluation starting (pending: {self.pending_count})")
            results = run_lighteval_cli(
                eval_checkpoint_path, 
                eval_step, 
                self.eval_gpu, 
                self.eval_tasks, 
                self.eval_max_samples, 
                self.hf_home
            )
            if results:
                return parse_and_log_results(results, eval_step)
            return {}
        
        self.pending_count += 1
        future = self.executor.submit(eval_and_log, step, checkpoint_path)
        self.futures.append((future, step))
        self.futures = [(f, s) for f, s in self.futures if not f.done()]
        
        logger.info(f"🎯 Evaluation queued for step {step} (pending: {self.pending_count})")
    
    def wait_for_completion(self):
        if self.futures:
            logger.info(f"⏳ Waiting for {len(self.futures)} evaluations to complete...")
            for future, step in self.futures:
                try:
                    future.result()
                    logger.info(f"✅ Evaluation completed for step {step}")
                except Exception as e:
                    logger.error(f"❌ Evaluation failed for step {step}: {e}")
        
        self.executor.shutdown(wait=True)
        logger.info("✅ All evaluations completed")