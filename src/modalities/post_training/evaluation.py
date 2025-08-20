"""Evaluation utilities for model assessment."""
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import wandb
from model_utils import generate_response

from config import EvaluationConfig

logger = logging.getLogger(__name__)


def run_lighteval_cli(
    checkpoint_path: str, step: int, eval_config: EvaluationConfig, hf_home: str = "/raid/s3/opengptx/mfrey/huggingface"
) -> Optional[Dict[str, Any]]:
    """Run LightEval CLI evaluation and return results."""
    try:
        logger.info(f"🔍 Starting CLI evaluation for step {step} on {checkpoint_path}")

        checkpoint_dir = Path(checkpoint_path)

        # Define the argument strings
        model_args = (
            f"model_name={checkpoint_path},"
            f"use_chat_template=True,"
            f"trust_remote_code=True,"
            f"batch_size={eval_config.batch_size}"
        )

        # Construct the full command
        cmd_string = (
            f"lighteval accelerate "
            f'"{model_args}" '
            f'"{eval_config.tasks}" '
            f"--max-samples {eval_config.max_samples} "
        )

        # Set environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(eval_config.gpu_id)
        env["HF_HOME"] = hf_home

        logger.info(f"🚀 Running command: CUDA_VISIBLE_DEVICES={eval_config.gpu_id} {cmd_string}")

        # Run the evaluation
        result = subprocess.run(
            cmd_string,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=eval_config.timeout,
            cwd=os.getcwd(),
        )

        if result.returncode != 0:
            logger.error(f"❌ Evaluation failed for step {step}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return None

        logger.info(f"✅ CLI evaluation completed for step {step}")

        # Find the results JSON file
        json_files = list(checkpoint_dir.glob("results_*.json"))
        if not json_files:
            logger.error(f"❌ No results JSON file found in {checkpoint_dir}")
            return None

        # Read the most recent results file
        results_file = max(json_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"📖 Reading results from {results_file}")

        with open(results_file, "r") as f:
            eval_results = json.load(f)

        return eval_results

    except subprocess.TimeoutExpired:
        logger.error(f"❌ Evaluation timed out for step {step}")
        return None
    except Exception as e:
        logger.error(f"❌ Evaluation failed for step {step}: {e}")
        return None


def parse_and_log_results(eval_results: Dict[str, Any], step: int) -> Dict[str, float]:
    """Parse LightEval results and log to WandB."""
    if not eval_results or "results" not in eval_results:
        logger.error(f"❌ No valid results to log for step {step}")
        return {}

    results_to_log = {}

    # Parse individual task results
    for task_name, metrics in eval_results["results"].items():
        if task_name == "all":  # Skip the aggregated results
            continue

        # Clean up task name for logging
        clean_task_name = task_name.replace("leaderboard|", "").split("|")[0]

        for metric_name, value in metrics.items():
            if not metric_name.endswith("_stderr"):  # Skip stderr metrics
                log_key = f"eval/{clean_task_name}_{metric_name}"
                results_to_log[log_key] = value

    # Also log some metadata
    if "config_general" in eval_results:
        config = eval_results["config_general"]
        if "total_evaluation_time_secondes" in config:
            results_to_log["eval/evaluation_time_seconds"] = float(config["total_evaluation_time_secondes"])
        if "model_size" in config:
            results_to_log["eval/model_size"] = config["model_size"]

    # Log to WandB if available
    if results_to_log and wandb.run is not None:
        wandb.log(results_to_log, step=step)
        logger.info(f"📊 Logged {len(results_to_log)} metrics to WandB for step {step}")
        for key, value in results_to_log.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.warning(f"❌ No metrics to log for step {step}")

    return results_to_log


def manual_evaluation(
    model,
    tokenizer,
    test_problems: List[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    num_samples: int = 5,
) -> Dict[str, Any]:
    # Limit to num_samples
    test_problems = test_problems[:num_samples]

    logger.info(f"Running manual evaluation on {len(test_problems)} problems")

    results = {"problems": [], "responses": [], "problem_response_pairs": []}

    for i, problem in enumerate(test_problems):
        logger.info(f"Evaluating problem {i+1}/{len(test_problems)}")

        try:
            response = generate_response(
                model, tokenizer, problem, max_new_tokens=max_new_tokens, temperature=temperature
            )

            results["problems"].append(problem)
            results["responses"].append(response)
            results["problem_response_pairs"].append({"problem": problem, "response": response, "problem_index": i})

            logger.info(f"Problem: {problem[:100]}...")
            logger.info(f"Response: {response[:200]}...")
            logger.info("-" * 50)

        except Exception as e:
            logger.error(f"Error evaluating problem {i}: {e}")
            results["problems"].append(problem)
            results["responses"].append(f"ERROR: {str(e)}")
            results["problem_response_pairs"].append(
                {"problem": problem, "response": f"ERROR: {str(e)}", "problem_index": i}
            )

    return results


def save_manual_evaluation_results(results: Dict[str, Any], save_path: str):
    """Save manual evaluation results to file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Manual evaluation results saved to: {save_path}")


class AsyncEvaluator:
    """Asynchronous evaluator for running evaluations in background."""

    def __init__(self, eval_config: EvaluationConfig, hf_home: str = "/raid/s3/opengptx/mfrey/huggingface"):
        self.eval_config = eval_config
        self.hf_home = hf_home
        self.executor = ThreadPoolExecutor(max_workers=eval_config.max_workers)
        self.futures = []

    def submit_evaluation(self, checkpoint_path: str, step: int):
        """Submit an evaluation job."""

        def eval_and_log():
            results = run_lighteval_cli(checkpoint_path, step, self.eval_config, self.hf_home)
            if results:
                return parse_and_log_results(results, step)
            return {}

        future = self.executor.submit(eval_and_log)
        self.futures.append((future, step))

        # Clean up completed futures
        self.futures = [(f, s) for f, s in self.futures if not f.done()]

        logger.info(f"🎯 Evaluation job submitted for step {step}")

    def wait_for_completion(self):
        """Wait for all evaluations to complete."""
        if self.futures:
            logger.info("⏳ Waiting for remaining evaluations to complete...")
            for future, step in self.futures:
                try:
                    future.result(timeout=self.eval_config.timeout)
                    logger.info(f"✅ Evaluation completed for step {step}")
                except Exception as e:
                    logger.error(f"❌ Evaluation failed for step {step}: {e}")

        self.executor.shutdown(wait=True)
        logger.info("✅ All evaluations completed")

    def get_completed_results(self) -> List[tuple]:
        """Get results from completed evaluations."""
        completed = []
        remaining = []

        for future, step in self.futures:
            if future.done():
                try:
                    result = future.result()
                    completed.append((step, result))
                except Exception as e:
                    logger.error(f"Error getting result for step {step}: {e}")
                    completed.append((step, None))
            else:
                remaining.append((future, step))

        self.futures = remaining
        return completed
