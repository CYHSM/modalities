Looking at the codebase structure, here's the clean implementation:

## 1. Async Evaluator Core

```python
# src/modalities/evaluation/async_evaluator.py
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from modalities.batch import EvaluationResultBatch, ResultItem
from modalities.logging_broker.messages import Message, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.logging_broker.subscriber import MessageSubscriberIF

logger = logging.getLogger(__name__)


class AsyncEvaluator(MessageSubscriberIF):
    def __init__(
        self,
        tasks: str,
        max_samples: int,
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        gpu_id: Optional[int] = None,
        max_workers: int = 1,
        timeout: int = 3600,
        hf_home: str = "/tmp/hf_cache"
    ):
        self.tasks = tasks
        self.max_samples = max_samples
        self.evaluation_result_publisher = evaluation_result_publisher
        self.gpu_id = gpu_id or self._get_free_gpu()
        self.max_workers = max_workers
        self.timeout = timeout
        self.hf_home = hf_home
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def _get_free_gpu(self) -> int:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
                capture_output=True, text=True, check=True
            )
            free_memory = [int(x) for x in result.stdout.strip().split('\n')]
            return free_memory.index(max(free_memory))
        except:
            logger.warning("Could not detect free GPU, using GPU 0")
            return 0

    def consume_message(self, message: Message):
        if message.message_type == MessageTypes.CHECKPOINT_SAVED:
            checkpoint_path = Path(message.payload["checkpoint_path"])
            step = message.payload["step"]
            self._submit_evaluation(checkpoint_path, step)

    def consume_dict(self, message_dict: dict[str, Any]):
        pass

    def _submit_evaluation(self, checkpoint_path: Path, step: int):
        future = self.executor.submit(self._run_evaluation, checkpoint_path, step)
        self.futures.append((future, step))
        self.futures = [(f, s) for f, s in self.futures if not f.done()]

    def _run_evaluation(self, checkpoint_path: Path, step: int) -> Optional[Dict[str, Any]]:
        try:
            model_args = f"model_name={checkpoint_path},use_chat_template=True,trust_remote_code=True"
            cmd = f'lighteval accelerate "{model_args}" "{self.tasks}" --max-samples {self.max_samples}'
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            env["HF_HOME"] = self.hf_home

            result = subprocess.run(
                cmd, shell=True, env=env, capture_output=True, 
                text=True, check=False, timeout=self.timeout
            )

            if result.returncode != 0:
                logger.error(f"Async evaluation failed for step {step}")
                return None

            json_files = list(checkpoint_path.glob("results_*.json"))
            if not json_files:
                logger.error(f"No results found for step {step}")
                return None

            results_file = max(json_files, key=lambda p: p.stat().st_mtime)
            with open(results_file, "r") as f:
                eval_results = json.load(f)

            self._publish_results(eval_results, step)
            return eval_results

        except Exception as e:
            logger.error(f"Async evaluation failed for step {step}: {e}")
            return None

    def _publish_results(self, eval_results: Dict[str, Any], step: int):
        if not eval_results or "results" not in eval_results:
            return

        metrics = {}
        for task_name, task_metrics in eval_results["results"].items():
            if task_name == "all":
                continue
            clean_task_name = task_name.replace("leaderboard|", "").split("|")[0]
            for metric_name, value in task_metrics.items():
                metrics[f"async_eval/{clean_task_name}_{metric_name}"] = ResultItem(torch.tensor(value))

        if metrics:
            evaluation_result = EvaluationResultBatch(
                losses={},
                metrics=metrics,
                throughput_metrics={},
                dataloader_tag="async_evaluation",
                num_train_steps_done=step,
            )
            self.evaluation_result_publisher.publish_message(
                payload=evaluation_result, 
                message_type=MessageTypes.EVALUATION_RESULT
            )

    def wait_for_completion(self):
        if self.futures:
            for future, step in self.futures:
                try:
                    future.result(timeout=self.timeout)
                except Exception as e:
                    logger.error(f"Async evaluation failed for step {step}: {e}")
        self.executor.shutdown(wait=True)
```

## 2. Configuration

```python
# src/modalities/evaluation/async_evaluation_config.py
from typing import Optional
from pydantic import BaseModel, Field


class AsyncEvaluatorConfig(BaseModel):
    tasks: str = "leaderboard|hellaswag|10|1,leaderboard|gsm8k|8|1"
    max_samples: int = Field(default=100, gt=0)
    gpu_id: Optional[int] = None
    max_workers: int = Field(default=1, gt=0)
    timeout: int = Field(default=3600, gt=0)
    hf_home: str = "/tmp/hf_cache"
    enabled: bool = True
```

## 3. Message Types Update

```python
# Add to src/modalities/logging_broker/messages.py
class MessageTypes(Enum):
    HIGH_LEVEL_PROGRESS_UPDATE = "HIGH_LEVEL_PROGRESS_UPDATE"
    BATCH_PROGRESS_UPDATE = "PROGRESS_UPDATE"
    ERROR_MESSAGE = "ERROR_MESSAGE"
    EVALUATION_RESULT = "EVALUATION_RESULT"
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"
```

## 4. Checkpoint Saving Integration

```python
# Modify src/modalities/checkpointing/checkpoint_saving.py
from modalities.logging_broker.messages import MessageTypes
from modalities.logging_broker.publisher import MessagePublisher


class CheckpointSaving:
    def __init__(
        self,
        checkpoint_saving_strategy: CheckpointSavingStrategyIF,
        checkpoint_saving_execution: CheckpointSavingExecutionABC,
        checkpoint_saved_publisher: Optional[MessagePublisher] = None,
    ):
        self.checkpoint_saving_strategy = checkpoint_saving_strategy
        self.checkpoint_saving_execution = checkpoint_saving_execution
        self.checkpoint_saved_publisher = checkpoint_saved_publisher

    def save_checkpoint(
        self,
        training_progress: TrainingProgress,
        evaluation_result: dict[str, EvaluationResultBatch],
        app_state: AppState,
        early_stoppping_criterion_fulfilled: bool = False,
    ):
        checkpointing_instruction = self.checkpoint_saving_strategy.get_checkpoint_instruction(
            training_progress=training_progress,
            evaluation_result=evaluation_result,
            early_stoppping_criterion_fulfilled=early_stoppping_criterion_fulfilled,
        )

        self.checkpoint_saving_execution.run_checkpoint_instruction(
            checkpointing_instruction=checkpointing_instruction,
            training_progress=training_progress,
            app_state=app_state,
        )

        if (checkpointing_instruction.save_current and 
            self.checkpoint_saved_publisher and 
            hasattr(self.checkpoint_saving_execution, 'get_last_checkpoint_path')):
            
            checkpoint_path = self.checkpoint_saving_execution.get_last_checkpoint_path(training_progress)
            payload = {
                "checkpoint_path": str(checkpoint_path),
                "step": training_progress.num_seen_steps_total
            }
            self.checkpoint_saved_publisher.publish_message(
                payload=payload, 
                message_type=MessageTypes.CHECKPOINT_SAVED
            )
```

## 5. Add Method to Checkpoint Execution Classes

```python
# Add to src/modalities/checkpointing/fsdp/fsdp_checkpoint_saving.py in both classes
def get_last_checkpoint_path(self, training_progress: TrainingProgress) -> Path:
    if hasattr(self, '_last_checkpoint_path'):
        return self._last_checkpoint_path
    # Fallback to constructing path
    return self._get_checkpointing_path(
        experiment_id=self.experiment_id,
        num_seen_steps=training_progress.num_seen_steps_total,
        num_seen_tokens=training_progress.num_seen_tokens_total,
        num_target_steps=training_progress.num_target_steps,
        num_target_tokens=training_progress.num_target_tokens,
        entity_type=CheckpointingEntityType.MODEL,
    )

# In _save_checkpoint methods, add:
# self._last_checkpoint_path = checkpoint_path
```

## 6. Training Components Config Update

```python
# Add to src/modalities/config/instantiation_models.py
from modalities.evaluation.async_evaluation_config import AsyncEvaluatorConfig

class TrainingComponentsInstantiationModel(BaseModel):
    # ... existing fields ...
    async_evaluator: Optional[AsyncEvaluatorConfig] = None
```

## 7. Main.py Integration

```python
# Modify src/modalities/main.py
from modalities.evaluation.async_evaluator import AsyncEvaluator

class Main:
    def run(self, components: TrainingComponentsInstantiationModel):
        evaluation_result_publisher, progress_publisher = self.get_logging_publishers(...)
        
        # Setup async evaluator if configured
        async_evaluator = None
        checkpoint_saved_publisher = None
        
        if components.async_evaluator and components.async_evaluator.enabled:
            checkpoint_saved_publisher = MessagePublisher(
                message_broker=message_broker,
                global_rank=components.settings.cuda_env.global_rank,
                local_rank=components.settings.cuda_env.local_rank,
            )
            
            async_evaluator = AsyncEvaluator(
                tasks=components.async_evaluator.tasks,
                max_samples=components.async_evaluator.max_samples,
                evaluation_result_publisher=evaluation_result_publisher,
                gpu_id=components.async_evaluator.gpu_id,
                max_workers=components.async_evaluator.max_workers,
                timeout=components.async_evaluator.timeout,
                hf_home=components.async_evaluator.hf_home,
            )
            
            message_broker.add_subscriber(
                subscription=MessageTypes.CHECKPOINT_SAVED, 
                subscriber=async_evaluator
            )

        # Update checkpoint saving with publisher
        original_checkpoint_saving = components.checkpoint_saving
        components.checkpoint_saving = CheckpointSaving(
            checkpoint_saving_strategy=original_checkpoint_saving.checkpoint_saving_strategy,
            checkpoint_saving_execution=original_checkpoint_saving.checkpoint_saving_execution,
            checkpoint_saved_publisher=checkpoint_saved_publisher,
        )

        # Run training
        gym.run(...)
        
        # Cleanup
        if async_evaluator:
            async_evaluator.wait_for_completion()
```

## 8. Example Configuration

```yaml
async_evaluator:
  enabled: true
  tasks: "leaderboard|hellaswag|10|1,leaderboard|gsm8k|8|1"
  max_samples: 100
  gpu_id: null  # Auto-detect
  max_workers: 1
  timeout: 3600
  hf_home: "/path/to/hf/cache"
```

This implementation integrates cleanly with the existing architecture, uses the established WandB logging through the message broker system, and provides automatic GPU detection when not specified.