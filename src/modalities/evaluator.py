import json
import logging
import subprocess
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from modalities.batch import DatasetBatch, EvaluationResultBatch, InferenceResultBatch, ResultItem
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import ExperimentStatus, MessageTypes, ProgressUpdate
from modalities.logging_broker.publisher import MessagePublisher
from modalities.models.model import model_predict_batch
from modalities.models.parallelism.pipeline_parallelism import Pipeline
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_parallel_degree
from modalities.running_env.fsdp.reducer import Reducer
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper
from modalities.util import TimeRecorder

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class which is responsible for evaluating the model on a set of datasets"""

    def __init__(
        self,
        progress_publisher: MessagePublisher[ProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        device_mesh: DeviceMesh | None = None,
    ) -> None:
        """Initializes the Evaluator class.

        Args:
            progress_publisher (MessagePublisher[ProgressUpdate]): Publisher for progress updates
            evaluation_result_publisher (MessagePublisher[EvaluationResultBatch]): Publisher for evaluation results
        """
        self.progress_publisher = progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
        if device_mesh is not None:
            self.dp_degree = get_parallel_degree(
                device_mesh, [ParallelismDegrees.DP_REPLICATE, ParallelismDegrees.DP_SHARD]
            )
            self.pp_degree = get_parallel_degree(device_mesh, [ParallelismDegrees.PP])
        else:  # TODO: we can remove the else part once we refactored out FSDP1
            self.dp_degree = dist.get_world_size()
            self.pp_degree = 1

    def evaluate_batch(
        self,
        batch: DatasetBatch,
        model: list[nn.Module],
        loss_fun: Callable[[InferenceResultBatch], torch.Tensor],
        scheduled_pipeline: Pipeline | None = None,
    ) -> torch.Tensor | None:
        """Evaluate a single batch by forwarding it through the model and calculating the loss.

        Args:
            batch (DatasetBatch): The batch to evaluate
            model (list[nn.Module]): The model (parts) to evaluate
            loss_fun (Callable[[InferenceResultBatch], torch.Tensor]): The loss function to calculate the loss
            scheduled_pipeline (Pipeline | None, optional): In case of pipeline parallelism, this is used to
                operate the model. Defaults to None.

        Returns:
            torch.Tensor | None: The loss of the batch
                None, if a non-last stage was processed in pipeline parallelism
        """
        with torch.no_grad():
            if scheduled_pipeline is not None:
                pp_schedule = scheduled_pipeline.pp_schedule
                targets, losses = (
                    (batch.targets[loss_fun.target_key].contiguous(), [])
                    if scheduled_pipeline.has_last_pp_stage
                    else (None, None)
                )

                if scheduled_pipeline.has_first_pp_stage:
                    pp_schedule.eval(batch.samples[model[0].sample_key].contiguous(), target=targets, losses=losses)
                else:
                    pp_schedule.eval(target=targets, losses=losses)
                loss = (
                    torch.mean(torch.stack(losses)).to(losses[0].device)
                    if scheduled_pipeline.has_last_pp_stage
                    else None
                )
            else:
                result_batch = model_predict_batch(model=model[0], batch=batch)
                loss = loss_fun(result_batch)
        return loss

    def evaluate(
        self,
        model: list[nn.Module] | nn.Module,
        data_loaders: list[LLMDataLoader],
        loss_fun: Callable[[InferenceResultBatch], torch.Tensor],
        num_train_steps_done: int,
        scheduled_pipeline: Pipeline | None = None,
    ) -> dict[str, EvaluationResultBatch]:
        """Evaluate the model on a set of datasets.

        Args:
            model (list[nn.Module] | nn.Module): The model or model parts to evaluate
            data_loaders (list[LLMDataLoader]): List of dataloaders to evaluate the model on
            loss_fun (Callable[[InferenceResultBatch], torch.Tensor]): The loss function to calculate the loss
            num_train_steps_done (int): The number of training steps done so far for logging purposes
            scheduled_pipeline (Pipeline | None, optional): In case of pipeline parallelism, this is used to
                operate the model. Defaults to None.

        Returns:
            dict[str, EvaluationResultBatch]: A dictionary containing the evaluation results for each dataloader
        """
        result_dict: dict[str, EvaluationResultBatch] = {}
        if not isinstance(model, list):
            assert scheduled_pipeline is None, "A non-scheduled pipeline should be processed with a single model."
            model = [model]
        for m in model:
            m.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for data_loader in data_loaders:
            local_num_seen_samples = 0
            cumulated_loss = torch.zeros(3).to(device)

            Evaluator._publish_progress(
                progress_publisher=self.progress_publisher,
                num_eval_steps_done=0,  # Reset progress bar
                dataloader_tag=data_loader.dataloader_tag,
            )
            with TimeRecorder() as forward_backward_timer_recorder:
                for batch_id, batch in enumerate(data_loader):
                    batch_loss = self.evaluate_batch(
                        batch=batch,
                        model=model,
                        loss_fun=loss_fun,
                        scheduled_pipeline=scheduled_pipeline,
                    )

                    # The batch_loss might be None if we use pipeline parallelism and are not the last stage.
                    if batch_loss is not None:
                        cumulated_loss[0] += batch_loss.item()  # sum up batch loss
                        cumulated_loss[1] += 1
                    local_num_seen_samples += torch.tensor(len(batch)).to(device)

                    Evaluator._publish_progress(
                        progress_publisher=self.progress_publisher,
                        num_eval_steps_done=batch_id + 1,
                        dataloader_tag=data_loader.dataloader_tag,
                    )
            # TODO: insert reducer from outside so Evaluator is independent of FSDP
            total_loss = Reducer.reduce(
                tensor=cumulated_loss,
                operation=dist.ReduceOp.SUM,
                post_processing_fun=lambda t: t[0] / t[1],
            )

            forward_backward_time = torch.tensor(forward_backward_timer_recorder.delta_t).to(device)
            global_num_seen_samples = local_num_seen_samples * self.dp_degree

            num_samples_per_second = global_num_seen_samples / forward_backward_time

            evaluation_result = EvaluationResultBatch(
                losses={loss_fun.tag: ResultItem(total_loss, decimal_places=2)},
                # TODO: hardcoded metric key
                throughput_metrics={
                    "evaluation_num_samples_per_second": ResultItem(num_samples_per_second, decimal_places=1)
                },
                dataloader_tag=data_loader.dataloader_tag,
                num_train_steps_done=num_train_steps_done,
            )
            Evaluator._publish_evaluation_result(
                evaluation_result_publisher=self.evaluation_result_publisher,
                evaluation_result=evaluation_result,
            )
            result_dict[data_loader.dataloader_tag] = evaluation_result

        for m in model:
            m.train()

        return result_dict

    @staticmethod
    def _publish_progress(
        progress_publisher: MessagePublisher[ProgressUpdate],
        num_eval_steps_done: int,
        dataloader_tag: str,
    ):
        payload = ProgressUpdate(
            num_steps_done=num_eval_steps_done,
            experiment_status=ExperimentStatus.EVALUATION,
            dataloader_tag=dataloader_tag,
        )
        progress_publisher.publish_message(payload=payload, message_type=MessageTypes.BATCH_PROGRESS_UPDATE)

    @staticmethod
    def _publish_evaluation_result(
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        evaluation_result: EvaluationResultBatch,
    ):
        evaluation_result_publisher.publish_message(
            payload=evaluation_result, message_type=MessageTypes.EVALUATION_RESULT
        )


class DownstreamEvaluator:
    """Evaluator that converts the latest DCP checkpoint to HF format and runs OLMES on it.

    Both steps are launched as a single dedicated sbatch job (see
    ``scripts/evaluation/run_convert_and_eval_sbatch.sh``), submitted asynchronously via a
    non-blocking ``subprocess.Popen`` so rank 0 never stalls the training process's own
    collectives. Conversion used to run synchronously inside the training loop (the old
    ``ModelConverter`` component); for checkpoints that take longer than NCCL's collective
    timeout to convert, that stalled rank 0 past the timeout and crashed the whole job.
    """

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        tasks: list[str],
        eval_interval: int,
        checkpoint_dir: Path,
        global_rank: int,
        olmes_command_template: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.tasks = tasks
        self.eval_interval = eval_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.global_rank = global_rank
        self.olmes_command_template = olmes_command_template
        self.active_processes: list[tuple[subprocess.Popen, int, Path]] = []

    def evaluate(self, num_train_steps_done: int) -> None:
        if num_train_steps_done == 0 or num_train_steps_done % self.eval_interval != 0:
            return
        if self.global_rank != 0:
            return

        checkpoint_path = self._resolve_checkpoint_path()
        if checkpoint_path is None:
            logger.warning(
                f"No checkpoint info found in {self.checkpoint_dir} at step {num_train_steps_done}, "
                "skipping conversion+downstream evaluation."
            )
            return

        config_path = self._resolve_config_path()
        hf_model_dir = checkpoint_path / "hf_checkpoint"

        tasks_str = " ".join(self.tasks)
        cmd = self.olmes_command_template.format(
            checkpoint_path=str(checkpoint_path),
            modalities_config=str(config_path),
            hf_model_dir=str(hf_model_dir),
            tasks=tasks_str,
            step=num_train_steps_done,
        )

        logger.info(f"Launching conversion+downstream evaluation: {cmd}")
        try:
            p = subprocess.Popen(cmd, shell=True)
            self.active_processes.append((p, num_train_steps_done, hf_model_dir))
            logger.info(f"Conversion+downstream evaluation launched for step {num_train_steps_done}.")
        except Exception as e:
            logger.error(f"Failed to launch conversion+downstream evaluation: {e}")

    def wait_for_evaluations(self) -> None:
        if not hasattr(self, "active_processes") or not self.active_processes:
            return

        logger.info(f"Waiting for {len(self.active_processes)} downstream evaluations to finish...")
        for p, step, hf_model_dir in self.active_processes:
            p.wait()
            if p.returncode == 0:
                self._sync_metrics_to_wandb(step, hf_model_dir)
            else:
                logger.warning(f"Downstream evaluation for step {step} exited with code {p.returncode}, skipping W&B sync.")
        logger.info("All downstream evaluations finished.")
        self.active_processes = []

    def _sync_metrics_to_wandb(self, step: int, hf_model_dir: Path) -> None:
        """Parse OLMES metrics-all.jsonl and log primary scores to the active W&B run."""
        metrics_file = hf_model_dir / f"olmes_eval_{step}" / "metrics-all.jsonl"
        if not metrics_file.exists():
            logger.warning(f"No metrics file found at {metrics_file}, skipping W&B sync for step {step}.")
            return

        metrics_dict = {}
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    alias = (
                        obj.get("task_config", {}).get("metadata", {}).get("alias")
                        or obj.get("task_name")
                    )
                    score = obj.get("metrics", {}).get("primary_score")
                    if alias and score is not None:
                        metrics_dict[f"downstream/{alias}"] = score
        except Exception as e:
            logger.error(f"Failed to parse metrics file {metrics_file}: {e}")
            return

        if not metrics_dict:
            logger.warning(f"No metrics extracted from {metrics_file} for step {step}.")
            return

        try:
            import wandb

            if wandb.run is not None:
                # Define a custom step metric so downstream/* metrics are decoupled from
                # the global training step counter (which is already past these steps).
                # NOT hidden: hidden=True also removes it from the manual-panel x-axis
                # picker in the W&B UI, which is the whole point of exposing it here.
                wandb.run.define_metric("downstream_step")
                wandb.run.define_metric("downstream/*", step_metric="downstream_step")
                metrics_dict["downstream_step"] = step
                wandb.run.log(metrics_dict)
                logger.info(f"Synced {len(metrics_dict)} OLMES metrics to W&B at step {step}: {metrics_dict}")
            else:
                logger.info(f"W&B not active, skipping metric sync for step {step}.")
        except ImportError:
            logger.info(f"wandb not installed, skipping metric sync for step {step}.")

    def _resolve_checkpoint_path(self) -> Path | None:
        """Read last_checkpoint_info.json and return the latest DCP checkpoint directory."""
        info_file = self.checkpoint_dir / "last_checkpoint_info.json"
        if not info_file.exists():
            return None

        try:
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            cp_str = (
                info.get("checkpoint_folder_path")
                or info.get("checkpoint_path")
                or info.get("model_checkpoint_path")
            )
            if not cp_str:
                return None
        except Exception as e:
            logger.error(f"Failed to read {info_file}: {e}")
            return None

        checkpoint_path = Path(cp_str)
        if checkpoint_path.is_file():
            checkpoint_path = checkpoint_path.parent
        return checkpoint_path

    def _resolve_config_path(self) -> Path:
        """Locate the resolved training config next to the checkpoints (same logic as the
        old ModelConverter): prefer the fully-resolved config dump, since the plain config
        copy still contains ${...} interpolations that cannot be resolved outside training.
        """
        config_path = next(iter(sorted(self.checkpoint_dir.glob("*.yaml.resolved"))), None)
        if config_path is not None:
            return config_path
        if (self.checkpoint_dir / "config.yaml").exists():
            return self.checkpoint_dir / "config.yaml"
        yaml_files = sorted(self.checkpoint_dir.glob("*.yaml"))
        return yaml_files[0] if yaml_files else self.checkpoint_dir / "config.yaml"

