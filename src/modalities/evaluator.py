from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from modalities.batch import DatasetBatch, EvaluationResultBatch, InferenceResultBatch, ResultItem
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.evaluation_metrics import EvaluationMetricIF
from modalities.logging_broker.messages import ExperimentStatus, MessageTypes, ProgressUpdate
from modalities.logging_broker.publisher import MessagePublisher
from modalities.models.model import model_predict_batch
from modalities.models.parallelism.pipeline_parallelism import Pipeline
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_parallel_degree
from modalities.running_env.fsdp.reducer import Reducer
from modalities.util import TimeRecorder


class Evaluator:
    """Evaluator class which is responsible for evaluating the model on a set of datasets"""

    def __init__(
        self,
        progress_publisher: MessagePublisher[ProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        device_mesh: DeviceMesh | None = None,
        metrics: list[EvaluationMetricIF] | None = None,
    ) -> None:
        """Initializes the Evaluator class.

        Args:
            progress_publisher (MessagePublisher[ProgressUpdate]): Publisher for progress updates
            evaluation_result_publisher (MessagePublisher[EvaluationResultBatch]): Publisher for evaluation results
            device_mesh (DeviceMesh | None, optional): The device mesh, used to determine the parallel degrees
            metrics (list[EvaluationMetricIF] | None, optional): Metrics reported alongside the loss. Each metric
                selects the dataloaders it applies to by tag, so a metric that assumes a particular target format
                is not computed on dataloaders that do not use it.
        """
        self.progress_publisher = progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
        self.metrics = list(metrics) if metrics is not None else []
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
        metrics: list[EvaluationMetricIF] | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """Evaluate a single batch by forwarding it through the model and calculating the loss.

        Args:
            batch (DatasetBatch): The batch to evaluate
            model (list[nn.Module]): The model (parts) to evaluate
            loss_fun (Callable[[InferenceResultBatch], torch.Tensor]): The loss function to calculate the loss
            scheduled_pipeline (Pipeline | None, optional): In case of pipeline parallelism, this is used to
                operate the model. Defaults to None.
            metrics (list[EvaluationMetricIF] | None, optional): Metrics to evaluate on this batch. Ignored under
                pipeline parallelism, where the schedule consumes the model output internally and never exposes
                the logits a metric would need.

        Returns:
            tuple[torch.Tensor | None, dict[str, torch.Tensor]]: The loss of the batch (None, if a non-last stage
                was processed in pipeline parallelism) and each metric's ``[numerator, denominator]`` contribution,
                keyed by metric tag.
        """
        metric_contributions: dict[str, torch.Tensor] = {}
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
                for metric in metrics or []:
                    metric_contributions[metric.tag] = metric(result_batch)
        return loss, metric_contributions

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
            dict[str, EvaluationResultBatch]: A dictionary containing the evaluation results for each dataloader.
                Each result carries the loss and, under ``metrics``, whichever of this Evaluator's metrics apply to
                that dataloader's tag.
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
            applicable_metrics = [metric for metric in self.metrics if metric.applies_to(data_loader.dataloader_tag)]
            # Per metric, the running [numerator, denominator]. Both are summed and only divided
            # after the cross-rank reduction, so the result does not depend on how the samples
            # happen to be split across batches or ranks.
            cumulated_metrics = {metric.tag: torch.zeros(2).to(device) for metric in applicable_metrics}

            Evaluator._publish_progress(
                progress_publisher=self.progress_publisher,
                num_eval_steps_done=0,  # Reset progress bar
                dataloader_tag=data_loader.dataloader_tag,
            )
            with TimeRecorder() as forward_backward_timer_recorder:
                for batch_id, batch in enumerate(data_loader):
                    batch_loss, batch_metrics = self.evaluate_batch(
                        batch=batch,
                        model=model,
                        loss_fun=loss_fun,
                        scheduled_pipeline=scheduled_pipeline,
                        metrics=applicable_metrics,
                    )

                    # The batch_loss might be None if we use pipeline parallelism and are not the last stage.
                    if batch_loss is not None:
                        cumulated_loss[0] += batch_loss.item()  # sum up batch loss
                        cumulated_loss[1] += 1
                    for metric_tag, contribution in batch_metrics.items():
                        cumulated_metrics[metric_tag] += contribution.to(device)
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
            metric_results: dict[str, ResultItem] = {}
            for metric_tag, cumulated_metric in cumulated_metrics.items():
                reduced = Reducer.reduce(tensor=cumulated_metric, operation=dist.ReduceOp.SUM)
                # A denominator of zero means no batch contributed a scored position, e.g. a
                # metric was pointed at a dataloader whose targets are never masked. Reporting a
                # NaN there would be indistinguishable from a diverged model, so skip it instead.
                if reduced[1] > 0:
                    metric_results[metric_tag] = ResultItem(reduced[0] / reduced[1], decimal_places=4)

            forward_backward_time = torch.tensor(forward_backward_timer_recorder.delta_t).to(device)
            global_num_seen_samples = local_num_seen_samples * self.dp_degree

            num_samples_per_second = global_num_seen_samples / forward_backward_time

            evaluation_result = EvaluationResultBatch(
                losses={loss_fun.tag: ResultItem(total_loss, decimal_places=2)},
                metrics=metric_results,
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
