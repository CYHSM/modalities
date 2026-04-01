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
from modalities.trainer import MetricsAccumulator, format_metrics
from modalities.util import TimeRecorder


class Evaluator:
    """Evaluator class which is responsible for evaluating the model on a set of datasets"""

    def __init__(
        self,
        progress_publisher: MessagePublisher[ProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        device_mesh: DeviceMesh | None = None,
        tokenizer: "PydanticTokenizerIFType | None" = None,
    ) -> None:
        self.progress_publisher = progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
        if device_mesh is not None:
            self.dp_degree = get_parallel_degree(
                device_mesh, [ParallelismDegrees.DP_REPLICATE, ParallelismDegrees.DP_SHARD]
            )
            self.pp_degree = get_parallel_degree(device_mesh, [ParallelismDegrees.PP])
        else:
            self.dp_degree = dist.get_world_size()
            self.pp_degree = 1

        self.tokenizer = tokenizer

        self.hardcoded_examples = [
            # Math reasoning (GSM8k style, with answer)
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
            "and bakes muffins for her friends every day with four. She sells the remainder "
            "at the farmers' market daily for $2 per fresh duck egg. How much in dollars does "
            "she make every day at the farmers' market? She sells 16 - 3 - 4 = 9 duck eggs "
            "a day. She makes 9 * 2 = $18 every day at the farmers' market.",
            # Factual knowledge across domains (history, physics, biology)
            "The Great Wall of China was built over many centuries, beginning in the 7th "
            "century BC. The speed of light in a vacuum is approximately 299,792 kilometers "
            "per second. Mitochondria are often called the powerhouse of the cell because "
            "they generate most of the cell's supply of adenosine triphosphate.",
        ]

    def evaluate_batch(
        self,
        batch: DatasetBatch,
        model: list[nn.Module],
        loss_fun: Callable[[InferenceResultBatch], torch.Tensor],
        scheduled_pipeline: Pipeline | None = None,
    ) -> torch.Tensor | None:
        """Evaluate a single batch."""
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

        result_dict: dict[str, EvaluationResultBatch] = {}
        if not isinstance(model, list):
            assert scheduled_pipeline is None
            model = [model]

        for m in model:
            m.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        underlying = model[0].module if hasattr(model[0], "module") else model[0]
        thresholds = [None]
        if hasattr(underlying, "adaptive_config") and underlying.adaptive_config and hasattr(underlying.adaptive_config, "eval_routing_thresholds"):
            for t in underlying.adaptive_config.eval_routing_thresholds:
                if t not in thresholds:
                    thresholds.append(t)

        has_adaptive = hasattr(underlying, "use_adaptive") and underlying.use_adaptive

        for threshold in thresholds:

            for m in model:
                m_underlying = m.module if hasattr(m, "module") else m
                if hasattr(m_underlying, "set_routing_threshold"):
                    m_underlying.set_routing_threshold(threshold)

            threshold_tag = f"@route_{threshold}" if threshold is not None else "@soft_blend"

            for data_loader in data_loaders:
                local_num_seen_samples = 0
                cumulated_loss = torch.zeros(3).to(device)
                current_tag = f"{data_loader.dataloader_tag}{threshold_tag}"

                Evaluator._publish_progress(
                    progress_publisher=self.progress_publisher,
                    num_eval_steps_done=0,
                    dataloader_tag=data_loader.dataloader_tag,
                )

                metrics_accum = MetricsAccumulator()

                with TimeRecorder() as forward_backward_timer_recorder:
                    for batch_id, batch in enumerate(data_loader):
                        batch_loss = self.evaluate_batch(
                            batch=batch, model=model, loss_fun=loss_fun,
                            scheduled_pipeline=scheduled_pipeline,
                        )

                        if batch_loss is not None:
                            cumulated_loss[0] += batch_loss.item()
                            cumulated_loss[1] += 1

                        if has_adaptive and hasattr(loss_fun, "get_metrics"):
                            try:
                                metrics_accum.accumulate(loss_fun.get_metrics())
                            except (KeyError, AttributeError):
                                pass

                        local_num_seen_samples += torch.tensor(len(batch)).to(device)

                        Evaluator._publish_progress(
                            progress_publisher=self.progress_publisher,
                            num_eval_steps_done=batch_id + 1,
                            dataloader_tag=data_loader.dataloader_tag,
                        )

                total_loss = Reducer.reduce(
                    tensor=cumulated_loss,
                    operation=dist.ReduceOp.SUM,
                    post_processing_fun=lambda t: t[0] / t[1],
                )

                forward_backward_time = torch.tensor(forward_backward_timer_recorder.delta_t).to(device)
                global_num_seen_samples = local_num_seen_samples * self.dp_degree
                num_samples_per_second = global_num_seen_samples / forward_backward_time

                eval_adaptive_losses = {}
                eval_adaptive_metrics = {}
                if has_adaptive and metrics_accum.count > 0:
                    (
                        sync_tensor, scalar_names, per_layer_names, per_layer_sizes,
                    ) = metrics_accum.build_sync_tensor(device)

                    reduce_scale = dist.get_world_size() / self.pp_degree
                    synced_tensor = Reducer.reduce(
                        tensor=sync_tensor,
                        operation=dist.ReduceOp.SUM,
                        post_processing_fun=lambda t: t / reduce_scale,
                    )

                    (
                        synced_ce, synced_ponder, synced_scalars, synced_per_layer,
                    ) = MetricsAccumulator.unpack_synced_tensor(
                        synced_tensor, scalar_names, per_layer_names, per_layer_sizes,
                    )

                    ponder_weight = 0.0
                    if underlying.adaptive_config is not None:
                        ponder_weight = getattr(underlying.adaptive_config, "ponder_penalty_weight", 0.0)

                    eval_adaptive_losses, eval_adaptive_metrics = format_metrics(
                        ce_loss=synced_ce,
                        ponder_loss=synced_ponder,
                        scalars=synced_scalars,
                        per_layer_scalars=synced_per_layer,
                        per_layer_vectors=metrics_accum.last_per_layer_vectors,
                        current_ponder_weight=ponder_weight,
                        summary_only=True,
                    )

                evaluation_result = EvaluationResultBatch(
                    losses={
                        loss_fun.tag: ResultItem(total_loss, decimal_places=2),
                        **eval_adaptive_losses,
                    },
                    throughput_metrics={
                        "evaluation_num_samples_per_second": ResultItem(num_samples_per_second, decimal_places=1)
                    },
                    metrics={**eval_adaptive_metrics},
                    dataloader_tag=current_tag,
                    num_train_steps_done=num_train_steps_done,
                )

                Evaluator._publish_evaluation_result(
                    evaluation_result_publisher=self.evaluation_result_publisher,
                    evaluation_result=evaluation_result,
                )
                result_dict[current_tag] = evaluation_result

            # ================================================================
            # Hardcoded examples — token routing visualization
            # ================================================================
            if self.tokenizer is not None and has_adaptive:
                hardcoded_tag = f"hardcoded_examples{threshold_tag}"

                encoded_examples = [self.tokenizer.tokenize(text) for text in self.hardcoded_examples]

                max_len = max(len(seq) for seq in encoded_examples)
                pad_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else 0
                padded_examples = []
                original_lengths = []
                for seq in encoded_examples:
                    original_lengths.append(len(seq))
                    padded_examples.append(seq + [pad_token_id] * (max_len - len(seq)))

                input_tensor = torch.tensor(padded_examples).to(device)

                dummy_batch = DatasetBatch(
                    samples={model[0].sample_key: input_tensor},
                    targets={loss_fun.target_key: input_tensor},
                )

                hardcoded_loss = self.evaluate_batch(
                    batch=dummy_batch, model=model, loss_fun=loss_fun,
                    scheduled_pipeline=scheduled_pipeline,
                )

                hc_eval_tokens = None
                hc_eval_gate_deep_probs = None
                hc_eval_gate_wide_probs = None
                hc_eval_expected_steps = None

                if hasattr(loss_fun, "get_metrics"):
                    m_bag = loss_fun.get_metrics().get("metrics")
                    if m_bag is not None:
                        if "eval_tokens" in m_bag:
                            hc_eval_tokens = m_bag["eval_tokens"].detach().cpu()
                        if "eval_gate_deep_probs" in m_bag:
                            hc_eval_gate_deep_probs = m_bag["eval_gate_deep_probs"].detach().cpu()
                        if "eval_gate_wide_probs" in m_bag:
                            hc_eval_gate_wide_probs = m_bag["eval_gate_wide_probs"].detach().cpu()
                        if "eval_expected_steps" in m_bag:
                            hc_eval_expected_steps = m_bag["eval_expected_steps"].detach().cpu()

                hc_evaluation_result = EvaluationResultBatch(
                    losses={
                        loss_fun.tag: ResultItem(
                            hardcoded_loss if hardcoded_loss is not None else torch.zeros(1),
                            decimal_places=2,
                        )
                    },
                    throughput_metrics={},
                    metrics={},
                    dataloader_tag=hardcoded_tag,
                    num_train_steps_done=num_train_steps_done,
                )

                if hc_eval_tokens is not None:
                    hc_evaluation_result.metrics["eval_tokens"] = ResultItem(hc_eval_tokens)
                if hc_eval_gate_deep_probs is not None:
                    hc_evaluation_result.metrics["eval_gate_deep_probs"] = ResultItem(hc_eval_gate_deep_probs.float())
                if hc_eval_gate_wide_probs is not None:
                    hc_evaluation_result.metrics["eval_gate_wide_probs"] = ResultItem(hc_eval_gate_wide_probs.float())
                if hc_eval_expected_steps is not None:
                    hc_evaluation_result.metrics["eval_expected_steps"] = ResultItem(hc_eval_expected_steps.float())

                Evaluator._publish_evaluation_result(
                    evaluation_result_publisher=self.evaluation_result_publisher,
                    evaluation_result=hc_evaluation_result,
                )
                result_dict[hardcoded_tag] = hc_evaluation_result

        # Reset threshold before returning to training
        for m in model:
            m_underlying = m.module if hasattr(m, "module") else m
            if hasattr(m_underlying, "set_routing_threshold"):
                m_underlying.set_routing_threshold(None)
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