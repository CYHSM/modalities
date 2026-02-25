from datetime import datetime
from enum import Enum
from typing import Callable, Optional
import math
import random

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.batch import DatasetBatch, EvaluationResultBatch, ResultItem
from modalities.checkpointing.stateful.app_state import AppState
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import ExperimentStatus, MessageTypes, ProgressUpdate
from modalities.logging_broker.publisher import MessagePublisher
from modalities.loss_functions import Loss
from modalities.models.model import model_predict_batch
from modalities.models.parallelism.pipeline_parallelism import Pipeline
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_parallel_degree
from modalities.running_env.fsdp.reducer import Reducer
from modalities.training.gradient_clipping.gradient_clipper import GradientClipperIF
from modalities.training.training_progress import TrainingProgress
from modalities.util import TimeRecorder, print_rank_0
from modalities.utils.mfu import MFUCalculatorABC
from modalities.utils.profilers.profilers import SteppableProfilerIF
from modalities.utils.typing_utils import FSDPX


# =============================================================================
# Ponder Schedulers (constant, random, asymmetric)
# =============================================================================

class ConstantPonderScheduler:
    def __init__(self, model_parts: list[FSDPX], constant_value: float = 0.0):
        self.constant_value = constant_value
        # Unwrap FSDP if needed — config lives on the first model part
        first = model_parts[0]
        self.config_module = first.module if hasattr(first, "module") else first

    def step(self, global_step: int) -> float:
        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = self.constant_value
        return self.constant_value


class RandomPonderScheduler:
    def __init__(self, model_parts: list[FSDPX], min_weight: float = -0.2, max_weight: float = 0.2, seed: int = 42):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.rng = random.Random(seed)
        first = model_parts[0]
        self.config_module = first.module if hasattr(first, "module") else first

    def step(self, global_step: int) -> float:
        weight = self.rng.uniform(self.min_weight, self.max_weight)
        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = weight
        return weight


class AsymmetricPonderScheduler:
    def __init__(
        self, model_parts: list[FSDPX], steps_per_cycle: int,
        base_amplitude: float = 0.05, negative_damping: float = 0.2,
    ):
        self.steps_per_cycle = steps_per_cycle
        self.base_amplitude = base_amplitude
        self.negative_damping = negative_damping
        first = model_parts[0]
        self.config_module = first.module if hasattr(first, "module") else first

    def step(self, global_step: int) -> float:
        cos_val = math.cos(2 * math.pi * global_step / self.steps_per_cycle)
        weight = self.base_amplitude * cos_val
        if weight < 0:
            weight *= self.negative_damping
        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = weight
        return weight


def create_ponder_scheduler(
    model_parts: list[FSDPX],
    scheduler_type: str,
    config_weight: float,
    num_target_steps: int,
    global_rank: int,
):
    """Factory function — add new scheduler types here."""
    if scheduler_type == "constant":
        return ConstantPonderScheduler(model_parts=model_parts, constant_value=config_weight)
    elif scheduler_type == "random":
        return RandomPonderScheduler(model_parts=model_parts, min_weight=1, max_weight=1, seed=42 + global_rank)
    elif scheduler_type == "asymmetric":
        return AsymmetricPonderScheduler(
            model_parts=model_parts, steps_per_cycle=10,
            base_amplitude=0.3, negative_damping=config_weight,
        )
    else:
        return ConstantPonderScheduler(model_parts=model_parts, constant_value=config_weight)


# =============================================================================
# Generic Metrics Accumulator
# =============================================================================
# Handles all three metric categories without knowing specific metric names.
# =============================================================================

class MetricsAccumulator:
    """
    Accumulates metrics from a MetricsBag across microbatches.

    Scalars and per-layer scalars are summed and averaged.
    Per-layer vectors are kept as last-batch snapshots only.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.ce_loss_sum: float = 0.0
        self.ponder_loss_sum: float = 0.0
        self.scalar_sums: dict[str, float] = {}
        self.per_layer_scalar_sums: dict[str, torch.Tensor] = {}
        self.last_per_layer_vectors: dict[str, torch.Tensor] = {}
        self.count: int = 0

    def accumulate(self, loss_metrics: dict):
        """
        Accepts the dict returned by loss_fun.get_metrics().
        """
        self.ce_loss_sum += loss_metrics["ce_loss"].item()
        self.ponder_loss_sum += loss_metrics["ponder_loss"].item()
        self.count += 1

        bag = loss_metrics.get("metrics")
        if bag is None:
            return

        # Scalars
        for name, tensor in bag.get("scalars", {}).items():
            self.scalar_sums[name] = self.scalar_sums.get(name, 0.0) + tensor.item()

        # Per-layer scalars
        for name, tensor in bag.get("per_layer_scalars", {}).items():
            if name not in self.per_layer_scalar_sums:
                self.per_layer_scalar_sums[name] = torch.zeros_like(tensor)
            self.per_layer_scalar_sums[name] += tensor

        # Per-layer vectors — just overwrite (last-batch snapshot)
        for name, tensor in bag.get("per_layer_vectors", {}).items():
            self.last_per_layer_vectors[name] = tensor

    def build_sync_tensor(self, device: torch.device) -> tuple[torch.Tensor, list[str], list[str], dict[str, int]]:
        """
        Packs all accumulated averages into a single flat tensor for one all-reduce.
        Returns (tensor, scalar_names, per_layer_names, per_layer_sizes).
        """
        if self.count == 0:
            return torch.zeros(2, device=device), [], [], {}

        n = self.count
        values = [self.ce_loss_sum / n, self.ponder_loss_sum / n]

        scalar_names = sorted(self.scalar_sums.keys())
        for name in scalar_names:
            values.append(self.scalar_sums[name] / n)

        per_layer_names = sorted(self.per_layer_scalar_sums.keys())
        per_layer_sizes = {}
        layer_tensors = []
        for name in per_layer_names:
            t = self.per_layer_scalar_sums[name] / n
            layer_tensors.append(t.to(device))
            per_layer_sizes[name] = t.numel()

        combined = torch.tensor(values, device=device, dtype=torch.float32)
        if layer_tensors:
            combined = torch.cat([combined, torch.cat(layer_tensors)])

        return combined, scalar_names, per_layer_names, per_layer_sizes

    @staticmethod
    def unpack_synced_tensor(
        synced: torch.Tensor,
        scalar_names: list[str],
        per_layer_names: list[str],
        per_layer_sizes: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Unpacks the all-reduced tensor back into named metrics.
        Returns (ce_loss, ponder_loss, scalars_dict, per_layer_scalars_dict).
        """
        idx = 0
        ce_loss = synced[idx]; idx += 1
        ponder_loss = synced[idx]; idx += 1

        scalars = {}
        for name in scalar_names:
            scalars[name] = synced[idx]; idx += 1

        per_layer_scalars = {}
        for name in per_layer_names:
            size = per_layer_sizes[name]
            per_layer_scalars[name] = synced[idx : idx + size]; idx += size

        return ce_loss, ponder_loss, scalars, per_layer_scalars


# =============================================================================
# Generic Metrics Formatter
# =============================================================================
# Converts synced metrics into the ResultItem dicts for W&B logging.
# =============================================================================

def format_metrics(
    ce_loss: torch.Tensor,
    ponder_loss: torch.Tensor,
    scalars: dict[str, torch.Tensor],
    per_layer_scalars: dict[str, torch.Tensor],
    per_layer_vectors: dict[str, torch.Tensor],
    current_ponder_weight: float,
) -> tuple[dict[str, ResultItem], dict[str, ResultItem]]:
    """
    Returns (losses_dict, metrics_dict) ready for EvaluationResultBatch.

    W&B key hierarchy:
        adaptive/       — global scalars (ponder_weight, expected_steps, ...)
        summary/        — averages across all layers (and loops for vectors)
        layer_{i}/      — per-layer detail: scalars + per-loop vectors + loop averages
        loop_{j}/       — per-loop-step detail: averaged across all layers

    This function never needs to change when new metrics are added in the model.
    """
    losses = {
        "loss/ce_avg": ResultItem(ce_loss, decimal_places=2),
        "ponder/loss_avg": ResultItem(ponder_loss, decimal_places=5),
    }

    metrics: dict[str, ResultItem] = {
        "adaptive/ponder_weight": ResultItem(torch.tensor(current_ponder_weight), 4),
    }

    # ---- adaptive/ : global scalars ----
    for name, val in scalars.items():
        metrics[f"adaptive/{name}"] = ResultItem(val, 4)

    # ---- per-layer scalars → layer_{i}/ + summary/ ----
    for name, vals in per_layer_scalars.items():
        metrics[f"summary/{name}"] = ResultItem(vals.mean(), 4)
        for i, v in enumerate(vals):
            metrics[f"layer_{i}/{name}"] = ResultItem(v, 4)

    # ---- per-layer vectors → layer_{i}/ + loop_{j}/ + summary/ ----
    for name, tensor in per_layer_vectors.items():
        if tensor.numel() == 0:
            continue
        t = tensor.float().cpu()  # (n_layers, max_loops)
        n_layers, n_loops = t.shape

        # summary/ : grand mean across all layers and loops
        metrics[f"summary/{name}"] = ResultItem(t.mean(), 4)

        # layer_{i}/ : average across loops + per-loop detail
        for i in range(n_layers):
            metrics[f"layer_{i}/avg_{name}"] = ResultItem(t[i].mean(), 4)
            for j in range(n_loops):
                metrics[f"layer_{i}/{name}_{j}"] = ResultItem(t[i, j], 4)

        # loop_{j}/ : average across layers
        for j in range(n_loops):
            metrics[f"loop_{j}/{name}"] = ResultItem(t[:, j].mean(), 4)

    return losses, metrics


# =============================================================================
# Throughput keys
# =============================================================================

class ThroughputAggregationKeys(Enum):
    NUM_SAMPLES = "NUM_SAMPLES"
    FORWARD_BACKWARD_TIME = "FORWARD_BACKWARD_TIME"


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    def __init__(
        self,
        global_rank: int,
        progress_publisher: MessagePublisher[ProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        gradient_acc_steps: int,
        global_num_tokens_per_train_step: int,
        device_mesh: DeviceMesh | None,
        num_seen_train_steps: int,
        global_num_seen_tokens: int,
        num_target_steps: int,
        num_target_tokens: int,
        gradient_clipper: GradientClipperIF,
        profiler: SteppableProfilerIF,
        mfu_calculator: MFUCalculatorABC | None = None,
    ) -> None:
        self.global_rank = global_rank
        if device_mesh is not None:
            self.dp_degree = get_parallel_degree(
                device_mesh, [ParallelismDegrees.DP_REPLICATE, ParallelismDegrees.DP_SHARD]
            )
            self.pp_degree = get_parallel_degree(device_mesh, [ParallelismDegrees.PP])
        else:  # TODO: we can remove the else part once we refactored out FSDP1
            self.dp_degree = dist.get_world_size()
            self.pp_degree = 1
        self.progress_publisher = progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
        self.gradient_acc_steps = gradient_acc_steps
        self.global_num_tokens_per_train_step = global_num_tokens_per_train_step
        self.num_seen_train_steps = num_seen_train_steps
        self.num_target_steps = num_target_steps
        self.num_target_tokens = num_target_tokens
        self.global_num_seen_tokens = global_num_seen_tokens
        self.gradient_clipper = gradient_clipper
        self.profiler = profiler
        self.mfu_calculator = mfu_calculator

    @staticmethod
    def _get_num_train_steps_done(micro_batch_id: int, gradient_acc_steps: int) -> int:
        return (micro_batch_id + 1) // gradient_acc_steps

    def _train_batch(
        self,
        batch: DatasetBatch,
        model_parts: list[FSDPX],
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        micro_batch_id: int,
        scheduled_pipeline: Optional[Pipeline] = None,
    ) -> tuple[bool, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if scheduled_pipeline is not None:
            pp_schedule = scheduled_pipeline.pp_schedule
            targets, losses = (
                (batch.targets[loss_fun.target_key].contiguous(), [])
                if scheduled_pipeline.has_last_pp_stage
                else (None, None)
            )

            if scheduled_pipeline.has_first_pp_stage:
                pp_schedule.step(batch.samples[model_parts[0].sample_key].contiguous(), target=targets, losses=losses)
            else:
                pp_schedule.step(target=targets, losses=losses)
            loss = (
                torch.mean(torch.stack(losses)).to(losses[0].device) if scheduled_pipeline.has_last_pp_stage else None
            )
        else:
            result_batch = model_predict_batch(model=model_parts[0], batch=batch)
            loss = loss_fun(result_batch)
            (loss / self.gradient_acc_steps).backward()

        if (micro_batch_id + 1) % self.gradient_acc_steps == 0:
            gradient_norm_score = self.gradient_clipper.clip_gradients()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step_performed = True
            gradient_norm_score = gradient_norm_score.detach().cpu()
        else:
            step_performed = False
            gradient_norm_score = None

        num_train_steps_done = Trainer._get_num_train_steps_done(
            micro_batch_id=micro_batch_id, gradient_acc_steps=self.gradient_acc_steps,
        )
        return step_performed, num_train_steps_done, loss, gradient_norm_score

    def train(
        self,
        app_state: AppState,
        train_loader: LLMDataLoader,
        loss_fun: Loss,
        training_log_interval_in_steps: int,
        evaluation_callback: Callable[[int], None],
        checkpointing_callback: Callable[[TrainingProgress], None],
        scheduled_pipeline: Pipeline | None = None,
    ):
        model_parts = app_state.model_parts
        optimizer = app_state.optimizer
        lr_scheduler = app_state.lr_scheduler
        if scheduled_pipeline is None:
            assert len(model_parts) == 1, "Expected a single model part when no scheduled pipeline is provided."
        for m in model_parts:
            m.train()

        # --- Ponder scheduler ---
        underlying_model = model_parts[0]
        if hasattr(underlying_model, "module"):
            underlying_model = underlying_model.module
        scheduler_type = "constant"
        config_weight = 0.0
        if hasattr(underlying_model, 'adaptive_config') and underlying_model.adaptive_config is not None:
            scheduler_type = underlying_model.adaptive_config.scheduler_type
            config_weight = getattr(underlying_model.adaptive_config, "ponder_penalty_weight", 0.0)

        ponder_scheduler = create_ponder_scheduler(
            model_parts, scheduler_type, config_weight, self.num_target_steps, self.global_rank,
        )
        current_ponder_weight = 0.0

        # --- Accumulators ---
        local_num_seen_samples = 0
        cumulated_losses = torch.zeros(3).cuda()
        metrics_accum = MetricsAccumulator()

        # --- Throughput ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        forward_backward_time_recorder = TimeRecorder()
        forward_backward_time_recorder.start()
        gradient_norm_scores = []

        # --- Initial callbacks ---
        evaluation_callback(num_train_steps_done=self.num_seen_train_steps)
        training_progress = TrainingProgress(
            num_seen_steps_previous_run=self.num_seen_train_steps,
            num_seen_tokens_previous_run=self.global_num_seen_tokens,
            num_seen_steps_current_run=0,
            num_seen_tokens_current_run=0,
            num_target_steps=self.num_target_steps,
            num_target_tokens=self.num_target_tokens,
        )
        checkpointing_callback(training_progress=training_progress)

        num_steps_todo = self.num_target_steps - self.num_seen_train_steps
        num_batches_todo = num_steps_todo * self.gradient_acc_steps

        batch: DatasetBatch
        with self.profiler as profiler_cm:
            for _, (micro_batch_id, batch) in zip(range(num_batches_todo), enumerate(train_loader)):
                current_ponder_weight = ponder_scheduler.step(training_progress.num_seen_steps_total)

                (
                    step_performed, num_train_steps_done, batch_loss, gradient_norm_score,
                ) = self._train_batch(
                    batch=batch,
                    model_parts=model_parts,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    loss_fun=loss_fun,
                    micro_batch_id=micro_batch_id,
                    scheduled_pipeline=scheduled_pipeline,
                )
                training_progress.num_seen_steps_current_run = num_train_steps_done
                training_progress.num_seen_tokens_current_run = (
                    self.global_num_tokens_per_train_step * num_train_steps_done
                )

                # --- Accumulate ---
                if batch_loss is not None:
                    cumulated_losses[0] += batch_loss.detach().item()
                    cumulated_losses[-1] += 1

                    if hasattr(loss_fun, 'get_metrics'):
                        metrics_accum.accumulate(loss_fun.get_metrics())

                if gradient_norm_score is not None:
                    gradient_norm_scores.append(gradient_norm_score.item())

                local_num_seen_samples += len(batch)

                self._publish_progress(
                    progress_publisher=self.progress_publisher,
                    num_train_steps_done=training_progress.num_seen_steps_total,
                    dataloader_tag=train_loader.dataloader_tag,
                )

                # ==============================================================
                # LOG INTERVAL — generic metric syncing and formatting
                # ==============================================================
                if training_progress.num_seen_steps_total % training_log_interval_in_steps == 0 and step_performed:
                    forward_backward_time_recorder.stop()
                    forward_backward_time = forward_backward_time_recorder.delta_t
                    forward_backward_time_recorder.reset()
                    forward_backward_time_recorder.start()

                    global_num_seen_samples = local_num_seen_samples * self.dp_degree
                    local_num_seen_samples = 0
                    global_num_samples_per_second = global_num_seen_samples / forward_backward_time

                    # --- Reduce total loss (same as upstream) ---
                    cumulated_losses[1] = batch_loss.detach().item() if batch_loss is not None else 0.0
                    reduced_losses = (
                        Reducer.reduce(
                            tensor=cumulated_losses,
                            operation=dist.ReduceOp.SUM,
                            post_processing_fun=lambda t: torch.stack(
                                [t[0] / t[-1], t[1] / dist.get_world_size() * self.pp_degree]
                            ),
                        )
                        .detach()
                        .cpu()
                    )
                    train_loss_avg = reduced_losses[0]
                    train_loss_last_batch = reduced_losses[1]

                    # --- Generic metric sync (single all-reduce) ---
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

                    # --- Format into W&B dicts ---
                    adaptive_losses, adaptive_metrics = format_metrics(
                        ce_loss=synced_ce,
                        ponder_loss=synced_ponder,
                        scalars=synced_scalars,
                        per_layer_scalars=synced_per_layer,
                        per_layer_vectors=metrics_accum.last_per_layer_vectors,
                        current_ponder_weight=current_ponder_weight,
                    )

                    # --- Combine all losses and metrics ---
                    losses = {
                        "train loss avg": ResultItem(train_loss_avg, decimal_places=2),
                        "train loss last": ResultItem(train_loss_last_batch, decimal_places=2),
                        **adaptive_losses,
                    }

                    consumed_tokens = torch.tensor(training_progress.num_seen_tokens_total)
                    metrics = {
                        "consumed tokens": ResultItem(consumed_tokens, 0),
                        "grad norm avg": ResultItem(torch.mean(torch.Tensor(gradient_norm_scores)), 2),
                        "grad norm last": ResultItem(torch.tensor(gradient_norm_scores[-1]), 2),
                        **adaptive_metrics,
                    }
                    gradient_norm_scores = []

                    # --- MFU ---
                    mfu_score = torch.tensor(-1.0)
                    if self.mfu_calculator is not None:
                        mfu_score = self.mfu_calculator.compute(num_samples_per_second=global_num_samples_per_second)

                    # --- Peak memory ---
                    if device.type == "cuda":
                        peak_memory_MB = torch.cuda.max_memory_allocated(device) / 1024**2
                        torch.cuda.reset_peak_memory_stats(device)
                    else:
                        try:
                            import resource
                            peak_memory_MB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                        except Exception:
                            peak_memory_MB = -1.0

                    training_metrics = EvaluationResultBatch(
                        losses=losses,
                        metrics=metrics,
                        throughput_metrics={
                            "train samples/s": ResultItem(torch.tensor(global_num_samples_per_second), 1),
                            "train mfu (16-bit)": ResultItem(torch.tensor(mfu_score), 2),
                            "lr mean": ResultItem(torch.tensor(lr_scheduler.get_last_lr()).mean()),
                            "peak memory rank 0 (MB)": ResultItem(torch.tensor(peak_memory_MB), 2),
                        },
                        dataloader_tag=train_loader.dataloader_tag,
                        num_train_steps_done=training_progress.num_seen_steps_total,
                    )
                    print_rank_0(f"{datetime.now().isoformat(timespec='seconds')} | {training_metrics}")
                    self._publish_evaluation_result(
                        evaluation_result_publisher=self.evaluation_result_publisher,
                        evaluation_result=training_metrics,
                    )

                    # --- Reset all accumulators ---
                    cumulated_losses.zero_()
                    metrics_accum.reset()

                if step_performed:
                    evaluation_callback(num_train_steps_done=training_progress.num_seen_steps_total)
                    checkpointing_callback(training_progress=training_progress)
                profiler_cm.step()

    @staticmethod
    def _publish_progress(
        progress_publisher: MessagePublisher[ProgressUpdate],
        num_train_steps_done: int,
        dataloader_tag: str,
    ):
        payload = ProgressUpdate(
            num_steps_done=num_train_steps_done,
            experiment_status=ExperimentStatus.TRAIN,
            dataloader_tag=dataloader_tag,
        )
        progress_publisher.publish_message(payload=payload, message_type=MessageTypes.BATCH_PROGRESS_UPDATE)

    @staticmethod
    def _publish_evaluation_result(
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        evaluation_result: EvaluationResultBatch,
    ):
        evaluation_result_publisher.publish_message(
            payload=evaluation_result, message_type=MessageTypes.EVALUATION_RESULT,
        )