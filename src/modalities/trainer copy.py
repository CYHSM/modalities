from datetime import datetime
from enum import Enum
from typing import Callable, Optional
import random
import math

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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
from modalities.util import Aggregator, TimeRecorder, print_rank_0
from modalities.utils.mfu import MFUCalculatorABC


class LinearDecayPonderScheduler:
    def __init__(
        self,
        model: torch.nn.Module,
        total_train_steps: int,
        start_weight: float = 0.1,
        end_weight: float = 0.01,
    ):
        """
        Curriculum approach:
        Starts with a high penalty to force efficient representations,
        then linearly relaxes the constraint to allow more thinking later.
        """
        self.model = model
        self.total_train_steps = total_train_steps
        self.start_weight = start_weight
        self.end_weight = end_weight
        
        if isinstance(model, FSDP):
            self.config_module = model.module
        else:
            self.config_module = model

    def step(self, global_step: int) -> float:
        progress = min(1.0, global_step / self.total_train_steps)
        # Linear interpolation
        weight = self.start_weight + progress * (self.end_weight - self.start_weight)
        
        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = weight
            
        return weight


class RandomPonderScheduler:
    def __init__(
        self, 
        model: torch.nn.Module, 
        min_weight: float = -0.2, 
        max_weight: float = 0.2,
        seed: int = 42
    ):
        """
        Random Baseline: Samples a weight uniformly between min and max at every step.
        """
        self.model = model
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.rng = random.Random(seed)
        
        # Unwrap FSDP if necessary
        if isinstance(model, FSDP):
             self.config_module = model.module
        else:
             self.config_module = model

    def step(self, global_step: int) -> float:
        # Uniform sample
        weight = self.rng.uniform(self.min_weight, self.max_weight)
        
        # Update model config in-place
        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = weight
            
        return weight


class ConstantPonderScheduler:
    def __init__(self, model: torch.nn.Module, constant_value: float = 0.0):
        self.constant_value = constant_value
        self.config_module = model.module if isinstance(model, FSDP) else model

    def step(self, global_step: int) -> float:
        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = self.constant_value
        return self.constant_value


class SimpleLinearScheduler:
    def __init__(
        self,
        model: torch.nn.Module,
        total_train_steps: int,
        start_weight: float = -0.01,
        end_weight: float = 0.01,
    ):
        """
        Linearly interpolates from start_weight (reward) to end_weight (penalty).
        """
        self.total_train_steps = total_train_steps
        self.start_weight = start_weight
        self.end_weight = end_weight
        
        # Unwrap FSDP if necessary
        if isinstance(model, FSDP):
            self.config_module = model.module
        else:
            self.config_module = model

    def step(self, global_step: int) -> float:
        if self.total_train_steps == 0:
            return self.end_weight

        progress = min(1.0, global_step / self.total_train_steps)
        weight = self.start_weight + progress * (self.end_weight - self.start_weight)
        
        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = weight
            
        return weight


class NegativeStartAsymmetricPonderScheduler:
    def __init__(
        self, 
        model: torch.nn.Module, 
        steps_per_cycle: int, 
        base_amplitude: float = 0.05, 
        negative_damping: float = 0.2
    ):
        """
        Inverse Asymmetric Scheduler:
        Starts at the negative trough (Reward) instead of the positive peak (Penalty).
        """
        self.model = model
        self.steps_per_cycle = steps_per_cycle
        self.base_amplitude = base_amplitude
        self.negative_damping = negative_damping
        
        if isinstance(model, FSDP):
             self.config_module = model.module
        else:
             self.config_module = model

    def step(self, global_step: int) -> float:
        # Standard Cosine: oscillates between +1.0 and -1.0
        cos_val = math.cos(2 * math.pi * global_step / self.steps_per_cycle)
        
        # Invert direction: Multiply by -1.0
        weight = -1.0 * self.base_amplitude * cos_val
        
        # Apply Asymmetry (Damping the negative reward)
        if weight < 0:
            weight = weight * self.negative_damping
            
        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = weight
            
        return weight


class AsymmetricPonderScheduler:
    def __init__(
        self, 
        model: torch.nn.Module, 
        steps_per_cycle: int, 
        base_amplitude: float = 0.05, 
        negative_damping: float = 0.2
    ):
        self.model = model
        self.steps_per_cycle = steps_per_cycle
        self.base_amplitude = base_amplitude
        self.negative_damping = negative_damping
        
        if isinstance(model, FSDP):
             self.config_module = model.module
        else:
             self.config_module = model

    def step(self, global_step: int) -> float:
        cos_val = math.cos(2 * math.pi * global_step / self.steps_per_cycle)
        weight = self.base_amplitude * cos_val
        
        # Apply Asymmetry (The "Cheating" Mitigation)
        if weight < 0:
            weight = weight * self.negative_damping
            
        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = weight
            
        return weight
    

class CycleThenConstantPonderScheduler:
    def __init__(
        self,
        model: torch.nn.Module,
        steps_per_cycle: int,
        base_amplitude: float = 0.05,
        negative_damping: float = 0.2,
        cycle_steps: int = 1000,
        constant_value: float = 0.0,
    ):
        self.model = model
        self.steps_per_cycle = steps_per_cycle
        self.base_amplitude = base_amplitude
        self.negative_damping = negative_damping
        self.cycle_steps = cycle_steps
        self.constant_value = constant_value
        
        if isinstance(model, FSDP):
            self.config_module = model.module
        else:
            self.config_module = model

    def step(self, global_step: int) -> float:
        if global_step < self.cycle_steps:
            cos_val = math.cos(2 * math.pi * global_step / self.steps_per_cycle)
            weight = self.base_amplitude * cos_val
            if weight < 0:
                weight = weight * self.negative_damping
        else:
            weight = self.constant_value
            
        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = weight
            
        return weight    


class DampedOscillationPonderScheduler:
    def __init__(
        self,
        model: torch.nn.Module,
        total_train_steps: int,
        steps_per_cycle: int,
        amplitude: float = 0.2,
    ):
        """
        Symmetric damped oscillation around 0.
        Starts at -amplitude, oscillates with decreasing amplitude, ends at 0.
        """
        self.model = model
        self.total_train_steps = total_train_steps
        self.steps_per_cycle = steps_per_cycle
        self.amplitude = amplitude
        self.config_module = model.module if isinstance(model, FSDP) else model

    def step(self, global_step: int) -> float:
        progress = min(1.0, global_step / self.total_train_steps)
        current_amplitude = self.amplitude * (1.0 - progress)
        weight = -current_amplitude * math.cos(2 * math.pi * global_step / self.steps_per_cycle)

        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = weight

        return weight


class DecreasingFrequencyPonderScheduler:
    def __init__(
        self,
        model: torch.nn.Module,
        initial_steps_per_cycle: int,
        frequency_decay_power: float = 0.75,
        base_amplitude: float = 0.05,
        negative_damping: float = 0.2,
    ):
        """
        Decreasing Frequency Scheduler (Chirp).
        The cycles start fast and gradually become longer (slower frequency).
        """
        self.model = model
        self.initial_steps_per_cycle = initial_steps_per_cycle
        self.frequency_decay_power = frequency_decay_power
        self.base_amplitude = base_amplitude
        self.negative_damping = negative_damping

        if isinstance(model, FSDP):
            self.config_module = model.module
        else:
            self.config_module = model

    def step(self, global_step: int) -> float:
        if global_step == 0:
            weight = self.base_amplitude
        else:
            phase = (global_step / self.initial_steps_per_cycle) ** self.frequency_decay_power
            cos_val = math.cos(2 * math.pi * phase)
            weight = self.base_amplitude * cos_val

        if weight < 0:
            weight = weight * self.negative_damping

        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = weight

        return weight

class ThroughputAggregationKeys(Enum):
    NUM_SAMPLES = "NUM_SAMPLES"
    FORWARD_BACKWARD_TIME = "FORWARD_BACKWARD_TIME"


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
        mfu_calculator: Optional[MFUCalculatorABC] = None,
    ) -> None:
        """
        Initializes the Trainer object.
        """
        self.global_rank = global_rank
        if device_mesh is not None:
            self.dp_degree = get_parallel_degree(
                device_mesh, [ParallelismDegrees.DP_REPLICATE, ParallelismDegrees.DP_SHARD]
            )
            self.pp_degree = get_parallel_degree(device_mesh, [ParallelismDegrees.PP])
        else:
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
        self.mfu_calculator = mfu_calculator

    @staticmethod
    def _get_num_train_steps_done(micro_batch_id: int, gradient_acc_steps: int) -> int:
        return (micro_batch_id + 1) // gradient_acc_steps

    def _train_batch(
        self,
        batch: DatasetBatch,
        model: FSDP,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        micro_batch_id: int,
        scheduled_pipeline: Optional[Pipeline] = None,
    ) -> tuple[bool, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Conducts a training step on batch of data.
        """
        if scheduled_pipeline is not None:
            pp_schedule = scheduled_pipeline.pp_schedule
            targets, losses = (
                (batch.targets[loss_fun.target_key].contiguous(), [])
                if scheduled_pipeline.is_last_pp_stage
                else (None, None)
            )

            if scheduled_pipeline.is_first_pp_stage:
                pp_schedule.step(batch.samples[model.sample_key].contiguous(), target=targets, losses=losses)
            else:
                pp_schedule.step(target=targets, losses=losses)
            loss = torch.mean(torch.stack(losses)).to(losses[0].device) if scheduled_pipeline.is_last_pp_stage else None
        else:
            # else continue with loss calculation
            result_batch = model_predict_batch(model=model, batch=batch)
            loss = loss_fun(result_batch)
            (loss / self.gradient_acc_steps).backward()

        if (micro_batch_id + 1) % self.gradient_acc_steps == 0:
            gradient_norm_score = self.gradient_clipper.clip_gradients()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step_performed = True
        else:
            step_performed = False
            gradient_norm_score = None

        num_train_steps_done = Trainer._get_num_train_steps_done(
            micro_batch_id=micro_batch_id, gradient_acc_steps=self.gradient_acc_steps
        )
        return step_performed, num_train_steps_done, loss, gradient_norm_score

    def train(
        self,
        app_state: AppState,
        train_loader: LLMDataLoader,
        loss_fun: Loss,
        training_log_interval_in_steps: int,
        evaluation_callback: Callable[[TrainingProgress], None],
        checkpointing_callback: Callable[[TrainingProgress], None],
        scheduled_pipeline: Pipeline | None = None,
    ):
        """
        Trains the model.
        """
        model = app_state.model
        optimizer = app_state.optimizer
        lr_scheduler = app_state.lr_scheduler
        model.train()

        # ==============================================================================
        # SCHEDULER SELECTION (Dynamic from Config)
        # ==============================================================================
        
        # 1. Access the underlying model config safely
        underlying_model = model.module if isinstance(model, FSDP) else model
        
        # 2. Get scheduler type string from config (default to "constant")
        scheduler_type = "constant"
        config_weight = 0.0
        if hasattr(underlying_model, 'adaptive_config') and underlying_model.adaptive_config is not None:
             scheduler_type = underlying_model.adaptive_config.scheduler_type
             config_weight = getattr(underlying_model.adaptive_config, "ponder_penalty_weight", 0.0)

        # 3. Select Scheduler
        if scheduler_type == "constant":
            ponder_scheduler = ConstantPonderScheduler(
                model=model,
                constant_value=config_weight,
            )
        elif scheduler_type == "random":
            ponder_scheduler = RandomPonderScheduler(
                model=model,
                min_weight=1, 
                max_weight=1,
                seed=42 + self.global_rank 
            )
        elif scheduler_type == "linear":
            ponder_scheduler = SimpleLinearScheduler(
                model=model,
                total_train_steps=self.num_target_steps,
                start_weight=-0.01,
                end_weight=0.01
            )
        elif scheduler_type == "negative_asymmetric":
            ponder_scheduler = NegativeStartAsymmetricPonderScheduler(
                model=model,
                steps_per_cycle=10, 
                base_amplitude=0.3, 
                negative_damping=0.2
            )
        elif scheduler_type == "decreasing":
            ponder_scheduler = DecreasingFrequencyPonderScheduler(
                model=model,
                initial_steps_per_cycle=10,
                frequency_decay_power=0.99,
                base_amplitude=0.3,
                negative_damping=0.2
            )
        elif scheduler_type == "constant_cycle":
            ponder_scheduler = CycleThenConstantPonderScheduler(
                model=model,
                steps_per_cycle=10, 
                base_amplitude=0.3, 
                negative_damping=0.2,
                cycle_steps=1000,
                constant_value=0.01
            )
        elif scheduler_type == "damped_oscillation":
            ponder_scheduler = DampedOscillationPonderScheduler(
                model=model,
                steps_per_cycle=10,
                total_train_steps=self.num_target_steps,
                amplitude=0.2,
            )
        elif scheduler_type == "linear_decay":
            ponder_scheduler = LinearDecayPonderScheduler(
                model=model,
                total_train_steps=self.num_target_steps,
                start_weight=0.1,
                end_weight=0.01
            )
        elif scheduler_type == "asymmetric":
            # Default Asymmetric Cosine
            ponder_scheduler = AsymmetricPonderScheduler(
                model=model,
                steps_per_cycle=10, 
                base_amplitude=0.3, 
                negative_damping=config_weight,
            )
            
        current_ponder_weight = 0.0
        # ==============================================================================

        cumulated_losses = self._reset_tracked_losses()
        
        # Track loss components separately
        cumulated_ce_loss = 0.0
        cumulated_ponder_loss = 0.0
        cumulated_ponder_cost_unweighted = 0.0
        cumulated_expected_steps = 0.0
        num_loss_accumulations = 0
        cumulated_normalized_steps = 0.0 
        
        cumulated_per_layer_costs: Optional[torch.Tensor] = None 
        cumulated_per_layer_sims: Optional[torch.Tensor] = None

        # throughput
        thoughput_aggregator = Aggregator[ThroughputAggregationKeys]()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # batch loop
        batch: DatasetBatch
        forward_backward_time_recorder = TimeRecorder()
        forward_backward_time_recorder.start()
        gradient_norm_scores = []

        # run evaluation callback and checkpointing callback before the first optimizer step
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
        
        for _, (micro_batch_id, batch) in zip(range(num_batches_todo), enumerate(train_loader)):
            
            # --- Update Ponder Weight ---
            current_ponder_weight = ponder_scheduler.step(training_progress.num_seen_steps_total)
            # ----------------------------

            # Train single batch
            (
                step_performed,
                num_train_steps_done,
                batch_loss,
                gradient_norm_score,
            ) = self._train_batch(
                batch=batch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                loss_fun=loss_fun,
                micro_batch_id=micro_batch_id,
                scheduled_pipeline=scheduled_pipeline,
            )
            forward_backward_time_recorder.stop()
            training_progress.num_seen_steps_current_run = num_train_steps_done
            training_progress.num_seen_tokens_current_run = self.global_num_tokens_per_train_step * num_train_steps_done

            # Accumulate loss metrics
            if batch_loss is not None:
                # Save the batch loss
                cumulated_losses[0] += batch_loss.item()
                cumulated_losses[-1] += 1  # number of local batches
                
                # Accumulate loss components if available
                if hasattr(loss_fun, 'get_loss_components'):
                    components = loss_fun.get_loss_components()
                    cumulated_ce_loss += components["ce_loss"].item()
                    cumulated_ponder_loss += components["ponder_loss"].item()
                    cumulated_ponder_cost_unweighted += components["ponder_cost_unweighted"].item()
                    cumulated_expected_steps += components["expected_steps"].item()
                    cumulated_normalized_steps += components.get("normalized_steps", torch.tensor(0.0)).item()

                    layer_costs = components.get("per_layer_ponder_costs", None)
                    if layer_costs is not None and layer_costs.numel() > 0:
                        if cumulated_per_layer_costs is None:
                            cumulated_per_layer_costs = torch.zeros_like(layer_costs)
                        cumulated_per_layer_costs += layer_costs

                    layer_sims = components.get("per_layer_cos_sims", None)
                    if layer_sims is not None and layer_sims.numel() > 0:
                        if cumulated_per_layer_sims is None:
                            cumulated_per_layer_sims = torch.zeros_like(layer_sims)
                        cumulated_per_layer_sims += layer_sims

                    num_loss_accumulations += 1

            # gradient norm is already synced across all ranks
            if gradient_norm_score is not None:
                gradient_norm_scores.append(gradient_norm_score.item())

            batch_length_tensor = torch.tensor(len(batch)).to(device)
            thoughput_aggregator.add_value(key=ThroughputAggregationKeys.NUM_SAMPLES, value=batch_length_tensor)

            self._publish_progress(
                progress_publisher=self.progress_publisher,
                num_train_steps_done=training_progress.num_seen_steps_total,
                dataloader_tag=train_loader.dataloader_tag,
            )
            
            # Check if model performance should be logged
            if training_progress.num_seen_steps_total % training_log_interval_in_steps == 0 and step_performed:
                forward_backward_time = torch.tensor(forward_backward_time_recorder.delta_t).to(device)
                forward_backward_time_recorder.reset()

                thoughput_aggregator.add_value(
                    key=ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, value=forward_backward_time
                )
                
                synced_num_samples = thoughput_aggregator.get_all_reduced_value(
                    ThroughputAggregationKeys.NUM_SAMPLES
                ) / (dist.get_world_size() / self.dp_degree)
                synced_forward_backward_time = thoughput_aggregator.get_all_reduced_value(
                    ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, reduce_operation=dist.ReduceOp.MAX
                )
                synced_num_samples_per_second = synced_num_samples / synced_forward_backward_time
                
                # Reduce total loss
                cumulated_losses[1] = batch_loss.item() if batch_loss is not None else 0.0

                reduced_losses = Reducer.reduce(
                    tensor=cumulated_losses,
                    operation=dist.ReduceOp.SUM,
                    post_processing_fun=lambda t: torch.stack(
                        [t[0] / t[-1], t[1] / dist.get_world_size() * self.pp_degree]
                    ),
                )

                train_loss_avg, train_loss_last_batch = (
                    reduced_losses[0],
                    reduced_losses[1],
                )
                
                # Compute and sync loss components
                if num_loss_accumulations > 0:
                    # 1. Prepare Scalars
                    avg_ce_loss = cumulated_ce_loss / num_loss_accumulations
                    avg_ponder_loss = cumulated_ponder_loss / num_loss_accumulations
                    avg_ponder_cost = cumulated_ponder_cost_unweighted / num_loss_accumulations
                    avg_expected_steps = cumulated_expected_steps / num_loss_accumulations
                    avg_normalized_steps = cumulated_normalized_steps / num_loss_accumulations
                    
                    scalars = torch.tensor(
                        [avg_ce_loss, avg_ponder_loss, avg_ponder_cost, avg_expected_steps, avg_normalized_steps],
                        device=device
                    )

                    # 2. Prepare Layer Vectors
                    tensors_to_sync = [scalars]
                    num_cost_layers = 0 
                    num_sim_layers = 0

                    if cumulated_per_layer_costs is not None:
                        avg_layer_costs = cumulated_per_layer_costs / num_loss_accumulations
                        tensors_to_sync.append(avg_layer_costs)
                        num_cost_layers = len(avg_layer_costs)

                    if cumulated_per_layer_sims is not None:
                        avg_layer_sims = cumulated_per_layer_sims / num_loss_accumulations
                        tensors_to_sync.append(avg_layer_sims)
                        num_sim_layers = len(avg_layer_sims)

                    combined_tensor = torch.cat(tensors_to_sync)
                    num_scalars = len(scalars)
                    
                    # 3. Sync everything at once
                    synced_tensor = Reducer.reduce(
                        tensor=combined_tensor,
                        operation=dist.ReduceOp.SUM,
                        post_processing_fun=lambda t: t / (dist.get_world_size() / self.pp_degree)
                    )
                    
                    # 4. Unpack Scalars
                    train_ce_loss_avg = synced_tensor[0]
                    train_ponder_loss_avg = synced_tensor[1]
                    train_ponder_cost_avg = synced_tensor[2]
                    train_expected_steps_avg = synced_tensor[3]
                    train_normalized_steps_avg = synced_tensor[4]
                    
                    # 5. Unpack Layer Vectors
                    current_idx = num_scalars
                    
                    if num_cost_layers > 0:
                        synced_layer_costs = synced_tensor[current_idx : current_idx + num_cost_layers]
                        current_idx += num_cost_layers
                    else:
                        synced_layer_costs = []

                    if num_sim_layers > 0:
                        synced_layer_sims = synced_tensor[current_idx : current_idx + num_sim_layers]
                        current_idx += num_sim_layers
                    else:
                        synced_layer_sims = []

                else:
                    train_ce_loss_avg = torch.tensor(0.0)
                    train_ponder_loss_avg = torch.tensor(0.0)
                    train_ponder_cost_avg = torch.tensor(0.0)
                    train_expected_steps_avg = torch.tensor(0.0)
                    train_normalized_steps_avg = torch.tensor(0.0)
                    synced_layer_costs = []
                    synced_layer_sims = []
                
                # ==============================================================================
                # LOGGING / METRICS CONSTRUCTION
                # ==============================================================================
                
                losses = {
                    "loss/avg": ResultItem(train_loss_avg, decimal_places=2),
                    "loss/last": ResultItem(train_loss_last_batch, decimal_places=2),
                    "loss/ce_avg": ResultItem(train_ce_loss_avg, decimal_places=2),
                    "ponder/loss_avg": ResultItem(train_ponder_loss_avg, decimal_places=5),
                }

                consumed_tokens = torch.tensor(training_progress.num_seen_tokens_total)
                
                metrics = {
                    "progress/consumed_tokens": ResultItem(consumed_tokens, 0),
                    "grads/norm_avg": ResultItem(torch.mean(torch.Tensor(gradient_norm_scores)), 2),
                    "grads/norm_last": ResultItem(torch.tensor(gradient_norm_scores[-1]), 2),
                    "adaptive/expected_steps_avg": ResultItem(train_expected_steps_avg, 2),
                    "adaptive/ponder_cost_avg": ResultItem(train_ponder_cost_avg, 2),
                    "adaptive/normalized_steps_avg": ResultItem(train_normalized_steps_avg, 3),
                    "adaptive/ponder_weight": ResultItem(torch.tensor(current_ponder_weight), 4),
                }

                # --- 1. Average Layer Stats (The Overview) ---
                if len(synced_layer_costs) > 0:
                    metrics["adaptive/avg_layer_cost"] = ResultItem(synced_layer_costs.mean(), 2)
                    metrics["adaptive/avg_layer_cos_sim"] = ResultItem(synced_layer_sims.mean(), 4)

                # --- 2. Detailed Layer Stats (The specific dashboard) ---
                for i, cost_val in enumerate(synced_layer_costs):
                    metrics[f"layers/{i}/ponder_cost"] = ResultItem(cost_val, 2)

                for i, sim_val in enumerate(synced_layer_sims):
                    metrics[f"layers/{i}/cos_sim"] = ResultItem(sim_val, 4)

                # --- 3. Complex Metrics (Vectors) with Averages ---
                if hasattr(loss_fun, 'get_loss_components'):
                    components = loss_fun.get_loss_components()
                    
                    # --- Loop Scales ---
                    batch_scales = components.get("loop_scales")
                    if batch_scales is not None and batch_scales.numel() > 0:
                        scales_cpu = batch_scales.float().cpu() # Shape: [n_layers, n_loops]
                        
                        # Overview: Average per step across all layers
                        avg_scales = torch.mean(scales_cpu, dim=0) 
                        for j, val in enumerate(avg_scales):
                            metrics[f"adaptive/avg_loop_scale_step_{j}"] = ResultItem(val, 4)

                        # Detailed: Per layer
                        for i, layer_scales in enumerate(scales_cpu):
                            for j, val in enumerate(layer_scales):
                                metrics[f"layers/{i}/loop_scale_{j}"] = ResultItem(val, 4)

                    # --- Halt Probs ---
                    batch_halt_probs = components.get("halt_probs")
                    if batch_halt_probs is not None and batch_halt_probs.numel() > 0:
                        probs_cpu = batch_halt_probs.float().cpu() # Shape: [n_layers, n_loops]

                        # Overview: Average per step across all layers
                        avg_probs = torch.mean(probs_cpu, dim=0)
                        for j, val in enumerate(avg_probs):
                            metrics[f"adaptive/avg_halt_prob_step_{j}"] = ResultItem(val, 4)

                        # Detailed: Per layer
                        for i, layer_probs in enumerate(probs_cpu):
                            for j, val in enumerate(layer_probs):
                                metrics[f"layers/{i}/halt_prob_{j}"] = ResultItem(val, 4)

                    # --- Local Mem Scales ---
                    batch_local_mem_scales = components.get("local_mem_scales")
                    if batch_local_mem_scales is not None and batch_local_mem_scales.numel() > 0:
                        scales_cpu = batch_local_mem_scales.float().cpu()
                        metrics["adaptive/avg_local_mem_scale"] = ResultItem(scales_cpu.mean(), 4)
                        for layer_idx, val in enumerate(scales_cpu):
                            metrics[f"layers/{layer_idx}/local_mem_scale"] = ResultItem(val, 4)

                    # --- Global Mem Scales ---
                    batch_global_mem_scales = components.get("global_mem_scales")
                    if batch_global_mem_scales is not None and batch_global_mem_scales.numel() > 0:
                        scales_cpu = batch_global_mem_scales.float().cpu()
                        metrics["adaptive/avg_global_mem_scale"] = ResultItem(scales_cpu.mean(), 4)
                        for layer_idx, val in enumerate(scales_cpu):
                            metrics[f"layers/{layer_idx}/global_mem_scale"] = ResultItem(val, 4)

                gradient_norm_scores = []
                mfu_score = torch.tensor(-1.0)
                if self.mfu_calculator is not None:
                    mfu_score = self.mfu_calculator.compute(num_samples_per_second=synced_num_samples_per_second)

                # Collect peak memory depending on device type
                if device.type == "cuda":
                    peak_memory_MB = torch.cuda.max_memory_allocated(device) / 1024**2  # in MB
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
                    # TODO: hardcoded metric key
                    throughput_metrics={
                        "train samples/s": ResultItem(synced_num_samples_per_second, 1),
                        "train mfu (16-bit)": ResultItem(mfu_score, 2),
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
                thoughput_aggregator.remove_keys()

                cumulated_losses = self._reset_tracked_losses()
                
                # Reset loss component accumulators
                cumulated_ce_loss = 0.0
                cumulated_ponder_loss = 0.0
                cumulated_ponder_cost_unweighted = 0.0
                cumulated_expected_steps = 0.0
                cumulated_normalized_steps = 0.0
                cumulated_per_layer_costs = None
                cumulated_per_layer_sims = None
                num_loss_accumulations = 0
                
            if step_performed:
                evaluation_callback(num_train_steps_done=training_progress.num_seen_steps_total)
                checkpointing_callback(training_progress=training_progress)
            
            forward_backward_time_recorder.start()


    def _reset_tracked_losses(self):
        # Initializes and returns a tensor representing the cumulated loss and gradient norm.
        cumulated_loss_and_gradient_norm = torch.zeros(3)
        if torch.cuda.is_available():
            cumulated_loss_and_gradient_norm = cumulated_loss_and_gradient_norm.to(torch.device("cuda"))
        else:
            cumulated_loss_and_gradient_norm = cumulated_loss_and_gradient_norm.to("cpu")
        return cumulated_loss_and_gradient_norm

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
            payload=evaluation_result, message_type=MessageTypes.EVALUATION_RESULT
        )