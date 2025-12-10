from datetime import datetime
from enum import Enum
from typing import Callable, Optional

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

import random

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


class AsymmetricPonderScheduler:
    def __init__(
        self, 
        model: torch.nn.Module, 
        steps_per_cycle: int, 
        base_amplitude: float = 0.05, 
        negative_damping: float = 0.2
    ):
        """
        Args:
            model: The model containing the adaptive_config.
            steps_per_cycle: Length of one full wave (e.g., 1000 steps).
            base_amplitude: The max positive penalty (e.g., 0.05).
            negative_damping: Multiplier when wave is negative. 
                              0.2 means the max reward is only 0.05 * 0.2 = -0.01.
        """
        self.model = model
        self.steps_per_cycle = steps_per_cycle
        self.base_amplitude = base_amplitude
        self.negative_damping = negative_damping
        
        # Unwrap FSDP if necessary to get to the config
        if isinstance(model, FSDP):
             self.config_module = model.module
        else:
             self.config_module = model

    def step(self, global_step: int) -> float:
        # 1. Standard Cosine: oscillates between +1.0 and -1.0
        cos_val = math.cos(2 * math.pi * global_step / self.steps_per_cycle)
        
        # 2. Calculate raw weight
        weight = self.base_amplitude * cos_val
        
        # 3. Apply Asymmetry (The "Cheating" Mitigation)
        if weight < 0:
            weight = weight * self.negative_damping
            
        # Update model config in-place
        # We check hasattr because the model might be wrapped or not have the config initialized yet
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
        start_amplitude: float = 0.2,
        final_target_weight: float = 0.01,
        negative_damping: float = 0.2,
    ):
        """
        Damped Oscillation:
        Oscillates around 'final_target_weight'. The amplitude of the oscillation
        linearly decreases from 'start_amplitude' to 0 over 'total_train_steps'.
        
        Formula:
        W(t) = target + (start_amp * (1 - t/T)) * cos(2pi * t / cycle)
        """
        self.model = model
        self.total_train_steps = total_train_steps
        self.steps_per_cycle = steps_per_cycle
        self.start_amplitude = start_amplitude
        self.final_target_weight = final_target_weight
        self.negative_damping = negative_damping

        # Unwrap FSDP if necessary
        if isinstance(model, FSDP):
            self.config_module = model.module
        else:
            self.config_module = model

    def step(self, global_step: int) -> float:
        # 1. Calculate progress (0.0 to 1.0)
        progress = min(1.0, global_step / self.total_train_steps)
        
        # 2. Calculate current Amplitude (Linear Decay)
        # At step 0: amplitude = start_amplitude
        # At last step: amplitude = 0
        current_amplitude = self.start_amplitude * (1.0 - progress)
        
        # 3. Calculate Oscillation
        cos_val = math.cos(2 * math.pi * global_step / self.steps_per_cycle)
        
        # 4. Combine: Center it on the target weight
        raw_weight = self.final_target_weight + (current_amplitude * cos_val)
        
        # 5. Apply Asymmetry (Mitigation for "Cheating")
        # If the oscillation swings below 0, we dampen it so the reward isn't too huge.
        if raw_weight < 0:
            raw_weight = raw_weight * self.negative_damping

        # Update model config
        if hasattr(self.config_module, 'adaptive_config') and self.config_module.adaptive_config is not None:
            self.config_module.adaptive_config.ponder_penalty_weight = raw_weight

        return raw_weight


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

        Args:
            model: The model containing the adaptive_config.
            initial_steps_per_cycle: The length (in steps) of the very first cycle.
            frequency_decay_power: A float between 0.0 and 1.0. 
                                   - 1.0 = Constant frequency (standard cosine).
                                   - 0.5 = Cycle length grows linearly over time.
                                   - Lower values cause the frequency to drop faster.
            base_amplitude: The max positive penalty.
            negative_damping: Multiplier when wave is negative.
        """
        self.model = model
        self.initial_steps_per_cycle = initial_steps_per_cycle
        self.frequency_decay_power = frequency_decay_power
        self.base_amplitude = base_amplitude
        self.negative_damping = negative_damping

        # Unwrap FSDP if necessary
        if isinstance(model, FSDP):
            self.config_module = model.module
        else:
            self.config_module = model

    def step(self, global_step: int) -> float:
        # Avoid division by zero or log errors at step 0
        if global_step == 0:
            weight = self.base_amplitude
        else:
            # 1. Calculate the 'stretched' phase
            # We normalize step by initial_period, then apply the power law.
            # This makes the argument grow slower than 't', stretching the wave.
            phase = (global_step / self.initial_steps_per_cycle) ** self.frequency_decay_power
            
            # 2. Compute Cosine
            cos_val = math.cos(2 * math.pi * phase)
            weight = self.base_amplitude * cos_val

        # 3. Apply Asymmetry (Mitigation)
        if weight < 0:
            weight = weight * self.negative_damping

        # Update model config in-place
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

        Args:
            global_rank (int): The global rank.
            progress_publisher (MessagePublisher[ProgressUpdate]): Progress publisher.
            evaluation_result_publisher (MessagePublisher[EvaluationResultBatch]): Evaluation result publisher.
            gradient_acc_steps (int): Gradient accumulation steps.
            global_num_tokens_per_train_step (int): Global number of tokens per train step.
            dp_degree (int): Data parallelism degree.
            pp_degree (int): Pipeline parallelism degree.
            num_seen_train_steps (int): Number of seen train steps.
            global_num_seen_tokens (int): Global number of seen tokens.
            num_target_steps (int): Number of target steps.
            num_target_tokens (int): Number of target tokens.
            gradient_clipper (GradientClipperIF): Gradient clipper.
            mfu_calculator (Optional[MFUCalculatorABC]): MFU calculator.

        Returns:
            None
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
        """
        Calculates the number of training steps done based on the micro batch ID and gradient accumulation steps.

        Args:
            micro_batch_id (int): The ID of the current micro batch.
            gradient_acc_steps (int): The number of gradient accumulation steps.

        Returns:
            int: The number of training steps done.
        """
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

        Args:
            batch (DatasetBatch): The input batch of data.
            model (FSDP): The model to train.
            optimizer (Optimizer): The optimizer used for training.
            scheduler (LRScheduler): The learning rate scheduler.
            loss_fun (Loss): The loss function used for training.
            micro_batch_id (int): The ID of the micro batch.
            scheduled_pipeline (Optional[Pipeline], optional): In case of pipeline parallelism, this is used to
                operate the model. Defaults to None.

        Returns:
            tuple[bool, int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                A tuple containing the following:
                    - step_performed (bool): Indicates whether a training step was performed.
                    - num_train_steps_done (int): The number of training steps done.
                    - loss (Optional[torch.Tensor]): The computed loss.
                        None, if a non-last stage was processes in pipeline parallelism.
                    - gradient_norm_score (Optional[torch.Tensor]): The gradient norm score,
                        if a training step was performed otherwise return None.
        """
        if scheduled_pipeline is not None:
            pp_schedule = scheduled_pipeline.pp_schedule
            # Pipeline Parallel forward / backward inside step() call
            # with self.train_context(optional_context_parallel_ctx):
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

        Args:
            app_state (AppState): The application state containing the model, optimizer and lr scheduler.
            train_loader (LLMDataLoader): The data loader containing the training data.
            loss_fun (Loss): The loss function used for training.
            training_log_interval_in_steps (int): The interval at which training progress is logged.
            evaluation_callback (Callable[[TrainingProgress], None]): A callback function for evaluation.
            checkpointing_callback (Callable[[TrainingProgress], None]): A callback function for checkpointing.
            scheduled_pipeline (Pipeline | None, optional): In case of pipeline parallelism, this is used to
                operate the model. Defaults to None.

        Returns:
            None
        """
        model = app_state.model
        optimizer = app_state.optimizer
        lr_scheduler = app_state.lr_scheduler
        model.train()

        # ==============================================================================
        # SCHEDULER SELECTION
        scheduler_type = "asymmetric"  # Options: "random", "constant_cycle", "decreasing", "asymmetric"
        
        if scheduler_type == "random":
            ponder_scheduler = RandomPonderScheduler(
                model=model,
                min_weight=-0.1, 
                max_weight=0.2,
                seed=42 + self.global_rank 
            )
        elif scheduler_type == "decreasing":
            # STARTS fast (short cycles) and slows down.
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
                total_train_steps=self.num_target_steps,
                steps_per_cycle=100,
                start_amplitude=0.2,
                final_target_weight=0.01,
                negative_damping=0.2
            )
        elif scheduler_type == "linear_decay":
            ponder_scheduler = LinearDecayPonderScheduler(
                model=model,
                total_train_steps=self.num_target_steps,
                start_weight=0.1,
                end_weight=0.01
            )
        else:
            # Default Asymmetric Cosine
            ponder_scheduler = AsymmetricPonderScheduler(
                model=model,
                steps_per_cycle=10, 
                base_amplitude=0.3, 
                negative_damping=0.2
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
        cumulated_step_gate = 0.0
        cumulated_per_layer_costs: Optional[torch.Tensor] = None 

        # throughput
        thoughput_aggregator = Aggregator[ThroughputAggregationKeys]()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # batch loop
        batch: DatasetBatch
        # TODO: why do we need a barrier here?
        # dist.barrier()
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
        # Because we might resume training, we add the starting batch id of the data loader
        for _, (micro_batch_id, batch) in zip(range(num_batches_todo), enumerate(train_loader)):
            
            # --- Update Ponder Weight ---
            # Update the weight based on total steps seen
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

            # The batch_loss might be None if we use pipeline parallelism and are not the last stage.
            if batch_loss is not None:
                # Save the batch loss
                cumulated_losses[0] += batch_loss.item()
                # This works, because we always drop the last batch in case it has less samples than the batch size
                cumulated_losses[-1] += 1  # number of local batches
                
                # Accumulate loss components if available
                if hasattr(loss_fun, 'get_loss_components'):
                    components = loss_fun.get_loss_components()
                    cumulated_ce_loss += components["ce_loss"].item()
                    cumulated_ponder_loss += components["ponder_loss"].item()
                    cumulated_ponder_cost_unweighted += components["ponder_cost_unweighted"].item()
                    cumulated_expected_steps += components["expected_steps"].item()

                    cumulated_normalized_steps += components.get("normalized_steps", torch.tensor(0.0)).item()
                    cumulated_step_gate += components.get("step_gate_mean", torch.tensor(0.0)).item()
                    layer_costs = components.get("per_layer_ponder_costs", None)
                    if layer_costs is not None and layer_costs.numel() > 0:
                        if cumulated_per_layer_costs is None:
                            cumulated_per_layer_costs = torch.zeros_like(layer_costs)
                        cumulated_per_layer_costs += layer_costs

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
                # we only want to sync the num samples across data parallel ranks
                # so we divide the world size by the dp degree
                synced_num_samples = thoughput_aggregator.get_all_reduced_value(
                    ThroughputAggregationKeys.NUM_SAMPLES
                ) / (dist.get_world_size() / self.dp_degree)
                synced_forward_backward_time = thoughput_aggregator.get_all_reduced_value(
                    ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, reduce_operation=dist.ReduceOp.MAX
                )
                synced_num_samples_per_second = synced_num_samples / synced_forward_backward_time
                # TODO: insert reducer from outside so Trainer is independent of FSDP
                # add the loss and gradient norm for the LAST batch

                cumulated_losses[1] = batch_loss.item() if batch_loss is not None else 0.0

                reduced_losses = Reducer.reduce(
                    tensor=cumulated_losses,
                    operation=dist.ReduceOp.SUM,
                    # 1.) summed batch loss / (num batches * (world size / dp_degree))
                    # 2.) last batch loss / (world size / pp_degree)
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
                    avg_step_gate = cumulated_step_gate / num_loss_accumulations
                    
                    scalars = torch.tensor(
                        [avg_ce_loss, avg_ponder_loss, avg_ponder_cost, avg_expected_steps, avg_normalized_steps, avg_step_gate],
                        device=device
                    )

                    # 2. Prepare Layer Vector
                    if cumulated_per_layer_costs is not None:
                        avg_layer_costs = cumulated_per_layer_costs / num_loss_accumulations
                        # Concatenate scalars and layer vector for a single ReduceOp
                        combined_tensor = torch.cat([scalars, avg_layer_costs])
                        num_scalars = len(scalars)
                    else:
                        combined_tensor = scalars
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
                    train_step_gate_avg = synced_tensor[5]
                    
                    # 5. Unpack Layer Vector (if it exists)
                    synced_layer_costs = synced_tensor[num_scalars:] if len(synced_tensor) > num_scalars else []

                else:
                    train_ce_loss_avg = torch.tensor(0.0)
                    train_ponder_loss_avg = torch.tensor(0.0)
                    train_ponder_cost_avg = torch.tensor(0.0)
                    train_expected_steps_avg = torch.tensor(0.0)
                    train_normalized_steps_avg = torch.tensor(0.0)
                    train_step_gate_avg = torch.tensor(0.0)
                    synced_layer_costs = torch.tensor([])
                
                losses = {
                    "train/loss_avg": ResultItem(train_loss_avg, decimal_places=2),
                    "train/loss_last": ResultItem(train_loss_last_batch, decimal_places=2),
                    "train/ce_loss_avg": ResultItem(train_ce_loss_avg, decimal_places=2),
                    "train/ponder_loss_avg": ResultItem(train_ponder_loss_avg, decimal_places=5),
                }

                consumed_tokens = torch.tensor(training_progress.num_seen_tokens_total)
                metrics = {
                    "train/consumed_tokens": ResultItem(consumed_tokens, 0),
                    "train/grad_norm_avg": ResultItem(torch.mean(torch.Tensor(gradient_norm_scores)), 2),
                    "train/grad_norm_last": ResultItem(torch.tensor(gradient_norm_scores[-1]), 2),
                    "train/expected_steps_avg": ResultItem(train_expected_steps_avg, 2),
                    "train/ponder_cost_avg": ResultItem(train_ponder_cost_avg, 2),
                    "train/normalized_steps_avg": ResultItem(train_normalized_steps_avg, 3),
                    "train/step_gate_avg": ResultItem(train_step_gate_avg, 4),
                    # --- [Log current weight to check scheduler] ---
                    "train/ponder_weight": ResultItem(torch.tensor(current_ponder_weight), 4),
                }

                for i, cost_val in enumerate(synced_layer_costs):
                    metrics[f"train/layer_{i}/ponder_cost"] = ResultItem(cost_val, 2)

                gradient_norm_scores = []
                mfu_score = torch.tensor(-1.0)
                if self.mfu_calculator is not None:
                    mfu_score = self.mfu_calculator.compute(num_samples_per_second=synced_num_samples_per_second)

                # Collect peak memory depending on device type. On CPU we fall back to RSS (if available) or -1.
                if device.type == "cuda":
                    peak_memory_MB = torch.cuda.max_memory_allocated(device) / 1024**2  # in MB
                    torch.cuda.reset_peak_memory_stats(device)
                else:
                    # ru_maxrss is in kilobytes on Linux; convert to MB. Use -1.0 if resource unavailable.
                    try:
                        import resource  # Standard lib (POSIX). Not available on some platforms.

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
                cumulated_per_layer_costs = None
                num_loss_accumulations = 0
                
            if step_performed:
                evaluation_callback(num_train_steps_done=training_progress.num_seen_steps_total)
                checkpointing_callback(training_progress=training_progress)
            # we start the time recoder here again to also capture the time spend loading
            # via the dataloader.
            forward_backward_time_recorder.start()


    def _reset_tracked_losses(self):
        # Initializes and returns a tensor representing the cumulated loss and gradient norm.
        # The tensor is initialized with zeros and its device is set based on the availability of CUDA.

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
        # Publishes the progress of the training, i.e., number of training steps done.

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
        # Publishes the evaluation result.

        evaluation_result_publisher.publish_message(
            payload=evaluation_result, message_type=MessageTypes.EVALUATION_RESULT
        )