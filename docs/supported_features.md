# Supported Features
In this file, we list the already implemented, planned and in-progress features w.r.t. to improving downstream performance, throughput, multi-modality, and alignment. 

## Throughput Features

| Name                                  | Status           | Description                                                                                                       |
|---------------------------------------|------------------|-------------------------------------------------------------------------------------------------------------------|
| Mixed Precision Training              | supported        | Utilizes both single (FP32) and half precision (FP16) floating-point formats to speed up arithmetic computations while maintaining model accuracy. Support for bf16|
| Fully Sharded Data Parallel (FSDP)    | supported        | Optimizes distributed training by sharding the model parameters, gradients, and optimizer states across all GPUs, reducing memory overhead and enabling the training of larger models. |
| Gradient Accumulation                 | supported        | Allows for the use of larger batch sizes than what might fit in memory by accumulating gradients over multiple mini-batches before updating model weights. |
| CPU Offloading via FSDP               | supported        | Moves parts of the model or computation from GPU to CPU or other storage to manage GPU memory constraints. |
| Memmap for efficient data loading     | supported        | Optimizes the data pipeline to reduce I/O bottlenecks. |
| Activation Checkpointing              | supported        | Saves intermediate activations to memory only at certain points during the forward pass and recomputes them during the backward pass, reducing memory usage at the cost of additional computation. |
| Flash Attention                       | supported        | A highly optimized attention mechanism that significantly reduces the computational burden and memory footprint of attention calculations, enabling faster training and inference on large models. |
| RMS and Layer Norm (pre-normalization) | supported        | Normalizes the pre-activation weights in a layer to stabilize training. |
| Adaptive Batch Size Exploration       | planned         | Dynamically increases the training batch size during the training process to identify the maximum batch size that can be accommodated by a given GPU setup without causing memory overflow or performance degradation. |
| Node Failure Recovery                 | planned         | Implements mechanisms to automatically detect and recover from failures (e.g., node or GPU failures) in distributed training environments, ensuring that training can continue with minimal interruption even if one or more nodes / GPUs in the cluster fail. |










## Downstream Performance Features

| Name                           | Status           | Description                                                                                                       |
|--------------------------------|------------------|-------------------------------------------------------------------------------------------------------------------|
| SwiGLU                         | supported         | A nonlinear activation function combining Gated Linear Units (GLU) with Swish for enhancing model capacity and learning efficiency. |
| Weight Decay                   | supported        | Regularization technique that adds a penalty on the size of weights, encouraging smaller weights to reduce overfitting and improve generalization. |
| RMSNorm (pre-normalization)    | supported        | Normalizes the pre-activation weights in a layer to stabilize training, often used as an alternative to LayerNorm for improved training dynamics. |
| Rotary Positional Embeddings (RoPE) | supported   | Encodes sequence position information into attention mechanisms, preserving relative positional information and improving model's understanding of sequence order. |
| Grouped-query Attention (GQA)  | supported      | Enhances attention mechanisms by grouping queries to reduce computation and memory footprint while maintaining or improving performance. |
| Learning Rate Scheduler        | supported      | Adjusts the learning rate during training according to a predefined schedule (e.g., step decay, exponential decay) to improve convergence and performance. |
| Gradient Clipping              | supported          | Prevents exploding gradients by clipping the gradients of an optimization algorithm to a maximum value, thereby stabilizing training. |
| Training Warmup                | supported          | Gradually increases the learning rate from a low to a high value during the initial phase of training to stabilize optimization. |
| Depth-wise Layer Loops         | supported          | Executes a run of layers several times while reusing one set of weights (Universal-Transformer-style weight sharing), trading parameters for effective depth. Written `[<symbols>]^<K>` in a hybrid layer pattern; see [nemotron_loops.md](components/nemotron_loops.md). |
| Synthetic Reasoning Evaluations | supported         | Deterministic held-out tasks (p-hop induction, variable binding) parameterized by the number of serial lookups the answer needs, so a control that requires no depth and a task that does can be compared directly. Reported as answer accuracy and answer NLL per evaluation dataloader; see [nemotron_loops.md](components/nemotron_loops.md#reasoning-evaluations). |
| Masked-target Benchmark Evaluations | supported     | Minerva MATH (4-shot, matching lm-evaluation-harness's `minerva_math`) and closed-book TriviaQA, scored as negative log-likelihood of the reference answer rather than accuracy, which is a constant zero at small scale. Tokenized once offline so every run is scored on byte-identical sequences; see [prepared_eval.py](../src/modalities/dataloader/prepared_eval.py). |
| Per-dataloader Evaluation Metrics | supported       | `eval_metrics` attaches metrics to named evaluation dataloaders, reported alongside the loss. Numerator and denominator are reduced separately across micro-batches and ranks, so the result does not depend on how samples are batched or sharded. |
| Loss Masking                   | planned          | Ignores or gives less weight to certain data points in the loss function, often used in tasks with variable-length sequences to ignore padding tokens or in more specific usecases such as GAtt. |
| Knowledge Distillation         | planned  | Transfers knowledge from a larger, complex model to a smaller, more efficient model, improving the smaller model's performance without the computational cost of the larger model.|
| Hyperparameter Optimization    | planned          | Grid search for various hyperparameter such as LR, Optimizer arguments etc. Also the integration of µP might be interesting |


## Multi-modality Features

## Alignment Features
