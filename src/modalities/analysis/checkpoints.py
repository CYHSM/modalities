"""Loading a trained arm's model and evaluation data in a single process.

Post-hoc analysis needs one thing the training entrypoints do not offer: a plain, unwrapped model on
one device, with a checkpoint's weights in it, without a process group, FSDP wrapping or an optimizer.
This module provides exactly that and is shared by every analysis driver, so the model under analysis
is built the same way each time.

Two details are load-bearing:

* Only the ``app.model`` subtree of the distributed checkpoint is read. Optimizer state is roughly two
  thirds of the bytes on disk and is irrelevant to inference-time analysis.
* ``load_state_dict`` is checked for missing *and* unexpected keys. A silent partial load produces a
  half-initialized model whose activations look entirely plausible, which is the failure mode most
  likely to survive into a result.
"""

import os
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.distributed.checkpoint.format_utils import (
    FileSystemReader,
    _EmptyStateDictLoadPlanner,
    _load_state_dict,
)

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.config.pydantic_if_types import PydanticDatasetIFType, PydanticPytorchModuleType
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry

# src/modalities/analysis/checkpoints.py -> repository root is four components up.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WAVE2_CONFIG_DIRECTORY = REPOSITORY_ROOT / "config_files" / "nemotron" / "loop_ablation_5b_cluster"


class ModelOnly(BaseModel):
    """Components model selecting just ``model_raw`` from a training config."""

    model_raw: PydanticPytorchModuleType


class DatasetOnly(BaseModel):
    """Components model selecting just ``test_dataset`` from a training config."""

    test_dataset: PydanticDatasetIFType


def arm_config_path(arm: str, config_directory: Path = WAVE2_CONFIG_DIRECTORY) -> Path:
    """
    Locates an arm's config, mapping a seed run back to the config it was generated from.

    Args:
        arm (str): Run name, e.g. ``A1_loop_mamba`` or ``A4_loop_mamba_moe_seed2_redo``.
        config_directory (Path): Where the wave's configs live.

    Raises:
        FileNotFoundError: If no config exists for the run.

    Returns:
        Path: The config path.
    """
    for candidate in (arm, arm.replace("_redo", "")):
        path = config_directory / f"config_{candidate}.yaml"
        if path.exists():
            return path
    raise FileNotFoundError(f"No config for arm '{arm}' under {config_directory}.")


def build_section(config_path: Path, section: str, model_type: Type[BaseModel]) -> BaseModel:
    """
    Builds a single top-level component from a config, without instantiating the rest of the tree.

    Building the whole tree would require a device mesh, a dataloader and an optimizer; ``model_raw``
    and ``test_dataset`` are each self-contained, so they are built in isolation.

    Args:
        config_path (Path): The config to read.
        section (str): The top-level key to build, e.g. ``"model_raw"``.
        model_type (Type[BaseModel]): The pydantic model describing that key.

    Returns:
        BaseModel: The built components model.
    """
    # The config's resolvers expect a launch environment even for a single-process build.
    for key, value in {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}.items():
        os.environ.setdefault(key, value)
    config_dict = load_app_config_dict(
        config_file_path=config_path,
        experiment_id="analysis",
        experiments_root_path=Path("/tmp/modalities_analysis"),
    )
    factory = ComponentFactory(registry=Registry(COMPONENTS))
    return factory.build_components(config_dict={section: config_dict[section]}, components_model_type=model_type)


def load_model_weights_(model: nn.Module, checkpoint_directory: Path) -> None:
    """
    Loads a distributed checkpoint's model weights into an unwrapped model, in place.

    Args:
        model (nn.Module): The model to load into.
        checkpoint_directory (Path): Directory holding the ``.distcp`` shards.

    Raises:
        RuntimeError: If the checkpoint does not match the model's parameters exactly.
    """
    state_dict: dict = {}
    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(checkpoint_directory),
        planner=_EmptyStateDictLoadPlanner(keys=["app.model"]),
        no_dist=True,
    )
    missing, unexpected = model.load_state_dict(state_dict["app"]["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match the model: missing={missing}, unexpected={unexpected}.")


def load_arm(arm: str, checkpoint_directory: Path, device: torch.device, config_path: Path = None) -> nn.Module:
    """
    Builds an arm's model and loads its trained weights, ready for inference.

    Args:
        arm (str): Run name.
        checkpoint_directory (Path): The run's checkpoint directory.
        device (torch.device): Where to place the model.
        config_path (Path | None): Overrides the config lookup.

    Returns:
        nn.Module: The model in eval mode on ``device``.
    """
    config_path = config_path or arm_config_path(arm)
    model = build_section(config_path, "model_raw", ModelOnly).model_raw
    load_model_weights_(model, checkpoint_directory)
    return model.to(device).eval()


def fixed_evaluation_batch(config_path: Path, sample_key: str, num_sequences: int) -> torch.Tensor:
    """
    Builds a fixed evaluation batch: the leading samples of the arm's test split.

    Identical for every arm, seed and setting, so that no comparison can be moved by which tokens were
    drawn. The test split is data no arm trained on.

    Args:
        config_path (Path): The arm's config, which names the test dataset.
        sample_key (str): Key under which token ids are stored in a sample.
        num_sequences (int): How many sequences to take.

    Returns:
        torch.Tensor: Token ids of shape ``(num_sequences, sequence_length + 1)``. A packed sample
        carries one token more than the model's context, so inputs and shifted targets both come from
        it; the caller slices it.
    """
    dataset = build_section(config_path, "test_dataset", DatasetOnly).test_dataset
    samples = [torch.as_tensor(dataset[index][sample_key]) for index in range(num_sequences)]
    return torch.stack(samples).to(torch.long)
