#!/usr/bin/env python
"""Reports whether `use_weight_tying` survives each model-construction stage, and a warmstart load.

Diagnostic for the run-vs-warmstart discrepancy found on 2026-08-14: the 5B-wave checkpoints show
`transformer.wte.weight` and `transformer.lm_head.weight` bit-identical after `modalities warmstart`
but genuinely different after `modalities run` (44% relative RMS apart, cosine 0.913, zero identical
elements), and the two paths reach measurably different quality (final train loss 2.76 vs 2.70,
grad norm 0.38 vs 0.48). NemotronLLM ties them by Python identity
(`self.transformer.wte.weight = self.transformer.lm_head.weight`, nemotron_model.py), and
`named_parameters()` deduplicates a tied tensor -- which is why the optimizer tracks it once, under
the `wte` name, with no Adam state for `lm_head.weight`. So something separates the two names
between construction and checkpoint saving. This script finds which stage.

Each stage is built in a SEPARATE ComponentFactory pass so stages cannot contaminate each other:
the config wires model_raw -> activation_checkpointed_model -> fsdp_model -> initialized_model, and
building any one of them builds its whole upstream chain. `is` identity is the ground truth here --
equal values are not enough, since two separate tensors start out equal and only drift later.

Run under torchrun with the same world size as training (FSDP2 needs a real process group):

    torchrun --nnodes=1 --nproc_per_node=4 inspect_weight_tying.py --config_file_path <cfg>
    torchrun ... inspect_weight_tying.py --config_file_path <warmstart cfg> \
        --last_checkpoint_info_file_path <path>   # adds the post-load check
"""

import json
from functools import partial
from pathlib import Path
from typing import Optional

import click
import click_pathlib
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from pydantic import BaseModel

from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticPytorchModuleType
from modalities.main import Main
from modalities.running_env.cuda_env import CudaEnv


def _warmstart_resolvers(last_checkpoint_info_file_path: Optional[Path]) -> dict:
    """Mirrors the `warmstart_env` resolver the real `modalities warmstart` command installs."""
    if last_checkpoint_info_file_path is None:
        return {}

    def resolve(var_name: str, path: Path) -> DictConfig:
        if var_name != "checkpoint_paths":
            raise ValueError(f"Unknown variable name {var_name}. Should be 'checkpoint_paths'.")
        with open(path, "r") as f:
            return DictConfig(json.load(f))

    return {"warmstart_env": partial(resolve, path=last_checkpoint_info_file_path)}

STAGES = ["model_raw", "activation_checkpointed_model", "fsdp_model", "initialized_model"]


def _make_single_component_model(instance_key: str) -> type[BaseModel]:
    """Builds a one-field pydantic model that resolves exactly `instance_key` from the config."""
    return type(
        "SingleComponentModel",
        (BaseModel,),
        {"__annotations__": {instance_key: PydanticPytorchModuleType}},
    )


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Digs the NemotronLLM out of whatever AC/FSDP wrappers sit on top of it."""
    seen = set()
    while True:
        if hasattr(module, "transformer"):
            return module
        nxt = getattr(module, "_checkpoint_wrapped_module", None) or getattr(module, "module", None)
        if nxt is None or id(nxt) in seen:
            return module
        seen.add(id(nxt))
        module = nxt


def _report(label: str, model: torch.nn.Module) -> None:
    m = _unwrap(model)
    if not hasattr(m, "transformer"):
        print(f"  {label:<32} <could not locate .transformer>", flush=True)
        return
    t = m.transformer
    wte, lmh = t.wte.weight, t.lm_head.weight
    tied = wte is lmh
    # A tied tensor is deduplicated by named_parameters(), so the count is a second, independent
    # witness: 1 => the optimizer would see one embedding, 2 => it would see two.
    n_named = sum(1 for n, _ in m.named_parameters() if n.endswith(("wte.weight", "lm_head.weight")))
    line = f"  {label:<32} tied(is)={str(tied):<5}  named_parameters entries={n_named}"
    if not tied:
        with torch.no_grad():
            a, b = wte.detach().float(), lmh.detach().float()
            if hasattr(a, "full_tensor"):  # DTensor -> materialize for comparison
                a, b = a.full_tensor(), b.full_tensor()
            line += f"  equal_values={torch.equal(a, b)}  max|diff|={(a - b).abs().max().item():.3e}"
    print(line, flush=True)


@click.command()
@click.option("--config_file_path", type=click_pathlib.Path(exists=True), required=True)
@click.option("--experiments_root_path", type=click_pathlib.Path(exists=True), required=True)
@click.option("--last_checkpoint_info_file_path", type=click_pathlib.Path(exists=True), default=None)
def main(config_file_path: Path, experiments_root_path: Path, last_checkpoint_info_file_path: Optional[Path]) -> None:
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        rank0 = dist.get_rank() == 0
        if rank0:
            mode = "WARMSTART" if last_checkpoint_info_file_path else "RUN"
            print(f"\n{'=' * 78}\n{mode}: {config_file_path.name}\n{'=' * 78}", flush=True)
            print("Stage-by-stage (each built in its own pass):", flush=True)

        resolvers = _warmstart_resolvers(last_checkpoint_info_file_path)
        for stage in STAGES:
            main_obj = Main(
                config_file_path, experiments_root_path=experiments_root_path, additional_resolver_funs=resolvers
            )
            try:
                components = main_obj.build_components(components_model_type=_make_single_component_model(stage))
            except Exception as e:  # a stage may not exist in every config; keep going
                if rank0:
                    print(f"  {stage:<32} <skipped: {type(e).__name__}: {str(e)[:90]}>", flush=True)
                dist.barrier()
                continue
            if rank0:
                _report(stage, getattr(components, stage))
            dist.barrier()

        # The decisive one for warmstart: identity AFTER the checkpoint has been loaded into the model.
        if last_checkpoint_info_file_path is not None:
            from modalities.__main__ import TrainingComponentsInstantiationModel

            main_obj = Main(
                config_file_path, experiments_root_path=experiments_root_path, additional_resolver_funs=resolvers
            )
            components = main_obj.build_components(components_model_type=TrainingComponentsInstantiationModel)
            if rank0:
                print("\nAfter checkpoint load into app_state:", flush=True)
                parts = components.app_state.model_parts
                _report("post-warmstart-load", parts[0] if isinstance(parts, list) else parts)
            dist.barrier()

        if rank0:
            print(f"{'=' * 78}\n", flush=True)


if __name__ == "__main__":
    main()
