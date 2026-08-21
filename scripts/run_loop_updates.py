#!/usr/bin/env python
"""Measures loop-update statistics for one Wave 2 arm from its trained checkpoint.

Builds the arm's ``model_raw`` from its own config, loads the distributed checkpoint into it, runs a
single forward pass over a fixed batch with
:class:`~modalities.analysis.loop_updates.LoopUpdateRecorder` attached, and writes the summary to
JSON. Nothing is trained and no optimizer state is read.

The batch is deliberately identical for every arm and seed -- same file, same leading samples -- so
that differences between arms cannot come from the tokens. Sequences are drawn from the arm's own
`test_dataset`, i.e. data the model never trained on.

Run under one GPU, from the repository root::

    python scripts/run_loop_updates.py --arm A1_loop_mamba \\
        --experiments-root /leonardo_scratch/large/userexternal/mfrey000/experiments_nemotron_5b_cluster

The companion Slurm launcher `scripts/run_loop_updates.sh` sweeps every arm and seed.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from modalities.analysis.checkpoints import arm_config_path, fixed_evaluation_batch, load_arm
from modalities.analysis.loop_updates import LoopUpdateRecorder

REPOSITORY_ROOT = Path(__file__).parents[1]

# Fixed evaluation batch. n_embd is 1024 and sequences are 2048 tokens, so 8 sequences give 16,384
# tokens -- ample for stable per-token medians, and small enough to hold every layer's captured
# activations in host memory. NOTE this is deliberately small because the metrics here are per-token
# medians over the batch; it is far too few sequences to estimate a *loss* precisely, which is why
# the loss below is only ever used as a load check.
NUM_SEQUENCES = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, help="Arm/run name, e.g. A1_loop_mamba or A1_loop_mamba_seed3")
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=REPOSITORY_ROOT / "docs" / "loopotron" / "loop_updates")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    config_path = arm_config_path(arguments.arm)
    checkpoint_info = arguments.experiments_root / arguments.arm / "checkpoints" / "last_checkpoint_info.json"
    checkpoint_directory = Path(json.loads(checkpoint_info.read_text())["checkpoint_folder_path"])

    print(f"[{arguments.arm}] config={config_path.name} checkpoint={checkpoint_directory.name}", flush=True)

    model = load_arm(arguments.arm, checkpoint_directory, device, config_path=config_path)

    samples = fixed_evaluation_batch(config_path, model.sample_key, NUM_SEQUENCES).to(device)
    inputs, targets = samples[:, : model.sequence_length], samples[:, 1 : model.sequence_length + 1]

    with torch.no_grad(), LoopUpdateRecorder(model=model) as recorder:
        logits = model({model.sample_key: inputs})[model.prediction_key]

        # Sanity, not a published figure: the fixed batch is a handful of test sequences rather than
        # the full split, so this will not equal the arm's reported loss. It is here to catch a
        # broken load -- a mangled checkpoint gives ~log(vocab_size), not ~2.5.
        loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())

    report = {
        "arm": arguments.arm,
        "config": config_path.name,
        "checkpoint": str(checkpoint_directory),
        "layer_pattern": model.layer_pattern,
        "n_built_layers": model.n_layer,
        "n_executed_layers": model.n_executed_layers,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch": {"num_sequences": int(inputs.shape[0]), "sequence_length": int(inputs.shape[1])},
        "sanity_batch_loss": loss.item(),
        "groups": recorder.group_report(),
        "stack": recorder.stack_report(),
        # Per-layer update magnitude against each layer's OWN input. On the unlooped baseline this is
        # the cheap predictor tested in docs/loopotron/update_norm_predictor.md: it is measurable
        # from one baseline checkpoint, before any loop arm is trained.
        "layer_profile": recorder.layer_profile(),
    }

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_directory / f"{arguments.arm}.json"
    output_path.write_text(json.dumps(report, indent=1))
    print(
        f"[{arguments.arm}] batch loss {loss.item():.4f}  executed {model.n_executed_layers} layers  "
        f"{len(report['groups'])} looped group(s) -> {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
