import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from modalities.util import print_rank_0

logger = logging.getLogger(__name__)


class ModelConverter:
    def __init__(
        self,
        command_template: str,
        checkpoint_dir: Path,
        global_rank: int,
        eval_interval: int,
    ):
        self.command_template = command_template
        self.checkpoint_dir = Path(checkpoint_dir)
        self.global_rank = global_rank
        self.eval_interval = eval_interval

    def convert(self, num_train_steps_done: int):
        if self.global_rank != 0:
            return

        if self.eval_interval <= 0 or num_train_steps_done % self.eval_interval != 0:
            return

        info_file = self.checkpoint_dir / "last_checkpoint_info.json"
        if not info_file.exists():
            logger.warning(f"ModelConverter: {info_file} does not exist. Skipping conversion.")
            return

        try:
            with open(info_file, "r") as f:
                info = json.load(f)
            cp_str = info.get("checkpoint_folder_path") or info.get("checkpoint_path") or info.get("model_checkpoint_path")
            if not cp_str:
                raise KeyError("No valid checkpoint path key found in info file")
            checkpoint_path = Path(cp_str)
        except Exception as e:
            logger.error(f"ModelConverter: Failed to read {info_file}: {e}")
            return

        output_dir = checkpoint_path / "hf_checkpoint"
        if output_dir.exists():
            logger.info(f"ModelConverter: Output dir {output_dir} already exists. Skipping.")
            return

        # Prefer the fully resolved config dump: the plain config copy still contains
        # ${...} interpolations (e.g. ${modalities_env:experiment_id}) that cannot be
        # resolved outside of the training process.
        config_path = next(iter(sorted(self.checkpoint_dir.glob("*.yaml.resolved"))), None)
        if config_path is None:
            if (self.checkpoint_dir / "config.yaml").exists():
                config_path = self.checkpoint_dir / "config.yaml"
            else:
                yaml_files = sorted(self.checkpoint_dir.glob("*.yaml"))
                config_path = yaml_files[0] if yaml_files else self.checkpoint_dir.parent / "config.yaml"

        cmd = self.command_template.format(
            checkpoint_path=str(checkpoint_path),
            output_dir=str(output_dir),
            modalities_config=str(config_path),
        )

        print_rank_0(f"ModelConverter: Executing command: {cmd}")
        try:
            res = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"ModelConverter output: {res.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"ModelConverter command failed with code {e.returncode}: {e.stderr}")
