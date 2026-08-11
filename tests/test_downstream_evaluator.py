import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modalities.conversion.model_converter import ModelConverter
from modalities.evaluator import DownstreamEvaluator


@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    cp_dir = tmp_path / "experiments" / "exp1"
    cp_dir.mkdir(parents=True)
    step_dir = cp_dir / "seen_steps_100"
    step_dir.mkdir()
    (cp_dir / "last_checkpoint_info.json").write_text(
        json.dumps({"checkpoint_path": str(step_dir), "checkpoint_folder_path": str(step_dir)})
    )
    (cp_dir / "config.yaml").write_text("dummy: config")
    return cp_dir


def test_model_converter_trigger(mock_checkpoint_dir):
    converter = ModelConverter(
        command_template="echo {checkpoint_path} -> {output_dir}",
        checkpoint_dir=mock_checkpoint_dir,
        global_rank=0,
        eval_interval=100,
    )
    with patch("subprocess.run") as mock_run:
        converter.convert(num_train_steps_done=100)
        mock_run.assert_called_once()


def test_model_converter_rank_gating(mock_checkpoint_dir):
    converter = ModelConverter(
        command_template="echo {checkpoint_path}",
        checkpoint_dir=mock_checkpoint_dir,
        global_rank=1,
        eval_interval=100,
    )
    with patch("subprocess.run") as mock_run:
        converter.convert(num_train_steps_done=100)
        mock_run.assert_not_called()


def test_model_converter_interval_gating(mock_checkpoint_dir):
    converter = ModelConverter(
        command_template="echo {checkpoint_path}",
        checkpoint_dir=mock_checkpoint_dir,
        global_rank=0,
        eval_interval=100,
    )
    with patch("subprocess.run") as mock_run:
        converter.convert(num_train_steps_done=50)
        mock_run.assert_not_called()


def test_downstream_evaluator_trigger(mock_checkpoint_dir):
    step_dir = mock_checkpoint_dir / "seen_steps_100"
    hf_dir = step_dir / "hf_checkpoint"
    hf_dir.mkdir()

    evaluator = DownstreamEvaluator(
        tokenizer=MagicMock(),
        tasks=["task1", "task2"],
        eval_interval=100,
        checkpoint_dir=mock_checkpoint_dir,
        global_rank=0,
        olmes_command_template="echo {hf_model_dir} '{tasks}' {step}",
    )

    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        evaluator.evaluate(num_train_steps_done=100)
        mock_popen.assert_called_once()
        assert len(evaluator.active_processes) == 1


def test_downstream_evaluator_wait_and_sync(mock_checkpoint_dir, tmp_path):
    step_dir = mock_checkpoint_dir / "seen_steps_100"
    hf_dir = step_dir / "hf_checkpoint"
    hf_dir.mkdir()
    eval_out_dir = hf_dir / "olmes_eval_100"
    eval_out_dir.mkdir()

    metrics_file = eval_out_dir / "metrics-all.jsonl"
    metrics_file.write_text(
        json.dumps(
            {
                "task_config": {"metadata": {"alias": "arc_challenge"}},
                "metrics": {"primary_score": 0.75},
            }
        )
        + "\n"
    )

    evaluator = DownstreamEvaluator(
        tokenizer=MagicMock(),
        tasks=["arc_challenge"],
        eval_interval=100,
        checkpoint_dir=mock_checkpoint_dir,
        global_rank=0,
        olmes_command_template="echo dummy",
    )

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    evaluator.active_processes = [(mock_proc, 100, hf_dir)]

    with patch("wandb.run") as mock_wandb_run:
        evaluator.wait_for_evaluations()
        mock_proc.wait.assert_called_once()
        mock_wandb_run.log.assert_called_once_with(
            {"downstream/arc_challenge": 0.75, "downstream_step": 100}
        )
        assert len(evaluator.active_processes) == 0
