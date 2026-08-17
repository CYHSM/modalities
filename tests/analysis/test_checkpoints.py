"""Tests for the shared analysis checkpoint plumbing.

These are path and lookup tests, not model tests. They exist because the first cluster submission of
the Phase 0 sweep died on every arm with `FileNotFoundError`: `REPOSITORY_ROOT` was computed one
directory too high, and the whole unit suite passed anyway because nothing exercised the lookup. A
path constant that is only ever evaluated on the cluster is a path constant that breaks there.
"""

import pytest

from modalities.analysis.checkpoints import (
    REPOSITORY_ROOT,
    WAVE2_CONFIG_DIRECTORY,
    arm_config_path,
)


def test_repository_root_points_at_the_repository():
    assert (REPOSITORY_ROOT / "pyproject.toml").is_file(), REPOSITORY_ROOT
    assert (REPOSITORY_ROOT / "src" / "modalities").is_dir(), REPOSITORY_ROOT


def test_wave2_config_directory_exists_and_holds_arm_configs():
    assert WAVE2_CONFIG_DIRECTORY.is_dir(), WAVE2_CONFIG_DIRECTORY
    assert list(WAVE2_CONFIG_DIRECTORY.glob("config_A*.yaml")), WAVE2_CONFIG_DIRECTORY


@pytest.mark.parametrize(
    "arm",
    [
        "A0_baseline",
        "A1_loop_mamba",
        "A1_loop_mamba_seed3",
        "A6_loop_attention_moe",
        # Redo runs have their own config; the fallback that strips "_redo" must not shadow it.
        "A4_loop_mamba_moe_seed2_redo",
        "N1_anchor_mamba",
    ],
)
def test_every_wave2_run_resolves_to_an_existing_config(arm):
    assert arm_config_path(arm).is_file()


def test_redo_runs_resolve_to_their_own_config_when_one_exists():
    resolved = arm_config_path("A4_loop_mamba_moe_seed2_redo")
    assert resolved.name == "config_A4_loop_mamba_moe_seed2_redo.yaml"


def test_unknown_arm_raises_rather_than_silently_falling_back():
    with pytest.raises(FileNotFoundError, match="No config for arm"):
        arm_config_path("Z9_does_not_exist")
