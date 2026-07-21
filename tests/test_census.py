"""Week 2: reachable census + one-command R1/R2 regeneration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solverrl.envs.keydoor import ACTION_TO_ID, DONE
from solverrl.exact import ExactMDP
from solverrl.teachers.census import (
    CensusDataset,
    build_census,
    load_census,
    regenerate_r1_r2_censuses,
    save_census,
)
from solverrl.teachers.ppo import build_ppo


@pytest.fixture(scope="module")
def mdp() -> ExactMDP:
    return ExactMDP.from_keydoor()


def test_build_census_labels_every_reachable_state(mdp: ExactMDP, tmp_path: Path):
    model = build_ppo(n_envs=2, seed=0)
    census = build_census(model, mdp=mdp, regime="smoke", seed=0)
    model.env.close()

    assert census.n_states == mdp.n
    assert census.actions.shape == (mdp.n,)
    assert census.actions.dtype == np.int64
    assert census.actions.min() >= 0
    assert census.actions.max() < 6
    assert int(census.actions[mdp.index[DONE]]) in range(6)
    assert census.regime == "smoke"

    path = save_census(census, tmp_path / "census.npz")
    loaded = load_census(path)
    assert loaded.n_states == census.n_states
    assert np.array_equal(loaded.actions, census.actions)
    assert loaded.regime == "smoke"


def test_regenerate_r1_r2_censuses_writes_both(tmp_path: Path, mdp: ExactMDP):
    # Tiny trains so the one-command path is testable without paper budgets.
    paths = regenerate_r1_r2_censuses(
        out_dir=tmp_path,
        n_envs=2,
        seed=0,
        r1_success_threshold=0.0,
        r1_max_timesteps=2048,
        r1_chunk_timesteps=2048,
        r2_timesteps=2048,
        mdp=mdp,
    )
    assert paths["r1"].is_file()
    assert paths["r2"].is_file()
    r1 = load_census(paths["r1"])
    r2 = load_census(paths["r2"])
    assert r1.regime == "r1"
    assert r2.regime == "r2"
    assert r1.n_states == mdp.n
    assert r2.n_states == mdp.n
