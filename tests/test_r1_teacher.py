"""Week 2: R1 early-stop when greedy exact success ≥ 0.95."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solverrl.envs.keydoor import ACTION_TO_ID, DONE
from solverrl.exact import ExactMDP, success_within_horizon
from solverrl.teachers.ppo import build_ppo, greedy_exact_success, greedy_policy_from_model, train_r1


@pytest.fixture(scope="module")
def mdp() -> ExactMDP:
    return ExactMDP.from_keydoor()


def test_left_policy_has_zero_exact_success(mdp: ExactMDP):
    policy = np.full(mdp.n, ACTION_TO_ID["left"], dtype=np.int64)
    assert success_within_horizon(mdp, policy) == pytest.approx(0.0)


def test_greedy_policy_from_model_shape(mdp: ExactMDP):
    model = build_ppo(n_envs=2, seed=0)
    policy = greedy_policy_from_model(model, mdp)
    assert policy.shape == (mdp.n,)
    assert policy.dtype == np.int64
    assert policy.min() >= 0
    assert policy.max() < 6
    assert int(policy[mdp.index[DONE]]) in range(6)
    model.env.close()


def test_train_r1_stops_and_saves_when_threshold_met(tmp_path: Path, mdp: ExactMDP):
    # threshold=0.0 ⇒ stop after the first eval (untrained greedy success ≥ 0).
    out = tmp_path / "r1.zip"
    result = train_r1(
        save_path=out,
        n_envs=2,
        seed=0,
        success_threshold=0.0,
        max_timesteps=10_000,
        chunk_timesteps=2048,
        mdp=mdp,
    )
    assert out.is_file()
    assert result.save_path == out
    assert result.timesteps <= 2048
    assert result.greedy_success >= 0.0
    assert result.stopped_early is True
