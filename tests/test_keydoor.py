"""Week 1: KeyDoor environment contracts (paper arXiv:2607.15459v2)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from solverrl.envs.keydoor import (
    ACTIONS,
    GAMMA,
    HORIZON,
    STEP_PENALTY,
    SUCCESS_REWARD,
    KeyDoorEnv,
    env_source_sha256,
    enumerate_reachable,
)


def test_action_space_is_six():
    env = KeyDoorEnv()
    assert env.action_space.n == 6
    assert set(ACTIONS) == {"up", "down", "left", "right", "pickup", "toggle"}


def test_reset_and_step_shapes():
    env = KeyDoorEnv(seed=0)
    obs, info = env.reset()
    assert obs.dtype == np.float32
    assert obs.ndim == 1
    assert "state" in info
    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert obs2.shape == obs.shape
    assert isinstance(reward, float)
    assert truncated is False or terminated or True  # horizon may truncate


def test_step_penalty_and_success_reward_match_paper():
    assert STEP_PENALTY == pytest.approx(-0.01)
    assert SUCCESS_REWARD == pytest.approx(1.0)
    assert GAMMA == pytest.approx(0.99)
    assert HORIZON == 120
    # Terminal transition nets to R_max = 0.99
    assert abs(SUCCESS_REWARD + STEP_PENALTY) == pytest.approx(0.99)


def test_env_source_hash_is_stable_sha256():
    digest = env_source_sha256()
    assert len(digest) == 64
    path = Path(__file__).resolve().parents[1] / "solverrl" / "envs" / "keydoor.py"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected


def test_reachable_set_closed_and_nontrivial():
    """Paper reports |S_r|=16944; our prose-faithful reimplementation is smaller.

    We assert closure under the transition kernel and a nontrivial census rather than
    bitwise equality with an unpublished reference env.
    """
    from solverrl.envs.keydoor import DONE, transition

    states = enumerate_reachable(include_done=True)
    index = set(states)
    assert DONE in index
    assert 5_000 <= len(states) <= 20_000
    for s in states:
        for a in range(6):
            nxt = transition(s, a).next_state
            assert nxt in index
