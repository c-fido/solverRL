"""Exact policy evaluation vs Monte Carlo on KeyDoor."""

from __future__ import annotations

import numpy as np
import pytest

from solverrl.envs.keydoor import (
    ACTIONS,
    DONE,
    GAMMA,
    ACTION_TO_ID,
    enumerate_reachable,
    initial_states,
    transition,
)
from solverrl.exact import ExactMDP, success_within_horizon


@pytest.fixture(scope="module")
def mdp() -> ExactMDP:
    return ExactMDP.from_keydoor()


def test_exact_return_constant_left_policy_finite(mdp: ExactMDP):
    # Always move left: never solves; still has a well-defined discounted return.
    policy = np.full(mdp.n, ACTION_TO_ID["left"], dtype=np.int64)
    policy[mdp.index[DONE]] = ACTION_TO_ID["left"]
    J = mdp.exact_return(policy)
    assert np.isfinite(J)
    assert J < 0.0


def test_exact_return_matches_monte_carlo_on_trivial_policy(mdp: ExactMDP):
    policy = np.full(mdp.n, ACTION_TO_ID["left"], dtype=np.int64)
    J_exact = mdp.exact_return(policy)

    rng = np.random.default_rng(0)
    returns = []
    initials = initial_states()
    # Infinite-horizon left-spin has J=-1; need a long rollout for MC to approach it.
    for _ in range(200):
        s = initials[int(rng.integers(0, len(initials)))]
        g = 0.0
        disc = 1.0
        for _t in range(2000):
            a = int(policy[mdp.index[s]])
            tr = transition(s, a)
            g += disc * tr.reward
            disc *= GAMMA
            s = tr.next_state
            if tr.terminated:
                break
        returns.append(g)
    J_mc = float(np.mean(returns))
    assert J_exact == pytest.approx(-1.0, abs=1e-6)
    assert J_exact == pytest.approx(J_mc, abs=0.02)


def test_optimalish_open_loop_beats_spinning(mdp: ExactMDP):
    spin = np.full(mdp.n, ACTION_TO_ID["left"], dtype=np.int64)
    # Prefer right when possible — still weak, but should not be worse than spinning left forever
    right = np.full(mdp.n, ACTION_TO_ID["right"], dtype=np.int64)
    assert mdp.exact_return(right) >= mdp.exact_return(spin) - 1e-9


def test_success_probability_in_unit_interval(mdp: ExactMDP):
    policy = np.full(mdp.n, ACTION_TO_ID["left"], dtype=np.int64)
    p = success_within_horizon(mdp, policy, horizon=120)
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(0.0)
