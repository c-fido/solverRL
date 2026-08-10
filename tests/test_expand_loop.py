"""EXPAND acceptance loop and R1 run tests."""

from __future__ import annotations

import numpy as np
import pytest

from solverrl.exact import ExactMDP
from solverrl.expand import certify_and_expand
from solverrl.ground import ground_atoms
from solverrl.run_r1 import run_r1_expand
from solverrl.teachers.census import build_census
from solverrl.teachers.ppo import build_ppo
import solverrl_core


@pytest.fixture(scope="module")
def mdp() -> ExactMDP:
    return ExactMDP.from_keydoor()


def test_certify_and_expand_return_curve_monotone(mdp: ExactMDP):
    model = build_ppo(n_envs=2, seed=0)
    census = build_census(model, mdp=mdp, regime="smoke", seed=0)
    model.env.close()

    atoms = ground_atoms(mdp.states)
    learner = solverrl_core.RuleLearner.keydoor()
    learner.fit(atoms, np.asarray(census.actions, dtype=np.int64))

    result = certify_and_expand(
        learner, mdp, atoms, np.asarray(census.actions, dtype=np.int64),
        tau=1e-9, max_iterations=20,
    )
    assert len(result.return_curve) >= 1
    for i in range(len(result.return_curve) - 1):
        assert result.return_curve[i + 1] >= result.return_curve[i] - 1e-12
    assert result.n_clauses >= 1
    assert "decision_list" in result.prolog


def test_certify_and_expand_stops_with_large_tau(mdp: ExactMDP):
    model = build_ppo(n_envs=2, seed=1)
    census = build_census(model, mdp=mdp, regime="smoke", seed=1)
    model.env.close()

    atoms = ground_atoms(mdp.states)
    learner = solverrl_core.RuleLearner.keydoor()
    learner.fit(atoms, np.asarray(census.actions, dtype=np.int64))

    result = certify_and_expand(
        learner, mdp, atoms, np.asarray(census.actions, dtype=np.int64),
        tau=1e9, max_iterations=10,
    )
    assert result.iterations == 0
    assert len(result.return_curve) == 1


def test_run_r1_expand_improves_or_matches_initial_return(mdp: ExactMDP):
    model = build_ppo(n_envs=2, seed=2)
    census = build_census(model, mdp=mdp, regime="r1_smoke", seed=2)
    model.env.close()

    report = run_r1_expand(census, mdp=mdp, max_iterations=30)
    assert report.final_return >= report.initial_return - 1e-12
    assert report.final_success >= report.initial_success - 1e-12
    assert len(report.expand.return_curve) >= 1
