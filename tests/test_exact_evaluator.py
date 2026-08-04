"""C++ ExactEvaluator vs Python ExactMDP reference."""

from __future__ import annotations

import numpy as np
import pytest

from solverrl.envs.keydoor import ACTION_TO_ID
from solverrl.exact import ExactMDP, success_within_horizon


@pytest.fixture(scope="module")
def mdp() -> ExactMDP:
    return ExactMDP.from_keydoor()


def test_cpp_exact_return_matches_python(mdp: ExactMDP):
    policy = np.full(mdp.n, ACTION_TO_ID["left"], dtype=np.int64)
    evalr = mdp.cpp_evaluator()
    j_py = mdp.exact_return(policy)
    j_cpp = float(evalr.exact_return(policy))
    assert j_cpp == pytest.approx(j_py, rel=0, abs=1e-8)


def test_cpp_success_matches_python(mdp: ExactMDP):
    policy = np.full(mdp.n, ACTION_TO_ID["left"], dtype=np.int64)
    evalr = mdp.cpp_evaluator()
    p_py = success_within_horizon(mdp, policy)
    p_cpp = float(evalr.success(policy))
    assert p_cpp == pytest.approx(p_py, rel=0, abs=1e-10)


def test_advantage_gap_cert_runs_on_distinct_policies(mdp: ExactMDP):
    left = np.full(mdp.n, ACTION_TO_ID["left"], dtype=np.int64)
    pickup = np.full(mdp.n, ACTION_TO_ID["pickup"], dtype=np.int64)
    cert = mdp.advantage_gap_cert(student=left, teacher=pickup)
    assert cert["n_disagree"] > 0
    assert np.isfinite(cert["weighted_gap"])
    assert np.isfinite(cert["return_gap"])
    assert cert["max_gap"] >= 0.0
