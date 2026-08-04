"""Tests for RuleLearner pybind, grounding, and census distillation."""

from __future__ import annotations

import numpy as np
import pytest

from solverrl.distill import distill_from_census
from solverrl.envs.keydoor import DONE, env_source_sha256
from solverrl.exact import ExactMDP
from solverrl.ground import encode_state, ground_atoms
from solverrl.teachers.census import CensusDataset, build_census
from solverrl.teachers.ppo import build_ppo


@pytest.fixture(scope="module")
def mdp() -> ExactMDP:
    return ExactMDP.from_keydoor()


def test_solverrl_core_rule_learner_import():
    import solverrl_core

    assert solverrl_core.ping() == "pong"
    assert solverrl_core.KEYDOOR_NUM_ATOMS == 17
    assert solverrl_core.KEYDOOR_NUM_ACTIONS == 6


def test_ground_atoms_shape_and_done_all_false(mdp: ExactMDP):
    sample = [mdp.states[0], mdp.states[100], DONE]
    atoms = ground_atoms(sample)
    assert atoms.shape == (3, 17)
    assert atoms.dtype == np.uint8
    assert atoms[2].sum() == 0


def test_rule_learner_fit_predict_roundtrip(mdp: ExactMDP):
    import solverrl_core

    states = mdp.states[:32]
    atoms = ground_atoms(states)
    actions = np.arange(len(states), dtype=np.int64) % 6

    learner = solverrl_core.RuleLearner.keydoor()
    learner.fit(atoms, actions)
    preds = learner.predict(atoms)
    assert preds.shape == (len(states),)
    assert learner.is_fitted
    assert learner.n_clauses >= 1
    pl = learner.to_prolog()
    assert "decision_list" in pl
    assert "act(" in pl


def test_distill_smoke_census_pipeline(mdp: ExactMDP):
    """Untrained PPO labels are near-random; verify the pipeline runs end-to-end."""
    model = build_ppo(n_envs=2, seed=0)
    census = build_census(model, mdp=mdp, regime="smoke", seed=0)
    model.env.close()

    result = distill_from_census(census, mdp=mdp)
    assert result.n_states == mdp.n
    assert result.n_clauses >= 1
    assert 0.0 <= result.fidelity <= 1.0
    assert result.fidelity > 1.0 / 6.0  # above uniform random (6 actions)
    assert "move(" in result.prolog or "pickup" in result.prolog


def test_distill_recovers_planted_teacher(mdp: ExactMDP):
    """A teacher induced by FOIL should distill back with near-perfect fidelity."""
    import solverrl_core

    atoms = ground_atoms(mdp.states)
    seed_labels = np.arange(mdp.n, dtype=np.int64) % 6

    planter = solverrl_core.RuleLearner.keydoor()
    planter.fit(atoms, seed_labels)
    planted = planter.predict(atoms)

    census = CensusDataset(
        actions=planted,
        regime="planted",
        seed=0,
        env_sha256=env_source_sha256(),
        n_states=mdp.n,
    )
    result = distill_from_census(census, mdp=mdp)
    assert result.fidelity >= 0.99


def test_prolog_emit_matches_cpp_predict_on_sample(mdp: ExactMDP):
    """predict() and fidelity() must agree on the same labeled sample."""
    import solverrl_core

    model = build_ppo(n_envs=2, seed=1)
    census = build_census(model, mdp=mdp, regime="agree", seed=1)
    model.env.close()

    atoms = ground_atoms(mdp.states)
    actions = np.asarray(census.actions, dtype=np.int64)
    learner = solverrl_core.RuleLearner.keydoor()
    learner.fit(atoms, actions)

    rng = np.random.default_rng(42)
    idx = rng.choice(mdp.n, size=min(200, mdp.n), replace=False)
    sub_atoms = atoms[idx]
    sub_actions = actions[idx]
    preds = learner.predict(sub_atoms)
    assert np.mean(preds == sub_actions) == pytest.approx(
        learner.fidelity(sub_atoms, sub_actions), rel=1e-9
    )


def test_encode_state_matches_cpp_grounding(mdp: ExactMDP):
    import solverrl_core

    for s in [mdp.states[0], mdp.states[500], DONE]:
        encoded = encode_state(s)[None, :]
        row = np.asarray(solverrl_core.ground_keydoor_states(encoded), dtype=np.uint8)
        assert np.array_equal(row, ground_atoms([s]))
