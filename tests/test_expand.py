"""EXPAND edit proposal tests."""

from __future__ import annotations

import numpy as np

from solverrl.expand import propose_edits
from solverrl.exact import ExactMDP
from solverrl.ground import ground_atoms
from solverrl.teachers.census import build_census
from solverrl.teachers.ppo import build_ppo
import solverrl_core


def test_expand_editor_proposes_all_kinds(mdp: ExactMDP = ExactMDP.from_keydoor()):
    model = build_ppo(n_envs=2, seed=0)
    census = build_census(model, mdp=mdp, regime="smoke", seed=0)
    model.env.close()

    atoms = ground_atoms(mdp.states)
    learner = solverrl_core.RuleLearner.keydoor()
    learner.fit(atoms, np.asarray(census.actions, dtype=np.int64))

    props = propose_edits(learner)
    kinds = {p.kind_name for p in props}
    assert "specialize" in kinds
    assert "reorder" in kinds
    assert "prune" in kinds


def test_proposal_rollout_shape(mdp: ExactMDP = ExactMDP.from_keydoor()):
    model = build_ppo(n_envs=2, seed=1)
    census = build_census(model, mdp=mdp, regime="smoke", seed=1)
    model.env.close()

    atoms = ground_atoms(mdp.states)
    learner = solverrl_core.RuleLearner.keydoor()
    learner.fit(atoms, np.asarray(census.actions, dtype=np.int64))
    props = propose_edits(learner)
    assert len(props) > 0
    rolled = np.asarray(props[0].rollout(atoms), dtype=np.int64)
    assert rolled.shape == (mdp.n,)
