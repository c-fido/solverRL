"""EXPAND edit proposals for decision lists."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from solverrl.ground import ground_atoms


def propose_edits(learner, *, max_body_literals: int = 3) -> Sequence:
    """Return specialize / reorder / prune proposals for a fitted RuleLearner."""
    import solverrl_core

    editor = solverrl_core.ExpandEditor(solverrl_core.KEYDOOR_NUM_ATOMS, max_body_literals)
    return editor.propose(learner)


def rollout_proposal(proposal, states) -> np.ndarray:
    """Roll out one proposal to a per-state action vector."""
    atoms = ground_atoms(states)
    return np.asarray(proposal.rollout(atoms), dtype=np.int64)
