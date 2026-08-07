"""EXPAND loop and certified expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from solverrl.exact import ExactMDP
from solverrl.ground import ground_atoms


@dataclass(frozen=True)
class ExpandResult:
    return_curve: List[float]
    success_curve: List[float]
    iterations: int
    final_return: float
    final_success: float
    accepted_any_edit: bool
    n_clauses: int
    prolog: str


def propose_edits(learner, *, max_body_literals: int = 3) -> Sequence:
    """Return specialize / reorder / prune proposals for a fitted RuleLearner."""
    import solverrl_core

    editor = solverrl_core.ExpandEditor(solverrl_core.KEYDOOR_NUM_ATOMS, max_body_literals)
    return editor.propose(learner)


def certify_and_expand(
    learner,
    mdp: ExactMDP,
    atoms: np.ndarray,
    *,
    tau: float = 1e-9,
    max_iterations: int = 200,
    max_body_literals: int = 3,
) -> ExpandResult:
    """Run EXPAND: accept edits iff exact ΔJ ≥ τ until no improvement or cap."""
    import solverrl_core

    evaluator = mdp.cpp_evaluator()
    loop = solverrl_core.ExpansionLoop(
        evaluator,
        np.ascontiguousarray(atoms, dtype=np.uint8),
        max_body_literals,
        float(tau),
        int(max_iterations),
    )
    result = loop.run(learner)
    learner.load_expansion_result(result)
    return ExpandResult(
        return_curve=[float(x) for x in result.return_curve],
        success_curve=[float(x) for x in result.success_curve],
        iterations=int(result.iterations),
        final_return=float(result.final_return),
        final_success=float(result.final_success),
        accepted_any_edit=bool(result.accepted_any_edit),
        n_clauses=int(learner.n_clauses),
        prolog=learner.to_prolog(),
    )


def rollout_proposal(proposal, states) -> np.ndarray:
    """Roll out one proposal to a per-state action vector."""
    atoms = ground_atoms(states)
    return np.asarray(proposal.rollout(atoms), dtype=np.int64)
