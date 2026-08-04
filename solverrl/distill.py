"""Distill greedy census labels into a Prolog decision list."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from solverrl.exact import ExactMDP
from solverrl.ground import ground_atoms
from solverrl.teachers.census import CensusDataset, load_census


@dataclass(frozen=True)
class DistillResult:
    fidelity: float
    n_clauses: int
    prolog: str
    regime: str
    n_states: int


def distill_from_census(
    census: CensusDataset,
    *,
    mdp: Optional[ExactMDP] = None,
    min_score: float = 1e-12,
    max_body_literals: int = 3,
) -> DistillResult:
    import solverrl_core

    if mdp is None:
        mdp = ExactMDP.from_keydoor()
    if census.n_states != mdp.n:
        raise ValueError(f"census n_states {census.n_states} != mdp.n {mdp.n}")

    atoms = ground_atoms(mdp.states)
    actions = np.asarray(census.actions, dtype=np.int64)

    learner = solverrl_core.RuleLearner.keydoor(min_score, max_body_literals)
    learner.fit(atoms, actions)
    fidelity = float(learner.fidelity(atoms, actions))

    return DistillResult(
        fidelity=fidelity,
        n_clauses=int(learner.n_clauses),
        prolog=learner.to_prolog(),
        regime=census.regime,
        n_states=census.n_states,
    )


def distill_census_file(
    path: Union[str, Path],
    *,
    mdp: Optional[ExactMDP] = None,
    out_prolog: Optional[Union[str, Path]] = None,
    **kwargs,
) -> DistillResult:
    census = load_census(path)
    result = distill_from_census(census, mdp=mdp, **kwargs)
    if out_prolog is not None:
        out = Path(out_prolog)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.prolog, encoding="utf-8")
    return result


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Distill census → Prolog decision list")
    p.add_argument("census", type=str, help="Path to census .npz")
    p.add_argument("--out", type=str, default=None, help="Write .pl here")
    p.add_argument("--max-literals", type=int, default=3)
    args = p.parse_args()

    result = distill_census_file(
        args.census,
        out_prolog=args.out,
        max_body_literals=args.max_literals,
    )
    print(f"regime={result.regime} fidelity={result.fidelity:.4f} clauses={result.n_clauses}")
    if args.out is None:
        print(result.prolog[:500] + ("..." if len(result.prolog) > 500 else ""))


if __name__ == "__main__":
    main()
