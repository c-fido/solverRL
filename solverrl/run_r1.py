"""Run KeyDoor R1: distill + EXPAND under exact return."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from stable_baselines3 import PPO

from solverrl.distill import distill_from_census
from solverrl.exact import ExactMDP, success_within_horizon
from solverrl.expand import ExpandResult, certify_and_expand
from solverrl.ground import ground_atoms
from solverrl.teachers.census import CensusDataset, load_census
from solverrl.teachers.ppo import greedy_policy_from_model


@dataclass(frozen=True)
class R1ExpandReport:
    distill_fidelity: float
    initial_return: float
    initial_success: float
    final_return: float
    final_success: float
    teacher_success: float
    expand: ExpandResult
    reached_success_one: bool


def run_r1_expand(
    census: CensusDataset,
    *,
    teacher_ckpt: Optional[Union[str, Path]] = None,
    mdp: Optional[ExactMDP] = None,
    tau: float = 1e-9,
    max_iterations: int = 200,
    max_body_literals: int = 3,
) -> R1ExpandReport:
    import solverrl_core

    if mdp is None:
        mdp = ExactMDP.from_keydoor()

    distilled = distill_from_census(census, mdp=mdp, max_body_literals=max_body_literals)
    atoms = ground_atoms(mdp.states)
    actions = np.asarray(census.actions, dtype=np.int64)

    learner = solverrl_core.RuleLearner.keydoor(max_body_literals=max_body_literals)
    learner.fit(atoms, actions)

    evaluator = mdp.cpp_evaluator()
    initial_policy = np.asarray(learner.predict(atoms), dtype=np.int64)
    initial_return = float(evaluator.exact_return(initial_policy))
    initial_success = float(evaluator.success(initial_policy))

    expanded = certify_and_expand(
        learner,
        mdp,
        atoms,
        tau=tau,
        max_iterations=max_iterations,
        max_body_literals=max_body_literals,
    )

    teacher_success = float("nan")
    if teacher_ckpt is not None:
        model = PPO.load(str(teacher_ckpt))
        teacher_pi = greedy_policy_from_model(model, mdp)
        teacher_success = float(success_within_horizon(mdp, teacher_pi))

    return R1ExpandReport(
        distill_fidelity=distilled.fidelity,
        initial_return=initial_return,
        initial_success=initial_success,
        final_return=expanded.final_return,
        final_success=expanded.final_success,
        teacher_success=teacher_success,
        expand=expanded,
        reached_success_one=expanded.final_success >= 1.0 - 1e-9,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="KeyDoor R1 distill + EXPAND")
    p.add_argument("census", type=str, help="Path to census_r1 .npz")
    p.add_argument("--teacher-ckpt", type=str, default=None)
    p.add_argument("--out-prolog", type=str, default=None)
    p.add_argument("--tau", type=float, default=1e-9)
    p.add_argument("--max-iterations", type=int, default=200)
    p.add_argument("--max-literals", type=int, default=3)
    args = p.parse_args()

    report = run_r1_expand(
        load_census(args.census),
        teacher_ckpt=args.teacher_ckpt,
        tau=args.tau,
        max_iterations=args.max_iterations,
        max_body_literals=args.max_literals,
    )

    print(f"distill_fidelity={report.distill_fidelity:.4f}")
    print(
        f"return {report.initial_return:.6f} -> {report.final_return:.6f}  "
        f"success {report.initial_success:.6f} -> {report.final_success:.6f}"
    )
    print(f"expand_iterations={report.expand.iterations} clauses={report.expand.n_clauses}")
    print(f"reached_success_one={report.reached_success_one}")
    if not np.isnan(report.teacher_success):
        print(f"teacher_success={report.teacher_success:.6f}")

    if args.out_prolog:
        Path(args.out_prolog).write_text(report.expand.prolog, encoding="utf-8")


if __name__ == "__main__":
    main()
