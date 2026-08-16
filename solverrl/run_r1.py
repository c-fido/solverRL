"""Run KeyDoor R1/R2: distill + EXPAND under exact return."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from stable_baselines3 import PPO

from solverrl.distill import distill_from_census
from solverrl.exact import ExactMDP, success_within_horizon
from solverrl.expand import ExpandResult, certify_and_expand
from solverrl.ground import ground_atoms
from solverrl.plot_expand import CurveSeries, plot_certified_return, save_curves
from solverrl.teachers.census import CensusDataset, load_census
from solverrl.teachers.ppo import greedy_policy_from_model


@dataclass(frozen=True)
class AdvantageGapReport:
    weighted_gap: float
    return_gap: float
    n_disagree: int
    max_gap: float

    @property
    def non_vacuous(self) -> bool:
        """True if student and teacher disagree on at least one state."""
        return self.n_disagree > 0


@dataclass(frozen=True)
class R1ExpandReport:
    distill_fidelity: float
    initial_return: float
    initial_success: float
    final_return: float
    final_success: float
    teacher_success: float
    teacher_return: float
    expand: ExpandResult
    reached_success_one: bool
    advantage_gap: Optional[AdvantageGapReport]


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
        actions,
        tau=tau,
        max_iterations=max_iterations,
        max_body_literals=max_body_literals,
    )

    student_policy = np.asarray(learner.predict(atoms), dtype=np.int64)
    teacher_policy = actions
    teacher_return = float(evaluator.exact_return(teacher_policy))
    teacher_success = float(evaluator.success(teacher_policy))

    # Optional ckpt reload: keep success from live greedy if provided (should match census).
    if teacher_ckpt is not None:
        model = PPO.load(str(teacher_ckpt))
        teacher_policy = greedy_policy_from_model(model, mdp)
        teacher_return = float(evaluator.exact_return(teacher_policy))
        teacher_success = float(success_within_horizon(mdp, teacher_policy))

    cert_dict = mdp.advantage_gap_cert(student=student_policy, teacher=teacher_policy)
    advantage_gap = AdvantageGapReport(
        weighted_gap=cert_dict["weighted_gap"],
        return_gap=cert_dict["return_gap"],
        n_disagree=cert_dict["n_disagree"],
        max_gap=cert_dict["max_gap"],
    )

    return R1ExpandReport(
        distill_fidelity=distilled.fidelity,
        initial_return=initial_return,
        initial_success=initial_success,
        final_return=expanded.final_return,
        final_success=expanded.final_success,
        teacher_success=teacher_success,
        teacher_return=teacher_return,
        expand=expanded,
        reached_success_one=expanded.final_success >= 1.0 - 1e-9,
        advantage_gap=advantage_gap,
    )


def print_report(report: R1ExpandReport) -> None:
    print(f"distill_fidelity={report.distill_fidelity:.4f}")
    print(
        f"return {report.initial_return:.6f} -> {report.final_return:.6f}  "
        f"success {report.initial_success:.6f} -> {report.final_success:.6f}"
    )
    print(f"expand_iterations={report.expand.iterations} clauses={report.expand.n_clauses}")
    print(f"reached_success_one={report.reached_success_one}")
    print(f"teacher_return={report.teacher_return:.6f} teacher_success={report.teacher_success:.6f}")
    if report.advantage_gap is not None:
        cert = report.advantage_gap
        print(
            f"advantage_gap return_gap={cert.return_gap:.6f} "
            f"weighted_gap={cert.weighted_gap:.6f} max_gap={cert.max_gap:.6f} "
            f"n_disagree={cert.n_disagree} non_vacuous={cert.non_vacuous}"
        )


def write_report_artifacts(
    report: R1ExpandReport,
    *,
    regime: str,
    label: str,
    out_prolog: Optional[Union[str, Path]] = None,
    out_cert: Optional[Union[str, Path]] = None,
    out_curves: Optional[Union[str, Path]] = None,
    out_plot: Optional[Union[str, Path]] = None,
) -> dict[str, Path]:
    """Write optional prolog / cert / curves / plot artifacts. Returns paths written."""
    written: dict[str, Path] = {}

    if out_prolog is not None:
        path = Path(out_prolog)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.expand.prolog, encoding="utf-8")
        written["prolog"] = path
        print(f"wrote_prolog={path}")

    if out_cert is not None and report.advantage_gap is not None:
        path = Path(out_cert)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **asdict(report.advantage_gap),
            "non_vacuous": report.advantage_gap.non_vacuous,
            "student_return": report.final_return,
            "student_success": report.final_success,
            "teacher_return": report.teacher_return,
            "teacher_success": report.teacher_success,
            "regime": regime,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written["cert"] = path
        print(f"wrote_cert={path}")

    if out_curves is not None:
        path = save_curves(
            out_curves,
            label=label,
            return_curve=report.expand.return_curve,
            success_curve=report.expand.success_curve,
            teacher_success=report.teacher_success,
        )
        written["curves"] = path
        print(f"wrote_curves={path}")

    if out_plot is not None:
        path = plot_certified_return(
            [
                CurveSeries(
                    label=label,
                    return_curve=report.expand.return_curve,
                    success_curve=report.expand.success_curve,
                    teacher_success=report.teacher_success,
                )
            ],
            out_plot,
            title=f"Certified EXPAND ({label})",
        )
        written["plot"] = path
        print(f"wrote_plot={path}")

    return written


def main() -> None:
    p = argparse.ArgumentParser(description="KeyDoor distill + EXPAND (+ advantage-gap cert)")
    p.add_argument("census", type=str, help="Path to census .npz")
    p.add_argument("--teacher-ckpt", type=str, default=None)
    p.add_argument("--out-prolog", type=str, default=None)
    p.add_argument("--out-plot", type=str, default=None, help="PNG certified-return/success plot")
    p.add_argument("--out-curves", type=str, default=None, help="Save curves .npz for later plots")
    p.add_argument("--out-cert", type=str, default=None, help="Write advantage-gap cert JSON")
    p.add_argument("--plot-label", type=str, default=None, help="Legend label for --out-plot")
    p.add_argument("--tau", type=float, default=1e-9)
    p.add_argument("--max-iterations", type=int, default=200)
    p.add_argument("--max-literals", type=int, default=3)
    args = p.parse_args()

    census = load_census(args.census)
    report = run_r1_expand(
        census,
        teacher_ckpt=args.teacher_ckpt,
        tau=args.tau,
        max_iterations=args.max_iterations,
        max_body_literals=args.max_literals,
    )

    print_report(report)
    write_report_artifacts(
        report,
        regime=census.regime,
        label=args.plot_label or census.regime.upper(),
        out_prolog=args.out_prolog,
        out_cert=args.out_cert,
        out_curves=args.out_curves,
        out_plot=args.out_plot,
    )


if __name__ == "__main__":
    main()
