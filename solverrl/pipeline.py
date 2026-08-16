"""One-command KeyDoor pipeline: census → distill → EXPAND → cert/plot."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from stable_baselines3 import PPO

from solverrl.exact import ExactMDP
from solverrl.plot_expand import CurveSeries, plot_certified_return
from solverrl.run_r1 import (
    R1ExpandReport,
    print_report,
    run_r1_expand,
    write_report_artifacts,
)
from solverrl.teachers.census import (
    build_census,
    load_census,
    regenerate_r1_r2_censuses,
    save_census,
)
from solverrl.teachers.ppo import train_r1, train_r2


@dataclass(frozen=True)
class RegimePaths:
    regime: str
    census: Path
    ckpt: Path
    prolog: Path
    cert: Path
    curves: Path
    plot: Path


def regime_paths(out_dir: Path, regime: str, seed: int) -> RegimePaths:
    census_dir = out_dir / "census"
    return RegimePaths(
        regime=regime,
        census=census_dir / f"census_{regime}_seed{seed}.npz",
        ckpt=census_dir / "checkpoints" / f"ppo_keydoor_{regime}_seed{seed}.zip",
        prolog=out_dir / "rules" / f"{regime}_expanded.pl",
        cert=out_dir / "certs" / f"{regime}_advantage_gap.json",
        curves=out_dir / "plots" / f"{regime}_curves.npz",
        plot=out_dir / "plots" / f"{regime}_certified.png",
    )


def ensure_regime_artifacts(
    paths: RegimePaths,
    *,
    seed: int,
    train: bool,
    n_envs: int,
    mdp: ExactMDP,
) -> None:
    """Train/load teacher and write census if missing (or if train=True)."""
    need_train = train or not paths.ckpt.is_file()
    need_census = train or not paths.census.is_file()

    if not need_train and not need_census:
        print(f"[{paths.regime}] reuse ckpt={paths.ckpt} census={paths.census}")
        return

    paths.ckpt.parent.mkdir(parents=True, exist_ok=True)
    paths.census.parent.mkdir(parents=True, exist_ok=True)

    if need_train:
        print(f"[{paths.regime}] training teacher → {paths.ckpt}")
        if paths.regime == "r1":
            train_r1(save_path=paths.ckpt, n_envs=n_envs, seed=seed, mdp=mdp)
        elif paths.regime == "r2":
            train_r2(save_path=paths.ckpt, n_envs=n_envs, seed=seed)
        else:
            raise ValueError(f"unknown regime {paths.regime!r}")

    if need_census or need_train:
        print(f"[{paths.regime}] building census → {paths.census}")
        model = PPO.load(str(paths.ckpt))
        census = build_census(model, regime=paths.regime, seed=seed, mdp=mdp)
        save_census(census, paths.census)


def run_regime(
    paths: RegimePaths,
    *,
    mdp: ExactMDP,
    tau: float,
    max_iterations: int,
    max_body_literals: int,
) -> R1ExpandReport:
    print(f"\n=== {paths.regime.upper()} distill + EXPAND ===")
    census = load_census(paths.census)
    report = run_r1_expand(
        census,
        teacher_ckpt=paths.ckpt,
        mdp=mdp,
        tau=tau,
        max_iterations=max_iterations,
        max_body_literals=max_body_literals,
    )
    print_report(report)
    write_report_artifacts(
        report,
        regime=paths.regime,
        label=paths.regime.upper(),
        out_prolog=paths.prolog,
        out_cert=paths.cert,
        out_curves=paths.curves,
        out_plot=paths.plot,
    )
    return report


def write_summary(
    out_path: Path,
    reports: Dict[str, R1ExpandReport],
) -> Path:
    rows = []
    for regime, report in reports.items():
        cert = report.advantage_gap
        rows.append(
            {
                "regime": regime,
                "distill_fidelity": report.distill_fidelity,
                "initial_return": report.initial_return,
                "final_return": report.final_return,
                "initial_success": report.initial_success,
                "final_success": report.final_success,
                "expand_iterations": report.expand.iterations,
                "n_clauses": report.expand.n_clauses,
                "reached_success_one": report.reached_success_one,
                "teacher_return": report.teacher_return,
                "teacher_success": report.teacher_success,
                "advantage_gap": None
                if cert is None
                else {
                    "return_gap": cert.return_gap,
                    "weighted_gap": cert.weighted_gap,
                    "max_gap": cert.max_gap,
                    "n_disagree": cert.n_disagree,
                    "non_vacuous": cert.non_vacuous,
                },
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"regimes": rows}, indent=2) + "\n", encoding="utf-8")
    return out_path


def run_pipeline(
    *,
    regimes: Sequence[str] = ("r1", "r2"),
    seed: int = 0,
    out_dir: Path | str = "data",
    train: bool = False,
    train_both_together: bool = False,
    n_envs: int = 8,
    tau: float = 1e-9,
    max_iterations: int = 50,
    max_body_literals: int = 3,
    mdp: Optional[ExactMDP] = None,
) -> Dict[str, R1ExpandReport]:
    """census → distill → EXPAND → cert/plot for each regime; combined comparison plot."""
    out_dir = Path(out_dir)
    if mdp is None:
        mdp = ExactMDP.from_keydoor()

    regime_list = [r.lower() for r in regimes]
    for r in regime_list:
        if r not in {"r1", "r2"}:
            raise ValueError(f"unsupported regime {r!r}; expected r1 and/or r2")

    if train_both_together and set(regime_list) == {"r1", "r2"}:
        print("=== training R1+R2 teachers and censuses ===")
        regenerate_r1_r2_censuses(out_dir=out_dir / "census", n_envs=n_envs, seed=seed, mdp=mdp)
    else:
        for regime in regime_list:
            ensure_regime_artifacts(
                regime_paths(out_dir, regime, seed),
                seed=seed,
                train=train,
                n_envs=n_envs,
                mdp=mdp,
            )

    reports: Dict[str, R1ExpandReport] = {}
    series: List[CurveSeries] = []
    for regime in regime_list:
        paths = regime_paths(out_dir, regime, seed)
        report = run_regime(
            paths,
            mdp=mdp,
            tau=tau,
            max_iterations=max_iterations,
            max_body_literals=max_body_literals,
        )
        reports[regime] = report
        series.append(
            CurveSeries(
                label=regime.upper(),
                return_curve=report.expand.return_curve,
                success_curve=report.expand.success_curve,
                teacher_success=report.teacher_success,
            )
        )

    summary_path = write_summary(out_dir / "plots" / f"pipeline_summary_seed{seed}.json", reports)
    print(f"wrote_summary={summary_path}")

    if series:
        combo = plot_certified_return(
            series,
            out_dir / "plots" / f"certified_return_seed{seed}.png",
            title=f"KeyDoor certified EXPAND (seed{seed})",
        )
        print(f"wrote_comparison_plot={combo}")

    return reports


def main() -> None:
    p = argparse.ArgumentParser(
        description="One-command KeyDoor pipeline: census → distill → EXPAND → cert/plot"
    )
    p.add_argument(
        "--regimes",
        type=str,
        default="r1,r2",
        help="Comma-separated regimes: r1, r2, or r1,r2 (default)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="data", help="Root for census/rules/certs/plots")
    p.add_argument(
        "--train",
        action="store_true",
        help="Retrain teachers and rebuild censuses even if files exist",
    )
    p.add_argument(
        "--train-both",
        action="store_true",
        help="Use regenerate_r1_r2_censuses (both teachers) when regimes=r1,r2",
    )
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--tau", type=float, default=1e-9)
    p.add_argument("--max-iterations", type=int, default=50)
    p.add_argument("--max-literals", type=int, default=3)
    args = p.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    reports = run_pipeline(
        regimes=regimes,
        seed=args.seed,
        out_dir=args.out_dir,
        train=args.train,
        train_both_together=args.train_both,
        n_envs=args.n_envs,
        tau=args.tau,
        max_iterations=args.max_iterations,
        max_body_literals=args.max_literals,
    )

    print("\n=== pipeline done ===")
    for regime, report in reports.items():
        print(
            f"{regime}: success {report.initial_success:.4f}->{report.final_success:.4f}  "
            f"J {report.initial_return:.4f}->{report.final_return:.4f}  "
            f"iters={report.expand.iterations} reached_one={report.reached_success_one}"
        )


if __name__ == "__main__":
    main()
