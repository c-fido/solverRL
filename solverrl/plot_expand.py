"""Plot certified EXPAND return / success curves."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class CurveSeries:
    label: str
    return_curve: list[float]
    success_curve: list[float]
    teacher_success: Optional[float] = None


def assert_monotone_nondecreasing(values: Sequence[float], *, atol: float = 1e-12) -> None:
    for i in range(len(values) - 1):
        if values[i + 1] + atol < values[i]:
            raise ValueError(
                f"curve not monotone non-decreasing at index {i}: "
                f"{values[i]} -> {values[i + 1]}"
            )


def plot_certified_return(
    series: Sequence[CurveSeries],
    out_path: Union[str, Path],
    *,
    title: str = "Certified EXPAND curves",
    verify_monotone: bool = True,
) -> Path:
    """Save a 2-panel figure: exact return J and success vs EXPAND iteration."""
    if not series:
        raise ValueError("plot_certified_return: empty series")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if verify_monotone:
        for s in series:
            assert_monotone_nondecreasing(s.return_curve)
            assert_monotone_nondecreasing(s.success_curve)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)

    for s in series:
        xs = list(range(len(s.return_curve)))
        axes[0].plot(xs, s.return_curve, marker="o", label=s.label)
        axes[1].plot(xs, s.success_curve, marker="o", label=s.label)
        if s.teacher_success is not None and np.isfinite(s.teacher_success):
            axes[1].axhline(
                s.teacher_success,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"{s.label} teacher",
            )

    axes[0].set_xlabel("EXPAND iteration")
    axes[0].set_ylabel("Exact return J")
    axes[0].set_title("Certified return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("EXPAND iteration")
    axes[1].set_ylabel("Exact success")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Certified success")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_curves(
    path: Union[str, Path],
    *,
    label: str,
    return_curve: Sequence[float],
    success_curve: Sequence[float],
    teacher_success: float = float("nan"),
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        label=np.asarray(label),
        return_curve=np.asarray(return_curve, dtype=np.float64),
        success_curve=np.asarray(success_curve, dtype=np.float64),
        teacher_success=np.asarray(teacher_success, dtype=np.float64),
    )
    return path


def load_curves(path: Union[str, Path]) -> CurveSeries:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    teacher = float(data["teacher_success"]) if "teacher_success" in data.files else float("nan")
    return CurveSeries(
        label=str(data["label"]),
        return_curve=[float(x) for x in data["return_curve"]],
        success_curve=[float(x) for x in data["success_curve"]],
        teacher_success=None if not np.isfinite(teacher) else teacher,
    )


def _parse_series_arg(spec: str) -> CurveSeries:
    """Parse LABEL:j0,j1,...:s0,s1,...[:teacher_success]."""
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "series must be LABEL:j0,j1,...:s0,s1,...[:teacher_success]"
        )
    label, j_part, s_part = parts[0], parts[1], parts[2]
    returns = [float(x) for x in j_part.split(",") if x != ""]
    success = [float(x) for x in s_part.split(",") if x != ""]
    if len(returns) != len(success) or not returns:
        raise argparse.ArgumentTypeError("return and success curves must be non-empty and aligned")
    teacher = float(parts[3]) if len(parts) == 4 else None
    return CurveSeries(label=label, return_curve=returns, success_curve=success, teacher_success=teacher)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot certified EXPAND return/success curves")
    p.add_argument(
        "curves",
        nargs="*",
        type=str,
        help="Optional .npz curve files from --out-curves",
    )
    p.add_argument(
        "--series",
        action="append",
        type=_parse_series_arg,
        default=[],
        help="Inline series LABEL:j0,j1,...:s0,s1,...[:teacher_success]",
    )
    p.add_argument("--out", type=str, required=True, help="Output PNG path")
    p.add_argument("--title", type=str, default="Certified EXPAND curves")
    args = p.parse_args()

    series: list[CurveSeries] = list(args.series)
    for path in args.curves:
        series.append(load_curves(path))
    if not series:
        raise SystemExit("provide at least one --series or curves .npz file")

    out = plot_certified_return(series, args.out, title=args.title)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
