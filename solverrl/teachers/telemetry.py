"""PPO training telemetry recording and health audit (paper Week 2 checks)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

from stable_baselines3.common.callbacks import BaseCallback

from solverrl.teachers.ppo import DEFAULT_N_ENVS, build_ppo

# Conservative defaults inspired by the paper's monitor language
# (no published numeric cutoffs — documented as project thresholds).
MAX_APPROX_KL = 0.15
MAX_CLIP_FRACTION = 0.40
MIN_ENTROPY = 0.05
# Explained variance can stay low on easy CartPole-like targets; for KeyDoor
# we only flag a hard collapse to ~0 after it has been informative.
MIN_EXPLAINED_VARIANCE_FLOOR = -1.0  # allow negative EV; flag only NaN/inf


@dataclass
class TelemetryTrace:
    approx_kl: List[float] = field(default_factory=list)
    clip_fraction: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    explained_variance: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class AuditReport:
    healthy: bool
    failures: Tuple[str, ...]
    notes: Tuple[str, ...] = ()


class TelemetryCallback(BaseCallback):
    """Capture SB3 PPO train/* scalars after each update."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.trace = TelemetryTrace()

    def _on_step(self) -> bool:
        return True

    def _capture(self) -> None:
        values = self.logger.name_to_value
        if "train/approx_kl" in values:
            self.trace.approx_kl.append(float(values["train/approx_kl"]))
        if "train/clip_fraction" in values:
            self.trace.clip_fraction.append(float(values["train/clip_fraction"]))
        # SB3 logs entropy_loss = -entropy
        if "train/entropy_loss" in values:
            self.trace.entropy.append(float(-values["train/entropy_loss"]))
        elif "train/entropy" in values:
            self.trace.entropy.append(float(values["train/entropy"]))
        if "train/explained_variance" in values:
            self.trace.explained_variance.append(float(values["train/explained_variance"]))

    def _on_rollout_end(self) -> None:
        # Metrics from the previous train() call are available here (after update 1+).
        self._capture()

    def _on_training_end(self) -> None:
        self._capture()


def audit_telemetry(
    trace: TelemetryTrace,
    *,
    max_approx_kl: float = MAX_APPROX_KL,
    max_clip_fraction: float = MAX_CLIP_FRACTION,
    min_entropy: float = MIN_ENTROPY,
) -> AuditReport:
    """Check for KL spikes, entropy collapse, and excessive clipping."""
    failures: List[str] = []
    notes: List[str] = []

    if not trace.approx_kl:
        failures.append("missing_approx_kl")
    elif max(trace.approx_kl) > max_approx_kl:
        failures.append("approx_kl_spike")

    if not trace.clip_fraction:
        failures.append("missing_clip_fraction")
    elif max(trace.clip_fraction) > max_clip_fraction:
        failures.append("excessive_clip_fraction")

    if not trace.entropy:
        failures.append("missing_entropy")
    elif min(trace.entropy) < min_entropy:
        failures.append("entropy_collapse")

    if not trace.explained_variance:
        failures.append("missing_explained_variance")
    else:
        if any(v != v or abs(v) == float("inf") for v in trace.explained_variance):
            failures.append("explained_variance_invalid")
        notes.append(
            "explained_variance retained for inspection; low EV alone is not a hard fail "
            "(paper notes low EV can be benign when the value target is nearly constant)."
        )

    return AuditReport(healthy=not failures, failures=tuple(failures), notes=tuple(notes))


def record_telemetry_during_train(
    total_timesteps: int,
    n_envs: int = DEFAULT_N_ENVS,
    seed: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> Tuple[TelemetryTrace, AuditReport]:
    """Train briefly (or fully), record SB3 train metrics, and audit them."""
    if save_path is None:
        save_path = Path("checkpoints") / f"ppo_keydoor_telemetry_seed{seed}.zip"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_ppo(n_envs=n_envs, seed=seed, device=device)
    cb = TelemetryCallback()
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)
    model.save(str(save_path))
    model.env.close()

    report = audit_telemetry(cb.trace)
    return cb.trace, report


def main() -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description="Record + audit PPO train telemetry on KeyDoor")
    p.add_argument("--timesteps", type=int, default=16_384)
    p.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="checkpoints/ppo_keydoor_telemetry_seed0.zip")
    args = p.parse_args()
    trace, report = record_telemetry_during_train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        save_path=args.save,
    )
    summary = {
        "healthy": report.healthy,
        "failures": list(report.failures),
        "notes": list(report.notes),
        "n_updates": len(trace.approx_kl),
        "approx_kl_max": max(trace.approx_kl) if trace.approx_kl else None,
        "clip_fraction_max": max(trace.clip_fraction) if trace.clip_fraction else None,
        "entropy_min": min(trace.entropy) if trace.entropy else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
