"""Reachable-state census: greedy teacher labels for every s ∈ S_r."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from stable_baselines3 import PPO

from solverrl.envs.keydoor import env_source_sha256
from solverrl.exact import ExactMDP
from solverrl.teachers.ppo import (
    R1_MAX_TIMESTEPS,
    R1_SUCCESS_THRESHOLD,
    R2_TIMESTEPS,
    greedy_policy_from_model,
    train_r1,
    train_r2,
)


@dataclass(frozen=True)
class CensusDataset:
    actions: np.ndarray  # (n_states,) int64 greedy labels
    regime: str
    seed: int
    env_sha256: str
    n_states: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", np.asarray(self.actions, dtype=np.int64))
        if self.actions.shape != (self.n_states,):
            raise ValueError(
                f"actions shape {self.actions.shape} != (n_states,)={(self.n_states,)}"
            )


def build_census(
    model: PPO,
    *,
    regime: str,
    seed: int,
    mdp: Optional[ExactMDP] = None,
) -> CensusDataset:
    """Label every reachable state with the model's deterministic greedy action."""
    if mdp is None:
        mdp = ExactMDP.from_keydoor()
    actions = greedy_policy_from_model(model, mdp)
    return CensusDataset(
        actions=actions,
        regime=regime,
        seed=seed,
        env_sha256=env_source_sha256(),
        n_states=mdp.n,
    )


def save_census(census: CensusDataset, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        actions=census.actions,
        regime=np.asarray(census.regime),
        seed=np.asarray(census.seed, dtype=np.int64),
        env_sha256=np.asarray(census.env_sha256),
        n_states=np.asarray(census.n_states, dtype=np.int64),
    )
    return path


def load_census(path: Union[str, Path]) -> CensusDataset:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    return CensusDataset(
        actions=data["actions"],
        regime=str(data["regime"]),
        seed=int(data["seed"]),
        env_sha256=str(data["env_sha256"]),
        n_states=int(data["n_states"]),
    )


def regenerate_r1_r2_censuses(
    out_dir: Union[str, Path] = "data/census",
    *,
    n_envs: int = 8,
    seed: int = 0,
    r1_success_threshold: float = R1_SUCCESS_THRESHOLD,
    r1_max_timesteps: int = R1_MAX_TIMESTEPS,
    r1_chunk_timesteps: int = 16_384,
    r2_timesteps: int = R2_TIMESTEPS,
    mdp: Optional[ExactMDP] = None,
) -> Dict[str, Path]:
    """One-shot: train R1 + R2 teachers and write greedy censuses for both."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mdp is None:
        mdp = ExactMDP.from_keydoor()

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    r1 = train_r1(
        save_path=ckpt_dir / f"ppo_keydoor_r1_seed{seed}.zip",
        n_envs=n_envs,
        seed=seed,
        success_threshold=r1_success_threshold,
        max_timesteps=r1_max_timesteps,
        chunk_timesteps=r1_chunk_timesteps,
        mdp=mdp,
    )
    r1_model = PPO.load(str(r1.save_path))
    r1_census = build_census(r1_model, regime="r1", seed=seed, mdp=mdp)
    r1_path = save_census(r1_census, out_dir / f"census_r1_seed{seed}.npz")

    r2 = train_r2(
        save_path=ckpt_dir / f"ppo_keydoor_r2_seed{seed}.zip",
        n_envs=n_envs,
        seed=seed,
        total_timesteps=r2_timesteps,
    )
    r2_model = PPO.load(str(r2.save_path))
    r2_census = build_census(r2_model, regime="r2", seed=seed, mdp=mdp)
    r2_path = save_census(r2_census, out_dir / f"census_r2_seed{seed}.npz")

    return {"r1": r1_path, "r2": r2_path, "r1_ckpt": r1.save_path, "r2_ckpt": r2.save_path}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Regenerate KeyDoor R1/R2 greedy censuses (Week 2 one-command)"
    )
    p.add_argument("--out-dir", type=str, default="data/census")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--r1-threshold", type=float, default=R1_SUCCESS_THRESHOLD)
    p.add_argument("--r1-max-timesteps", type=int, default=R1_MAX_TIMESTEPS)
    p.add_argument("--r1-chunk-timesteps", type=int, default=16_384)
    p.add_argument("--r2-timesteps", type=int, default=R2_TIMESTEPS)
    args = p.parse_args()
    paths = regenerate_r1_r2_censuses(
        out_dir=args.out_dir,
        n_envs=args.n_envs,
        seed=args.seed,
        r1_success_threshold=args.r1_threshold,
        r1_max_timesteps=args.r1_max_timesteps,
        r1_chunk_timesteps=args.r1_chunk_timesteps,
        r2_timesteps=args.r2_timesteps,
    )
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
