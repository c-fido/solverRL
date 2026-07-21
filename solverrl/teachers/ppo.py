"""PPO teacher on KeyDoor (Week 2).

Paper-like setup: flat one-hot observations, two-layer tanh actor-critic,
eight synchronous environments by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from solverrl.envs.keydoor import KeyDoorEnv
from solverrl.exact import ExactMDP, success_within_horizon

# Exposed for tests / callers that want the paper defaults.
DEFAULT_N_ENVS = 8
DEFAULT_NET_ARCH = dict(pi=[64, 64], vf=[64, 64])
DEFAULT_ACTIVATION = nn.Tanh
# Paper: early-stop when greedy exact success ≥ 0.95; train up to ~800k.
R1_SUCCESS_THRESHOLD = 0.95
R1_MAX_TIMESTEPS = 800_000


def make_keydoor_vec(n_envs: int = DEFAULT_N_ENVS, seed: int = 0) -> DummyVecEnv:
    return make_vec_env(KeyDoorEnv, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)


def build_ppo(
    n_envs: int = DEFAULT_N_ENVS,
    seed: int = 0,
    device: str = "cpu",
) -> PPO:
    env = make_keydoor_vec(n_envs=n_envs, seed=seed)
    policy_kwargs = dict(
        net_arch=DEFAULT_NET_ARCH,
        activation_fn=DEFAULT_ACTIVATION,
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        verbose=0,
    )
    # Keep architecture knobs on the instance for tests / logging.
    model.policy.net_arch = DEFAULT_NET_ARCH  # type: ignore[attr-defined]
    model.policy.activation_fn = DEFAULT_ACTIVATION  # type: ignore[attr-defined]
    return model


def greedy_policy_from_model(model: PPO, mdp: ExactMDP) -> np.ndarray:
    """Argmax (deterministic) action at every reachable state, via one-hot obs."""
    obs_batch = np.eye(mdp.n, dtype=np.float32)
    actions, _ = model.predict(obs_batch, deterministic=True)
    return np.asarray(actions, dtype=np.int64).reshape(mdp.n)


def greedy_exact_success(model: PPO, mdp: ExactMDP, horizon: Optional[int] = None) -> float:
    policy = greedy_policy_from_model(model, mdp)
    if horizon is None:
        return success_within_horizon(mdp, policy)
    return success_within_horizon(mdp, policy, horizon=horizon)


@dataclass(frozen=True)
class TrainR1Result:
    save_path: Path
    timesteps: int
    greedy_success: float
    stopped_early: bool


def train_r1(
    save_path: Optional[Union[str, Path]] = None,
    n_envs: int = DEFAULT_N_ENVS,
    seed: int = 0,
    success_threshold: float = R1_SUCCESS_THRESHOLD,
    max_timesteps: int = R1_MAX_TIMESTEPS,
    chunk_timesteps: int = 16_384,
    device: str = "cpu",
    mdp: Optional[ExactMDP] = None,
) -> TrainR1Result:
    """Train until greedy exact success ≥ threshold (paper R1) or hit max steps."""
    if save_path is None:
        save_path = Path("checkpoints") / f"ppo_keydoor_r1_seed{seed}.zip"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if mdp is None:
        mdp = ExactMDP.from_keydoor()

    model = build_ppo(n_envs=n_envs, seed=seed, device=device)
    trained = 0
    last_success = 0.0
    stopped_early = False

    while trained < max_timesteps:
        chunk = min(chunk_timesteps, max_timesteps - trained)
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=(trained == 0),
            progress_bar=False,
        )
        trained += chunk
        last_success = greedy_exact_success(model, mdp)
        if last_success >= success_threshold:
            stopped_early = True
            break

    model.save(str(save_path))
    model.env.close()
    return TrainR1Result(
        save_path=save_path,
        timesteps=trained,
        greedy_success=last_success,
        stopped_early=stopped_early,
    )


def train_ppo(
    total_timesteps: int,
    n_envs: int = DEFAULT_N_ENVS,
    seed: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> Path:
    """Train PPO on KeyDoor and optionally save a Stable-Baselines3 zip."""
    model = build_ppo(n_envs=n_envs, seed=seed, device=device)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    if save_path is None:
        save_path = Path("checkpoints") / f"ppo_keydoor_seed{seed}.zip"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    model.env.close()
    return save_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train PPO on KeyDoor")
    sub = p.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser("smoke", help="Fixed-timestep train (Week 2 task 1)")
    smoke.add_argument("--timesteps", type=int, default=50_000)
    smoke.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--save", type=str, default="checkpoints/ppo_keydoor_seed0.zip")

    r1 = sub.add_parser("r1", help="R1: early-stop on greedy exact success ≥ 0.95")
    r1.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    r1.add_argument("--seed", type=int, default=0)
    r1.add_argument("--save", type=str, default="checkpoints/ppo_keydoor_r1_seed0.zip")
    r1.add_argument("--threshold", type=float, default=R1_SUCCESS_THRESHOLD)
    r1.add_argument("--max-timesteps", type=int, default=R1_MAX_TIMESTEPS)
    r1.add_argument("--chunk-timesteps", type=int, default=16_384)

    args = p.parse_args()
    if args.cmd == "smoke":
        path = train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            save_path=args.save,
        )
        print(f"saved {path}")
    elif args.cmd == "r1":
        result = train_r1(
            save_path=args.save,
            n_envs=args.n_envs,
            seed=args.seed,
            success_threshold=args.threshold,
            max_timesteps=args.max_timesteps,
            chunk_timesteps=args.chunk_timesteps,
        )
        print(
            f"saved {result.save_path}  timesteps={result.timesteps}  "
            f"greedy_success={result.greedy_success:.4f}  early_stop={result.stopped_early}"
        )


if __name__ == "__main__":
    main()
