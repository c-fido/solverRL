"""PPO teacher on KeyDoor (Week 2).

Paper-like setup: flat one-hot observations, two-layer tanh actor-critic,
eight synchronous environments by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from solverrl.envs.keydoor import KeyDoorEnv

# Exposed for tests / callers that want the paper defaults.
DEFAULT_N_ENVS = 8
DEFAULT_NET_ARCH = dict(pi=[64, 64], vf=[64, 64])
DEFAULT_ACTIVATION = nn.Tanh


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
    """Short default train so `python -m solverrl.teachers.ppo` is runnable."""
    import argparse

    p = argparse.ArgumentParser(description="Train PPO on KeyDoor (Week 2 task 1)")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="checkpoints/ppo_keydoor_seed0.zip")
    args = p.parse_args()
    path = train_ppo(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        save_path=args.save,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
