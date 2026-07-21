"""Week 2 task 1: PPO teacher smoke — architecture + short train on KeyDoor."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from solverrl.teachers.ppo import build_ppo, train_ppo


def test_build_ppo_uses_tanh_two_layer_mlp():
    model = build_ppo(n_envs=2, seed=0)
    # SB3 stores activation on the policy features extractor / mlp extractor
    pi = model.policy
    assert pi.__class__.__name__ == "ActorCriticPolicy"
    # net_arch two hidden layers of 64 (paper-like small tanh MLP)
    assert model.policy.net_arch == dict(pi=[64, 64], vf=[64, 64])
    assert model.policy.activation_fn is nn.Tanh
    # flat one-hot observation
    assert model.observation_space.shape == (model.env.observation_space.shape[0],)
    assert model.observation_space.shape[0] > 1000  # full reachable one-hot


def test_train_ppo_smoke_saves_checkpoint(tmp_path: Path):
    out = tmp_path / "ppo_smoke.zip"
    path = train_ppo(
        total_timesteps=2048,
        n_envs=2,
        seed=0,
        save_path=out,
    )
    assert path == out
    assert out.is_file()
    assert out.stat().st_size > 0
