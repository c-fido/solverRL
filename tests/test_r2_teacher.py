"""Week 2: R2 capped-budget PPO teacher."""

from __future__ import annotations

from pathlib import Path

from solverrl.teachers.ppo import R2_TIMESTEPS, train_r2


def test_r2_timesteps_match_paper():
    assert R2_TIMESTEPS == 299_008


def test_train_r2_saves_checkpoint(tmp_path: Path):
    out = tmp_path / "r2.zip"
    result = train_r2(
        save_path=out,
        n_envs=2,
        seed=0,
        total_timesteps=2048,  # short smoke; default remains paper budget
    )
    assert out.is_file()
    assert result.save_path == out
    assert result.timesteps == 2048
    assert result.capped is True
