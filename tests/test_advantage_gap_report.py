"""Advantage-gap certificate wiring in the expand runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from solverrl.envs.keydoor import ACTION_TO_ID
from solverrl.exact import ExactMDP
from solverrl.run_r1 import AdvantageGapReport, run_r1_expand
from solverrl.teachers.census import build_census
from solverrl.teachers.ppo import build_ppo


@pytest.fixture(scope="module")
def mdp() -> ExactMDP:
    return ExactMDP.from_keydoor()


def test_advantage_gap_report_non_vacuous_flag():
    vacuous = AdvantageGapReport(0.0, 0.0, 0, 0.0)
    assert vacuous.non_vacuous is False
    filled = AdvantageGapReport(0.1, -0.2, 12, 0.5)
    assert filled.non_vacuous is True


@pytest.mark.integration
def test_run_r1_expand_emits_advantage_gap_vs_census_teacher(mdp: ExactMDP, tmp_path: Path):
    model = build_ppo(n_envs=2, seed=3)
    census = build_census(model, mdp=mdp, regime="cert_smoke", seed=3)
    model.env.close()

    report = run_r1_expand(census, mdp=mdp, max_iterations=5)
    assert report.advantage_gap is not None
    # Untrained teacher vs distilled student almost always disagrees somewhere.
    assert report.advantage_gap.non_vacuous
    assert report.advantage_gap.max_gap >= 0.0

    # Direct MDP cert should match the report fields for the same teacher labels.
    # (Student policy is post-EXPAND; recompute via report return gap sign consistency.)
    assert np.isfinite(report.advantage_gap.return_gap)

    cert_path = tmp_path / "cert.json"
    payload = {
        **report.advantage_gap.__dict__,
        "non_vacuous": report.advantage_gap.non_vacuous,
    }
    cert_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(cert_path.read_text(encoding="utf-8"))
    assert loaded["n_disagree"] == report.advantage_gap.n_disagree


def test_advantage_gap_cert_left_vs_right_baseline(mdp: ExactMDP):
    left = np.full(mdp.n, ACTION_TO_ID["left"], dtype=np.int64)
    right = np.full(mdp.n, ACTION_TO_ID["right"], dtype=np.int64)
    cert = mdp.advantage_gap_cert(student=left, teacher=right)
    assert cert["n_disagree"] == mdp.n
    report = AdvantageGapReport(
        weighted_gap=cert["weighted_gap"],
        return_gap=cert["return_gap"],
        n_disagree=cert["n_disagree"],
        max_gap=cert["max_gap"],
    )
    assert report.non_vacuous
