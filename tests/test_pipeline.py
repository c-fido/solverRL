"""One-command KeyDoor pipeline helpers."""

from __future__ import annotations

from pathlib import Path

from solverrl.expand import ExpandResult
from solverrl.pipeline import regime_paths, write_summary
from solverrl.run_r1 import AdvantageGapReport, R1ExpandReport


def test_regime_paths_layout(tmp_path: Path):
    paths = regime_paths(tmp_path, "r2", seed=0)
    assert paths.census.name == "census_r2_seed0.npz"
    assert paths.ckpt.name == "ppo_keydoor_r2_seed0.zip"
    assert paths.prolog.name == "r2_expanded.pl"
    assert paths.cert.name == "r2_advantage_gap.json"
    assert "plots" in str(paths.plot)


def test_write_summary_json(tmp_path: Path):
    expand = ExpandResult(
        return_curve=[-1.0, 0.8],
        success_curve=[0.0, 1.0],
        iterations=1,
        final_return=0.8,
        final_success=1.0,
        accepted_any_edit=True,
        n_clauses=10,
        prolog="% test",
    )
    report = R1ExpandReport(
        distill_fidelity=0.3,
        initial_return=-1.0,
        initial_success=0.0,
        final_return=0.8,
        final_success=1.0,
        teacher_success=0.2,
        teacher_return=-0.5,
        expand=expand,
        reached_success_one=True,
        advantage_gap=AdvantageGapReport(
            weighted_gap=-0.1, return_gap=-1.3, n_disagree=100, max_gap=0.4
        ),
    )
    out = write_summary(tmp_path / "summary.json", {"r2": report})
    text = out.read_text(encoding="utf-8")
    assert "r2" in text
    assert "reached_success_one" in text
    assert "non_vacuous" in text
