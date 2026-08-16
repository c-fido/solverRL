"""Certified EXPAND plot helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from solverrl.plot_expand import (
    CurveSeries,
    assert_monotone_nondecreasing,
    load_curves,
    plot_certified_return,
    save_curves,
)


def test_assert_monotone_accepts_flat_and_increasing():
    assert_monotone_nondecreasing([0.8])
    assert_monotone_nondecreasing([-1.0, 0.2, 0.8])
    assert_monotone_nondecreasing([0.5, 0.5])


def test_assert_monotone_rejects_decrease():
    with pytest.raises(ValueError):
        assert_monotone_nondecreasing([1.0, 0.5])


def test_plot_and_curves_roundtrip(tmp_path: Path):
    series = CurveSeries(
        label="R2",
        return_curve=[-1.0, 0.815778],
        success_curve=[0.0, 1.0],
        teacher_success=0.22,
    )
    curves_path = save_curves(
        tmp_path / "r2.npz",
        label=series.label,
        return_curve=series.return_curve,
        success_curve=series.success_curve,
        teacher_success=series.teacher_success or float("nan"),
    )
    loaded = load_curves(curves_path)
    assert loaded.label == "R2"
    assert loaded.return_curve == pytest.approx(series.return_curve)
    assert loaded.success_curve == pytest.approx(series.success_curve)
    assert loaded.teacher_success == pytest.approx(0.22)

    png = plot_certified_return(
        [series, CurveSeries("R1", [0.815778], [1.0], teacher_success=0.956)],
        tmp_path / "certified.png",
        title="test",
    )
    assert png.is_file()
    assert png.stat().st_size > 0
