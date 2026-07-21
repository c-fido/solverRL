"""Week 2: PPO training telemetry audit (KL, entropy, clip, explained variance)."""

from __future__ import annotations

from pathlib import Path

import pytest

from solverrl.teachers.telemetry import (
    TelemetryTrace,
    audit_telemetry,
    record_telemetry_during_train,
)


def test_audit_passes_on_healthy_trace():
    trace = TelemetryTrace(
        approx_kl=[0.01, 0.02, 0.015],
        clip_fraction=[0.1, 0.12, 0.08],
        entropy=[0.5, 0.48, 0.47],
        explained_variance=[0.2, 0.3, 0.25],
    )
    report = audit_telemetry(trace)
    assert report.healthy is True
    assert report.failures == ()


def test_audit_fails_on_kl_spike_and_entropy_collapse():
    trace = TelemetryTrace(
        approx_kl=[0.01, 0.5],  # spike
        clip_fraction=[0.1, 0.1],
        entropy=[0.5, 1e-6],  # collapse
        explained_variance=[0.2, 0.2],
    )
    report = audit_telemetry(trace)
    assert report.healthy is False
    assert "approx_kl_spike" in report.failures
    assert "entropy_collapse" in report.failures


def test_short_train_records_telemetry_and_audits(tmp_path: Path):
    trace, report = record_telemetry_during_train(
        total_timesteps=2048,
        n_envs=2,
        seed=0,
        save_path=tmp_path / "tel.zip",
    )
    assert len(trace.approx_kl) >= 1
    assert len(trace.clip_fraction) >= 1
    assert len(trace.entropy) >= 1
    assert len(trace.explained_variance) >= 1
    # Short randomish runs can look noisy; we only require a structured report.
    assert isinstance(report.healthy, bool)
    assert isinstance(report.failures, tuple)
