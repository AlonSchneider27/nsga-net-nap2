"""Smoke tests for scripts/analyze_summary.py.

Verify the script:
  - exits 0 on a well-formed summary.json (new envelope shape)
  - exits 0 on the older flat layout (forward compatibility)
  - exits 1 with a clear stderr message on a missing input
  - prints the documented sections on the new envelope
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "analyze_summary.py"


@pytest.fixture
def envelope_summary(tmp_path):
    """summary.json in the new {architectures, metrics} layout."""
    payload = {
        "architectures": {
            "1": {"valid_acc": 50.0, "pred_acc": 0.30, "flops": 10.0,
                  "param_size_mb": 0.10, "genotype": "Genotype(...)"},
            "2": {"valid_acc": 60.0, "pred_acc": 0.40, "flops": 15.0,
                  "param_size_mb": 0.20, "genotype": "Genotype(...)"},
            "3": {"valid_acc": 70.0, "pred_acc": None, "flops": 20.0,
                  "param_size_mb": 0.25, "genotype": "Genotype(...)"},
            "4": {"valid_acc": 80.0, "pred_acc": 0.60, "flops": 25.0,
                  "param_size_mb": 0.30, "genotype": "Genotype(...)"},
        },
        "metrics": {
            "kendall_tau": 1.0,
            "spearman_rho": 1.0,
            "top_10pct_accuracy": 1.0,
            "num_architectures": 3,
            "num_failed_predictions": 1,
        },
    }
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(payload))
    return p


@pytest.fixture
def flat_summary(tmp_path):
    """summary.json in the old flat {arch_id: {...}} layout."""
    payload = {
        "1": {"valid_acc": 50.0, "pred_acc": 0.30, "flops": 10.0,
              "param_size_mb": 0.10, "genotype": "Genotype(...)"},
        "2": {"valid_acc": 60.0, "pred_acc": None, "flops": 15.0,
              "param_size_mb": 0.20, "genotype": "Genotype(...)"},
    }
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(payload))
    return p


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_envelope_summary_exits_zero(envelope_summary):
    r = _run(envelope_summary)
    assert r.returncode == 0, r.stderr
    assert "Architectures parsed       : 4" in r.stdout
    assert "Ranking metrics" in r.stdout
    assert "kendall_tau" in r.stdout
    assert "Top-5 by valid_acc" in r.stdout
    assert "Top-5 by pred_acc" in r.stdout


def test_envelope_summary_with_directory_arg(envelope_summary):
    """Pass the parent directory; the script should resolve summary.json."""
    r = _run(envelope_summary.parent)
    assert r.returncode == 0, r.stderr
    assert "Architectures parsed" in r.stdout


def test_flat_summary_still_works(flat_summary):
    """Old flat layout: script exits 0 but reports metrics not present."""
    r = _run(flat_summary)
    assert r.returncode == 0, r.stderr
    assert "old flat layout" in r.stdout


def test_missing_summary_exits_one(tmp_path):
    r = _run(tmp_path / "does_not_exist.json")
    assert r.returncode == 1
    assert "error" in r.stderr.lower()


def test_top_flag_respected(envelope_summary):
    r = _run(envelope_summary, "--top", "2")
    assert r.returncode == 0, r.stderr
    assert "Top-2 by valid_acc" in r.stdout


def test_histogram_flag(envelope_summary):
    r = _run(envelope_summary, "--histogram")
    assert r.returncode == 0, r.stderr
    assert "pred_acc histogram" in r.stdout


def test_bottom_flag(envelope_summary):
    r = _run(envelope_summary, "--bottom", "2")
    assert r.returncode == 0, r.stderr
    assert "Bottom-2" in r.stdout
