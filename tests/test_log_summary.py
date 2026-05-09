"""Tests for misc/log_summary.py.

Uses an in-memory fixture log shaped exactly like a real evolution_search
run, including the awkward cases:
  - normal arch with all fields populated
  - arch where the predictor failed (pred_acc=n/a)
  - arch missing the preceding ``flops = ...`` line (corrupted log)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from misc.log_summary import resolve_log_path, scrape, write_summary


# Shaped after a real log line. Timestamp prefix kept because the regexes
# anchor on the trailing tokens, not the date — and we want to verify
# that.
SAMPLE_LOG = """\
05/06 08:36:00 PM args = Namespace(...)
05/06 08:36:01 PM Network id = 1
05/06 08:36:01 PM Architecture = Genotype(normal=[('sep_conv_3x3', 0)], normal_concat=[2], reduce=[('avg_pool_3x3', 0)], reduce_concat=[2])
05/06 08:36:01 PM param size = 0.349076MB
05/06 08:36:14 PM nap2 pred_acc = 0.0024 (steps=5)
05/06 08:36:14 PM epoch 0 lr 2.469e-02
05/06 08:36:14 PM train_acc 12.0
05/06 08:38:00 PM valid_acc 58.28
05/06 08:38:00 PM flops = 67.0994
05/06 08:38:00 PM arch 1: valid_acc=58.2800 pred_acc=0.0024
05/06 08:39:00 PM Network id = 2
05/06 08:39:00 PM Architecture = Genotype(normal=[('max_pool_3x3', 0)], normal_concat=[2], reduce=[('skip_connect', 1)], reduce_concat=[2])
05/06 08:39:00 PM param size = 0.103444MB
05/06 08:39:13 PM nap2 prediction failed
Traceback (most recent call last):
  File "...", line 1, in <module>
TypeError: ufunc 'isinf' not supported for the input types ...
05/06 08:39:13 PM epoch 0 lr 2.469e-02
05/06 08:41:00 PM valid_acc 52.83
05/06 08:41:00 PM flops = 18.3114
05/06 08:41:00 PM arch 2: valid_acc=52.8300 pred_acc=n/a
05/06 08:42:00 PM Network id = 3
05/06 08:42:00 PM Architecture = Genotype(normal=[('skip_connect', 0)], normal_concat=[2], reduce=[('dil_conv_3x3', 0)], reduce_concat=[2])
05/06 08:42:00 PM param size = 0.220000MB
05/06 08:42:13 PM nap2 pred_acc = 0.0030 (steps=5)
05/06 08:43:00 PM valid_acc 60.10
05/06 08:43:00 PM arch 3: valid_acc=60.1000 pred_acc=0.0030
05/06 08:44:00 PM generation = 1
"""


@pytest.fixture
def sample_log(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text(SAMPLE_LOG)
    return p


def test_scrape_normal_arch(sample_log):
    result = scrape(sample_log)
    arch1 = result["1"]
    assert arch1["valid_acc"] == pytest.approx(58.28)
    assert arch1["pred_acc"] == pytest.approx(0.0024)
    assert arch1["flops"] == pytest.approx(67.0994)
    assert arch1["param_size_mb"] == pytest.approx(0.349076)
    assert arch1["genotype"].startswith("Genotype(normal=")
    assert arch1["genotype"].endswith(")")


def test_scrape_failed_predictor_keeps_arch(sample_log):
    """The pre-fix scraper silently dropped pred_acc=n/a archs entirely."""
    result = scrape(sample_log)
    assert "2" in result, "arch 2 must appear even with pred_acc=n/a"
    arch2 = result["2"]
    assert arch2["pred_acc"] is None
    assert arch2["valid_acc"] == pytest.approx(52.83)
    assert arch2["flops"] == pytest.approx(18.3114)
    assert arch2["param_size_mb"] == pytest.approx(0.103444)


def test_scrape_missing_flops_yields_none(sample_log):
    """arch 3 in the fixture has no preceding `flops = ...` line."""
    result = scrape(sample_log)
    arch3 = result["3"]
    assert arch3["flops"] is None
    assert arch3["valid_acc"] == pytest.approx(60.10)
    assert arch3["pred_acc"] == pytest.approx(0.0030)


def test_scrape_orders_by_arch_id(sample_log):
    result = scrape(sample_log)
    assert list(result.keys()) == ["1", "2", "3"]


def test_scrape_genotype_is_verbatim(sample_log):
    result = scrape(sample_log)
    assert (
        "[('sep_conv_3x3', 0)]" in result["1"]["genotype"]
        and "normal_concat=[2]" in result["1"]["genotype"]
    )


def test_write_summary_round_trip(sample_log, tmp_path):
    out = tmp_path / "summary.json"
    data = write_summary(sample_log, out)
    on_disk = json.loads(out.read_text())
    assert on_disk == data
    assert on_disk["2"]["pred_acc"] is None  # JSON null


def test_resolve_log_path_directory(sample_log):
    result = resolve_log_path(sample_log.parent)
    assert result == sample_log


def test_resolve_log_path_file(sample_log):
    result = resolve_log_path(sample_log)
    assert result == sample_log


def test_resolve_log_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_log_path(tmp_path / "nonexistent")
