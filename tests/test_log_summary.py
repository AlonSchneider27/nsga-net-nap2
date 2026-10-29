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

from misc.log_summary import (
    compute_run_metrics,
    resolve_log_path,
    scrape,
    write_summary,
)


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
05/06 08:43:30 PM Network id = 4
05/06 08:43:30 PM Architecture = NB201Genotype(arch_str='|nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|nor_conv_3x3~0|skip_connect~1|none~2|')
05/06 08:43:30 PM param size = 0.045000MB
05/06 08:43:45 PM nap2 pred_acc = 0.4500 (steps=5)
05/06 08:44:00 PM valid_acc 70.50
05/06 08:44:00 PM flops = 12.5000
05/06 08:44:00 PM arch 4: valid_acc=70.5000 pred_acc=0.4500
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
    assert arch1["valid_acc"] == pytest.approx(0.5828)
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
    assert arch2["valid_acc"] == pytest.approx(0.5283)
    assert arch2["flops"] == pytest.approx(18.3114)
    assert arch2["param_size_mb"] == pytest.approx(0.103444)


def test_scrape_missing_flops_yields_none(sample_log):
    """arch 3 in the fixture has no preceding `flops = ...` line."""
    result = scrape(sample_log)
    arch3 = result["3"]
    assert arch3["flops"] is None
    assert arch3["valid_acc"] == pytest.approx(0.6010)
    assert arch3["pred_acc"] == pytest.approx(0.0030)


def test_scrape_orders_by_arch_id(sample_log):
    result = scrape(sample_log)
    assert list(result.keys()) == ["1", "2", "3", "4"]


def test_scrape_captures_nb201_genotype_verbatim(sample_log):
    """NB201Genotype(arch_str='|...') must come through verbatim."""
    result = scrape(sample_log)
    arch4 = result["4"]
    assert arch4["genotype"].startswith("NB201Genotype(arch_str=")
    # The NB201 arch_str delimiters must round-trip cleanly.
    assert "|nor_conv_3x3~0|" in arch4["genotype"]
    assert arch4["genotype"].count("+") == 2
    assert arch4["valid_acc"] == pytest.approx(0.7050)
    assert arch4["pred_acc"] == pytest.approx(0.4500)
    assert arch4["flops"] == pytest.approx(12.5)


def test_scrape_genotype_is_verbatim(sample_log):
    result = scrape(sample_log)
    assert (
        "[('sep_conv_3x3', 0)]" in result["1"]["genotype"]
        and "normal_concat=[2]" in result["1"]["genotype"]
    )


def test_write_summary_round_trip(sample_log, tmp_path):
    out = tmp_path / "summary.json"
    payload = write_summary(sample_log, out)
    on_disk = json.loads(out.read_text())
    assert on_disk == payload
    # Top-level shape is the documented {architectures, metrics} envelope.
    assert set(on_disk) == {"architectures", "metrics"}
    assert on_disk["architectures"]["2"]["pred_acc"] is None  # JSON null
    # Metrics block exists with the documented keys (or an "error" key
    # if compute failed). Values themselves checked elsewhere.
    metrics = on_disk["metrics"]
    if "error" not in metrics:
        for key in ("kendall_tau", "spearman_rho", "top_10pct_accuracy",
                    "num_architectures", "num_failed_predictions"):
            assert key in metrics


def test_compute_run_metrics_happy_path():
    """Two-or-more paired observations -> real KT/Spearman values."""
    archs = {
        "1": {"valid_acc": 50.0, "pred_acc": 0.30},
        "2": {"valid_acc": 60.0, "pred_acc": 0.40},
        "3": {"valid_acc": 70.0, "pred_acc": 0.50},
        "4": {"valid_acc": 80.0, "pred_acc": 0.60},
    }
    m = compute_run_metrics(archs)
    # Perfect rank correlation -> KT and Spearman both == 1.0.
    assert m["kendall_tau"] == pytest.approx(1.0)
    assert m["spearman_rho"] == pytest.approx(1.0)
    assert m["num_architectures"] == 4
    assert m["num_failed_predictions"] == 0


def test_compute_run_metrics_excludes_failed_predictions():
    """pred_acc=None archs are tallied but excluded from correlation."""
    archs = {
        "1": {"valid_acc": 50.0, "pred_acc": 0.30},
        "2": {"valid_acc": 60.0, "pred_acc": None},     # failed predictor
        "3": {"valid_acc": 70.0, "pred_acc": 0.50},
        "4": {"valid_acc": 80.0, "pred_acc": None},     # failed predictor
        "5": {"valid_acc": 90.0, "pred_acc": 0.70},
    }
    m = compute_run_metrics(archs)
    assert m["num_failed_predictions"] == 2
    assert m["num_architectures"] == 3   # only the 3 with both signals


def test_compute_run_metrics_too_few_paired_returns_nulls():
    """Below 2 paired observations, correlations are null (not NaN)."""
    archs = {
        "1": {"valid_acc": 50.0, "pred_acc": None},
        "2": {"valid_acc": 60.0, "pred_acc": 0.40},   # only 1 paired
    }
    m = compute_run_metrics(archs)
    assert m["kendall_tau"] is None
    assert m["spearman_rho"] is None
    assert m["top_10pct_accuracy"] is None
    assert m["num_architectures"] == 1
    assert m["num_failed_predictions"] == 1


def test_compute_run_metrics_all_failed_returns_nulls():
    archs = {
        "1": {"valid_acc": 50.0, "pred_acc": None},
        "2": {"valid_acc": 60.0, "pred_acc": None},
    }
    m = compute_run_metrics(archs)
    assert m["kendall_tau"] is None
    assert m["num_architectures"] == 0
    assert m["num_failed_predictions"] == 2


def test_write_summary_metrics_in_payload(sample_log, tmp_path):
    """End-to-end: metrics block populated from the fixture log."""
    out = tmp_path / "summary.json"
    payload = write_summary(sample_log, out)
    metrics = payload["metrics"]
    assert "error" not in metrics
    # Fixture has 4 archs; arch 2 has pred_acc=n/a -> 3 paired observations.
    assert metrics["num_architectures"] == 3
    assert metrics["num_failed_predictions"] == 1


def test_resolve_log_path_directory(sample_log):
    result = resolve_log_path(sample_log.parent)
    assert result == sample_log


def test_resolve_log_path_file(sample_log):
    result = resolve_log_path(sample_log)
    assert result == sample_log


def test_resolve_log_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_log_path(tmp_path / "nonexistent")


# ---- learning-curve baseline fitness lines --------------------------------

FITNESS_LOG = """\
05/06 08:36:01 PM Network id = 1
05/06 08:36:01 PM Architecture = NB201Genotype(arch_str='|nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|nor_conv_3x3~0|skip_connect~1|none~2|')
05/06 08:36:01 PM param size = 0.349076MB
05/06 08:38:00 PM flops = 67.0994
05/06 08:38:00 PM arch 1: valid_acc=58.2800 pred_acc=n/a
05/06 08:38:00 PM arch 1 fitness: sotl=-123.456000 sotl_e=-45.610000 early_stop=0.421300 lce_m=0.871200
05/06 08:39:00 PM Network id = 2
05/06 08:39:00 PM Architecture = NB201Genotype(arch_str='|none~0|+|none~0|none~1|+|none~0|none~1|none~2|')
05/06 08:39:00 PM param size = 0.103444MB
05/06 08:41:00 PM flops = 18.3114
05/06 08:41:00 PM arch 2: valid_acc=70.8300 pred_acc=n/a
05/06 08:41:00 PM arch 2 fitness: sotl=-100.000000 sotl_e=-30.500000 early_stop=0.650000 lce_m=0.900000
05/06 08:42:00 PM Network id = 3
05/06 08:42:00 PM Architecture = NB201Genotype(arch_str='|none~0|+|none~0|none~1|+|none~0|skip_connect~1|none~2|')
05/06 08:42:00 PM param size = 0.220000MB
05/06 08:43:00 PM flops = 20.0
05/06 08:43:00 PM arch 3: valid_acc=80.1000 pred_acc=n/a
05/06 08:43:00 PM arch 3 fitness: sotl=-50.000000 sotl_e=-10.000000 early_stop=0.780000 lce_m=0.950000
"""


@pytest.fixture
def fitness_log(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text(FITNESS_LOG)
    return p


def test_scrape_attaches_fitness_scores(fitness_log):
    result = scrape(fitness_log)
    assert result["1"]["fitness"] == pytest.approx({
        "sotl": -123.456, "sotl_e": -45.61,
        "early_stop": 0.4213, "lce_m": 0.8712,
    })


def test_scrape_without_fitness_lines_has_no_fitness_key(sample_log):
    result = scrape(sample_log)
    assert all("fitness" not in v for v in result.values())


def test_fitness_metrics_per_method(fitness_log, tmp_path):
    payload = write_summary(fitness_log, tmp_path / "summary.json")
    fm = payload["fitness_metrics"]
    assert set(fm) == {"sotl", "sotl_e", "early_stop", "lce_m"}
    # All four methods rank the 3 archs in the same order as valid_acc.
    for method, metrics in fm.items():
        assert metrics["kendall_tau"] == pytest.approx(1.0), method
        assert metrics["num_architectures"] == 3


def test_payload_shape_unchanged_without_fitness(sample_log, tmp_path):
    payload = write_summary(sample_log, tmp_path / "summary.json")
    assert set(payload) == {"architectures", "metrics"}


BUDGET_FITNESS_LINES = """\
05/06 08:37:00 PM Network id = 1
05/06 08:37:10 PM Architecture = NB201Genotype(arch_str='|none~0|+|none~0|none~1|+|none~0|none~1|none~2|')
05/06 08:37:20 PM param size = 0.1MB
05/06 08:37:30 PM flops = 1.0
05/06 08:38:00 PM arch 1: valid_acc=61.2000 pred_acc=0.8412
05/06 08:38:00 PM arch 1 fitness: sotl@1=-32.100000 sotl@5=-123.456000 nap2@1=0.812300 nap2@5=0.841200 synflow=1.5e+30
"""


def test_scrape_budget_suffixed_fitness_keys(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(BUDGET_FITNESS_LINES)
    from misc.log_summary import scrape
    arch = scrape(log)["1"]
    assert arch["fitness"] == {
        "sotl@1": -32.1, "sotl@5": -123.456,
        "nap2@1": 0.8123, "nap2@5": 0.8412,
        "synflow": 1.5e+30,
    }


OBJECTIVE_LINE = ("05/06 08:35:00 PM objective_method = sotl_e "
                  "(objs[0] = -score at budget 5; flops objective unchanged)\n")


def test_scrape_objective_method(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(OBJECTIVE_LINE + BUDGET_FITNESS_LINES)
    from misc.log_summary import scrape_objective_method
    assert scrape_objective_method(log) == "sotl_e"


def test_scrape_objective_method_absent(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(BUDGET_FITNESS_LINES)
    from misc.log_summary import scrape_objective_method, write_summary
    assert scrape_objective_method(log) is None
    payload = write_summary(log, tmp_path / "s.json")
    assert "objective_method" not in payload


def test_write_summary_includes_objective_method(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(OBJECTIVE_LINE + BUDGET_FITNESS_LINES)
    from misc.log_summary import write_summary
    payload = write_summary(log, tmp_path / "s.json")
    assert payload["objective_method"] == "sotl_e"


def test_write_summary_tolerates_bad_bytes_in_log(tmp_path):
    # Real search logs can contain stray non-UTF-8 bytes; the objective
    # scrape must not kill the whole summary (regression guard).
    log = tmp_path / "log.txt"
    log.write_bytes(b"\xff\xfe garbage line\n"
                    + OBJECTIVE_LINE.encode()
                    + BUDGET_FITNESS_LINES.encode())
    from misc.log_summary import write_summary
    payload = write_summary(log, tmp_path / "s.json")
    assert payload["objective_method"] == "sotl_e"
    assert len(payload["architectures"]) == 1


def test_scrape_epoch_native_keys(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(BUDGET_FITNESS_LINES.replace(
        "sotl@1=-32.100000 sotl@5=-123.456000",
        "sotl@e1=-32.100000 sotl_e@e20=-3.500000"))
    from misc.log_summary import scrape
    fit = scrape(log)["1"]["fitness"]
    assert fit["sotl@e1"] == -32.1
    assert fit["sotl_e@e20"] == -3.5
