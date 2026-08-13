"""Tests for the learning-curve fitness scorers (fitness/scorers.py + registry).

Covers the scoring math on synthetic traces (sign conventions are the classic
failure mode), the SoTL-E window/fallback behavior, the needs_* flags each
scorer declares, and build_scorers() parsing.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from fitness import ALL_BASELINES, FITNESS_SCORERS, build_scorers
from fitness.trace import TrainingTrace


def make_trace(losses, val_curve=(), final_val=None, epoch_len=196, interval=100):
    return TrainingTrace(
        minibatch_losses=list(losses),
        val_acc_curve=list(val_curve),
        final_val_acc=final_val,
        epoch_len=epoch_len,
        snapshot_interval=interval,
    )


# ---- SoTL ----------------------------------------------------------------

def test_sotl_is_negated_sum_of_all_losses():
    scorer = FITNESS_SCORERS['sotl']()
    trace = make_trace([1.0, 2.0, 3.0])
    assert scorer.score(trace) == -6.0


def test_sotl_lower_losses_score_higher():
    scorer = FITNESS_SCORERS['sotl']()
    good = make_trace([0.5] * 10)
    bad = make_trace([2.0] * 10)
    assert scorer.score(good) > scorer.score(bad)


# ---- SoTL-E --------------------------------------------------------------

def test_sotle_sums_only_last_epoch_window():
    scorer = FITNESS_SCORERS['sotl_e']()
    # 2 epochs of 3 minibatches: only the last 3 losses count.
    trace = make_trace([10.0, 10.0, 10.0, 1.0, 2.0, 3.0], epoch_len=3)
    assert scorer.score(trace) == -6.0


def test_sotle_mid_epoch_budget_uses_trailing_window():
    scorer = FITNESS_SCORERS['sotl_e']()
    # 5 losses, epoch_len 3: window = the most recent 3.
    trace = make_trace([10.0, 10.0, 1.0, 2.0, 3.0], epoch_len=3)
    assert scorer.score(trace) == -6.0


def test_sotle_falls_back_to_sotl_below_one_epoch():
    # Budget < 1 epoch: no full window exists, degenerates to SoTL.
    trace = make_trace([1.0, 2.0], epoch_len=196)
    assert FITNESS_SCORERS['sotl_e']().score(trace) == \
        FITNESS_SCORERS['sotl']().score(trace) == -3.0


# ---- Early-Stop ----------------------------------------------------------

def test_early_stop_returns_final_val_acc():
    scorer = FITNESS_SCORERS['early_stop']()
    trace = make_trace([1.0], final_val=0.42)
    assert scorer.score(trace) == 0.42


# ---- registry / needs flags ---------------------------------------------

def test_all_baselines_registered_with_needs_flags():
    assert set(ALL_BASELINES) <= set(FITNESS_SCORERS)
    expectations = {
        'sotl': (False, False),
        'sotl_e': (False, False),
        'early_stop': (False, True),
        'lce_m': (True, False),
        'lc_pfn': (True, False),
    }
    for name, (needs_curve, needs_final) in expectations.items():
        cls = FITNESS_SCORERS[name]
        assert cls.needs_val_curve is needs_curve, name
        assert cls.needs_final_val is needs_final, name
        assert cls.name == name


# ---- build_scorers -------------------------------------------------------

def test_build_scorers_parses_comma_list_in_order():
    scorers = build_scorers('sotl_e,early_stop,sotl_e')
    assert [s.name for s in scorers] == ['sotl_e', 'early_stop']  # deduped


def test_build_scorers_all_excludes_nap2(tmp_path):
    ckpt = tmp_path / 'fake.pt'
    ckpt.write_bytes(b'x')
    scorers = build_scorers('all', lcpfn_ckpt=str(ckpt))
    assert [s.name for s in scorers] == ALL_BASELINES


def test_build_scorers_rejects_unknown_and_nap2():
    with pytest.raises(ValueError, match='Unknown fitness method'):
        build_scorers('sotl,bogus')
    with pytest.raises(ValueError, match='nap2'):
        build_scorers('nap2')


def test_build_scorers_lcpfn_requires_existing_checkpoint():
    with pytest.raises(ValueError, match='checkpoint'):
        build_scorers('lc_pfn', lcpfn_ckpt='/nonexistent/path.pt')
