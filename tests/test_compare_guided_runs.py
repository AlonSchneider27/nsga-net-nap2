"""Tests for scripts/compare_guided_runs.py pure analysis functions."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import compare_guided_runs as cgr


GT = {'|a~0|': 90.0, '|b~0|': 94.0, '|c~0|': 80.0, '|d~0|': 85.0}


def test_dedupe_keep_first():
    v = [('|a~0|', 1.0), ('|b~0|', 2.0), ('|a~0|', 3.0)]
    assert cgr.dedupe_keep_first(v) == [('|a~0|', 1.0), ('|b~0|', 2.0)]


def test_convergence_curve_monotone_with_unknown_archs():
    v = [('|c~0|', None), ('|zzz|', None), ('|b~0|', None), ('|a~0|', None)]
    assert cgr.convergence_curve(v, GT) == [80.0, 80.0, 94.0, 94.0]


def test_jaccard():
    assert cgr.jaccard(['x', 'y'], ['y', 'z']) == round(1 / 3, 4)
    assert cgr.jaccard([], []) == 0.0


def test_op_profile_normalized():
    prof = cgr.op_profile(['|none~0|+|skip_connect~0|none~1|'
                           '+|nor_conv_3x3~0|none~1|none~2|'])
    assert prof['none'] == round(4 / 6, 4)
    assert prof['nor_conv_3x3'] == round(1 / 6, 4)
    assert abs(sum(prof.values()) - 1.0) < 1e-3   # 4-decimal rounding slack


def test_random_baseline_bounds_and_determinism():
    b1 = cgr.random_baseline(GT, budget=2, draws=200, seed=1)
    b2 = cgr.random_baseline(GT, budget=2, draws=200, seed=1)
    assert b1 == b2
    assert 80.0 <= b1['p05'] <= b1['mean_best'] <= b1['p95'] <= 94.0


def test_method_report_end_to_end():
    run = {'method': 'sotl_e',
           'visited': [('|c~0|', -10.0), ('|a~0|', -5.0), ('|b~0|', -2.0),
                       ('|c~0|', -10.0)],   # duplicate visit
           'front': ['|b~0|', '|c~0|']}
    r = cgr.method_report(run, GT, top_k=2)
    assert r['n_evals'] == 4 and r['n_unique'] == 3
    assert r['best_gt_visited'] == 94.0
    assert r['top2_mean_gt'] == round((94.0 + 90.0) / 2, 4)
    assert r['best_gt_front'] == 94.0
    # picked = by own score: |b~0| (-2) then |a~0| (-5)
    assert r['picked_topk'][0]['arch'] == '|b~0|'
    assert r['best_gt_picked'] == 94.0
    assert r['fidelity_kt'] == 1.0     # score order == GT order here
    assert r['convergence'] == [80.0, 90.0, 94.0, 94.0]
