"""Tests for the method-guided GA objective (--fitness_objective)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


@pytest.fixture
def nas_module(monkeypatch):
    # evolution_search parses CLI args at import time; give it a clean argv.
    monkeypatch.setattr(sys, 'argv', ['evolution_search.py'])
    from search import evolution_search
    return evolution_search


def perf(**over):
    base = {'valid_acc': 55.0, 'params': 0.1, 'flops': 10.0,
            'pred_acc': 0.84, 'fitness_scores': {'sotl_e': -45.61,
                                                 'synflow': 1.5e30,
                                                 'sotl_e@5': -40.0},
            'genotype': "NB201Genotype(arch_str='|none~0|+|none~0|none~1|+|none~0|none~1|none~2|')"}
    base.update(over)
    return base


# ---------------- objective_score ----------------

def test_objective_score_plain_key(nas_module):
    assert nas_module.objective_score(perf(), 'sotl_e') == -45.61


def test_objective_score_budget_list_uses_max(nas_module):
    assert nas_module.objective_score(perf(), 'sotl_e', [1, 5]) == -40.0


def test_objective_score_zero_cost_plain_fallback_in_list_mode(nas_module):
    # synflow has no @budget key even in list mode.
    assert nas_module.objective_score(perf(), 'synflow', [1, 5]) == 1.5e30


def test_objective_score_nap2_reads_pred_acc(nas_module):
    assert nas_module.objective_score(perf(), 'nap2') == 0.84
    assert nas_module.objective_score(perf(), 'nap2', [1, 5]) == 0.84


def test_objective_score_missing_returns_none(nas_module):
    assert nas_module.objective_score(perf(fitness_scores=None), 'sotl_e') is None
    assert nas_module.objective_score(perf(pred_acc=None), 'nap2') is None


# ---------------- NAS._evaluate guided mode ----------------

def guided_problem(nas_module, monkeypatch, fake_main, **kwargs):
    monkeypatch.setattr(nas_module.train_search, 'main', fake_main)
    return nas_module.NAS(search_space='nb201', n_var=6, n_obj=2,
                          n_constr=0, lb=np.zeros(6), ub=np.full(6, 4.0),
                          **kwargs)


X_ONE = np.array([[0, 1, 2, 3, 4, 0]])


def test_guided_objective_is_negated_score(nas_module, monkeypatch):
    problem = guided_problem(nas_module, monkeypatch, lambda **kw: perf(),
                             fitness_objective='sotl_e')
    out = {}
    problem._evaluate(X_ONE, out)
    np.testing.assert_allclose(out['F'][0, 0], 45.61)
    np.testing.assert_allclose(out['F'][0, 1], 10.0)   # flops unchanged


def test_guided_objective_budget_list_variant(nas_module, monkeypatch):
    problem = guided_problem(nas_module, monkeypatch, lambda **kw: perf(),
                             fitness_objective='sotl_e', nap2_steps_list=[1, 5])
    out = {}
    problem._evaluate(X_ONE, out)
    np.testing.assert_allclose(out['F'][0, 0], 40.0)


def test_guided_objective_nap2(nas_module, monkeypatch):
    problem = guided_problem(nas_module, monkeypatch, lambda **kw: perf(),
                             fitness_objective='nap2')
    out = {}
    problem._evaluate(X_ONE, out)
    np.testing.assert_allclose(out['F'][0, 0], -0.84)


@pytest.mark.parametrize('bad', [None, float('nan')])
def test_guided_objective_penalty_on_bad_score(nas_module, monkeypatch, bad):
    fake = lambda **kw: perf(fitness_scores={'sotl_e': bad})
    problem = guided_problem(nas_module, monkeypatch, fake,
                             fitness_objective='sotl_e')
    out = {}
    problem._evaluate(X_ONE, out)
    np.testing.assert_allclose(out['F'][0, 0], nas_module.OBJECTIVE_PENALTY)


def test_default_mode_unchanged(nas_module, monkeypatch):
    problem = guided_problem(nas_module, monkeypatch, lambda **kw: perf())
    out = {}
    problem._evaluate(X_ONE, out)
    np.testing.assert_allclose(out['F'][0, 0], 45.0)   # 100 - valid_acc


def test_guided_cache_hit_consistent(nas_module, monkeypatch):
    calls = []

    def fake_main(**kw):
        calls.append(1)
        return perf()

    problem = guided_problem(nas_module, monkeypatch, fake_main,
                             fitness_objective='sotl_e')
    x = np.array([[0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 0]])
    out = {}
    problem._evaluate(x, out)
    assert len(calls) == 1
    np.testing.assert_allclose(out['F'][:, 0], 45.61)


# ---------------- save_final_population ----------------

class FakePop:
    def __init__(self, X, F):
        self._d = {'X': np.asarray(X), 'F': np.asarray(F)}

    def get(self, key):
        return self._d[key]


class FakeRes:
    def __init__(self, pop_x, pop_f, front_x):
        self.pop = FakePop(pop_x, pop_f)
        self.X = None if front_x is None else np.asarray(front_x)


def test_save_final_population_schema(nas_module, tmp_path):
    pop_x = [[0, 1, 2, 3, 4, 0], [3, 3, 3, 3, 3, 3]]
    pop_f = [[45.61, 10.0], [30.0, 50.0]]
    res = FakeRes(pop_x, pop_f, front_x=[pop_x[1]])
    path = tmp_path / 'final_pop.json'

    payload = nas_module.save_final_population(res, 'nb201', str(path),
                                               objective_method='sotl_e')
    on_disk = json.loads(path.read_text())
    assert on_disk == payload
    assert payload['search_space'] == 'nb201'
    assert payload['objective_method'] == 'sotl_e'
    assert payload['objectives'] == ['-sotl_e', 'flops']
    assert len(payload['population']) == 2
    front_flags = [e['on_pareto_front'] for e in payload['population']]
    assert front_flags == [False, True]
    for e in payload['population']:
        assert 'arch_str' in e and e['arch_str'].startswith('|')
        assert all(isinstance(v, int) for v in e['X'])
        assert all(isinstance(v, float) for v in e['F'])


def test_save_final_population_none_front(nas_module, tmp_path):
    res = FakeRes([[0, 0, 0, 0, 0, 0]], [[1.0, 2.0]], front_x=None)
    payload = nas_module.save_final_population(
        res, 'nb201', str(tmp_path / 'fp.json'))
    assert payload['population'][0]['on_pareto_front'] is False
    assert payload['objectives'] == ['100-valid_acc', 'flops']


def test_guided_failed_score_not_cached(nas_module, monkeypatch):
    """A transient score failure must not become a permanent cached penalty."""
    calls = []

    def flaky_main(**kw):
        calls.append(1)
        # First evaluation fails to score; the retry succeeds.
        if len(calls) == 1:
            return perf(fitness_scores=None)
        return perf()

    problem = guided_problem(nas_module, monkeypatch, flaky_main,
                             fitness_objective='sotl_e')
    out = {}
    problem._evaluate(X_ONE, out)
    np.testing.assert_allclose(out['F'][0, 0], nas_module.OBJECTIVE_PENALTY)
    problem._evaluate(X_ONE, out)          # re-sampled -> re-evaluated
    assert len(calls) == 2                 # NOT served from cache
    np.testing.assert_allclose(out['F'][0, 0], 45.61)
    problem._evaluate(X_ONE, out)          # good result IS cached now
    assert len(calls) == 2
