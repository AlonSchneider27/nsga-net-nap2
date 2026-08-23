"""Test NAS._evaluate's genome-keyed performance cache (evolution_search.py)."""

from __future__ import annotations

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


def test_same_genome_trains_once(nas_module, monkeypatch):
    calls = []

    def fake_main(**kwargs):
        calls.append(kwargs['genome'])
        return {'valid_acc': 55.0, 'params': 0.1, 'flops': 10.0,
                'pred_acc': None, 'fitness_scores': {'sotl': -3.0},
                'genotype': 'NB201Genotype(arch_str=fake)'}

    monkeypatch.setattr(nas_module.train_search, 'main', fake_main)

    problem = nas_module.NAS(search_space='nb201', n_var=6, n_obj=2,
                             n_constr=0, lb=np.zeros(6), ub=np.full(6, 4.0))
    x = np.array([[0, 1, 2, 3, 4, 0],
                  [0, 1, 2, 3, 4, 0]])   # identical genomes

    out = {}
    problem._evaluate(x, out)
    problem._evaluate(x, out)            # re-sampled in a later generation

    assert len(calls) == 1               # trained exactly once
    assert out['F'].shape == (2, 2)
    np.testing.assert_allclose(out['F'][:, 0], 45.0)   # 100 - valid_acc
    np.testing.assert_allclose(out['F'][:, 1], 10.0)


def test_distinct_genomes_not_conflated(nas_module, monkeypatch):
    calls = []

    def fake_main(**kwargs):
        calls.append(kwargs['genome'])
        return {'valid_acc': 50.0 + len(calls), 'params': 0.1, 'flops': 1.0,
                'pred_acc': None, 'fitness_scores': None,
                'genotype': 'NB201Genotype(arch_str=fake)'}

    monkeypatch.setattr(nas_module.train_search, 'main', fake_main)

    problem = nas_module.NAS(search_space='nb201', n_var=6, n_obj=2,
                             n_constr=0, lb=np.zeros(6), ub=np.full(6, 4.0))
    x = np.array([[0, 0, 0, 0, 0, 0],
                  [4, 4, 4, 4, 4, 4]])

    out = {}
    problem._evaluate(x, out)
    assert len(calls) == 2
    assert out['F'][0, 0] != out['F'][1, 0]
