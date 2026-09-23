"""Tests for scripts/train_top_archs.py selection logic (training stubbed)."""

from __future__ import annotations

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import train_top_archs as tta

NB201 = "NB201Genotype(arch_str='|nor_conv_3x3~0|+|none~0|none~1|+|none~0|none~1|none~2|')"


def arch(gen=NB201, valid=0.5, pred=None, fitness=None):
    return {'genotype': gen, 'valid_acc': valid, 'pred_acc': pred,
            'flops': 1.0, 'param_size_mb': 0.1,
            **({'fitness': fitness} if fitness else {})}


def summary(archs, objective=None):
    s = {'architectures': archs, 'metrics': {}}
    if objective:
        s['objective_method'] = objective
    return s


def test_extract_arch_str():
    assert tta.extract_arch_str(NB201).startswith('|nor_conv_3x3')
    with pytest.raises(ValueError):
        tta.extract_arch_str('Genotype(normal=[...], reduce=[...])')


def test_resolve_rank_key_exact_and_budget():
    archs = {'1': arch(fitness={'sotl_e': -3.0, 'lce_m@5': 0.8, 'lce_m@23': 0.9})}
    assert tta.resolve_rank_key(archs, 'sotl_e') == ('fitness', 'sotl_e')
    assert tta.resolve_rank_key(archs, 'lce_m') == ('fitness', 'lce_m@23')
    assert tta.resolve_rank_key(archs, 'pred_acc') == ('attr', 'pred_acc')
    assert tta.resolve_rank_key(archs, 'nap2') == ('attr', 'pred_acc')
    with pytest.raises(KeyError):
        tta.resolve_rank_key(archs, 'nonexistent')


def gen_for(op):
    return f"NB201Genotype(arch_str='|{op}~0|+|none~0|none~1|+|none~0|none~1|none~2|')"


def test_select_archs_ranking_dedupe_topk():
    archs = {
        '1': arch(gen=gen_for('nor_conv_3x3'), fitness={'sotl_e': -10.0}),
        '2': arch(gen=gen_for('skip_connect'), fitness={'sotl_e': -5.0}),
        '3': arch(gen=gen_for('nor_conv_3x3'), fitness={'sotl_e': -7.0}),  # dup, worse
        '4': arch(gen=gen_for('avg_pool_3x3'), fitness={'sotl_e': -20.0}),
    }
    rows = tta.select_archs(summary(archs, objective='sotl_e'), top_k=2)
    assert [r['arch_id'] for r in rows] == ['2', '3']
    # dup arch '3' (-7.0) beats '1' (-10.0) and keeps the better score.
    assert rows[1]['score'] == -7.0
    assert rows[0]['rank_key'] == 'sotl_e'


def test_select_archs_auto_falls_back_to_pred_acc():
    archs = {'1': arch(pred=0.8), '2': arch(gen=gen_for('skip_connect'), pred=0.9)}
    rows = tta.select_archs(summary(archs), top_k=2, rank_by='auto')
    assert rows[0]['arch_id'] == '2'
    assert rows[0]['rank_key'] == 'pred_acc'


def test_select_archs_hard_error_when_scores_absent():
    # Key not resolvable at all -> KeyError from resolve_rank_key.
    with pytest.raises(KeyError):
        tta.select_archs(summary({'1': arch()}, objective='sotl_e'), top_k=1)
    # Key resolvable but every score is None -> SystemExit.
    archs = {'1': arch(fitness={'sotl_e': None})}
    with pytest.raises(SystemExit):
        tta.select_archs(summary(archs, objective='sotl_e'), top_k=1)


def test_select_from_final_pop():
    fp = {'population': [
        {'arch_str': '|a~0|', 'F': [-0.9, 1.0], 'on_pareto_front': True},
        {'arch_str': '|b~0|', 'F': [-0.5, 2.0], 'on_pareto_front': False},
        {'arch_str': '|a~0|', 'F': [-0.9, 1.0], 'on_pareto_front': True},  # dup
        {'arch_str': '|c~0|', 'F': [-0.7, 0.5], 'on_pareto_front': True},
    ]}
    rows = tta.select_from_final_pop(fp)
    assert [r['arch_str'] for r in rows] == ['|a~0|', '|c~0|']
    assert rows[0]['score'] == 0.9


def test_main_wiring(tmp_path, monkeypatch):
    archs = {'1': arch(fitness={'sotl_e': -3.0})}
    (tmp_path / 'summary.json').write_text(json.dumps(summary(archs, 'sotl_e')))

    def fake_train(rows, args):
        for r in rows:
            r['trained_acc'] = 0.77
            r['t_min'] = 0.0
        return rows

    monkeypatch.setattr(tta, 'train_selected', fake_train)
    rc = tta.main(['--summary', str(tmp_path), '--top-k', '1'])
    assert rc == 0
    out = json.loads((tmp_path / 'top_archs.json').read_text())
    assert out[0]['trained_acc'] == 0.77
    assert (tmp_path / 'top_archs.csv').exists()


def test_select_from_final_pop_full_front_by_default_and_truncation():
    fp = {'population': [
        {'arch_str': f'|{i}~0|', 'F': [-i, 1.0], 'on_pareto_front': True}
        for i in range(8)]}
    assert len(tta.select_from_final_pop(fp)) == 8          # whole front
    assert len(tta.select_from_final_pop(fp, top_k=0)) == 8  # 0 = no limit
    assert len(tta.select_from_final_pop(fp, top_k=3)) == 3  # explicit only


def test_select_archs_topk_zero_means_all():
    archs = {str(i): arch(gen=gen_for(op), fitness={'sotl_e': -float(i)})
             for i, op in enumerate(['none', 'skip_connect', 'nor_conv_1x1'])}
    assert len(tta.select_archs(summary(archs, 'sotl_e'), top_k=0)) == 3
