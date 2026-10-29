#!/usr/bin/env python3
"""Aggregate the table-backed nap2-guided budget sweep (research_v2/guided_lut): steps 1..31 x 3 predictors x
3 targets x seeds -> per-run CSV, per-(target, predictor, step) mean/std CSV, and an interactive HTML page.

Every cell comes from the lookup-table runs (no live guided cells are mixed in). Sizes (FLOPs, params) are the
values the GA logged in summary.json; fronts come from final_pop.json. Reference lines use ground truth only
(random-620 baseline, whole-space optimum).

    .venv/bin/python scripts/guided_lut_aggregate.py --root ~/Downloads/research_v2/guided_lut \
        --gt data/nb201_nats_gt.json --out ~/Downloads/research_v2/analysis/guided_lut
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guided_summary_html import analyse, KEY, NAME, KS  # noqa: E402
from compare_guided_runs import random_baseline  # noqa: E402

ARCH_RE = re.compile(r"arch_str='([^']+)'")
RUN_RE = re.compile(r'search-(?P<tgt>c10|c100|in16)_nap2lut31_pred-(?P<pred>c10|c100|in16)_s(?P<step>\d+)_e0_seed(?P<seed>\d+)-')
MB_PER_EPOCH = {'c10': 157, 'c100': 157, 'in16': 593}
INTERVAL = 75
CAPMAX = {'c10': 200.0, 'c100': 200.0, 'in16': 50.0}
METRICS = ['n_unique', 'unique_ratio', 'best_visited', 'fid_kt', 'fid_kt_top10',
           'pick@1', 'pick@5', 'pick@10', 'pick@20', 'oracle@1', 'oracle@5', 'oracle@10', 'oracle@20', 'gap@20',
           'visited_flops', 'visited_params', 'pick1_flops', 'pick1_params', 'top20_flops', 'top20_params',
           'kt_score_flops', 'kt_score_params', 'kt_gt_flops_visited', 'kt_within_flops_q',
           'pick_capq', 'oracle_capq', 'pick_caph', 'oracle_caph', 'pick_capf', 'oracle_capf',
           'front_n', 'front_flops_min', 'front_flops_max', 'front_best_gt', 'front_mean_gt', 'front_regret', 'hv_ratio']


def kt(x, y):
    return float(kendalltau(x, y).correlation) if len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else float('nan')


def hv(points, ref):
    """2-D hypervolume of a (flops, gt) set: maximise gt, minimise flops; ref = (flops_max, gt_min)."""
    pts = sorted(set((f, g) for f, g in points if f <= ref[0] and g >= ref[1]))
    running = ref[1]; area = 0.0
    for i, (f, g) in enumerate(pts):
        nxt = pts[i + 1][0] if i + 1 < len(pts) else ref[0]
        running = max(running, g)
        area += (nxt - f) * (running - ref[1])
    return area


def size_metrics(r, run_dir, ds, gt):
    """Same definitions as guided_seeds_aggregate.py: sizes from the GA logs, fronts from final_pop.json."""
    s = json.load(open(os.path.join(run_dir, 'summary.json')))
    logged = {}
    for aid in sorted(s['architectures'], key=int):
        a = s['architectures'][aid]; m = ARCH_RE.search(a.get('genotype', ''))
        if m and m.group(1) not in logged:
            logged[m.group(1)] = (a.get('flops'), a.get('param_size_mb'))
    A = r['_archs']; sc = r['_scores']; g = r['_gt']
    fl = np.array([logged.get(a, (np.nan, np.nan))[0] for a in A], dtype=float)
    pa = np.array([logged.get(a, (np.nan, np.nan))[1] for a in A], dtype=float)
    ok = ~np.isnan(fl); sc2, g2, fl2, pa2 = sc[ok], g[ok], fl[ok], pa[ok]
    by_score = np.argsort(-sc2)
    q = np.quantile(fl2, [0.25, 0.5, 0.75]); bins = np.digitize(fl2, q); within = []
    for b in range(4):
        m = bins == b
        if m.sum() >= 10:
            v = kt(sc2[m], g2[m])
            if not np.isnan(v): within.append(v)
    r.update({'visited_flops': float(fl2.mean()), 'visited_params': float(pa2.mean()),
              'pick1_flops': float(fl2[by_score[0]]), 'pick1_params': float(pa2[by_score[0]]),
              'top20_flops': float(fl2[by_score[:20]].mean()), 'top20_params': float(pa2[by_score[:20]].mean()),
              'kt_score_flops': kt(sc2, fl2), 'kt_score_params': kt(sc2, pa2), 'kt_gt_flops_visited': kt(g2, fl2),
              'kt_within_flops_q': float(np.mean(within)) if within else float('nan')})
    for name, frac in (('q', 0.125), ('h', 0.25), ('f', 0.5)):
        cap = CAPMAX[ds] * frac; m = fl2 <= cap
        r[f'pick_cap{name}'] = float(g2[m][np.argmax(sc2[m])]) if m.any() else float('nan')
        r[f'oracle_cap{name}'] = float(g2[m].max()) if m.any() else float('nan')
    front = []
    fp_path = os.path.join(run_dir, 'final_pop.json')
    if os.path.exists(fp_path):
        for e in json.load(open(fp_path))['population']:
            if e.get('on_pareto_front') and e.get('arch_str') in gt:
                front.append((float(e['F'][1]), gt[e['arch_str']]))
    gt_pts = list(zip(fl2.tolist(), g2.tolist()))
    ref = (float(max(f for f, _ in gt_pts)), float(min(gv for _, gv in gt_pts)))
    hv_front = hv(front, ref) if front else float('nan'); hv_oracle = hv(gt_pts, ref)
    regs = [float(g2[fl2 <= f + 1e-9].max() - gv) for f, gv in front if (fl2 <= f + 1e-9).any()]
    r.update({'front_n': len(front), 'front_flops_min': float(min(f for f, _ in front)) if front else float('nan'),
              'front_flops_max': float(max(f for f, _ in front)) if front else float('nan'),
              'front_best_gt': float(max(gv for _, gv in front)) if front else float('nan'),
              'front_mean_gt': float(np.mean([gv for _, gv in front])) if front else float('nan'),
              'front_regret': float(np.mean(regs)) if regs else float('nan'),
              'hv_ratio': float(hv_front / hv_oracle) if (front and hv_oracle > 0) else float('nan')})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='<...>/guided_lut containing seed_<S>/ trees')
    ap.add_argument('--seeds', default='42,123,456,789')
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    os.makedirs(args.out, exist_ok=True)

    gt_all = {a: v for a, v in json.load(open(args.gt)).items() if not a.startswith('_')}
    GT = {ds: {a: v[KEY[ds]] for a, v in gt_all.items()} for ds in KEY}
    opt = {ds: round(max(GT[ds].values()), 2) for ds in KEY}
    rb = {ds: random_baseline(GT[ds], 620) for ds in KEY}
    space = {ds: {'kt_gt_flops': kt([v['flop'] for v in gt_all.values()], [GT[ds][a] for a in gt_all]),
                  'kt_gt_params': kt([v['params'] for v in gt_all.values()], [GT[ds][a] for a in gt_all])} for ds in KEY}

    runs = []
    for seed in seeds:
        for d in sorted(glob.glob(os.path.join(args.root, f'seed_{seed}', '*', '*', 'search-*'))):
            m = RUN_RE.search(os.path.basename(d))
            if not m or not os.path.exists(os.path.join(d, 'summary.json')):
                continue
            tgt, pred, step = m['tgt'], m['pred'], int(m['step'])
            r = size_metrics(analyse(d, tgt, pred, GT[tgt]), d, tgt, GT[tgt])
            for k in KS:
                r[f'oracle@{k}'] = r.get(f'oracle@{k}', float('nan'))
            r.update({'seed': seed, 'step': step, 'pred': pred, 'tgt': tgt})
            runs.append(r)
    print(f'{len(runs)} runs parsed')
    with open(os.path.join(args.out, 'guided_lut_runs.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['tgt', 'pred', 'step', 'seed'] + METRICS)
        for r in sorted(runs, key=lambda r: (r['tgt'], r['pred'], r['step'], r['seed'])):
            w.writerow([r['tgt'], r['pred'], r['step'], r['seed']] + [r.get(c, float('nan')) for c in METRICS])

    AGG = {}  # tgt -> pred -> {metric: {'mean': [31], 'std': [31], 'n': [31]}}
    with open(os.path.join(args.out, 'guided_lut_aggregate.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['tgt', 'pred', 'step', 'n_seeds'] + [f'{c}_{s}' for c in METRICS for s in ('mean', 'std')])
        for tgt in KEY:
            for pred in KEY:
                cell = AGG.setdefault(tgt, {}).setdefault(pred, {c: {'mean': [], 'std': [], 'n': []} for c in METRICS})
                for step in range(1, 32):
                    rs = [r for r in runs if r['tgt'] == tgt and r['pred'] == pred and r['step'] == step]
                    row = [tgt, pred, step, len(rs)]
                    for c in METRICS:
                        v = np.array([r.get(c, float('nan')) for r in rs], dtype=float); v = v[~np.isnan(v)]
                        mu = float(v.mean()) if len(v) else None; sd = float(v.std()) if len(v) else None
                        cell[c]['mean'].append(mu); cell[c]['std'].append(sd); cell[c]['n'].append(int(len(v)))
                        row += [mu, sd]
                    w.writerow(row)

    # ---- findings (numbers only from AGG) ----
    def m(tgt, pred, c, step):
        return AGG[tgt][pred][c]['mean'][step - 1]

    F = []
    for tgt in KEY:
        p10 = [m(tgt, tgt, 'pick@10', s) for s in range(1, 32)]
        best_s = int(np.nanargmax(p10)) + 1; top = np.nanmax(p10)
        sat = next((s for s in range(1, 32) if p10[s - 1] >= top - 0.1), best_s)
        g1, g5, g31 = p10[0], p10[4], p10[30]
        F.append(f"<b>{NAME[tgt]} (in-dataset)</b>: pick@10 {g1:.2f} at step 1 → {g5:.2f} at step 5 → {g31:.2f} at step 31 "
                 f"(best {top:.2f} at step {best_s}; within 0.1 of best from step {sat}, = {sat * INTERVAL} mb ≈ "
                 f"{sat * INTERVAL / MB_PER_EPOCH[tgt]:.1f} epochs). Random-620 mean best {rb[tgt]['mean_best']:.2f}, oracle@10 at step 31 "
                 f"{m(tgt, tgt, 'oracle@10', 31):.2f}, fidelity KT {m(tgt, tgt, 'fid_kt', 1):.2f} → {m(tgt, tgt, 'fid_kt', 31):.2f}.")
        others = [p for p in KEY if p != tgt]
        F.append(f"&nbsp;&nbsp;cross-dataset at step 31: pick@10 " + ', '.join(f"{p} predictor {m(tgt, p, 'pick@10', 31):.2f}" for p in others)
                 + f" vs in-dataset {g31:.2f}; fidelity KT " + ', '.join(f"{p} {m(tgt, p, 'fid_kt', 31):.2f}" for p in others) + '.')
        F.append(f"&nbsp;&nbsp;size: KT(score, FLOPs) {m(tgt, tgt, 'kt_score_flops', 1):.2f} at step 1 → {m(tgt, tgt, 'kt_score_flops', 31):.2f} at step 31 "
                 f"(whole-space KT(GT, FLOPs) {space[tgt]['kt_gt_flops']:.2f}); mean visited FLOPs {m(tgt, tgt, 'visited_flops', 1):.1f} → "
                 f"{m(tgt, tgt, 'visited_flops', 31):.1f} MFLOPs, top-20 picks {m(tgt, tgt, 'top20_flops', 1):.1f} → {m(tgt, tgt, 'top20_flops', 31):.1f}; "
                 f"params of top-20 {m(tgt, tgt, 'top20_params', 1):.3f} → {m(tgt, tgt, 'top20_params', 31):.3f} MB; "
                 f"size-controlled KT (within FLOPs quartiles) {m(tgt, tgt, 'kt_within_flops_q', 1):.2f} → {m(tgt, tgt, 'kt_within_flops_q', 31):.2f}.")
        F.append(f"&nbsp;&nbsp;bi-objective: front regret {m(tgt, tgt, 'front_regret', 1):.2f} → {m(tgt, tgt, 'front_regret', 31):.2f}, "
                 f"HV ratio {m(tgt, tgt, 'hv_ratio', 1):.2f} → {m(tgt, tgt, 'hv_ratio', 31):.2f}, front size {m(tgt, tgt, 'front_n', 31):.1f}; "
                 f"capped pick (half cap) {m(tgt, tgt, 'pick_caph', 1):.2f} → {m(tgt, tgt, 'pick_caph', 31):.2f} vs capped oracle {m(tgt, tgt, 'oracle_caph', 31):.2f}.")

    D = {'agg': AGG, 'rb': {ds: rb[ds]['mean_best'] for ds in KEY}, 'opt': opt, 'space': space, 'name': NAME,
         'mbpe': MB_PER_EPOCH, 'interval': INTERVAL, 'seeds': seeds, 'n_runs': len(runs), 'findings': F}
    html = HTML.replace('__DATA__', json.dumps(D))
    out = os.path.join(args.out, 'guided_lut_aggregate.html')
    open(out, 'w').write(html)
    print('wrote', out)
    for line in F:
        print(re.sub('<[^>]+>|&nbsp;', '', line))


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>nap2 guided budget sweep (lookup tables)</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:18px;color:#222;background:#fff;max-width:1500px}
h1{font-size:20px}h2{font-size:16px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
.row{display:flex;flex-wrap:wrap;gap:8px}.p{width:480px;height:340px}.note{color:#555;font-size:13px}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:3px 6px;text-align:right}th{background:#f3f3f3}
td:first-child{text-align:left}select,label{font-size:13px}.best{font-weight:bold;background:#eef7ee}
.find li{margin:4px 0;font-size:13px}.defs td{text-align:left}</style></head><body>
<h1>nap2-guided GA, budget sweep steps 1–31 (lookup tables, 75-mb cadence)</h1>
<p class="note" id="hdr"></p>
<h2>1. In-dataset: picked accuracy vs budget</h2>
<p class="note">Lines = mean over seeds, bands = ±1 std. Dashed grey = oracle@20 (true best 20 among visited), dotted = random-620 baseline (mean best GT of 620 random archs: compare with pick@1), thin top line = whole-space optimum. x = snapshot step (75 mini-batches each); hover shows epochs.</p>
<label>band metric: <select id="selk"><option value="pick@10" selected>pick@10</option><option value="pick@1">pick@1</option><option value="pick@5">pick@5</option><option value="pick@20">pick@20</option></select></label>
<div class="row" id="sec1"></div>
<h2>2. Cross-dataset: which predictor to use for a target, by budget</h2>
<label>metric: <select id="selx"><option value="pick@10" selected>pick@10</option><option value="pick@1">pick@1</option><option value="pick@5">pick@5</option><option value="pick@20">pick@20</option><option value="fid_kt">fidelity KT</option><option value="fid_kt_top10">KT top-10%</option></select></label>
<div class="row" id="sec2"></div>
<h2>3. Fidelity vs budget (in-dataset)</h2>
<div class="row" id="sec3"></div>
<h2>4. Size analysis: FLOPs and parameters vs budget (logged by the GA)</h2>
<p class="note">Sizes are the FLOPs / param_size_mb the GA logged for each evaluated architecture. KT(score, FLOPs) = how much the predictor's ranking follows size at that budget; the grey reference is KT(GT, FLOPs) on the visited set. "within FLOPs quartiles" = mean KT(score, GT) inside each FLOPs quartile of the visited set = ranking skill with size held roughly fixed.</p>
<div class="row" id="sec4"></div>
<h2>5. Bi-objective (score × FLOPs) vs budget</h2>
<p class="note">Front = final-population Pareto set evaluated in GT space. Front regret = mean over front points of (best visited GT with ≤ that FLOPs − point GT). HV ratio = hypervolume of the front / hypervolume of the visited oracle set. Capped pick = GT of the best-scored visited arch under a FLOPs cap (quarter / half / full of the dataset's cap) vs the capped oracle.</p>
<div class="row" id="sec5"></div>
<h2>6. Findings (numbers from the tables below)</h2><ul class="find" id="find"></ul>
<h2>7. Table</h2>
<label>step: <select id="selstep"></select></label> <label>target: <select id="seltgt"><option value="all">all</option><option>c10</option><option>c100</option><option>in16</option></select></label>
<div id="tbl"></div>
<h2>8. Column definitions</h2>
<table class="defs"><tr><th>column</th><th>meaning</th></tr>
<tr><td>step</td><td>nap2 budget = number of 75-mini-batch snapshots used by the predictor (k·75 mb; CIFAR epoch = 157 mb, IN16 epoch = 593 mb).</td></tr>
<tr><td>pick@k</td><td>Mean ground truth (NATS 200-epoch, seed average) of the k highest-scored unique visited architectures. pick@1 = deploy choice.</td></tr>
<tr><td>oracle@k</td><td>Mean ground truth of the true best k among the visited architectures (ceiling of a perfect ranker on the same visited set).</td></tr>
<tr><td>gap@20</td><td>oracle@20 − pick@20.</td></tr>
<tr><td>best_visited</td><td>Best ground truth among visited architectures (search quality independent of the final pick).</td></tr>
<tr><td>unique_ratio</td><td>Unique architectures / 620 evaluations.</td></tr>
<tr><td>fid_kt / fid_kt_top10</td><td>Kendall tau between nap2 score and ground truth over all visited archs / over the top 10% by ground truth.</td></tr>
<tr><td>visited_flops, visited_params</td><td>Mean logged FLOPs (MFLOPs) / params (MB) of the visited architectures = where the search spent its evaluations.</td></tr>
<tr><td>pick1_*, top20_*</td><td>Size of the pick@1 architecture / mean size of the 20 highest-scored.</td></tr>
<tr><td>kt_score_flops, kt_score_params</td><td>Kendall tau between nap2 score and size on the visited set (size bias of the predictor).</td></tr>
<tr><td>kt_gt_flops_visited</td><td>Kendall tau between ground truth and FLOPs on the visited set (how much size truly matters there).</td></tr>
<tr><td>kt_within_flops_q</td><td>Mean KT(score, GT) within FLOPs quartiles of the visited set (size-controlled ranking skill).</td></tr>
<tr><td>pick_cap{q,h,f} / oracle_cap{q,h,f}</td><td>GT of the best-scored visited arch with FLOPs ≤ cap / best GT under the same cap; caps = 1/8, 1/4, 1/2 of the dataset cap (200 MFLOPs CIFAR, 50 IN16).</td></tr>
<tr><td>front_*</td><td>Final Pareto front (score × FLOPs): size, FLOPs range, best / mean GT, regret, HV ratio.</td></tr></table>
<script>
const D=__DATA__; const T=['c10','c100','in16']; const COL={c10:'#1f77b4',c100:'#ff7f0e',in16:'#d62728'};
const STEPS=[...Array(31).keys()].map(i=>i+1);
document.getElementById('hdr').textContent=`${D.n_runs} runs = 3 predictors × 3 targets × 31 steps × seeds ${D.seeds.join('/')}. All cells are lookup-table runs (one fixed partial training per architecture, torch seed 42); live guided runs are not mixed in. Ground truth = NATS-Bench 200-epoch seed average.`;
function g(t,p,c,s){return D.agg[t][p][c][s];}
function ep(t){return STEPS.map(s=>(s*D.interval/D.mbpe[t]).toFixed(2));}
function band(t,p,c,color,name,dash){const mu=g(t,p,c,'mean'),sd=g(t,p,c,'std');const tr=[{x:STEPS,y:mu,name:name,mode:'lines+markers',marker:{size:4},line:{color:color,dash:dash||'solid'},text:ep(t).map(e=>e+' ep'),hovertemplate:'step %{x} (%{text}): %{y:.2f}<extra>'+name+'</extra>'}];
 if(sd&&sd[0]!=null){tr.push({x:[...STEPS,...STEPS.slice().reverse()],y:[...mu.map((v,i)=>v+sd[i]),...mu.map((v,i)=>v-sd[i]).reverse()],fill:'toself',fillcolor:color+'22',line:{color:'transparent'},hoverinfo:'skip',showlegend:false});}return tr;}
function div(parent,id){const d=document.createElement('div');d.className='p';d.id=id;document.getElementById(parent).appendChild(d);return d;}
const L=(title,y)=>({title:{text:title,font:{size:13}},margin:{t:34,l:48,r:10,b:40},xaxis:{title:'step (×75 mb)'},yaxis:{title:y},legend:{font:{size:10},orientation:'h',y:-0.25}});
function sec1(){const k=document.getElementById('selk').value;document.getElementById('sec1').innerHTML='';
 T.forEach(t=>{const tr=[...band(t,t,k,COL[t],k)];['pick@1','pick@5','pick@10','pick@20'].filter(x=>x!==k).forEach((x,i)=>tr.push({x:STEPS,y:g(t,t,x,'mean'),name:x,mode:'lines',line:{color:COL[t],width:1,dash:['dot','dashdot','longdash'][i]},opacity:.6}));
  tr.push({x:STEPS,y:g(t,t,'oracle@20','mean'),name:'oracle@20',mode:'lines',line:{color:'#888',dash:'dash'}});tr.push({x:STEPS,y:STEPS.map(_=>D.rb[t]),name:'random-620 mean best (vs pick@1)',mode:'lines',line:{color:'#aaa',dash:'dot'}});tr.push({x:STEPS,y:STEPS.map(_=>D.opt[t]),name:'optimum',mode:'lines',line:{color:'#ccc',width:1}});
  Plotly.newPlot(div('sec1','s1'+t),tr,L(D.name[t]+' — in-dataset predictor','ground truth (%)'),{responsive:true,displayModeBar:false});});}
function sec2(){const c=document.getElementById('selx').value;document.getElementById('sec2').innerHTML='';
 T.forEach(t=>{const tr=[];T.forEach(p=>tr.push(...band(t,p,c,COL[p],p+' predictor'+(p===t?' (in-dataset)':''),p===t?'solid':'dash')));
  if(c.startsWith('pick')){tr.push({x:STEPS,y:STEPS.map(_=>D.rb[t]),name:'random-620 mean best (vs pick@1)',mode:'lines',line:{color:'#aaa',dash:'dot'}});}
  Plotly.newPlot(div('sec2','s2'+t),tr,L('target '+D.name[t]+' — '+c,c),{responsive:true,displayModeBar:false});});}
function sec3(){T.forEach(t=>{const tr=[...band(t,t,'fid_kt',COL[t],'KT all visited'),...band(t,t,'fid_kt_top10',COL[t],'KT top-10% (GT)','dash'),{x:STEPS,y:g(t,t,'unique_ratio','mean'),name:'unique ratio',mode:'lines',line:{color:'#999',dash:'dot'},yaxis:'y2'}];
 const l=L(D.name[t]+' — fidelity','Kendall tau');l.yaxis2={title:'unique ratio',overlaying:'y',side:'right',range:[0,1]};Plotly.newPlot(div('sec3','s3'+t),tr,l,{responsive:true,displayModeBar:false});});}
function sec4(){T.forEach(t=>{const tr=[...band(t,t,'visited_flops',COL[t],'visited mean'),...band(t,t,'top20_flops','#2ca02c','top-20 picks'),...band(t,t,'pick1_flops','#9467bd','pick@1','dot')];
 Plotly.newPlot(div('sec4','s4a'+t),tr,L(D.name[t]+' — FLOPs of visited / picked','MFLOPs (logged)'),{responsive:true,displayModeBar:false});});
 T.forEach(t=>{const tr=[...band(t,t,'visited_params',COL[t],'visited mean'),...band(t,t,'top20_params','#2ca02c','top-20 picks'),...band(t,t,'pick1_params','#9467bd','pick@1','dot')];
 Plotly.newPlot(div('sec4','s4b'+t),tr,L(D.name[t]+' — params of visited / picked','params (MB, logged)'),{responsive:true,displayModeBar:false});});
 T.forEach(t=>{const tr=[...band(t,t,'kt_score_flops',COL[t],'KT(score, FLOPs)'),...band(t,t,'kt_score_params','#2ca02c','KT(score, params)','dash'),{x:STEPS,y:g(t,t,'kt_gt_flops_visited','mean'),name:'KT(GT, FLOPs) visited',mode:'lines',line:{color:'#888',dash:'dot'}},...band(t,t,'kt_within_flops_q','#9467bd','KT(score,GT) within FLOPs quartiles','dashdot')];
 Plotly.newPlot(div('sec4','s4c'+t),tr,L(D.name[t]+' — size bias and size-controlled skill','Kendall tau'),{responsive:true,displayModeBar:false});});}
function sec5(){T.forEach(t=>{const tr=[...band(t,t,'front_regret',COL[t],'front regret'),...band(t,t,'hv_ratio','#2ca02c','HV ratio','dash'),{x:STEPS,y:g(t,t,'front_n','mean'),name:'front size',mode:'lines',line:{color:'#999',dash:'dot'},yaxis:'y2'}];
 const l=L(D.name[t]+' — front quality','regret (pts) / HV ratio');l.yaxis2={title:'front size',overlaying:'y',side:'right'};Plotly.newPlot(div('sec5','s5a'+t),tr,l,{responsive:true,displayModeBar:false});});
 T.forEach(t=>{const tr=[];[['q','1/8 cap','#1f77b4'],['h','1/4 cap','#2ca02c'],['f','1/2 cap','#d62728']].forEach(([c,n,col])=>{tr.push(...band(t,t,'pick_cap'+c,col,'pick '+n));tr.push({x:STEPS,y:g(t,t,'oracle_cap'+c,'mean'),name:'oracle '+n,mode:'lines',line:{color:col,dash:'dot',width:1}});});
 Plotly.newPlot(div('sec5','s5b'+t),tr,L(D.name[t]+' — picks under FLOPs caps','ground truth (%)'),{responsive:true,displayModeBar:false});});}
document.getElementById('find').innerHTML=D.findings.map(f=>'<li>'+f+'</li>').join('');
const TC=['pick@1','pick@5','pick@10','pick@20','oracle@20','gap@20','best_visited','unique_ratio','fid_kt','fid_kt_top10','visited_flops','top20_flops','pick1_flops','top20_params','kt_score_flops','kt_within_flops_q','front_regret','hv_ratio','pick_caph','oracle_caph'];
const ss=document.getElementById('selstep');STEPS.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;if(s===31)o.selected=true;ss.appendChild(o);});
function tbl(){const s=+ss.value,tf=document.getElementById('seltgt').value;let h='<table><tr><th>target</th><th>predictor</th>'+TC.map(c=>'<th>'+c+'</th>').join('')+'</tr>';
 T.filter(t=>tf==='all'||t===tf).forEach(t=>T.forEach(p=>{h+='<tr><td>'+D.name[t]+'</td><td>'+p+(p===t?' (in)':'')+'</td>'+TC.map(c=>{const v=g(t,p,c,'mean')[s-1],sd=g(t,p,c,'std')[s-1];return '<td>'+(v==null?'n/a':(v.toFixed(c.startsWith('fid')||c.startsWith('kt')||c==='unique_ratio'||c==='hv_ratio'?3:2)+(sd!=null?' <span style="color:#999">±'+sd.toFixed(2)+'</span>':'')))+'</td>';}).join('')+'</tr>';}));
 document.getElementById('tbl').innerHTML=h+'</table>';}
document.getElementById('selk').onchange=sec1;document.getElementById('selx').onchange=sec2;ss.onchange=tbl;document.getElementById('seltgt').onchange=tbl;
sec1();sec2();sec3();sec4();sec5();tbl();
</script></body></html>'''


if __name__ == '__main__':
    main()
