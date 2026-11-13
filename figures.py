"""Paper figures from the saved runs.

  python figures.py [overview avgperf forward analysis]   -> figures/*.png
"""
import argparse
import pickle
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr

from common import genotype_means, per_env_r
from evaluate import load_model, seen_mask

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID, SURF, REF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff', '#a8a7a1'
COL = {'GBLUP': ('#eb6834', 's', '#fbe0d4'), 'CARD': ('#8a8984', 'D', '#e6e5e1'),
       'GNN only': ('#1baf7a', '^', '#d3efe4'), 'GxEDiff': ('#2a78d6', 'o', '#d6e6fa')}
MODEL_DIR = {'GBLUP': ('gblup', 'GBLUP'), 'CARD': ('card', None), 'GNN only': ('gnn_only', None), 'GxEDiff': ('gxediff', None)}
YEARS = [2018, 2020, 2022, 2024]
plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif', 'font.sans-serif': ['Nimbus Sans', 'DejaVu Sans'],
                     'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.png', facecolor=SURF, dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('saved', out / f'{name}.png')


def axstyle(ax, grid='y'):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK, width=0.6, length=2.5, pad=1.5)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, loc='left', fontsize=7.5, color=INK, pad=3)


def overview(out):
    steps = [('1  Yield table', 'Genotype × environment\nyields; most cells are\nempty. Inputs: SNP\nmarkers, LSTM weather\nvectors'),
             ('2  Graph', 'Nodes: genotypes and\nenvironments. Edges:\nmeasured yields, the 10\nclosest relatives, the 10\nmost similar seasons'),
             ('3  Training', 'Hide some measured\nyields and add noise;\nthe graph network learns\nto recover them'),
             ('4  Filling', 'Empty cells start as\nnoise and are cleaned in\n50 rounds; each round\npasses its estimates on.\nPrediction: mean of 1,000\nfilled tables')]
    fig, ax = plt.subplots(figsize=(7.0, 1.17))
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 1.17)
    ax.axis('off')
    W, H, gap = 1.6, 1.12, 0.17
    for k, (head, body) in enumerate(steps):
        x = 0.02 + k * (W + gap)
        ax.add_patch(FancyBboxPatch((x, 0.02), W, H, boxstyle='round,pad=0,rounding_size=0.06', facecolor='white',
                                    edgecolor='black', linewidth=1.2))
        ax.text(x + 0.07, 0.02 + H - 0.11, head, fontsize=8.5, fontweight='bold', va='center')
        ax.text(x + 0.07, 0.02 + H - 0.22, body, fontsize=7.2, va='top', linespacing=1.2)
        if k < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + W + 0.015, 0.02 + H / 2), (x + W + gap - 0.015, 0.02 + H / 2),
                                         arrowstyle='-|>', mutation_scale=9, color='black', linewidth=1))
    save(fig, out, 'Fig_overview')


def model_preds(design, names, seeds=None):
    out, ref = {}, None
    for n in names:
        model, key = MODEL_DIR[n]
        res = load_model(design, model, key, seeds=None if model == 'gblup' else seeds)
        if res is None:
            continue
        if ref is None:
            ref = res[:3]
        assert (res[0] == ref[0]).all() and (res[1] == ref[1]).all() and (res[2] == ref[2]).all(), (design, n)
        out[n] = np.mean(res[3], 0)
    return ref, out


def avgperf(out, runs, cells, groups=5):
    (y, h, e), preds = model_preds(runs / 'g2f2024', ['GBLUP', 'CARD', 'GNN only', 'GxEDiff'])
    m = ~seen_mask(cells, h, 2024)
    y, h, e = y[m], h[m], e[m]
    true = genotype_means(y, h, e)
    fig, ax = plt.subplots(figsize=(3.3, 2.05))
    axstyle(ax)
    x = np.arange(1, groups + 1)
    ideal = np.array([g.mean() for g in np.array_split(np.sort(true.values), groups)])
    ax.plot(x, ideal, color=REF, linestyle='--', linewidth=0.9, label='perfect ranking')
    labels = {'GBLUP': 'GBLUP, GBLUP + W'}
    for n, p in preds.items():
        pm = genotype_means(p[m], h, e).loc[true.index]
        order = np.argsort(pm.values, kind='stable')
        ys = np.array([true.values[ix].mean() for ix in np.array_split(order, groups)])
        r = spearmanr(pm.values, true.values).statistic
        c, mk, _ = COL[n]
        ax.plot(x, ys, color=c, marker=mk, linewidth=1.1, markersize=3, label=f'{labels.get(n, n)} ($\\rho$ {r:.2f})')
        print(f'{n}: top 20% {ys[-1]:+.2f} t/ha, bottom 20% {ys[0]:+.2f} t/ha')
    ax.axhline(0, color=REF, linewidth=0.6)
    ax.set_xticks(x, ['lowest\n20%', '', 'middle', '', 'highest\n20%'])
    ax.set_xlabel('hybrids grouped by predicted average', color=INK)
    ax.set_ylabel('average yield above field mean (t/ha)', color=INK)
    ax.legend(loc='upper left', frameon=False, fontsize=5.6, handlelength=1.8, labelspacing=0.25)
    save(fig, out, 'Fig_avgperf')


def env_scores(design, cells, year, names, unseen_only=True):
    ref, preds = model_preds(design, names, seeds=(0, 1, 2))
    if ref is None:
        return None
    y, h, e = ref
    m = ~seen_mask(cells, h, year) if unseen_only else np.ones(len(y), bool)
    return {n: per_env_r(y[m], p[m], e[m]) for n, p in preds.items()}


def boxes(ax, groups, names, w=0.27, dot=2.0):
    offs = np.linspace(-w, w, len(names))
    r0 = np.random.default_rng(0)
    for gi, (label, r) in enumerate(groups):
        for mi, n in enumerate(names):
            if n not in r:
                continue
            v = np.array(list(r[n].values()))
            c, _, tint = COL[n]
            ax.boxplot([v], positions=[gi + offs[mi]], widths=w * 0.85, patch_artist=True, showfliers=False,
                       whis=(0, 100), medianprops=dict(color=INK, linewidth=0.9),
                       whiskerprops=dict(color=c, linewidth=0.6), capprops=dict(color=c, linewidth=0.6),
                       boxprops=dict(facecolor=tint, edgecolor=c, linewidth=0.6))
            ax.scatter(gi + offs[mi] + r0.uniform(-w * 0.28, w * 0.28, len(v)), v, s=dot, color=c, alpha=0.6,
                       linewidths=0, zorder=3)
            ax.scatter(gi + offs[mi], v.mean(), marker='D', s=7, color=SURF, edgecolor=INK, linewidth=0.5, zorder=4)
    ax.axhline(0, color=REF, linewidth=0.7)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlim(-0.55, len(groups) - 0.45)


def forward(out, runs, cells):
    names = ['GBLUP', 'GNN only', 'GxEDiff']
    full = [(f"{Y}\n'14-'{str(Y - 1)[2:]}", env_scores(runs / f'g2f{Y}', cells, Y, names)) for Y in YEARS]
    win = []
    for Y in YEARS:
        start = max(2014, Y - 6)
        d = runs / (f'g2f{Y}_from{start}' if start > 2014 else f'g2f{Y}')
        win.append((f"{Y}\n'{str(start)[2:]}-'{str(Y - 1)[2:]}", env_scores(d, cells, Y, names)))
    panels = [('(a) All earlier years', [g for g in full if g[1]]), ('(b) Six earlier years', [g for g in win if g[1]])]
    loyo = [(str(Y), env_scores(runs / f'g2f_loyo{Y}', cells, Y, names, unseen_only=False)) for Y in range(2014, 2024)]
    loyo = [g for g in loyo if g[1]]
    if loyo:
        panels.append(('(c) Leave-one-year-out', loyo))
    fig = plt.figure(figsize=(7.0, 1.6))
    gs = fig.add_gridspec(1, len(panels), width_ratios=[1, 1, 2.25][:len(panels)], wspace=0.07)
    ax0 = None
    for k, (head, groups) in enumerate(panels):
        ax = fig.add_subplot(gs[k], sharey=ax0)
        ax0 = ax0 or ax
        axstyle(ax)
        boxes(ax, groups, names, dot=1.6 if k < 2 else 1.2)
        title(ax, head)
        if k:
            ax.tick_params(labelleft=False)
    ax0.set_ylim(-0.35, 0.8)
    ax0.set_ylabel('within-site-year r', color=INK)
    handles = [plt.Line2D([], [], color=COL[n][0], marker='s', linestyle='none', markersize=5,
                          markerfacecolor=COL[n][2], markeredgecolor=COL[n][0]) for n in names]
    fig.legend(handles, names, loc='upper center', ncol=len(names), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.1), handletextpad=0.2, columnspacing=1.1)
    save(fig, out, 'Fig_forward')


def analysis(out, runs, cells):
    design = runs / 'g2f2024'
    (y, h, e), preds = model_preds(design, ['GBLUP', 'CARD', 'GNN only', 'GxEDiff'])
    allm, new = np.ones(len(y), bool), ~seen_mask(cells, h, 2024)
    score = lambda p, m: float(np.mean(list(per_env_r(y[m], p[m], e[m]).values())))
    fig = plt.figure(figsize=(7.0, 1.38))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    axstyle(ax)
    variants = ['CARD', 'GNN only', 'GxEDiff']
    for gi, m in enumerate([allm, new]):
        for vi, n in enumerate(variants):
            ax.bar(gi * 4 + vi, score(preds[n], m), color=COL[n][0], width=0.8, edgecolor=SURF, linewidth=0.3)
        gbv = score(preds['GBLUP'], m)
        ax.plot([gi * 4 - 0.5, gi * 4 + 2.5], [gbv, gbv], color=COL['GBLUP'][0], linestyle='--', linewidth=0.9)
    ax.set_xticks([1.0, 5.0], ['all of 2024', 'unseen hybrids'])
    ax.set_ylim(0, 0.62)
    ax.set_ylabel('within-site-year r', color=INK)
    title(ax, '(a) Graph and diffusion')
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=COL[n][0]) for n in variants] +
              [plt.Line2D([], [], color=COL['GBLUP'][0], linestyle='--', linewidth=0.9)],
              variants + ['GBLUP'], loc='upper left', ncol=2, frameon=False,
              fontsize=5.4, handlelength=0.9, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.1)

    ax = fig.add_subplot(gs[1])
    axstyle(ax)
    for m, lab, ls in [(allm, 'all', '-'), (new, 'unseen', '--')]:
        xs, ys = [], []
        for L, model, seeds in [(3, 'gxediff_l3', None), (5, 'gxediff', (0, 1, 2)), (15, 'gxediff_l15', None)]:
            res = load_model(design, model, seeds=seeds)
            if res is None:
                continue
            xs.append(L)
            ys.append(score(np.mean(res[3], 0), m))
            for p in res[3]:
                ax.scatter(L, score(p, m), s=3, color=COL['GxEDiff'][0], alpha=0.35, linewidths=0)
        ax.plot(xs, ys, color=COL['GxEDiff'][0], linestyle=ls, marker='o', linewidth=1, markersize=3)
        ax.text(xs[-1] + 0.6, ys[-1], lab, color=INK, fontsize=6, va='center')
    ax.set_xticks([3, 5, 15])
    ax.set_xlim(2, 18)
    ax.set_xlabel('message-passing layers', color=INK)
    title(ax, '(b) Depth')

    ax = fig.add_subplot(gs[2])
    axstyle(ax)
    per = load_model(design, 'gxediff')[3]
    rng = np.random.default_rng(0)
    for m, lab, ls in [(allm, 'all', '-'), (new, 'unseen', '--')]:
        med, lo, hi = [], [], []
        for K in range(1, len(per) + 1):
            combs = list(combinations(range(len(per)), K))
            if len(combs) > 60:
                combs = [tuple(rng.choice(len(per), K, replace=False)) for _ in range(60)]
            vals = [score(np.mean([per[i] for i in c], 0), m) for c in combs]
            med.append(np.median(vals))
            lo.append(np.percentile(vals, 10))
            hi.append(np.percentile(vals, 90))
        Ks = np.arange(1, len(per) + 1)
        ax.plot(Ks, med, color=COL['GxEDiff'][0], linestyle=ls, linewidth=1)
        ax.fill_between(Ks, lo, hi, color=COL['GxEDiff'][0], alpha=0.15, linewidth=0)
        ax.text(Ks[-1] + 0.3, med[-1], lab, color=INK, fontsize=6, va='center')
    ax.set_xticks([1, 5, 10])
    ax.set_xlim(0.5, 11.8)
    ax.set_xlabel('trained models averaged', color=INK)
    title(ax, '(c) Ensemble size')
    save(fig, out, 'Fig_analysis')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('which', nargs='*', default=['overview', 'avgperf', 'forward', 'analysis'])
    ap.add_argument('--data-dir', default=str(HERE / 'data'))
    ap.add_argument('--runs', default=str(HERE / 'runs'))
    ap.add_argument('--out', default=str(HERE / 'figures'))
    a = ap.parse_args()
    out, runs = Path(a.out), Path(a.runs)
    cells = pickle.load(open(Path(a.data_dir) / 'g2f.pkl', 'rb'))['cells']
    if 'overview' in a.which:
        overview(out)
    if 'avgperf' in a.which:
        avgperf(out, runs, cells)
    if 'forward' in a.which:
        forward(out, runs, cells)
    if 'analysis' in a.which:
        analysis(out, runs, cells)


if __name__ == '__main__':
    main()
