"""Accuracy of all models: Table 2, single-model spread, forward tests, self-conditioning and depth.

  r_bar : mean within-environment Pearson r (environments with 10+ test cells)
  rho   : Spearman r of the genotype averages of environment-centred yields (genotypes with 3+ test cells)
  python evaluate.py   -> results/*.csv and a printed summary
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import genotype_means, per_env_r

HERE = Path(__file__).resolve().parent
MODELS = [('GBLUP', 'gblup', 'GBLUP'), ('GBLUP + W', 'gblup', 'GBLUP_W'), ('CARD', 'card', None),
          ('GNN only', 'gnn_only', None), ('GxEDiff', 'gxediff', None)]


def load_model(design, model, key=None, seeds=None):
    """(y, h, e, [prediction per trained model]) for one model on one test design; None if not run yet."""
    if model == 'gblup':
        f = design / 'gblup/preds.npz'
        if not f.exists():
            return None
        a = np.load(f)
        return a['y'], a['h'], a['e'], [a[f'{key}_mean']]
    files = sorted(design.glob(f'{model}_s*/test_samples.npz'), key=lambda f: int(f.parent.name.rsplit('_s', 1)[1]))
    files = [f for f in files if seeds is None or int(f.parent.name.rsplit('_s', 1)[1]) in seeds]
    if not files:
        return None
    preds = []
    for f in files:
        a = np.load(f)
        preds.append(a['S'].mean(0))
    return a['y'], a['h'], a['e'], preds


def r_bar(y, p, e):
    return float(np.mean(list(per_env_r(y, p, e).values())))


def rho(y, p, h, e):
    t, q = genotype_means(y, h, e), genotype_means(p, h, e)
    return float(spearmanr(q.loc[t.index], t).statistic)


def scores(res, mask):
    y, h, e, preds = res
    p = np.mean(preds, 0)
    one = [(r_bar(y[mask], q[mask], e[mask]), rho(y[mask], q[mask], h[mask], e[mask])) for q in preds]
    return dict(r_bar=r_bar(y[mask], p[mask], e[mask]), rho=rho(y[mask], p[mask], h[mask], e[mask]), models=len(preds),
                single_r_bar=np.mean([o[0] for o in one]), single_r_bar_sd=np.std([o[0] for o in one], ddof=1) if len(one) > 1 else np.nan,
                single_rho=np.mean([o[1] for o in one]), single_rho_sd=np.std([o[1] for o in one], ddof=1) if len(one) > 1 else np.nan)


def seen_mask(cells, h, year):
    return np.isin(h, cells[cells.Year < year].h.unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(HERE / 'data'))
    ap.add_argument('--runs', default=str(HERE / 'runs'))
    ap.add_argument('--out', default=str(HERE / 'results'))
    a = ap.parse_args()
    RUNS, OUT = Path(a.runs), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    g2f = pickle.load(open(Path(a.data_dir) / 'g2f.pkl', 'rb'))['cells']
    pd.set_option('display.width', 200)

    rows = []
    for label, model, key in MODELS:
        for test, design, subsets in [('G2F 2024', RUNS / 'g2f2024', ['seen', 'unseen']), ('MaizeGEP', RUNS / 'maizegep', ['unseen'])]:
            res = load_model(design, model, key)
            if res is None:
                continue
            seen = seen_mask(g2f, res[1], 2024) if test == 'G2F 2024' else np.zeros(len(res[0]), bool)
            for sub in subsets:
                m = seen if sub == 'seen' else ~seen
                rows.append(dict(model=label, test=test, genotypes=sub, n=len(np.unique(res[1][m])), **scores(res, m)))
    t2 = pd.DataFrame(rows)
    t2.round(4).to_csv(OUT / 'table2.csv', index=False)
    print('Table 2 (all trained models averaged) and single trained models (mean, SD)')
    print(t2.round(3).to_string(index=False), '\n')

    rows = []
    for year in [2018, 2020, 2022, 2024]:
        for window, design in [('all earlier years', RUNS / f'g2f{year}'),
                               ('six earlier years', RUNS / (f'g2f{year}_from{year - 6}' if year >= 2022 else f'g2f{year}'))]:
            for label, model, key in [MODELS[0], MODELS[3], MODELS[4]]:
                res = load_model(design, model, key, seeds=None if model == 'gblup' else (0, 1, 2))
                if res is None:
                    continue
                m = ~seen_mask(g2f, res[1], year)
                rows.append(dict(year=year, window=window, model=label, r_bar=r_bar(res[0][m], np.mean(res[3], 0)[m], res[2][m])))
    if rows:
        fw = pd.DataFrame(rows).pivot_table(index=['window', 'year'], columns='model', values='r_bar')
        fw.round(4).to_csv(OUT / 'forward_tests.csv')
        print('Forward tests, unseen hybrids, r_bar (3 trained models each)')
        print(fw.round(3).to_string(), '\n')

    design = RUNS / 'g2f2024'
    rows = []
    for label, model, seeds in [('GxEDiff, 3 layers', 'gxediff_l3', None), ('GxEDiff, 5 layers', 'gxediff', (0, 1, 2)),
                                ('GxEDiff, 15 layers', 'gxediff_l15', None),
                                ('GxEDiff without self-conditioning', 'gxediff_nosc', None)]:
        res = load_model(design, model, seeds=seeds)
        if res is None:
            continue
        m = ~seen_mask(g2f, res[1], 2024)
        rows.append(dict(variant=label, models=len(res[3]), r_bar_all=r_bar(res[0], np.mean(res[3], 0), res[2]),
                         r_bar_unseen=r_bar(res[0][m], np.mean(res[3], 0)[m], res[2][m])))
    if rows:
        ab = pd.DataFrame(rows)
        ab.round(4).to_csv(OUT / 'ablation.csv', index=False)
        print('Depth and self-conditioning (G2F 2024)')
        print(ab.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
