"""G2F inputs: train 2014-2023, test on the official 2024 yields.

  cells   : one value per hybrid x site-year (replicate-adjusted mean of the plots), plot count, plot variance
  markers : the release's 2,425-marker panel, coded 0/1/2 and centred; 10 closest relatives (VanRaden)
  weather : weekly means of 5 daily variables, 14 days before to 150 days after planting (LSTM input)
  ec      : 16 variables averaged in six crop stages (GBLUP + W)
  python prepare_g2f.py --raw <G2F 2024 competition folder>   -> data/g2f.pkl
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
FEATS = ['ALLSKY_SFC_PAR_TOT', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DNI']
RAIN = FEATS.index('PRECTOTCORR')
K_KIN = 10
PRE_DAYS, POST_DAYS = 14, 150
EC_VARS = ['RH2M', 'T2M_MAX', 'ALLSKY_SFC_SW_DWN', 'T2MWET', 'GWETTOP', 'QV2M', 'GWETPROF', 'T2M_MIN', 'T2MDEW', 'PS',
           'T2M', 'GWETROOT', 'ALLSKY_SFC_PAR_TOT', 'WS2M', 'ALLSKY_SFC_SW_DNI', 'PRECTOTCORR']
STAGES = [0, 14, 35, 60, 90, 120, POST_DAYS + 1]


def stage1(tr):
    """Per site-year least squares: plot yield ~ hybrid + replicate."""
    rows, s2 = [], {}
    for env, d in tr.groupby('Env'):
        hyb = pd.Categorical(d.Hybrid)
        rep = pd.Categorical(d.Replicate.astype(str))
        hc, rc = hyb.codes.astype(np.int64), rep.codes.astype(np.int64)
        nh, nr, n = len(hyb.categories), len(rep.categories), len(d)
        X = np.zeros((n, nh + nr - 1))
        X[np.arange(n), hc] = 1
        if nr > 1:
            m = rc > 0
            X[np.where(m)[0], nh + rc[m] - 1] = 1
        y = d.Yield_Mg_ha.values
        coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        df = n - rank
        s2[env] = (resid @ resid) / df if df >= 5 else np.nan
        rep_eff = np.r_[0.0, coef[nh:]] if nr > 1 else np.array([0.0])
        blue = coef[:nh] + rep_eff.mean()
        cnt = np.bincount(hc, minlength=nh)
        rows += [(env, h, blue[k], cnt[k]) for k, h in enumerate(hyb.categories)]
    cells = pd.DataFrame(rows, columns=['Env', 'Hybrid', 'y', 'n'])
    s2 = pd.Series(s2)
    cells['s2_env'] = cells.Env.map(s2).fillna(np.nanmedian(s2.values))
    return cells, s2


def mode_date(s):
    s = pd.to_datetime(s, format='mixed', errors='coerce').dropna()
    return s.mode().iloc[0] if len(s) else pd.NaT


def weekly_weather(wx, envs, plant):
    wx = wx.copy()
    wx['date'] = pd.to_datetime(wx.Date.astype(str), format='%Y%m%d')
    seqs = []
    for e in envs:
        d = wx[wx.Env == e].set_index('date')[FEATS].astype(float).sort_index()
        p = plant[e]
        s = d.loc[p - pd.Timedelta(days=PRE_DAYS): p + pd.Timedelta(days=POST_DAYS)].values
        assert len(s) == PRE_DAYS + POST_DAYS + 1, (e, len(s))
        seqs.append(s)
    X = np.stack(seqs)
    X = np.where(np.isnan(X), np.nanmean(X, axis=(0, 1)), X)
    nw = int(np.ceil(X.shape[1] / 7))
    W = np.zeros((len(envs), nw, len(FEATS)), np.float32)
    for k in range(nw):
        blk = X[:, 7 * k: 7 * (k + 1)]
        W[:, k] = blk.mean(1)
        W[:, k, RAIN] = np.log1p(np.clip(blk[..., RAIN], 0, None).sum(1))
    return W


def stage_ecs(wx, envs, plant):
    rows = []
    for e in envs:
        d = wx[wx.Env == e].set_index('date')[EC_VARS].astype(float).sort_index()
        p = plant[e]
        v = []
        for a, b in zip(STAGES[:-1], STAGES[1:]):
            blk = d.loc[p + pd.Timedelta(days=a): p + pd.Timedelta(days=b - 1)]
            m = blk.mean()
            m['PRECTOTCORR'] = blk.PRECTOTCORR.clip(lower=0).sum()
            v.append(m.values)
        rows.append(np.concatenate(v))
    X = np.array(rows, dtype=np.float64)
    X = np.where(np.isnan(X), np.nanmean(X, 0), X)
    names = [f'{v}_stage{k + 1}' for k in range(len(STAGES) - 1) for v in EC_VARS]
    return X.astype(np.float32), names


def kinship_neighbours(M, p):
    with torch.no_grad():
        Mt = torch.tensor(M, device='cuda' if torch.cuda.is_available() else 'cpu')
        K = Mt @ Mt.T / float(2 * (p * (1 - p)).sum())
        K.fill_diagonal_(-1e9)
        val, nb = K.topk(K_KIN, dim=1)
    nb, val = nb.cpu().numpy(), val.cpu().numpy()
    return nb.ravel(), np.repeat(np.arange(len(M)), K_KIN), val.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=str(HERE / 'raw/G2F_2024'))
    ap.add_argument('--out', default=str(HERE / 'data/g2f.pkl'))
    a = ap.parse_args()
    RAW = Path(a.raw)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    g = pd.read_csv(RAW / 'Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt', sep='\t', skiprows=1,
                    index_col=0, low_memory=False)
    g.index = g.index.astype(str)
    g = g.apply(pd.to_numeric, errors='coerce')
    g = g.loc[:, g.std() > 0]

    wx = pd.concat([pd.read_csv(RAW / 'Training_data/4_Training_Weather_Data_2014_2023_full_year.csv'),
                    pd.read_csv(RAW / 'Testing_data/4_Testing_Weather_Data_2024_full_year.csv')], ignore_index=True)
    wx['date'] = pd.to_datetime(wx.Date.astype(str), format='%Y%m%d')

    tr = pd.read_csv(RAW / 'Training_data/1_Training_Trait_Data_2014_2023.csv', low_memory=False)
    tr['Hybrid'] = tr.Hybrid.astype(str)
    tr = tr[tr.Yield_Mg_ha.notna() & tr.Hybrid.isin(g.index) & tr.Env.isin(set(wx.Env))].copy()
    cells_tr, s2 = stage1(tr)

    te = pd.read_csv(RAW / 'Testing_data/7_Testing_Observed_Values.csv')
    te = te[te.Yield_Mg_ha.notna()].rename(columns={'Yield_Mg_ha': 'y'})
    te['Hybrid'] = te.Hybrid.astype(str)
    site_tr = cells_tr.Env.str[:-5]
    n_site = cells_tr.groupby(site_tr).n.median()
    s2_site = s2.groupby(s2.index.str[:-5]).median()
    te_site = te.Env.str[:-5]
    te['n'] = te_site.map(n_site).fillna(cells_tr.n.median()).values
    te['s2_env'] = te_site.map(s2_site).fillna(np.nanmedian(s2.values)).values
    cells = pd.concat([cells_tr, te[['Env', 'Hybrid', 'y', 'n', 's2_env']]], ignore_index=True)

    hybrids = sorted(cells.Hybrid.unique())
    envs = sorted(cells.Env.unique())
    cells['h'] = pd.Index(hybrids).get_indexer(cells.Hybrid)
    cells['e'] = pd.Index(envs).get_indexer(cells.Env)
    cells['Year'] = cells.Env.str[-4:].astype(int)
    cells['test'] = cells.Year == 2024

    meta24 = pd.read_csv(RAW / 'Testing_data/2_Testing_Meta_Data_2024.csv', encoding='latin1').set_index('Env')
    plant = tr.groupby('Env').Date_Planted.agg(mode_date)
    plant = pd.concat([plant, pd.to_datetime(meta24.Date_Planted, format='%m/%d/%Y')])
    plant = plant[~plant.index.duplicated()].reindex(envs)
    env_meta = pd.DataFrame({'Env': envs, 'Site': [e[:-5] for e in envs], 'Year': [int(e[-4:]) for e in envs],
                             'plant': plant.values})
    W = weekly_weather(wx.drop(columns='date'), envs, plant)
    EC, ec_names = stage_ecs(wx, envs, plant)

    G = g.loc[hybrids]
    G = (G.fillna(G.mean()) * 2).values.astype(np.float32)
    p = G.mean(0) / 2
    M = G - G.mean(0, keepdims=True)
    kin = kinship_neighbours(M, p)

    meta = tr.drop_duplicates('Hybrid').set_index('Hybrid')[['Hybrid_Parent1', 'Hybrid_Parent2']]
    hyb_meta = pd.DataFrame({'Hybrid': hybrids}).join(meta, on='Hybrid')
    sp = hyb_meta.Hybrid.str.split('/', n=1, expand=True)
    hyb_meta['Hybrid_Parent1'] = hyb_meta.Hybrid_Parent1.fillna(sp[0])
    hyb_meta['Hybrid_Parent2'] = hyb_meta.Hybrid_Parent2.fillna(sp[1])

    pickle.dump(dict(cells=cells, hybrids=hybrids, envs=envs, env_meta=env_meta, hyb_meta=hyb_meta, M=M, weather=W,
                     kin=kin, feats=FEATS, ec=EC, ec_names=ec_names), open(a.out, 'wb'))
    print(f'{len(cells)} cells, {len(hybrids)} hybrids, {len(envs)} site-years, {M.shape[1]} markers -> {a.out}')


if __name__ == '__main__':
    main()
