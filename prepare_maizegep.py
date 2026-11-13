"""MaizeGEP inputs (2017-2023): the 70 varieties never grown in 2017-2020 are the test genotypes (newvar).

  cells   : mean of the survey records per variety x county-year (z-scored yields), record count, within-cell variance
  markers : 12,232 SNPs (0/1/2), centred; 10 closest relatives (VanRaden)
  weather : weekly means (35 weeks, March-October) of 5 daily variables (LSTM input)
  ec      : monthly means of 9 variables (GBLUP + W)
  python prepare_maizegep.py --raw <MaizeGEP download folder>   -> data/maizegep.pkl
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_g2f import kinship_neighbours

HERE = Path(__file__).resolve().parent
FEATS = ['SSD', 'tem_Max', 'tem_Min', 'PRE', 'RHU']
EC_VARS = ['tem_Max', 'tem_Min', 'tem_Avg', 'win_S', 'win_Max', 'PRS', 'RHU', 'PRE', 'SSD']
MONTHS = [('March', 31), ('April', 30), ('May', 31), ('June', 30), ('July', 31), ('August', 31), ('September', 30),
          ('October', 31)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=str(HERE / 'raw/MaizeGEP'))
    ap.add_argument('--out', default=str(HERE / 'data/maizegep.pkl'))
    a = ap.parse_args()
    RAW = Path(a.raw)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    ph = pd.read_excel(RAW / 'normalized_PHE.xlsx')
    ph = ph[ph.Yield.notna() & (ph.Year <= 2023)].copy()
    ph['Env'] = ph['Year-adcode'].astype(str)
    ph['Hybrid'] = ph.Name_Code.astype(str)
    g = pd.read_csv(RAW / 'extracted/ID_gen_260_12233.txt', sep=r'\s+', header=None, index_col=0)
    g.index = g.index.astype(str)
    g = g.loc[:, g.std() > 0]

    grp = ph.groupby(['Env', 'Hybrid'])
    cells = grp.Yield.agg(y='mean', n='size').reset_index()
    dev = ph.Yield - grp.Yield.transform('mean')
    ss = (dev ** 2).groupby(ph.Env).sum()
    df = ph.groupby('Env').size() - cells.groupby('Env').size()
    s2 = (ss / df).where(df >= 5)
    cells['s2_env'] = cells.Env.map(s2).fillna(np.nanmedian(s2.values))

    hybrids = sorted(cells.Hybrid.unique())
    envs = sorted(cells.Env.unique())
    cells['h'] = pd.Index(hybrids).get_indexer(cells.Hybrid)
    cells['e'] = pd.Index(envs).get_indexer(cells.Env)
    cells['Year'] = cells.Env.str[:4].astype(int)
    cells['newvar'] = ~cells.Hybrid.isin(set(cells[cells.Year <= 2020].Hybrid))
    cells['test'] = cells.newvar

    env = pd.read_excel(RAW / 'extracted/normalized_ENV.xlsx')
    env['Env'] = env['Year-adcode'].astype(str)
    env = env.set_index('Env').loc[envs]
    day = lambda v: [v] + [f'{v}.{k}' for k in range(1, 245)]
    W = np.stack([env[day(v)].values.reshape(len(envs), 35, 7).mean(2) for v in FEATS], -1).astype(np.float32)
    ec, ec_names, start = [], [], 0
    for mname, nd in MONTHS:
        for v in EC_VARS:
            ec.append(env[day(v)].values[:, start:start + nd].mean(1))
            ec_names.append(f'{v}_{mname}')
        start += nd
    EC = np.stack(ec, 1).astype(np.float32)
    env_meta = pd.DataFrame({'Env': envs, 'Site': env.ADCODE.astype(str).values, 'Year': [int(x[:4]) for x in envs],
                             'plant': pd.NaT, 'lon': env.Longitude.values, 'lat': env.Latitude.values,
                             'zone': env['Agro.ecological.zone'].values})

    G = g.loc[hybrids].values.astype(np.float32)
    p = G.mean(0) / 2
    M = G - G.mean(0, keepdims=True)
    kin = kinship_neighbours(M, p)
    hyb_meta = pd.DataFrame({'Hybrid': hybrids, 'Hybrid_Parent1': np.nan, 'Hybrid_Parent2': np.nan})

    pickle.dump(dict(cells=cells, hybrids=hybrids, envs=envs, env_meta=env_meta, hyb_meta=hyb_meta, M=M, weather=W,
                     kin=kin, feats=FEATS, ec=EC, ec_names=ec_names), open(a.out, 'wb'))
    print(f'{len(cells)} cells, {len(hybrids)} varieties, {len(envs)} county-years, {M.shape[1]} markers -> {a.out}')


if __name__ == '__main__':
    main()
