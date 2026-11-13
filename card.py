"""CARD baseline (Han, Zheng & Zhou 2022): diffusion without the graph.

Each cell is predicted from its genotype's markers, its environment vector and plot features:
stage 1 fits a first guess f(x), stage 2 trains a diffusion model around it.
  python card.py --seed 0 [--data maizegep]
Writes runs/<design>/card_s<seed>/test_samples.npz.
"""
import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import (additive_fit, cosine_schedule, env_mean_module, env_vectors, run_dir, select_cells,
                    timestep_embedding, within_env_r)

HERE = Path(__file__).resolve().parent
T_START = time.time()


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")} +{(time.time() - T_START) / 60:5.1f} min] {msg}', flush=True)


class Encoder(nn.Module):
    def __init__(self, n_markers, env_dim, d=128, marker_dropout=0.5, dropout=0.2):
        super().__init__()
        self.marker_drop = nn.Dropout(marker_dropout)
        self.geno_proj = nn.Linear(n_markers, d)
        self.env_proj = nn.Linear(env_dim, d)
        self.feat_proj = nn.Linear(2, d)
        self.mix = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Linear(3 * d, 2 * d), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(2 * d, d), nn.SiLU())

    def forward(self, M, Z, F):
        return self.mix(torch.cat([self.geno_proj(self.marker_drop(M)), self.env_proj(Z), self.feat_proj(F)], 1))


class FirstGuess(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.enc = Encoder(*a, **k)
        self.out = nn.Linear(self.enc.geno_proj.out_features, 1)

    def forward(self, M, Z, F):
        return self.out(self.enc(M, Z, F)).squeeze(-1)


class EpsNet(nn.Module):
    def __init__(self, *a, d=128, temb_dim=64, dropout=0.2, **k):
        super().__init__()
        self.temb_dim = temb_dim
        self.enc = Encoder(*a, d=d, dropout=dropout, **k)
        self.temb = nn.Sequential(nn.Linear(temb_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.inp = nn.Linear(2, d)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(2 * d, 2 * d), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(2 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, 1))

    def forward(self, hx, yt, fx, t):
        tt = self.temb(timestep_embedding(t.float(), self.temb_dim))
        return self.net(torch.cat([hx + tt, self.inp(torch.stack([yt, fx], 1)) + tt], 1)).squeeze(-1)


def make_opt(model, a):
    gp = list(model.enc.geno_proj.parameters())
    ids = {id(p) for p in gp}
    opt = torch.optim.AdamW([{'params': gp, 'weight_decay': a.wd_marker},
                             {'params': [p for p in model.parameters() if id(p) not in ids], 'weight_decay': a.wd}], lr=a.lr)
    warm = min(200, a.steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(k, a.steps) / a.steps))))
    return opt, sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', choices=['g2f', 'maizegep'], default='g2f')
    ap.add_argument('--year', type=int, default=2024)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--T', type=int, default=50)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--marker-dropout', type=float, default=0.5)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--wd-marker', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--env-vectors', default=None, help='environment vectors to reuse (G2F default: those of GxEDiff seed 0)')
    ap.add_argument('--data-dir', default=str(HERE / 'data'))
    ap.add_argument('--runs', default=str(HERE / 'runs'))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.eval_every, a.samples = 60, 20, 4
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    base = run_dir(Path(a.runs), a.data, a.year)
    OUT = base / (f'card_s{a.seed}' + ('_smoke' if a.smoke else ''))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(OUT / 'args.json', 'w'), indent=1)

    D = pickle.load(open(Path(a.data_dir) / f'{a.data}.pkl', 'rb'))
    cells, test = select_cells(D, a.data, a.year)
    NG, NE = len(D['hybrids']), len(D['envs'])
    h, e, y, years = cells.h.values, cells.e.values, cells.y.values, cells.Year.values
    env_site, env_year = D['env_meta'].Site.values, D['env_meta'].Year.values
    train = ~test
    pair = h.astype(np.int64) * 10_000 + years
    tp = np.unique(pair[train])
    es = train & np.isin(pair, tp[np.random.default_rng(2000 + a.seed).random(len(tp)) < 0.05])
    prac = train & ~es
    known = np.zeros(NE, bool)
    known[np.unique(e[train])] = True
    zf = Path(a.env_vectors) if a.env_vectors else (base / 'gxediff_s0/env_vectors.npy' if a.data == 'g2f' else None)
    if zf is not None and zf.exists():
        Z = np.load(zf)
        log(f'environment vectors from {zf}')
    else:
        Z = env_vectors(D['weather'], known, epochs=200 if a.smoke else 2500, device=dev, log=log)
    mu_known = additive_fit(h[train], e[train], y[train], NG, NE)
    mu_e, env_err = env_mean_module(mu_known, known, env_site, Z, env_year)
    s2_obs = cells.groupby('e').s2_env.first().reindex(range(NE)).values
    s2_eff = s2_obs.copy()
    site_s2 = pd.Series(s2_obs[known]).groupby(env_site[known]).median()
    s2_eff[~known] = pd.Series(env_site[~known]).map(site_s2).fillna(np.nanmedian(s2_obs[known])).values
    n_med = cells[train].groupby('e').n.median().reindex(range(NE))
    n_med = n_med.fillna(pd.Series(env_site).map(n_med.groupby(env_site).median())).fillna(cells.n.median()).values
    n_cell = np.where(test, n_med[e], cells.n.values)
    r = y - mu_e[e]
    s = float(r[train].std())
    F = np.stack([np.log1p(n_cell), s2_eff[e] / n_cell / s ** 2], 1).astype(np.float32)
    target = (r / s).astype(np.float32)
    log(f'training {prac.sum()}, early stopping {es.sum()}, test {test.sum()} cells; s = {s:.3f}')

    Mt, Zt = torch.tensor(D['M'], device=dev), torch.tensor(Z, device=dev)
    Ft, Yt = torch.tensor(F, device=dev), torch.tensor(target, device=dev)
    ht, et = torch.tensor(h, device=dev), torch.tensor(e, device=dev)
    inputs = lambda idx: (Mt[ht[idx]], Zt[et[idx]], Ft[idx])
    prac_i, es_i, te_i = (torch.tensor(np.where(m)[0], device=dev) for m in (prac, es, test))
    torch.manual_seed(a.seed)
    kw = dict(marker_dropout=a.marker_dropout, dropout=a.dropout)
    batches = lambda idx, b=16384: [idx[i:i + b] for i in range(0, len(idx), b)]

    f = FirstGuess(D['M'].shape[1], Z.shape[1], **kw).to(dev)
    opt, sched = make_opt(f, a)

    @torch.no_grad()
    def predict_f(idx):
        f.eval()
        return torch.cat([f(*inputs(b)) for b in batches(idx)])

    best = (np.inf, None, 0)
    for step in range(1, a.steps + 1):
        f.train()
        bi = prac_i[torch.randint(len(prac_i), (a.batch,), device=dev)]
        loss = ((f(*inputs(bi)) - Yt[bi]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            p = predict_f(es_i)
            rmse = float(((p - Yt[es_i]) ** 2).mean().sqrt()) * s
            if rmse < best[0]:
                best = (rmse, {k: v.clone() for k, v in f.state_dict().items()}, step)
    f.load_state_dict(best[1])
    fx_all = torch.zeros(len(cells), device=dev)
    for idx in (prac_i, es_i, te_i):
        fx_all[idx] = predict_f(idx)
    log(f'first guess: best step {best[2]}, early-stop RMSE {best[0]:.3f}')

    sch = cosine_schedule(a.T)
    abar = torch.tensor(sch['abar'], dtype=torch.float32, device=dev)
    abar_prev = torch.tensor(sch['abar_prev'], dtype=torch.float32, device=dev)
    alpha = torch.tensor(sch['alpha'], dtype=torch.float32, device=dev)
    beta = torch.tensor(sch['beta'], dtype=torch.float32, device=dev)
    eps_net = EpsNet(D['M'].shape[1], Z.shape[1], **kw).to(dev)
    opt, sched = make_opt(eps_net, a)

    def noisy(idx, t, z):
        ab = abar[t - 1]
        return ab.sqrt() * Yt[idx] + (1 - ab.sqrt()) * fx_all[idx] + (1 - ab).sqrt() * z

    g = torch.Generator(device=dev).manual_seed(77)
    t_val = torch.randint(1, a.T + 1, (len(es_i),), device=dev, generator=g)
    z_val = torch.randn(len(es_i), device=dev, generator=g)

    @torch.no_grad()
    def val_loss():
        eps_net.eval()
        out = []
        for k in range(0, len(es_i), 16384):
            idx, t, z = es_i[k:k + 16384], t_val[k:k + 16384], z_val[k:k + 16384]
            out.append((eps_net(eps_net.enc(*inputs(idx)), noisy(idx, t, z), fx_all[idx], t) - z) ** 2)
        return float(torch.cat(out).mean())

    best = (np.inf, None, 0)
    for step in range(1, a.steps + 1):
        eps_net.train()
        bi = prac_i[torch.randint(len(prac_i), (a.batch,), device=dev)]
        t = torch.randint(1, a.T + 1, (a.batch,), device=dev)
        z = torch.randn(a.batch, device=dev)
        loss = ((eps_net(eps_net.enc(*inputs(bi)), noisy(bi, t, z), fx_all[bi], t) - z) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eps_net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            vl = val_loss()
            if vl < best[0]:
                best = (vl, {k: v.clone() for k, v in eps_net.state_dict().items()}, step)
    eps_net.load_state_dict(best[1])
    log(f'diffusion: best step {best[2]}, early-stop noise loss {best[0]:.4f}')

    @torch.no_grad()
    def sample(idx, K):
        eps_net.eval()
        torch.manual_seed(10_000 + a.seed)
        out = []
        for b in batches(idx, 2048):
            hx = eps_net.enc(*inputs(b)).repeat(K, 1)
            fx = fx_all[b].repeat(K)
            yt = fx + torch.randn_like(fx)
            for t in range(a.T, 0, -1):
                k = t - 1
                tt = torch.full_like(fx, t, dtype=torch.long)
                eps = eps_net(hx, yt, fx, tt)
                y0 = ((yt - (1 - abar[k].sqrt()) * fx - (1 - abar[k]).sqrt() * eps) / abar[k].sqrt()).clamp(-6, 6)
                if t == 1:
                    yt = y0
                    break
                g0 = beta[k] * abar_prev[k].sqrt() / (1 - abar[k])
                g1 = (1 - abar_prev[k]) * alpha[k].sqrt() / (1 - abar[k])
                g2 = 1 + (abar[k].sqrt() - 1) * (alpha[k].sqrt() + abar_prev[k].sqrt()) / (1 - abar[k])
                var = (1 - abar_prev[k]) / (1 - abar[k]) * beta[k]
                yt = g0 * y0 + g1 * yt + g2 * fx + var.sqrt() * torch.randn_like(yt)
            out.append(yt.view(K, -1))
        return torch.cat(out, 1).cpu().numpy()

    te = te_i.cpu().numpy()
    S = sample(te_i, a.samples)
    uy, yi = np.unique(env_year, return_inverse=True)
    rs = np.random.default_rng(5000)
    shift = rs.normal(0, env_err['year_sd'], (a.samples, len(uy)))[:, yi] + rs.normal(0, env_err['site_sd'], (a.samples, NE))
    Y = mu_e[e[te]][None] + s * S + np.where(known[e[te]], 0.0, 1.0)[None] * shift[:, e[te]]
    np.savez_compressed(OUT / 'test_samples.npz', y=y[te], S=Y.astype(np.float32), h=h[te], e=e[te], year=years[te])
    df = cells.iloc[te][['Env', 'y']].reset_index(drop=True)
    summ = dict(model='card', seed=a.seed, r_bar=within_env_r(df.assign(p=Y.mean(0)), 'p'),
                minutes=round((time.time() - T_START) / 60, 1))
    json.dump(summ, open(OUT / 'summary.json', 'w'), indent=1)
    log(f'done: {summ}')


if __name__ == '__main__':
    main()
