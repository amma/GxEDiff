"""Train GxEDiff (or an ablation) on one test design and fill the test genotypes' empty cells.

  python gxediff.py --seed 0                          G2F 2024 test
  python gxediff.py --year 2022 [--train-from 2016]   G2F forward test
  python gxediff.py --loyo 2019                       G2F leave-one-year-out (2014-2023)
  python gxediff.py --data maizegep --seed 0          MaizeGEP unseen varieties
  --gnn-only: the same GNN trained to predict hidden cells in one pass (no diffusion)
  --no-self-cond: diffusion without self-conditioning
Writes runs/<design>/<model>_s<seed>/test_samples.npz (true yields y and 100 filled values S per test cell).
"""
import argparse
import copy
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (MaskSampler, additive_fit, correlated_noise, cosine_schedule, env_mean_module, env_vectors,
                    run_dir, select_cells, weather_check, weather_edges, within_env_r)
from denoiser import R_CTX, R_CTX_R, R_KIN, R_QRY, R_QRY_R, R_WX, Denoiser

HERE = Path(__file__).resolve().parent
T_START = time.time()


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")} +{(time.time() - T_START) / 60:5.1f} min] {msg}', flush=True)


class Graph:
    """Builds the graph for one training step or one filling pass; `reps` copies are stacked to fill several tables."""

    def __init__(self, D, Z, wx, dev, self_cond=False):
        self.dev, self.NG, self.NE = dev, len(D['hybrids']), len(D['envs'])
        self.nf = 8 if self_cond else 6
        self.M = torch.tensor(D['M'], device=dev)
        self.Z = torch.tensor(Z, device=dev)
        ks, kd, kw = D['kin']
        kw = (kw - kw.mean()) / kw.std()
        self.kin = [torch.tensor(v, device=dev) for v in (ks, kd)] + [torch.tensor(kw, dtype=torch.float32, device=dev)]
        ws, wd, ww = wx
        self.wx = [torch.tensor(v, device=dev) for v in (ws, wd)] + [torch.tensor(ww, dtype=torch.float32, device=dev)]

    def build(self, ch, ce, cfeat, qh, qe, qfeat, reps):
        dev, NG, NE = self.dev, self.NG, self.NE
        og = (torch.arange(reps, device=dev) * NG)[:, None]
        oe = (torch.arange(reps, device=dev) * NE)[:, None]
        cg, cev = (ch[None] + og).reshape(-1), (ce[None] + oe).reshape(-1)
        qg, qev = (qh[None] + og).reshape(-1), (qe[None] + oe).reshape(-1)
        ctx_attr = torch.zeros(len(ch), self.nf, device=dev)
        ctx_attr[:, 0], ctx_attr[:, 4], ctx_attr[:, 5] = cfeat[:, 2], cfeat[:, 0], cfeat[:, 1]
        qry_attr = torch.zeros(reps, len(qh), self.nf, device=dev)
        qry_attr[..., 1] = 1.0
        qry_attr[..., 4], qry_attr[..., 5] = qfeat[:, 0], qfeat[:, 1]
        ks, kd, kw = self.kin
        ws, wd, ww = self.wx
        return dict(
            reps=reps, M=self.M, Zenv=self.Z, t=0,
            cnt_g=torch.log1p(torch.bincount(ch, minlength=NG).float()).repeat(reps),
            cnt_e=torch.log1p(torch.bincount(ce, minlength=NE).float()).repeat(reps),
            edges={R_CTX: torch.stack([cev, cg]), R_CTX_R: torch.stack([cg, cev]),
                   R_QRY: torch.stack([qev, qg]), R_QRY_R: torch.stack([qg, qev]),
                   R_KIN: torch.stack([(ks[None] + og).reshape(-1), (kd[None] + og).reshape(-1)]),
                   R_WX: torch.stack([(ws[None] + oe).reshape(-1), (wd[None] + oe).reshape(-1)])},
            ctx_attr=ctx_attr.repeat(reps, 1), qry_attr=qry_attr.reshape(-1, self.nf),
            kin_w=kw.repeat(reps), wx_w=ww.repeat(reps), qry_h=qg, qry_e=qev)

    @staticmethod
    def set_noise(g, x, t, sab, s1ab):
        g['t'] = t
        g['qry_attr'][:, 0] = x
        g['qry_attr'][:, 2] = sab
        g['qry_attr'][:, 3] = s1ab

    @staticmethod
    def set_selfcond(g, x0_prev):
        if g['qry_attr'].shape[1] < 8:
            return
        if x0_prev is None:
            g['qry_attr'][:, 6:8] = 0.0
        else:
            g['qry_attr'][:, 6] = x0_prev
            g['qry_attr'][:, 7] = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', choices=['g2f', 'maizegep'], default='g2f')
    ap.add_argument('--year', type=int, default=2024, help='G2F test year (2018, 2020, 2022 or 2024)')
    ap.add_argument('--train-from', type=int, default=None, help='G2F: first training year')
    ap.add_argument('--loyo', type=int, default=None, help='G2F: year held out, trained on the other years 2014-2023')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gnn-only', action='store_true')
    ap.add_argument('--no-self-cond', action='store_true')
    ap.add_argument('--layers', type=int, default=5)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--T', type=int, default=50)
    ap.add_argument('--shared-noise', type=float, default=0.5)
    ap.add_argument('--batch-hybrids', type=int, default=None, help='genotypes filled together (G2F 200, MaizeGEP 40)')
    ap.add_argument('--chunk', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--marker-dropout', type=float, default=0.5)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--wd-marker', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--data-dir', default=str(HERE / 'data'))
    ap.add_argument('--runs', default=str(HERE / 'runs'))
    ap.add_argument('--smoke', action='store_true', help='tiny run to test the code')
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.eval_every, a.T, a.samples, a.chunk = 30, 10, 5, 2, 2
    if a.batch_hybrids is None:
        a.batch_hybrids = 40 if a.data == 'maizegep' else 200
    direct, self_cond, sd, lam = a.gnn_only, not (a.gnn_only or a.no_self_cond), a.seed, a.shared_noise
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = 'cuda'

    model_name = 'gnn_only' if direct else 'gxediff' + ('' if self_cond else '_nosc')
    model_name += '' if a.layers == 5 else f'_l{a.layers}'
    OUT = run_dir(Path(a.runs), a.data, a.year, a.train_from, a.loyo) / (f'{model_name}_s{sd}' + ('_smoke' if a.smoke else ''))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(OUT / 'args.json', 'w'), indent=1)

    D = pickle.load(open(Path(a.data_dir) / f'{a.data}.pkl', 'rb'))
    cells, test = select_cells(D, a.data, a.year, a.train_from, a.loyo)
    NG, NE = len(D['hybrids']), len(D['envs'])
    h, e, y, years = cells.h.values, cells.e.values, cells.y.values, cells.Year.values
    env_site, env_year = D['env_meta'].Site.values, D['env_meta'].Year.values
    log(f'{OUT.relative_to(Path(a.runs))}: {len(cells)} cells, {NG} genotypes, {NE} environments')

    train = ~test
    pair = h.astype(np.int64) * 10_000 + years
    tp = np.unique(pair[train])
    es = train & np.isin(pair, tp[np.random.default_rng(2000 + sd).random(len(tp)) < 0.05])
    prac = train & ~es
    known = np.zeros(NE, bool)
    known[np.unique(e[train])] = True
    log(f'training {prac.sum()}, early stopping {es.sum()}, test {test.sum()} cells; '
        f'environments with yields {known.sum()}, without {(~known).sum()}')

    Z = env_vectors(D['weather'], known, epochs=200 if a.smoke else 2500, device=dev, log=log)
    np.save(OUT / 'env_vectors.npy', Z)
    log(f'environment vectors {Z.shape}, weather R2 {weather_check(D["weather"], Z, known):.3f}')
    wx = weather_edges(Z)

    mu_known = additive_fit(h[train], e[train], y[train], NG, NE)
    mu_e, env_err = env_mean_module(mu_known, known, env_site, Z, env_year)
    s2_obs = cells.groupby('e').s2_env.first().reindex(range(NE)).values
    s2_eff = s2_obs.copy()
    site_s2 = pd.Series(s2_obs[known]).groupby(env_site[known]).median()
    s2_eff[~known] = pd.Series(env_site[~known]).map(site_s2).fillna(np.nanmedian(s2_obs[known])).values
    n_med = cells[train].groupby('e').n.median().reindex(range(NE))
    n_med = n_med.fillna(pd.Series(env_site).map(n_med.groupby(env_site).median())).fillna(cells.n.median()).values
    r = y - mu_e[e]
    s = float(r[train].std())
    feat = np.stack([np.log1p(cells.n.values), s2_eff[e] / cells.n.values / s ** 2, r / s], 1).astype(np.float32)
    log(f'residual SD s = {s:.3f}')

    G = Graph(D, Z, wx, dev, self_cond=self_cond)
    sch = cosine_schedule(a.T)
    th, tev, tfeat = (torch.tensor(v, device=dev) for v in (h, e, feat))
    model = Denoiser(D['M'].shape[1], Z.shape[1], heads=a.heads, layers=a.layers,
                     marker_dropout=a.marker_dropout, dropout=a.dropout, self_cond=self_cond).to(dev)

    torch.manual_seed(sd)
    rng = np.random.default_rng(3000 + sd)
    gp = list(model.geno_proj.parameters())
    gp_ids = {id(p) for p in gp}
    opt = torch.optim.AdamW([{'params': gp, 'weight_decay': a.wd_marker},
                             {'params': [p for p in model.parameters() if id(p) not in gp_ids], 'weight_decay': a.wd}],
                            lr=a.lr)
    warm = min(200, a.steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(k, a.steps) / a.steps))))
    prac_idx, es_idx = np.where(prac)[0], np.where(es)[0]
    sampler = MaskSampler(h[prac], e[prac], years[prac], seed=sd)
    ci_all, qi_es = torch.tensor(prac_idx, device=dev), torch.tensor(es_idx, device=dev)
    g_es = G.build(th[ci_all], tev[ci_all], tfeat[ci_all], th[qi_es], tev[qi_es], tfeat[qi_es], 1)
    es_df = cells.iloc[es_idx][['Env', 'y']].reset_index(drop=True)

    def es_eval():
        model.eval()
        with torch.no_grad():
            ab = sch['abar'][-1]
            if direct:
                Graph.set_noise(g_es, torch.zeros(len(es_idx), device=dev), a.T, 0.0, 1.0)
            else:
                Graph.set_noise(g_es, correlated_noise(NG, th[qi_es], lam, dev), a.T, math.sqrt(ab), math.sqrt(1 - ab))
            Graph.set_selfcond(g_es, None)
            p = model(g_es)
            if self_cond:
                Graph.set_selfcond(g_es, p.clamp(-6, 6))
                p = model(g_es)
            p = p.cpu().numpy() * s + mu_e[e[es_idx]]
        return float(np.sqrt(np.mean((p - y[es_idx]) ** 2))), within_env_r(es_df.assign(p=p), 'p')

    best, hist, losses = (np.inf, None, 0), [], []
    for step in range(1, a.steps + 1):
        model.train()
        _, q = sampler.sample()
        qi, ci = torch.tensor(prac_idx[q], device=dev), torch.tensor(prac_idx[~q], device=dev)
        t = int(rng.integers(1, a.T + 1))
        ab = sch['abar'][t - 1]
        x0q = tfeat[qi, 2]
        g = G.build(th[ci], tev[ci], tfeat[ci], th[qi], tev[qi], tfeat[qi], 1)
        if direct:
            Graph.set_noise(g, torch.zeros_like(x0q), a.T, 0.0, 1.0)
        else:
            xt = math.sqrt(ab) * x0q + math.sqrt(1 - ab) * correlated_noise(NG, th[qi], lam, dev)
            Graph.set_noise(g, xt, t, math.sqrt(ab), math.sqrt(1 - ab))
        Graph.set_selfcond(g, None)
        if self_cond and rng.random() < 0.5:
            with torch.no_grad():
                first = model(g).clamp(-6, 6)
            Graph.set_selfcond(g, first)
        loss = ((model(g) - x0q) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % a.eval_every == 0 or step == a.steps:
            rmse, rw = es_eval()
            hist.append(dict(step=step, loss=float(np.mean(losses)), es_rmse=rmse, es_r=rw))
            if rmse < best[0]:
                best = (rmse, copy.deepcopy(model.state_dict()), step)
            log(f'step {step:5d} | loss {np.mean(losses):.3f} | early-stop RMSE {rmse:.3f}, r {rw:.3f}')
            losses = []
    pd.DataFrame(hist).to_csv(OUT / 'training_history.csv', index=False)
    model.load_state_dict(best[1])
    torch.save(best[1], OUT / 'denoiser.pt')
    log(f'best step {best[2]} (early-stop RMSE {best[0]:.3f})')
    if direct:
        model.eval()
        with torch.no_grad():
            Graph.set_noise(g_es, torch.zeros(len(es_idx), device=dev), a.T, 0.0, 1.0)
            eps = feat[es_idx, 2] - model(g_es).cpu().numpy()
        blk = pd.Series(eps).groupby(h[es_idx].astype(np.int64) * 10_000 + years[es_idx])
        kb, mb = blk.size().values, blk.mean().values
        ok = kb >= 2
        V, ik = float(np.mean(eps ** 2)), float(np.mean(1 / kb[ok]))
        sh2 = max((float(np.mean(mb[ok] ** 2)) - V * ik) / (1 - ik), 0.0)
        direct_sd = (math.sqrt(sh2), math.sqrt(max(V - sh2, 1e-6)))
    del g_es
    torch.cuda.empty_cache()

    model.eval()
    torch.manual_seed(10_000)
    obs = np.zeros((NG, NE), bool)
    obs[h[train], e[train]] = True
    mh, me = np.where(~obs)
    fill_h = np.unique(h[test])
    if a.smoke:
        fill_h = np.random.default_rng(0).choice(fill_h, min(200, len(fill_h)), replace=False)
    keep = np.isin(mh, fill_h)
    mh, me = mh[keep], me[keep]
    nmiss = len(mh)
    starts = np.searchsorted(mh, np.arange(NG + 1))
    ctx = torch.tensor(np.where(train)[0], device=dev)
    qfeat_all = np.stack([np.log1p(n_med[me]), s2_eff[me] / n_med[me] / s ** 2, np.zeros(nmiss)], 1).astype(np.float32)
    samples = np.zeros((a.samples, nmiss), np.float16)
    hyb_order = np.random.default_rng(4000 + sd).permutation(np.unique(mh))
    batches = [hyb_order[i:i + a.batch_hybrids] for i in range(0, len(hyb_order), a.batch_hybrids)]
    al, ab_, abp, be = sch['alpha'], sch['abar'], sch['abar_prev'], sch['beta']
    c1, c2, pv = np.sqrt(abp) * be / (1 - ab_), np.sqrt(al) * (1 - abp) / (1 - ab_), be * (1 - abp) / (1 - ab_)
    log(f'filling {nmiss:,} empty cells of {len(fill_h)} genotypes, {a.samples} tables, {a.T} rounds')
    rng_direct = np.random.default_rng(6000 + sd)
    with torch.no_grad():
        for bi, bh in enumerate(batches):
            sel = np.concatenate([np.arange(starts[k], starts[k + 1]) for k in bh])
            qh, qe = torch.tensor(mh[sel], device=dev), torch.tensor(me[sel], device=dev)
            qf = torch.tensor(qfeat_all[sel], device=dev)
            if direct:
                g = G.build(th[ctx], tev[ctx], tfeat[ctx], qh, qe, qf, 1)
                Graph.set_noise(g, torch.zeros(len(sel), device=dev), a.T, 0.0, 1.0)
                pred = model(g).cpu().numpy()
                zh = rng_direct.normal(0, 1, (a.samples, NG))[:, mh[sel]]
                zc = rng_direct.normal(0, 1, (a.samples, len(sel)))
                samples[:, sel] = (pred[None] + direct_sd[0] * zh + direct_sd[1] * zc).astype(np.float16)
                continue
            for c0 in range(0, a.samples, a.chunk):
                reps = min(a.chunk, a.samples - c0)
                g = G.build(th[ctx], tev[ctx], tfeat[ctx], qh, qe, qf, reps)
                x = correlated_noise(reps * NG, g['qry_h'], lam, dev)
                x0h = None
                for t in range(a.T, 0, -1):
                    k = t - 1
                    Graph.set_noise(g, x, t, math.sqrt(ab_[k]), math.sqrt(1 - ab_[k]))
                    Graph.set_selfcond(g, x0h)
                    x0h = model(g).clamp(-6, 6)
                    if t > 1:
                        x = c1[k] * x0h + c2[k] * x + math.sqrt(pv[k]) * correlated_noise(reps * NG, g['qry_h'], lam, dev)
                    else:
                        x = x0h
                samples[c0:c0 + reps, sel] = x.view(reps, -1).cpu().numpy().astype(np.float16)
            log(f'filled batch {bi + 1}/{len(batches)}')

    Y = mu_e[me][None] + s * samples.astype(np.float32)
    new_cells = ~known[me]
    if new_cells.any():
        rs = np.random.default_rng(5000)
        uy, yi = np.unique(env_year, return_inverse=True)
        shift = rs.normal(0, env_err['year_sd'], (a.samples, len(uy)))[:, yi] + rs.normal(0, env_err['site_sd'], (a.samples, NE))
        Y[:, new_cells] += shift[:, me[new_cells]]

    pos = np.full((NG, NE), -1, np.int64)
    pos[mh, me] = np.arange(nmiss)
    te_i = np.where(test)[0]
    te_i = te_i[pos[h[te_i], e[te_i]] >= 0]
    S = Y[:, pos[h[te_i], e[te_i]]]
    np.savez_compressed(OUT / 'test_samples.npz', y=y[te_i], S=S.astype(np.float32), h=h[te_i], e=e[te_i], year=years[te_i])
    df = cells.iloc[te_i][['Env', 'y']].reset_index(drop=True)
    summ = dict(model=model_name, seed=sd, best_step=best[2], early_stop_rmse=best[0], residual_sd=s,
                test_cells=len(te_i), r_bar=within_env_r(df.assign(p=S.mean(0)), 'p'),
                minutes=round((time.time() - T_START) / 60, 1))
    json.dump(summ, open(OUT / 'summary.json', 'w'), indent=1, default=float)
    log(f'done: {json.dumps(summ, default=float)}')


if __name__ == '__main__':
    main()
