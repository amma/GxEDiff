"""GBLUP and GBLUP + W baselines.

  GBLUP     : y = mu + site + environment + line + g + e,  g = Z beta (markers)
  GBLUP + W : the same + environmental covariates w' gamma (Jarquin et al. 2014)
Variance components by maximum marginal likelihood on the GPU.
  python gblup.py                       G2F 2024 test
  python gblup.py --year 2022 [--train-from 2016] | --loyo 2019 | --data maizegep
Writes runs/<design>/gblup/preds.npz.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from common import run_dir, select_cells

HERE = Path(__file__).resolve().parent
FIXED_VAR = 1e4
F64 = torch.float64


class MixedModel:
    """Linear mixed model with dense blocks (intercept, covariates, markers) and one-hot blocks (site, environment, line)."""

    def __init__(self, var_names, dev):
        self.var_names, self.dev = var_names + ['s2_ge', 's'], dev

    def _setup(self, dense, onehot):
        dev = self.dev
        self.D = torch.cat([b.to(F64) for _, b, _ in dense], 1)
        self.oh = [(torch.as_tensor(i, device=dev), k) for _, i, k, _ in onehot]
        self.pd = self.D.shape[1]
        self.offs = np.cumsum([self.pd] + [k for _, k in self.oh])
        self.P = int(self.offs[-1])
        cv = [torch.full((b.shape[1],), -1 if v is None else self.var_names.index(v), device=dev) for _, b, v in dense]
        cv += [torch.full((k,), self.var_names.index(v), device=dev) for _, _, k, v in onehot]
        self.col_var = torch.cat(cv)

    def _gram(self, w):
        dev, pd, nq = self.dev, self.pd, self.P - self.pd
        Dw = self.D * w[:, None]
        Add = Dw.T @ self.D
        Adq = torch.cat([torch.zeros(pd, k, dtype=F64, device=dev).index_add(1, i, Dw.T) for i, k in self.oh], 1)
        qi = [i + int(o) - pd for (i, _), o in zip(self.oh, self.offs[:-1])]
        Aqq = torch.zeros(nq, nq, dtype=F64, device=dev)
        for a in range(len(qi)):
            for b in range(a, len(qi)):
                Aqq = Aqq.index_put((qi[a], qi[b]), w, accumulate=True)
                if b != a:
                    Aqq = Aqq.index_put((qi[b], qi[a]), w, accumulate=True)
        return torch.cat([torch.cat([Add, Adq], 1), torch.cat([Adq.T, Aqq], 1)], 0)

    def _xty(self, wy):
        parts = [self.D.T @ wy]
        for i, k in self.oh:
            parts.append(torch.zeros(k, dtype=F64, device=self.dev).index_add(0, i, wy))
        return torch.cat(parts)

    def _prior_var(self, logp):
        pv = torch.exp(logp)[self.col_var.clamp(min=0)]
        return torch.where(self.col_var < 0, torch.full_like(pv, FIXED_VAR), pv)

    def _nll(self, logp):
        logp = logp.clamp(-25.0, 12.0)
        th = {k: torch.exp(logp[i]) for i, k in enumerate(self.var_names)}
        r = th['s2_ge'] + th['s'] * self.v
        w = 1.0 / r
        pv = self._prior_var(logp)
        A = self._gram(w) + torch.diag(1.0 / pv)
        b = self._xty(w * self.y)
        L = torch.linalg.cholesky(A)
        alpha = torch.cholesky_solve(b[:, None], L)[:, 0]
        quad = (w * self.y ** 2).sum() - b @ alpha
        logdet = 2 * torch.log(torch.diagonal(L)).sum() + torch.log(r).sum() + torch.log(pv).sum()
        self._cache = (L, alpha, th)
        return 0.5 * (quad + logdet)

    def fit(self, dense, onehot, y, v, init, max_iter=60, rounds=4):
        self._setup(dense, onehot)
        self.y = torch.as_tensor(y, dtype=F64, device=self.dev)
        self.v = torch.as_tensor(v, dtype=F64, device=self.dev)
        logp = torch.tensor(np.log([init[k] for k in self.var_names]), dtype=F64, device=self.dev, requires_grad=True)
        opt = torch.optim.LBFGS([logp], lr=1, max_iter=max_iter, tolerance_change=1e-6, tolerance_grad=1e-3,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = self._nll(logp)
            loss.backward()
            return loss
        last = np.inf
        for _ in range(rounds):
            opt.step(closure)
            with torch.no_grad():
                cur = float(self._nll(logp))
            if last - cur < 0.01:
                break
            last = cur
        with torch.no_grad():
            self.nll = float(self._nll(logp))
            L, alpha, th = self._cache
            self.theta = {k: float(v) for k, v in th.items()}
            self.coef, self.Sigma = alpha, torch.cholesky_inverse(L)
        del self.D
        torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def predict(self, dense, onehot):
        dev = self.dev
        parts = [b.to(F64) for _, b, _ in dense]
        for _, i, k, _ in onehot:
            i = torch.as_tensor(i, device=dev)
            X = torch.zeros(len(i), k, dtype=F64, device=dev)
            ok = i >= 0
            X[torch.arange(len(i), device=dev)[ok], i[ok]] = 1
            parts.append(X)
        X = torch.cat(parts, 1)
        return (X @ self.coef).cpu().numpy(), ((X @ self.Sigma) * X).sum(1).cpu().numpy()


def lookup(x, ref):
    pos = np.searchsorted(ref, x).clip(max=len(ref) - 1)
    return np.where(ref[pos] == x, pos, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', choices=['g2f', 'maizegep'], default='g2f')
    ap.add_argument('--year', type=int, default=2024)
    ap.add_argument('--train-from', type=int, default=None)
    ap.add_argument('--loyo', type=int, default=None)
    ap.add_argument('--data-dir', default=str(HERE / 'data'))
    ap.add_argument('--runs', default=str(HERE / 'runs'))
    a = ap.parse_args()
    dev = 'cuda'
    OUT = run_dir(Path(a.runs), a.data, a.year, a.train_from, a.loyo) / 'gblup'
    OUT.mkdir(parents=True, exist_ok=True)
    D = pickle.load(open(Path(a.data_dir) / f'{a.data}.pkl', 'rb'))
    cells, test = select_cells(D, a.data, a.year, a.train_from, a.loyo)
    tr, te = np.where(~test)[0], np.where(test)[0]
    h, e, y, n = cells.h.values, cells.e.values, cells.y.values, cells.n.values
    site = D['env_meta'].Site.values
    tr_env, tr_hyb, tr_site = np.unique(e[tr]), np.unique(h[tr]), np.unique(site[e[tr]])

    v = (cells.s2_env.values / n).astype(np.float64)
    M = D['M'].astype(np.float64)
    if M.shape[1] > M.shape[0]:
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        Mk = U * S
    else:
        Mk = M
    Zm = torch.tensor(Mk / np.sqrt(M.shape[1]), dtype=F64, device=dev)
    EC = D['ec'].astype(np.float64)
    known_env = np.zeros(len(D['envs']), bool)
    known_env[tr_env] = True
    EC = (EC - EC[known_env].mean(0)) / (EC[known_env].std(0) + 1e-9)
    Wm = torch.tensor(EC / np.sqrt(EC.shape[1]), dtype=F64, device=dev)

    def design(idx, with_w):
        dense = [('mu', torch.ones(len(idx), 1, dtype=F64, device=dev), None)]
        if with_w:
            dense.append(('W', Wm[torch.as_tensor(e[idx], device=dev)], 's2_W'))
        dense.append(('markers', Zm[torch.as_tensor(h[idx], device=dev)], 's2_a'))
        onehot = [('site', lookup(site[e[idx]], tr_site), len(tr_site), 's2_S'),
                  ('env', lookup(e[idx], tr_env), len(tr_env), 's2_E'),
                  ('line', lookup(h[idx], tr_hyb), len(tr_hyb), 's2_L')]
        return dense, onehot

    res = dict(y=y[te], h=h[te], e=e[te])
    thetas = {}
    for name, with_w in [('GBLUP', False), ('GBLUP_W', True)]:
        t0 = time.time()
        var_names = ['s2_S', 's2_E'] + (['s2_W'] if with_w else []) + ['s2_a', 's2_L']
        m = MixedModel(var_names, dev)
        init = dict(s2_S=2.0, s2_E=3.0, s2_a=0.5, s2_L=0.2, s2_ge=0.9, s=1.0)
        if with_w:
            init = dict(thetas['GBLUP']['theta'], s2_W=0.5)
        m.fit(*design(tr, with_w), y[tr], v[tr], init)
        mean, var = m.predict(*design(te, with_w))
        th = m.theta
        new_site = lookup(site[e[te]], tr_site) < 0
        new_hyb = lookup(h[te], tr_hyb) < 0
        var = var + th['s2_E'] + np.where(new_site, th['s2_S'], 0.0) + np.where(new_hyb, th['s2_L'], 0.0)
        sd = np.sqrt(var + th['s2_ge'] + th['s'] * v[te])
        res[f'{name}_mean'], res[f'{name}_sd'] = mean, sd
        thetas[name] = dict(theta=th, nll=m.nll, seconds=round(time.time() - t0))
        r_bar = np.mean([np.corrcoef(mean[e[te] == k], y[te][e[te] == k])[0, 1] for k in np.unique(e[te])
                         if (e[te] == k).sum() >= 10])
        print(f'{name}: {time.time() - t0:.0f}s, r_bar {r_bar:.3f}, theta {json.dumps({k: round(x, 3) for k, x in th.items()})}',
              flush=True)
        del m
        torch.cuda.empty_cache()
    np.savez(OUT / 'preds.npz', **res)
    json.dump(thetas, open(OUT / 'theta.json', 'w'), indent=1)
    print('saved', OUT / 'preds.npz', flush=True)


if __name__ == '__main__':
    main()
