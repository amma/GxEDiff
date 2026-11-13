import math

import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from torch import nn


class EnvAE(nn.Module):
    """LSTM autoencoder: the last hidden state of the encoder is the environment vector."""

    def __init__(self, n_feat, dim):
        super().__init__()
        self.enc = nn.LSTM(n_feat, dim, batch_first=True)
        self.dec = nn.LSTM(dim, dim, batch_first=True)
        self.out = nn.Linear(dim, n_feat)

    def encode(self, x):
        _, (h, _) = self.enc(x)
        return h[-1]

    def forward(self, x):
        z = self.encode(x)
        d, _ = self.dec(z.unsqueeze(1).expand(-1, x.size(1), -1))
        return self.out(d), z


def env_vectors(W, train_envs, dim=21, seeds=(0, 1, 2), epochs=2500, lr=2e-3, device='cuda', log=print):
    """Train on the training environments only, encode all; three runs aligned by Procrustes and averaged."""
    mean = W[train_envs].reshape(-1, W.shape[-1]).mean(0)
    std = W[train_envs].reshape(-1, W.shape[-1]).std(0) + 1e-9
    X = torch.tensor((W - mean) / std, dtype=torch.float32, device=device)
    tr = torch.tensor(np.where(train_envs)[0], device=device)
    codes = []
    for s in seeds:
        torch.manual_seed(s)
        m = EnvAE(W.shape[-1], dim).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.05)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            rec, _ = m(X[tr])
            loss = ((rec - X[tr]) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
        m.eval()
        with torch.no_grad():
            codes.append(m.encode(X).cpu().numpy().astype(np.float64))
        log(f'  environment LSTM seed {s}: reconstruction MSE {loss.item():.3f}')
    zs = lambda z: (z - z[train_envs].mean(0)) / (z[train_envs].std(0) + 1e-9)
    ref = zs(codes[0])
    aligned = [ref]
    for z in codes[1:]:
        R, _ = orthogonal_procrustes(zs(z)[train_envs], ref[train_envs])
        aligned.append(zs(z) @ R)
    Z = np.mean(aligned, 0)
    Z = (Z - Z[train_envs].mean(0)) / (Z[train_envs].std(0) + 1e-9)
    return Z.astype(np.float32)


def weather_check(W, Z, train_envs):
    """Mean cross-validated R2 of season weather summaries predicted from the environment vectors."""
    season = W[:, 2:19].mean(1)
    flower = W[:, 10:13].mean(1)[:, [1, 3]]
    Y = np.hstack([season, flower])[train_envs]
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    Zt = Z[train_envs]
    r2 = []
    for j in range(Y.shape[1]):
        p = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), Zt, Y[:, j], cv=KFold(5, shuffle=True, random_state=0))
        r2.append(1 - np.mean((Y[:, j] - p) ** 2))
    return float(np.mean(r2))


def weather_edges(Z, k=10):
    k = min(k, len(Z) - 1)
    d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nb = np.argsort(d2, 1)[:, :k]
    dst = np.repeat(np.arange(len(Z)), k)
    src = nb.ravel()
    dd = d2[dst, src]
    w = np.exp(-dd / np.median(dd))
    return src, dst, w.astype(np.float32)


def additive_fit(h, e, y, n_h, n_e, iters=50):
    """y = mu + env + genotype by alternating least squares; returns mu + env for every environment."""
    mu = y.mean()
    g = np.zeros(n_h)
    for _ in range(iters):
        r = y - mu - g[h]
        ee = np.bincount(e, r, n_e) / np.maximum(np.bincount(e, minlength=n_e), 1)
        r = y - mu - ee[e]
        g = np.bincount(h, r, n_h) / np.maximum(np.bincount(h, minlength=n_h), 1)
        g -= g[np.bincount(h, minlength=n_h) > 0].mean()
    return mu + ee


MIN_ENVS_WEATHER = 30


def env_mean_module(mu_known, known, site, Z, years):
    """Means of environments without yields: ridge on the site, then ridge on the weather vector.
    Also returns the leave-one-year-out error split into a year part and a site part."""
    def fit_predict(tr, te):
        sites = np.unique(site[tr])
        oh = lambda s: (s[:, None] == sites[None]).astype(float)
        m1 = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(oh(site[tr]), mu_known[tr])
        p_tr, p_te = m1.predict(oh(site[tr])), m1.predict(oh(site[te]))
        if tr.sum() < MIN_ENVS_WEATHER:
            return p_te
        zm, zs = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        m2 = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit((Z[tr] - zm) / zs, mu_known[tr] - p_tr)
        return p_te + m2.predict((Z[te] - zm) / zs)

    out = mu_known.copy()
    if (~known).any():
        out[~known] = fit_predict(known, ~known)
    errs, year_err, site_err = [], [], []
    for Y in np.unique(years[known]):
        te, tr = known & (years == Y), known & (years != Y)
        err = mu_known[te] - fit_predict(tr, te)
        errs.append(err)
        year_err.append(err.mean())
        site_err.append(err - err.mean())
    total = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
    return out, dict(total_sd=total, year_sd=float(np.sqrt(np.mean(np.square(year_err)))),
                     site_sd=float(np.sqrt(np.mean(np.concatenate(site_err) ** 2))))


def correlated_noise(n_hyb_total, hyb_idx, lam, dev):
    """Standard normal noise; a share lam of the variance is shared by all cells of a genotype."""
    z = torch.randn(len(hyb_idx), device=dev)
    if lam > 0:
        z = math.sqrt(1 - lam) * z + math.sqrt(lam) * torch.randn(n_hyb_total, device=dev)[hyb_idx]
    return z


def cosine_schedule(T, s=0.008):
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    beta = np.clip(1 - abar[1:] / abar[:-1], 1e-5, 0.999)
    alpha = 1 - beta
    abar = np.cumprod(alpha)
    return dict(beta=beta, alpha=alpha, abar=abar, abar_prev=np.r_[1.0, abar[:-1]])


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[..., None] * freqs
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class MaskSampler:
    """Hiding schemes for one training step (scheme, share of steps):
    scatter 30%: 20% of cells | hybyear 25%: all cells of 15% of genotype-year pairs |
    thin 20%: 15% of genotypes keep 1-3 cells | newvar 15%: all cells of 8% of genotypes |
    newenv 10%: all cells of 5% of environments."""
    SCHEMES = ['scatter', 'hybyear', 'newvar', 'thin', 'newenv']
    PROBS = [0.30, 0.25, 0.15, 0.20, 0.10]

    def __init__(self, h, e, year, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n = len(h)
        self.by_h = pd.Series(np.arange(self.n)).groupby(h).apply(np.array).to_dict()
        self.by_e = pd.Series(np.arange(self.n)).groupby(e).apply(np.array).to_dict()
        self.by_hy = pd.Series(np.arange(self.n)).groupby([h, year]).apply(np.array).to_dict()
        self.keys_h, self.keys_e, self.keys_hy = list(self.by_h), list(self.by_e), list(self.by_hy)

    def _pick(self, groups, keys, frac):
        sel = self.rng.choice(len(keys), max(1, int(frac * len(keys))), replace=False)
        return np.concatenate([groups[keys[i]] for i in sel])

    def sample(self):
        s = self.rng.choice(self.SCHEMES, p=self.PROBS)
        q = np.zeros(self.n, bool)
        if s == 'scatter':
            q[self.rng.random(self.n) < 0.20] = True
        elif s == 'hybyear':
            q[self._pick(self.by_hy, self.keys_hy, 0.15)] = True
        elif s == 'newvar':
            q[self._pick(self.by_h, self.keys_h, 0.08)] = True
        elif s == 'thin':
            sel = self.rng.choice(len(self.keys_h), int(0.15 * len(self.keys_h)), replace=False)
            for i in sel:
                idx = self.by_h[self.keys_h[i]]
                keep = self.rng.choice(len(idx), min(len(idx), self.rng.integers(1, 4)), replace=False)
                m = np.ones(len(idx), bool)
                m[keep] = False
                q[idx[m]] = True
        else:
            q[self._pick(self.by_e, self.keys_e, 0.05)] = True
        if q.sum() == 0:
            q[self.rng.integers(self.n)] = True
        return s, q


def within_env_r(df, pred_col, min_n=10):
    out = []
    for _, g in df.groupby('Env'):
        if len(g) >= min_n and g[pred_col].std() > 0 and g.y.std() > 0:
            out.append(np.corrcoef(g[pred_col], g.y)[0, 1])
    return float(np.mean(out)) if out else np.nan


def per_env_r(y, p, e, min_n=10):
    """{environment: Pearson r} for environments with at least min_n test cells (r-bar is their mean)."""
    out = {}
    for k in np.unique(e):
        m = e == k
        if m.sum() >= min_n and y[m].std() > 0 and p[m].std() > 0:
            out[k] = np.corrcoef(p[m], y[m])[0, 1]
    return out


def centre(v, e):
    out = v.astype(float).copy()
    for k in np.unique(e):
        m = e == k
        out[m] -= out[m].mean()
    return out


def genotype_means(y, h, e, min_envs=3):
    """Average of the environment-centred values per genotype, genotypes with at least min_envs test cells."""
    g = pd.DataFrame({'h': h, 'v': centre(y, e)}).groupby('h').v.agg(['size', 'mean'])
    return g.loc[g['size'] >= min_envs, 'mean']


def run_dir(root, data, year=None, train_from=None, loyo=None):
    if data == 'maizegep':
        return root / 'maizegep'
    if loyo is not None:
        return root / f'g2f_loyo{loyo}'
    return root / (f'g2f{year}' + (f'_from{train_from}' if train_from else ''))


def select_cells(D, data, year=2024, train_from=None, loyo=None):
    """Training and test cells of one test design; returns (cells, test mask)."""
    cells = D['cells']
    if data == 'maizegep':
        cells = cells.reset_index(drop=True)
        return cells, cells.newvar.values
    if loyo is not None:
        cells = cells[cells.Year <= 2023].reset_index(drop=True)
        return cells, cells.Year.values == loyo
    cells = cells[cells.Year <= year]
    if train_from:
        cells = cells[cells.Year >= train_from]
    cells = cells.reset_index(drop=True)
    return cells, cells.Year.values == year
