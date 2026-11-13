"""Denoiser: heterogeneous graph transformer over genotype and environment nodes, plus a cell MLP.

Edges: ctx = visible measured cells, qry = cells being denoised (both directions),
kin = 10 closest relatives of each genotype, wx = 10 closest environments by weather vector.
"""
import torch
from torch import nn
from torch_geometric.nn import HeteroConv, TransformerConv

from common import timestep_embedding

R_CTX, R_CTX_R = ('env', 'ctx', 'geno'), ('geno', 'ctx_r', 'env')
R_QRY, R_QRY_R = ('env', 'qry', 'geno'), ('geno', 'qry_r', 'env')
R_KIN, R_WX = ('geno', 'kin', 'geno'), ('env', 'wx', 'env')
CELL_FEATS = 6   # value, is_query, sqrt(abar), sqrt(1 - abar), log1p(plot count), measurement variance
SC_FEATS = 2     # self-conditioning: previous estimate, flag


class Denoiser(nn.Module):
    def __init__(self, n_markers, env_dim, d=128, heads=4, layers=3, edge_dim=32,
                 marker_dropout=0.3, dropout=0.1, temb_dim=64, self_cond=False):
        super().__init__()
        self.temb_dim, self.self_cond = temb_dim, self_cond
        nf = CELL_FEATS + SC_FEATS * self_cond
        nrow = 2 + self_cond
        self.marker_drop = nn.Dropout(marker_dropout)
        self.geno_proj = nn.Linear(n_markers, d)
        self.env_proj = nn.Linear(env_dim, d)
        self.cnt_g, self.cnt_e = nn.Linear(1, d), nn.Linear(1, d)
        self.row_g, self.row_e = nn.Linear(nrow, d), nn.Linear(nrow, d)
        self.temb = nn.Sequential(nn.Linear(temb_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.cell_edge = nn.Sequential(nn.Linear(nf, edge_dim), nn.SiLU(), nn.Linear(edge_dim, edge_dim))
        self.kin_edge, self.wx_edge = nn.Linear(1, edge_dim), nn.Linear(1, edge_dim)

        def bip():
            return TransformerConv((d, d), d // heads, heads=heads, edge_dim=edge_dim, root_weight=False)

        def same():
            return TransformerConv(d, d // heads, heads=heads, edge_dim=edge_dim, root_weight=False)

        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict(dict(
                conv=HeteroConv({R_CTX: bip(), R_CTX_R: bip(), R_QRY: bip(), R_QRY_R: bip(),
                                 R_KIN: same(), R_WX: same()}, aggr='sum'),
                tg=nn.Linear(d, d), te=nn.Linear(d, d),
                n1g=nn.LayerNorm(d), n1e=nn.LayerNorm(d), n2g=nn.LayerNorm(d), n2e=nn.LayerNorm(d),
                fg=nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)),
                fe=nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)))))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(3 * d + 5 + SC_FEATS * self_cond, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(),
                                  nn.Linear(128, 1))

    def forward(self, g):
        reps = g['reps']
        hg = self.geno_proj(self.marker_drop(g['M'])).repeat(reps, 1) + self.cnt_g(g['cnt_g'][:, None])
        he = self.env_proj(g['Zenv']).repeat(reps, 1) + self.cnt_e(g['cnt_e'][:, None])
        xq = g['qry_attr'][:, 0]
        for name, idx, n in (('g', g['qry_h'], len(hg)), ('e', g['qry_e'], len(he))):
            cnt = torch.zeros(n, device=xq.device).index_add_(0, idx, torch.ones_like(xq))
            mean = torch.zeros(n, device=xq.device).index_add_(0, idx, xq) / cnt.clamp(min=1)
            row = [mean, (cnt > 0).float()]
            if self.self_cond:
                row.append(torch.zeros(n, device=xq.device).index_add_(0, idx, g['qry_attr'][:, 6]) / cnt.clamp(min=1))
            row = torch.stack(row, 1)
            if name == 'g':
                hg = hg + self.row_g(row)
            else:
                he = he + self.row_e(row)
        tt = self.temb(timestep_embedding(torch.tensor([float(g['t'])], device=hg.device), self.temb_dim))
        ctx_a, qry_a = self.cell_edge(g['ctx_attr']), self.cell_edge(g['qry_attr'])
        ea = {R_CTX: ctx_a, R_CTX_R: ctx_a, R_QRY: qry_a, R_QRY_R: qry_a,
              R_KIN: self.kin_edge(g['kin_w'][:, None]), R_WX: self.wx_edge(g['wx_w'][:, None])}
        for L in self.layers:
            out = L['conv']({'geno': hg + L['tg'](tt), 'env': he + L['te'](tt)}, g['edges'], edge_attr_dict=ea)
            hg = L['n1g'](hg + self.drop(out['geno']))
            he = L['n1e'](he + self.drop(out['env']))
            hg = L['n2g'](hg + self.drop(L['fg'](hg)))
            he = L['n2e'](he + self.drop(L['fe'](he)))
        qa = g['qry_attr'][:, [0, 2, 3, 4, 5, 6, 7] if self.self_cond else [0, 2, 3, 4, 5]]
        z = torch.cat([hg[g['qry_h']], he[g['qry_e']], tt.expand(len(qa), -1), qa], 1)
        return self.head(z).squeeze(-1)
