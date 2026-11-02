"""Temporal encoder (Conv1d + GRU) with noise-aware, PU and self-training objectives; unsupervised baselines."""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn
from sklearn.ensemble import IsolationForest

torch.set_num_threads(4)


class TemporalEncoder(nn.Module):
    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(d, hidden, 5, padding=2), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def embed(self, x):  # x (B, L, d)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, hn = self.gru(h); return hn[-1]

    def forward(self, x): return self.head(self.embed(x)).squeeze(-1)


def _seed(s): torch.manual_seed(s); np.random.seed(s)


def train_soft(X, target, weight, seed, epochs=15, lr=2e-3):
    """Noise-aware training: BCE against probabilistic targets with per-sample confidence weights."""
    _seed(seed); m = TemporalEncoder(X.shape[2]); opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.from_numpy(X); T = torch.tensor(target, dtype=torch.float32); Wt = torch.tensor(weight, dtype=torch.float32)
    keep = torch.nonzero(Wt > 0).squeeze(1)
    for _ in range(epochs):
        perm = keep[torch.randperm(len(keep))]
        for b in range(0, len(perm), 128):
            i = perm[b:b + 128]
            loss = (nn.functional.binary_cross_entropy_with_logits(m(Xt[i]), T[i], reduction="none") * Wt[i]).sum() / Wt[i].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def train_nnpu(X, pos_mask, prior, seed, epochs=15, lr=2e-3):
    """Non-negative PU risk (Kiryo et al., 2017) with sigmoid loss."""
    _seed(seed); m = TemporalEncoder(X.shape[2]); opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.from_numpy(X); P = torch.nonzero(torch.tensor(pos_mask)).squeeze(1); U = torch.nonzero(torch.tensor(~pos_mask)).squeeze(1)
    loss_f = lambda z, y: torch.sigmoid(-y * z)
    nb = max(1, len(U) // 128)
    for _ in range(epochs):
        pu = U[torch.randperm(len(U))]
        for b in range(nb):
            iu = pu[b * 128:(b + 1) * 128]; ip = P[torch.randint(len(P), (min(128, max(len(P), 1)),))]
            zp, zu = m(Xt[ip]), m(Xt[iu])
            rp = loss_f(zp, 1).mean(); rn = loss_f(zu, -1).mean() - prior * loss_f(zp, -1).mean()
            loss = prior * rp + torch.clamp(rn, min=0) if rn >= 0 else -rn
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def predict(m, X):
    out = []
    for b in range(0, len(X), 1024): out.append(torch.sigmoid(m(torch.from_numpy(X[b:b + 1024]))).numpy())
    return np.concatenate(out)


def self_train(X, lm_prob, true_idx, true_y, seed, rounds=3, c_start=0.95, c_end=0.85, epochs=15, use_lf=True, curriculum=True):
    """WeakSense combo: LF posteriors + few true labels + confidence-curriculum self-training."""
    n = len(X)
    target = lm_prob.copy() if use_lf else np.zeros(n)
    weight = (np.abs(lm_prob - 0.5) * 2) if use_lf else np.zeros(n)
    target[true_idx] = true_y; weight[true_idx] = 3.0
    m = train_soft(X, target, weight, seed, epochs)
    for r in range(rounds):
        c = c_start + (c_end - c_start) * (r / max(rounds - 1, 1)) if curriculum else c_end
        p = predict(m, X)
        conf = np.maximum(p, 1 - p) >= c
        t2, w2 = target.copy(), weight.copy()
        t2[conf] = (p[conf] > 0.5).astype(float); w2[conf] = np.maximum(w2[conf], np.maximum(p, 1 - p)[conf])
        t2[true_idx] = true_y; w2[true_idx] = 3.0
        m = train_soft(X, t2, w2, seed + 100 * (r + 1), epochs)
    return m


class DenseAE(nn.Module):
    def __init__(self, D):
        super().__init__(); self.net = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, D))
    def forward(self, x): return self.net(x)


def ae_scores(Xtr, Xte, seed, epochs=30):
    _seed(seed); A = torch.from_numpy(Xtr.reshape(len(Xtr), -1)); B = torch.from_numpy(Xte.reshape(len(Xte), -1))
    m = DenseAE(A.shape[1]); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(A))
        for b in range(0, len(A), 256):
            i = perm[b:b + 256]; loss = ((m(A[i]) - A[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return ((m(B) - B) ** 2).mean(1).numpy()


def iforest_scores(Xtr, Xte, seed):
    f = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr.reshape(len(Xtr), -1))
    return -f.score_samples(Xte.reshape(len(Xte), -1))
