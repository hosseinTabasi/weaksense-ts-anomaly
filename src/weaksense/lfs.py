"""Programmatic labeling functions. Votes: 1 anomaly, 0 normal, -1 abstain.

All thresholds are fixed a priori in reference-scaled (z) units; none are tuned on labels.
"""
from __future__ import annotations
import numpy as np

ABSTAIN = -1


class CrossSensorModel:
    """Predict each sensor from the others with ridge regression fit on reference (normal) data."""
    def fit(self, Z: np.ndarray, alpha: float = 1.0):
        d = Z.shape[1]; self.B = []
        for j in range(d):
            X = np.delete(Z, j, 1); X1 = np.c_[X, np.ones(len(X))]
            b = np.linalg.solve(X1.T @ X1 + alpha * np.eye(X1.shape[1]), X1.T @ Z[:, j])
            self.B.append(b)
        res = self.residual(Z); self.sd = res.std(0) + 1e-6
        return self

    def residual(self, Z):
        out = np.zeros_like(Z)
        for j, b in enumerate(self.B):
            X = np.delete(Z, j, -1); out[..., j] = Z[..., j] - (X @ b[:-1] + b[-1])
        return out


def _vote(hi, lo=None):
    v = np.full(hi.shape, ABSTAIN); v[hi] = 1
    if lo is not None: v[lo & ~hi] = 0
    return v


def generic_lfs(W: np.ndarray, xsm: CrossSensorModel, per_sensor: bool):
    """W: (N, L, d) z-scaled windows. Returns (N, m) votes and LF names."""
    last = np.abs(W[:, -1, :]); absW = np.abs(W)
    votes, names = [], []
    votes.append(_vote(last.max(1) > 5)); names.append("threshold_any_sensor>5sd")
    votes.append(_vote(np.zeros(len(W), bool), absW.max((1, 2)) < 2)); names.append("all_sensors_within_2sd")
    dz = np.abs(np.diff(W, axis=1)).max((1, 2))
    votes.append(_vote(dz > 3)); names.append("rate_of_change>3sd")
    res = np.abs(xsm.residual(W[:, -8:, :]) / xsm.sd).mean(1).max(1)
    votes.append(_vote(res > 3, res < 1)); names.append("cross_sensor_inconsistency")
    F = np.abs(np.fft.rfft(W - W.mean(1, keepdims=True), axis=1)) ** 2
    hf = F[:, F.shape[1] // 2:, :].sum(1) / (F.sum(1) + 1e-9)
    votes.append(_vote(hf.max(1) > 0.6)); names.append("spectral_high_freq_share>0.6")
    ms = np.abs(W.mean(1)).mean(1)
    votes.append(_vote(ms > 2, ms < 0.5)); names.append("mean_shift")
    vs = W.std(1).max(1)
    votes.append(_vote(vs > 3, vs < 1.2)); names.append("variance_burst")
    if per_sensor:
        for j in range(W.shape[2]):
            votes.append(_vote(last[:, j] > 4, absW[:, :, j].max(1) < 1.5)); names.append(f"threshold_sensor{j}")
    return np.stack(votes, 1), names


def lf_matrix(W, rid, refs, per_sensor):
    """Cross-sensor model fit per run on its reference segment."""
    out = np.zeros((len(W), 0)); names = None; parts = []
    for k in np.unique(rid):
        xsm = CrossSensorModel().fit(refs[k])
        v, names = generic_lfs(W[rid == k], xsm, per_sensor); parts.append((np.where(rid == k)[0], v))
    out = np.full((len(W), parts[0][1].shape[1]), ABSTAIN)
    for idx, v in parts: out[idx] = v
    return out, names


def lf_stats(L: np.ndarray, y: np.ndarray, names, learned_acc=None):
    rows = []
    act = L != ABSTAIN
    for j, n in enumerate(names):
        a = act[:, j]; others = act.copy(); others[:, j] = False
        overlap = (a & others.any(1)).mean()
        conflict = np.mean(a & np.any((L != L[:, [j]]) & others, 1))
        acc = (L[a, j] == y[a]).mean() if a.any() else np.nan
        r = dict(lf=n, coverage=a.mean(), overlap=overlap, conflict=conflict, empirical_acc=acc,
                 polarity="/".join(str(v) for v in sorted(set(L[a, j]))))
        if learned_acc is not None: r["learned_acc"] = learned_acc[j]
        rows.append(r)
    return rows
