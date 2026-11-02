"""Label models over LF votes (1/0/-1 abstain)."""
from __future__ import annotations
import numpy as np


def majority_vote(L: np.ndarray, prior: float = 0.5) -> np.ndarray:
    pos = (L == 1).sum(1); neg = (L == 0).sum(1); n = pos + neg
    return np.where(n > 0, (pos + prior) / (n + 1), prior)


class DawidSkene:
    """Binary generative label model with class-conditional LF accuracies, fit by EM.

    P(y=1)=pi;  P(L_j=1|y=1)=tpr_j, P(L_j=0|y=0)=tnr_j, abstains ignored (conditionally independent LFs).
    Optional anchors: a few true labels clamp the posterior during EM (semi-supervised).
    """
    def __init__(self, iters: int = 100, smooth: float = 1.0):
        self.iters, self.smooth = iters, smooth

    def fit(self, L: np.ndarray, anchors: dict | None = None):
        q = majority_vote(L)
        idx = np.array(list(anchors)) if anchors else None
        for _ in range(self.iters):
            if idx is not None: q[idx] = np.array([anchors[i] for i in idx])
            self.pi = np.clip(q.mean(), 0.02, 0.98)
            pos1 = ((L == 1) * q[:, None]).sum(0); act1 = ((L != -1) * q[:, None]).sum(0)
            neg0 = ((L == 0) * (1 - q)[:, None]).sum(0); act0 = ((L != -1) * (1 - q)[:, None]).sum(0)
            s = self.smooth
            self.tpr = np.clip((pos1 + s) / (act1 + 2 * s), 0.01, 0.99)
            self.tnr = np.clip((neg0 + s) / (act0 + 2 * s), 0.01, 0.99)
            q = self.predict_proba(L)
        if idx is not None: q[idx] = np.array([anchors[i] for i in idx])
        self.q = q
        return self

    def predict_proba(self, L):
        lp1 = np.log(self.pi) + np.where(L == 1, np.log(self.tpr), 0) .sum(1) + np.where(L == 0, np.log(1 - self.tpr), 0).sum(1)
        lp0 = np.log(1 - self.pi) + np.where(L == 0, np.log(self.tnr), 0).sum(1) + np.where(L == 1, np.log(1 - self.tnr), 0).sum(1)
        return 1 / (1 + np.exp(np.clip(lp0 - lp1, -50, 50)))

    def learned_accuracy(self):
        """Balanced accuracy of each LF implied by the fitted parameters."""
        return 0.5 * (self.tpr + self.tnr)
