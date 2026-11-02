from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def evaluate(y, score, prob=True):
    out = dict(auroc=roc_auc_score(y, score), auprc=average_precision_score(y, score))
    out["f1@0.5"] = f1_score(y, score > 0.5) if prob else np.nan
    return out
