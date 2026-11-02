"""SKAB and TEP loaders producing per-run reference-scaled sliding windows."""
from __future__ import annotations
import glob, os
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd

ROOT = os.environ.get("INDUSTRIAL_TS_DATA", "/workspace/industrial-ts-data")
SKAB_DIR = Path(ROOT) / "SKAB/data"
TEP_DIR = Path(ROOT) / "TEP/braatz"


@dataclass
class Run:
    name: str
    x: np.ndarray  # (T, d) raw
    y: np.ndarray  # (T,) 0/1


def skab_runs(groups=("valve1", "valve2", "other")) -> list[Run]:
    runs = []
    for g in groups:
        for f in sorted(glob.glob(str(SKAB_DIR / g / "*.csv")), key=lambda p: int(Path(p).stem)):
            d = pd.read_csv(f, sep=";")
            x = d.drop(columns=["datetime", "anomaly", "changepoint"]).values.astype(float)
            runs.append(Run(f"{g}/{Path(f).stem}", x, d["anomaly"].values.astype(int)))
    return runs


def tep_runs(faults) -> list[Run]:
    runs = []
    for f in faults:
        x = np.loadtxt(TEP_DIR / f"d{f:02d}_te.dat")
        y = np.zeros(len(x), int); y[160:] = 1
        runs.append(Run(f"tep/IDV{f}", x, y))
    return runs


def windows(run: Run, L: int, ref: int, stride: int = 2):
    """Scale by the run's reference (first `ref` samples) and cut windows; label = last-step label."""
    mu = run.x[:ref].mean(0); sd = run.x[:ref].std(0) + 1e-6
    z = (run.x - mu) / sd
    idx = np.arange(L, len(z) + 1, stride)
    W = np.stack([z[i - L:i] for i in idx]).astype(np.float32)
    return W, run.y[idx - 1], z[:ref]


def build(runs: list[Run], L: int, ref: int, stride: int = 2):
    Ws, ys, refs, rid = [], [], [], []
    for k, r in enumerate(runs):
        W, y, zr = windows(r, L, ref, stride)
        Ws.append(W); ys.append(y); refs.append(zr); rid += [k] * len(y)
    return np.concatenate(Ws), np.concatenate(ys), refs, np.array(rid)
