"""WeakSense experiments E1-E5 on SKAB (+ TEP transfer)."""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from weaksense import data, lfs, label_model as lmod, models, metrics

cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
L, REF, SEEDS, EP = cfg["window"], cfg["reference"], cfg["seeds"], cfg["epochs"]
ST = dict(rounds=cfg["self_train_rounds"], c_start=cfg["confidence_start"], c_end=cfg["confidence_end"], epochs=EP)
RES = ROOT / "experiments/results"; FIG = ROOT / "figures"; RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
ONLY = sys.argv[1:] or ["E1", "E2", "E3", "E4", "E5"]

def prep(runs, per_sensor, stride=2):
    X, y, refs, rid = data.build(runs, L, REF, stride)
    Lm, names = lfs.lf_matrix(X, rid, refs, per_sensor)
    return X, y, Lm, names

def reveal(n, budget, y, seed):
    k = int(round(budget * n))
    if k == 0: return np.array([], int), np.array([])
    idx = np.random.default_rng(seed).choice(n, k, replace=False); return idx, y[idx].astype(float)

def run_methods(Xtr, ytr, Ltr, Xte, yte, budget, seed, which):
    out = {}
    idx, yl = reveal(len(Xtr), budget, ytr, seed)
    anchors = {int(i): float(v) for i, v in zip(idx, yl)}
    ds = lmod.DawidSkene().fit(Ltr, anchors or None); q = ds.q
    if "Supervised (budget only)" in which and len(idx) >= 2 and len(set(yl)) == 2:
        w = np.zeros(len(Xtr)); w[idx] = 1; t = np.zeros(len(Xtr)); t[idx] = yl
        out["Supervised (budget only)"] = models.predict(models.train_soft(Xtr, t, w, seed, EP), Xte)
    if "LF label model -> encoder" in which:
        out["LF label model -> encoder"] = models.predict(models.train_soft(Xtr, q, np.abs(q - .5) * 2, seed, EP), Xte)
    if "PU (nnPU)" in which:
        pos = q > 0.9
        if len(idx): pos[idx[yl == 1]] = True; pos[idx[yl == 0]] = False
        out["PU (nnPU)"] = models.predict(models.train_nnpu(Xtr, pos, float(np.clip(q.mean(), .05, .6)), seed, EP), Xte)
    if "WeakSense (LF+labels+curriculum)" in which:
        out["WeakSense (LF+labels+curriculum)"] = models.predict(models.self_train(Xtr, q, idx, yl, seed, **ST), Xte)
    return out, ds

summary_rows = []
def record(exp, setting, method, seed, y, s, prob=True, **kw):
    r = dict(exp=exp, setting=setting, method=method, seed=seed, **metrics.evaluate(y, s, prob), **kw); summary_rows.append(r); return r

runs = data.skab_runs()
tr_runs = [r for i, r in enumerate(runs) if i % 2 == 0]; te_runs = [r for i, r in enumerate(runs) if i % 2 == 1]
Xtr, ytr, Ltr, names = prep(tr_runs, True); Xte, yte, Lte, _ = prep(te_runs, True)
log("SKAB windows", Xtr.shape, Xte.shape, "anomaly rate", ytr.mean().round(3), yte.mean().round(3))
ALL = ["Supervised (budget only)", "LF label model -> encoder", "PU (nnPU)", "WeakSense (LF+labels+curriculum)"]

if "E1" in ONLY:
    for s in SEEDS:
        for b in [0.0, 0.005, 0.01, 0.05, 0.10]:
            which = ALL[1:] if b == 0 else [w for w in ALL if w != "LF label model -> encoder"]
            out, _ = run_methods(Xtr, ytr, Ltr, Xte, yte, b, s, which)
            for m, sc in out.items(): record("E1", f"budget={b}", m, s, yte, sc, budget=b)
            log("E1", s, b)
        full = models.predict(models.train_soft(Xtr, ytr.astype(float), np.ones(len(ytr)), s, EP), Xte)
        record("E1", "budget=1.0", "Fully supervised", s, yte, full, budget=1.0)
        record("E1", "unsupervised", "IsolationForest", s, yte, models.iforest_scores(Xtr, Xte, s), prob=False, budget=0.0)
        record("E1", "unsupervised", "Dense autoencoder", s, yte, models.ae_scores(Xtr, Xte, s), prob=False, budget=0.0)
        record("E1", "unsupervised", "LF majority vote (no training)", s, yte, lmod.majority_vote(Lte), budget=0.0)
        record("E1", "unsupervised", "LF Dawid-Skene posterior (no training)", s, yte, lmod.DawidSkene().fit(Lte).q, budget=0.0)
    pd.DataFrame([r for r in summary_rows if r["exp"] == "E1"]).to_csv(RES / "e1_label_budget_raw.csv", index=False)

if "E2" in ONLY:
    ds = lmod.DawidSkene().fit(Ltr)
    st = pd.DataFrame(lfs.lf_stats(Ltr, ytr, names, ds.learned_accuracy())); st.to_csv(RES / "e2_lf_analysis.csv", index=False)
    lm = pd.DataFrame([dict(label_model=n, **metrics.evaluate(ytr, p)) for n, p in [("Majority vote", lmod.majority_vote(Ltr)), ("Dawid-Skene EM", ds.q)]]
                      + [dict(label_model="LF coverage (any non-abstain)", auroc=np.nan, auprc=np.nan, **{"f1@0.5": (Ltr != -1).any(1).mean()})])
    lm.to_csv(RES / "e2_label_model_quality.csv", index=False); log("E2 done")

if "E3" in ONLY:
    rows = []
    best = int(np.nanargmax([(Ltr[Ltr[:, j] != -1, j] == ytr[Ltr[:, j] != -1]).mean() if (Ltr[:, j] != -1).any() else -1 for j in range(Ltr.shape[1])]))
    for kind in ["random", "adversarial"]:
        for k in [0, 2, 4, 8]:
            for s in SEEDS:
                rng = np.random.default_rng(s)
                def corrupt(Lm):
                    extra = []
                    for _ in range(k):
                        if kind == "random": extra.append(rng.choice([-1, 0, 1], size=len(Lm), p=[.4, .3, .3]))
                        else: v = Lm[:, best].copy(); v[v != -1] = 1 - v[v != -1]; extra.append(v)
                    return np.c_[Lm, np.array(extra).T] if extra else Lm
                Lc = corrupt(Ltr)
                mv = metrics.evaluate(ytr, lmod.majority_vote(Lc))["auroc"]; dsq = lmod.DawidSkene().fit(Lc).q
                idx, yl = reveal(len(Xtr), 0.01, ytr, s)
                dsa = lmod.DawidSkene().fit(Lc, {int(i): float(v) for i, v in zip(idx, yl)}).q
                sc = models.predict(models.self_train(Xtr, dsa, idx, yl, s, **ST), Xte)
                rows.append(dict(kind=kind, n_bad_lfs=k, seed=s, mv_train_auroc=mv, ds_train_auroc=metrics.evaluate(ytr, dsq)["auroc"],
                                 ds_anchored_train_auroc=metrics.evaluate(ytr, dsa)["auroc"], weaksense_test_auroc=metrics.evaluate(yte, sc)["auroc"]))
                log("E3", kind, k, s)
    pd.DataFrame(rows).to_csv(RES / "e3_noisy_lfs_raw.csv", index=False)

if "E4" in ONLY:
    rows = []
    v1 = [r for r in runs if r.name.startswith("valve1")]; oth = [r for r in runs if not r.name.startswith("valve1")]
    tep_tr = data.tep_runs([1, 2, 4, 5, 6, 7, 8, 10, 11, 12]); tep_te = data.tep_runs([13, 14, 16, 17, 18, 19, 20, 21])
    for name, (a, b, ps, stride) in {"SKAB valve1 -> valve2+other": (v1, oth, True, 2), "TEP faults {1,2,4-8,10-12} -> {13,14,16-21}": (tep_tr, tep_te, False, 4)}.items():
        Xa, ya, La, _ = prep(a, ps, stride); Xb, yb, Lb, _ = prep(b, ps, stride)
        for s in SEEDS:
            out, _ = run_methods(Xa, ya, La, Xb, yb, 0.01, s, ALL)
            for m, sc in out.items(): rows.append(dict(transfer=name, method=m, seed=s, **metrics.evaluate(yb, sc)))
            rows.append(dict(transfer=name, method="Fully supervised (source)", seed=s, **metrics.evaluate(yb, models.predict(models.train_soft(Xa, ya.astype(float), np.ones(len(ya)), s, EP), Xb))))
            rows.append(dict(transfer=name, method="IsolationForest", seed=s, **metrics.evaluate(yb, models.iforest_scores(Xa, Xb, s), False)))
            log("E4", name, s)
    pd.DataFrame(rows).to_csv(RES / "e4_transfer_raw.csv", index=False)

if "E5" in ONLY:
    rows = []
    for s in SEEDS:
        idx, yl = reveal(len(Xtr), 0.01, ytr, s); anchors = {int(i): float(v) for i, v in zip(idx, yl)}
        q = lmod.DawidSkene().fit(Ltr, anchors).q; qmv = lmod.majority_vote(Ltr); qmv[idx] = yl
        var = {
            "Full WeakSense": lambda: models.self_train(Xtr, q, idx, yl, s, **ST),
            "No LFs (labels + self-training)": lambda: models.self_train(Xtr, q, idx, yl, s, use_lf=False, **ST),
            "No self-training": lambda: models.self_train(Xtr, q, idx, yl, s, **{**ST, "rounds": 0}),
            "No curriculum (fixed threshold)": lambda: models.self_train(Xtr, q, idx, yl, s, curriculum=False, **ST),
            "Majority vote instead of Dawid-Skene": lambda: models.self_train(Xtr, qmv, idx, yl, s, **ST),
            "No true labels (budget 0)": lambda: models.self_train(Xtr, lmod.DawidSkene().fit(Ltr).q, np.array([], int), np.array([]), s, **ST),
        }
        for n, f in var.items():
            rows.append(dict(variant=n, seed=s, **metrics.evaluate(yte, models.predict(f(), Xte)))); log("E5", n, s)
    pd.DataFrame(rows).to_csv(RES / "e5_ablation_raw.csv", index=False)
log("done")
