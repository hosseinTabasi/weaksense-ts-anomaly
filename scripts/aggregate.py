"""Aggregate raw experiment CSVs into mean/std summaries, figures and reports/RESULTS.md."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "experiments/results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
out = ["# WeakSense — Results", "", "All numbers are aggregated from `experiments/results/*_raw.csv` (seeds 0,1,2; mean ± std).", ""]
def ms(df, keys, cols):
    g = df.groupby(keys, sort=False)[cols].agg(["mean", "std"]).round(3)
    g.columns = [f"{a}_{b}" for a, b in g.columns]; return g.reset_index()
def md(df): return df.to_markdown(index=False)

f = RES / "e1_label_budget_raw.csv"
if f.exists():
    d = pd.read_csv(f); s = ms(d, ["setting", "method"], ["auroc", "auprc", "f1@0.5"]); s.to_csv(RES / "e1_label_budget_summary.csv", index=False)
    out += ["## E1 Label-budget curve (SKAB test runs)", "", md(s), ""]
    plt.figure(figsize=(6, 4))
    for m, g in d[d.setting.str.startswith("budget")].groupby("method"):
        a = g.groupby("budget").auroc.agg(["mean", "std"])
        if len(a) > 1: plt.errorbar(a.index * 100, a["mean"], a["std"], marker="o", capsize=3, label=m)
        else: plt.axhline(a["mean"].iloc[0], ls="--", c="k", label=m)
    for m, g in d[d.setting == "unsupervised"].groupby("method"): plt.axhline(g.auroc.mean(), ls=":", lw=1, label=m, c=plt.cm.tab10(hash(m) % 10))
    plt.xlabel("% of training windows labelled"); plt.ylabel("test AUROC"); plt.xscale("symlog", linthresh=0.5); plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(FIG / "e1_label_budget.png", dpi=150); plt.close()
for n in ["e2_lf_analysis", "e2_label_model_quality"]:
    f = RES / f"{n}.csv"
    if f.exists(): out += [f"## E2 {n}", "", md(pd.read_csv(f).round(3)), ""]
f = RES / "e2_lf_analysis.csv"
if f.exists():
    d = pd.read_csv(f); print(d.columns.tolist())
f = RES / "e3_noisy_lfs_raw.csv"
if f.exists():
    d = pd.read_csv(f); cols = ["mv_train_auroc", "ds_train_auroc", "ds_anchored_train_auroc", "weaksense_test_auroc"]
    s = ms(d, ["kind", "n_bad_lfs"], cols); s.to_csv(RES / "e3_noisy_lfs_summary.csv", index=False); out += ["## E3 Noisy / adversarial LFs", "", md(s), ""]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for a, (k, g) in zip(ax, d.groupby("kind")):
        for c in cols:
            m = g.groupby("n_bad_lfs")[c].agg(["mean", "std"]); a.errorbar(m.index, m["mean"], m["std"], marker="o", capsize=3, label=c)
        a.set_title(f"{k} bad LFs"); a.set_xlabel("# injected LFs")
    ax[0].set_ylabel("AUROC"); ax[1].legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / "e3_noisy_lfs.png", dpi=150); plt.close()
f = RES / "e4_transfer_raw.csv"
if f.exists():
    d = pd.read_csv(f); s = ms(d, ["transfer", "method"], ["auroc", "auprc", "f1@0.5"]); s.to_csv(RES / "e4_transfer_summary.csv", index=False)
    out += ["## E4 Transfer", "", md(s), ""]
    p = s.pivot(index="method", columns="transfer", values="auroc_mean"); p.plot.barh(figsize=(7, 4)); plt.xlabel("target AUROC"); plt.tight_layout(); plt.savefig(FIG / "e4_transfer.png", dpi=150); plt.close()
f = RES / "e5_ablation_raw.csv"
if f.exists():
    d = pd.read_csv(f); s = ms(d, ["variant"], ["auroc", "auprc", "f1@0.5"]); s.to_csv(RES / "e5_ablation_summary.csv", index=False)
    out += ["## E5 Ablation (budget 1%)", "", md(s), ""]
    plt.figure(figsize=(6, 3.5)); plt.barh(s.variant, s.auroc_mean, xerr=s.auroc_std.fillna(0)); plt.xlim(0.5, 1); plt.xlabel("test AUROC"); plt.tight_layout(); plt.savefig(FIG / "e5_ablation.png", dpi=150); plt.close()
(ROOT / "reports").mkdir(exist_ok=True); (ROOT / "reports/RESULTS.md").write_text("\n".join(out))
print("\n".join(out))
