"""Draw Figure 2 (incident prediction) from the deposited CSVs.

Inputs : Incident_primary.csv, Figure2_importance.csv in T2D_OUT_DIR
Output : ../figures/Figure2_incident.png
Panel A reports POOLED held-out AUROC with 95% percentile bootstrap intervals, the same
quantities as the pooled rows of Table 2A; panels B and C report gain shares from a model
fitted to the whole population of each sex.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT

ROWS = ["LightGBM, 175 predictors", "LightGBM, diagnosis <5 years",
        "LightGBM, diagnosis 5-10 years", "LightGBM, diagnosis >=10 years",
        "Five-variable comparator"]
TICKS = ["All cases", "<5 y", "5-10 y", ">=10 y", "Five-variable\ncomparator"]
REN = {"Body mass index (BMI) [23104]": "BMI (impedance)",
       "Body mass index (BMI) [21001]": "BMI (measured)",
       "Age when attended assessment centre [21003]": "Age",
       "Gamma glutamyltransferase [30730]": "Gamma-glutamyltransferase",
       "Forced expiratory volume in 1-second (FEV1) [3063]": "FEV1",
       "Forced vital capacity (FVC) [3062]": "FVC",
       "Sex hormone-binding globulin [30830]": "SHBG"}
COL = {"Male": "#1f4e79", "Female": "#c0504d"}
lab = lambda p: REN.get(p, p.rsplit(" [", 1)[0])


def main():
    IP = pd.read_csv(os.path.join(OUT, "Incident_primary.csv"))
    IMP = pd.read_csv(os.path.join(OUT, "Figure2_importance.csv"))
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.46, height_ratios=[1.0, 1.25])
    axA, axB, axC = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for k, sex in enumerate(("Male", "Female")):
        xs, ys, lo, hi = [], [], [], []
        for i, a in enumerate(ROWS):
            r = IP[(IP.Sex == sex) & (IP.Analysis == a)].iloc[0]
            l, h = [float(v) for v in r.pooled_95CI.split("-")]
            xs.append(i + (k - 0.5) * 0.17); ys.append(r.pooled_AUROC)
            lo.append(r.pooled_AUROC - l); hi.append(h - r.pooled_AUROC)
        axA.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=5, color=COL[sex], ecolor=COL[sex],
                     elinewidth=1.1, capsize=2.5, label=sex.lower())
    axA.set_xticks(range(len(ROWS))); axA.set_xticklabels(TICKS); axA.set_ylabel("AUROC")
    axA.set_title("Incident prediction: pooled held-out AUROC with 95% confidence intervals",
                  loc="left")
    axA.legend(frameon=False, fontsize=8); axA.margins(x=0.08)
    axA.grid(axis="y", alpha=0.25, lw=0.6)
    for ax, sex in ((axB, "Male"), (axC, "Female")):
        s = IMP[IMP.Sex == sex].sort_values("Gain, %")
        ax.barh(np.arange(len(s)), s["Gain, %"], color=COL[sex], height=0.72)
        ax.set_yticks(np.arange(len(s))); ax.set_yticklabels([lab(p) for p in s.Predictor], fontsize=7)
        ax.set_xlabel("Share of total gain, %")
        ax.set_title("Men" if sex == "Male" else "Women", loc="left"); ax.margins(y=0.02)
    for ax, lt in ((axA, "a"), (axB, "b"), (axC, "c")):
        ax.text(-0.02, 1.06, lt, transform=ax.transAxes, fontweight="bold", fontsize=11,
                ha="right", va="bottom")
    figdir = os.path.join(OUT, "..", "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "Figure2_incident.png")
    fig.savefig(out, dpi=400, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
