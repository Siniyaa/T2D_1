"""Table 5 and Figure 5: calibration, clinical utility and treatment leakage.

Predicted probabilities from a case-control development sample are not risks. They are first
corrected to the validation-sample prevalence with the standard case-control offset on the
logit scale, then calibration and net benefit are assessed.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, argparse, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from common import (load_prevalent, load_nhanes, score_nhanes, nhanes_diagnosed,
                    nhanes_undiagnosed, MAP, OUT)


def offset_correct(p, pi1, delta1):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p)) - np.log(pi1 / (1 - pi1)) + np.log(delta1 / (1 - delta1))
    return 1 / (1 + np.exp(-lo))


def calibration(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p))
    slope = sm.Logit(np.asarray(y, float), sm.add_constant(lo)).fit(disp=0).params[1]
    inter = sm.Logit(np.asarray(y, float), np.ones((len(y), 1)), offset=lo).fit(disp=0).params[0]
    return float(inter), float(slope), float(brier_score_loss(y, p))


def net_benefit(y, p, pt):
    y = np.asarray(y); n = len(y); pred = p >= pt
    tp = ((pred) & (y == 1)).sum(); fp = ((pred) & (y == 0)).sum()
    return tp / n - (fp / n) * (pt / (1 - pt))


def main(make_figure=False):
    prev = load_prevalent()
    pi1 = float(prev.HasT2D.mean())
    N = score_nhanes(load_nhanes(), prev)
    cal, dca, ops = [], [], []
    for sex in ("Male", "Female"):
        for lbl, dd in [("Diagnosed", nhanes_diagnosed(N, sex)),
                        ("Undiagnosed", nhanes_undiagnosed(N, sex))]:
            prevalence = dd.y.mean()
            praw = dd.p_lgbm.values
            pcor = offset_correct(praw, pi1, prevalence)
            for tag, pv in [("raw", praw), ("prevalence-corrected", pcor)]:
                i, s, b = calibration(dd.y, pv)
                cal.append({"outcome": lbl, "sex": sex, "probs": tag,
                            "prevalence": round(prevalence, 4), "intercept": round(i, 2),
                            "slope": round(s, 2), "Brier": round(b, 4)})
            for pt in (0.02, 0.05, 0.075, 0.10, 0.15, 0.20):
                dca.append({"outcome": lbl, "sex": sex, "threshold": pt,
                            "NB_model": round(net_benefit(dd.y, pcor, pt), 4),
                            "NB_screen_all": round(net_benefit(dd.y, np.ones(len(dd)), pt), 4)})
            for pt in (0.05, 0.075, 0.10):
                pred = pcor >= pt; y = dd.y.values
                tp = ((pred) & (y == 1)).sum(); fp = ((pred) & (y == 0)).sum()
                fn = ((~pred) & (y == 1)).sum(); tn = ((~pred) & (y == 0)).sum()
                ops.append({"outcome": lbl, "sex": sex, "threshold": pt,
                            "tested_%": round(pred.mean() * 100, 1),
                            "sensitivity_%": round(100 * tp / max(tp + fn, 1), 1),
                            "specificity_%": round(100 * tn / max(tn + fp, 1), 1),
                            "PPV_%": round(100 * tp / max(tp + fp, 1), 1),
                            "NPV_%": round(100 * tn / max(tn + fn, 1), 1),
                            "NNT": round((tp + fp) / max(tp, 1), 1)})
    for frame, name in [(cal, "Calibration.csv"), (dca, "DecisionCurve.csv"),
                        (ops, "OperatingPoints.csv")]:
        pd.DataFrame(frame).to_csv(os.path.join(OUT, name), index=False)
        print("wrote", os.path.join(OUT, name))
    if make_figure:
        figure5(N, pi1)


def figure5(N, pi1):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from leakage import leakage_table
    L = leakage_table(N).sort_values("gap")
    fig = plt.figure(figsize=(7.2, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30, height_ratios=[0.75, 1.45])
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    col = {"Male": "#1f4e79", "Female": "#c0504d"}
    ths = np.linspace(0.01, 0.25, 60)
    for sex in ("Male", "Female"):
        dd = nhanes_undiagnosed(N, sex)
        p = offset_correct(dd.p_lgbm.values, pi1, dd.y.mean())
        q = pd.qcut(p, 8, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": dd.y.values, "q": q}).groupby("q", observed=True).mean()
        axA.plot(g.p, g.y, "o-", color=col[sex], ms=4, lw=1.4, label=sex)
        axB.plot(ths, [net_benefit(dd.y, p, t) for t in ths], color=col[sex], lw=1.8,
                 label=f"Model, {sex.lower()}")
        axB.plot(ths, [net_benefit(dd.y, np.ones(len(dd)), t) for t in ths], color=col[sex],
                 lw=1.0, ls=":", label=f"Test everyone, {sex.lower()}")
    axA.plot([0, .35], [0, .35], "--", color="0.5", lw=1)
    axA.set_xlabel("Predicted probability"); axA.set_ylabel("Observed proportion")
    axA.set_title("Calibration", loc="left"); axA.legend(frameon=False, fontsize=7)
    axB.axhline(0, color="0.4", lw=0.9); axB.set_ylim(-0.05, 0.08)
    axB.set_xlabel("Threshold probability"); axB.set_ylabel("Net benefit")
    axB.set_title("Clinical utility", loc="left"); axB.legend(frameon=False, fontsize=6.3)
    y = np.arange(len(L))
    axC.barh(y, L.gap, color=["#c0504d" if v > 0 else "#2e9bb5" for v in L.gap], height=0.74)
    axC.set_yticks(y); axC.set_yticklabels([p[:40] for p in L.predictor], fontsize=7)
    axC.axvline(0, color="0.3", lw=0.9); axC.margins(y=0.01)
    axC.set_xlabel("Leakage gap: AUROC(diagnosed) - AUROC(undiagnosed)")
    axC.set_title("Treatment signal in individual predictors", loc="left")
    figdir = os.path.join(OUT, "..", "figures"); os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, "Figure5.png"), dpi=400, bbox_inches="tight")
    print("wrote", os.path.join(figdir, "Figure5.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true", help="also render Figure 5")
    main(ap.parse_args().figure)
