"""Table 5 and Figure 5: calibration, clinical utility and treatment leakage.

Predicted probabilities from a case-control development sample are not risks. They are first
corrected to the validation-sample prevalence with the standard case-control offset on the
logit scale, then calibration and net benefit are assessed.

The offset uses the SEX-SPECIFIC case fraction of the development sample, because the models
are sex-specific; using a pooled fraction would mis-centre both models. All quantities in this
script are UNWEIGHTED: the prevalence offset, calibration, net benefit and operating points are
computed on the raw NHANES records, not on survey-weighted estimates. Survey weighting is
applied only to the discrimination estimates in nhanes_weighted.py.

Brier scores are reported alongside the score of a non-informative reference that predicts the
sample prevalence for everyone, p(1-p), so that the reader can see whether the model improves
on a constant prediction.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, argparse, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from common import (load_prevalent, load_nhanes, score_nhanes, nhanes_diagnosed,
                    nhanes_undiagnosed, MAP, OUT, SEX)


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
    # sex-specific development case fractions (the models are sex-specific)
    PI1 = {s: float(prev.loc[prev[SEX] == s, "HasT2D"].mean()) for s in ("Male", "Female")}
    N = score_nhanes(load_nhanes(), prev)
    cal, dca, ops = [], [], []
    for sex in ("Male", "Female"):
        pi1 = PI1[sex]
        for lbl, dd in [("Diagnosed", nhanes_diagnosed(N, sex)),
                        ("Undiagnosed", nhanes_undiagnosed(N, sex))]:
            prevalence = dd.y.mean()
            praw = dd.p_lgbm.values
            pcor = offset_correct(praw, pi1, prevalence)
            for tag, pv in [("raw", praw), ("prevalence-corrected", pcor)]:
                i, s, b = calibration(dd.y, pv)
                cal.append({"outcome": lbl, "sex": sex, "probs": tag,
                            "prevalence": round(prevalence, 6), "intercept": round(i, 2),
                            "slope": round(s, 2), "Brier": round(b, 5),
                            "Brier_null": round(prevalence * (1 - prevalence), 5),
                            "n": int(len(dd)), "events": int(dd.y.sum())})
            for pt in np.round(np.arange(0.005, 0.2501, 0.005), 4):
                dca.append({"outcome": lbl, "sex": sex, "threshold": float(pt),
                            "NB_model": round(net_benefit(dd.y, pcor, pt), 4),
                            "NB_screen_all": round(net_benefit(dd.y, np.ones(len(dd)), pt), 4)})
            for pt in (0.05, 0.075, 0.10):
                pred = pcor >= pt; y = dd.y.values
                _rng = np.random.default_rng(7); _se, _te = [], []
                for _ in range(1000):
                    _b = _rng.integers(0, len(y), len(y))
                    _yb, _pb = y[_b], pred[_b]
                    if _yb.sum() > 0:
                        _se.append(100 * (_pb & (_yb == 1)).sum() / _yb.sum())
                        _te.append(100 * _pb.mean())
                tp = ((pred) & (y == 1)).sum(); fp = ((pred) & (y == 0)).sum()
                fn = ((~pred) & (y == 1)).sum(); tn = ((~pred) & (y == 0)).sum()
                ops.append({"outcome": lbl, "sex": sex, "threshold": pt,
                            "tested_%": round(pred.mean() * 100, 1),
                            "sensitivity_%": round(100 * tp / max(tp + fn, 1), 1),
                            "sensitivity_95CI": f"{np.percentile(_se,2.5):.1f}-{np.percentile(_se,97.5):.1f}",
                            "tested_95CI": f"{np.percentile(_te,2.5):.1f}-{np.percentile(_te,97.5):.1f}",
                            "specificity_%": round(100 * tn / max(tn + fp, 1), 1),
                            "PPV_%": round(100 * tp / max(tp + fp, 1), 1),
                            "NPV_%": round(100 * tn / max(tn + fn, 1), 1),
                            "NNT": round((tp + fp) / max(tp, 1), 1)})
    for frame, name in [(cal, "Calibration.csv"), (dca, "DecisionCurve.csv"),
                        (ops, "OperatingPoints.csv")]:
        pd.DataFrame(frame).to_csv(os.path.join(OUT, name), index=False)
        print("wrote", os.path.join(OUT, name))
    if make_figure:
        figure5(N, PI1)

def _lowess_band(y, p, grid, B=400, seed=7, frac=0.30):
    """Loess calibration curve with a percentile bootstrap band.

    it=0 is required: the statsmodels default of three robustifying iterations treats the cases
    of a binary outcome as outliers and collapses the fitted curve onto zero.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    y, p = np.asarray(y, float), np.asarray(p, float)

    def f(yy, pp):
        s = lowess(yy, pp, frac=frac, it=0, return_sorted=True)
        return np.interp(grid, s[:, 0], s[:, 1])

    obs = f(y, p)
    rng = np.random.default_rng(seed); n = len(y); out = []
    for _ in range(B):
        b = rng.integers(0, n, n)
        if len(np.unique(y[b])) > 1:
            out.append(f(y[b], p[b]))
    o = np.vstack(out)
    return obs, np.percentile(o, 2.5, axis=0), np.percentile(o, 97.5, axis=0)


def _nb_band(y, p, ths, B=500, seed=7):
    """Net benefit with a percentile bootstrap band."""
    y, p = np.asarray(y), np.asarray(p)
    obs = np.array([net_benefit(y, p, t) for t in ths])
    rng = np.random.default_rng(seed); n = len(y); out = []
    for _ in range(B):
        b = rng.integers(0, n, n)
        if y[b].sum() > 0:
            out.append([net_benefit(y[b], p[b], t) for t in ths])
    o = np.vstack(out)
    return obs, np.percentile(o, 2.5, axis=0), np.percentile(o, 97.5, axis=0)


def figure5(N, PI1):
    """Flexible (loess) calibration with bootstrap bands for both outcomes, the distribution of
    predicted probabilities, and the decision curve with bootstrap uncertainty."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from figure_style import apply_figure_style; apply_figure_style()
    except Exception:
        pass
    col = {"Male": "#1f4e79", "Female": "#c0504d"}
    fig = plt.figure(figsize=(7.2, 8.2))
    outer = fig.add_gridspec(2, 1, hspace=0.34, height_ratios=[1.0, 0.78])
    top = outer[0].subgridspec(2, 2, hspace=0.06, wspace=0.28, height_ratios=[1.0, 0.26])
    axA, axB = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    hA, hB = fig.add_subplot(top[1, 0]), fig.add_subplot(top[1, 1])
    axC = fig.add_subplot(outer[1])
    for title, sel, ax, hax, L in [("Diagnosed diabetes", nhanes_diagnosed, axA, hA, "A"),
                                   ("Undiagnosed diabetes", nhanes_undiagnosed, axB, hB, "B")]:
        hi = 0.0
        for sex in ("Male", "Female"):
            dd = sel(N, sex)
            p = offset_correct(dd.p_lgbm.values, PI1[sex], dd.y.mean())
            g = np.linspace(np.percentile(p, 1), np.percentile(p, 97), 60)
            o, lo, up = _lowess_band(dd.y.values, p, g)
            ax.plot(g, o, color=col[sex], lw=1.7, label=sex.lower(), zorder=3)
            ax.fill_between(g, lo, up, color=col[sex], alpha=0.20, lw=0, zorder=2)
            hax.hist(np.clip(p, 0, g.max()), bins=40, range=(0, g.max()),
                     color=col[sex], alpha=0.55, lw=0)
            hi = max(hi, g.max())
        ax.plot([0, hi], [0, hi], "--", color="0.55", lw=1, zorder=1)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel("Observed proportion")
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        ax.tick_params(labelbottom=False)
        hax.set_xlim(0, hi); hax.set_yticks([])
        hax.set_xlabel("Predicted probability")
        hax.set_ylabel("Count", labelpad=2, fontsize=8)
        for sp in ("top", "right", "left"):
            hax.spines[sp].set_visible(False)
        ax.text(-0.20, 1.04, L, transform=ax.transAxes, fontsize=11, fontweight="bold")
    ths = np.linspace(0.01, 0.25, 60)
    for sex in ("Male", "Female"):
        dd = nhanes_undiagnosed(N, sex)
        p = offset_correct(dd.p_lgbm.values, PI1[sex], dd.y.mean())
        o, lo, up = _nb_band(dd.y.values, p, ths)
        axC.plot(ths, o, color=col[sex], lw=1.8, label=f"model, {sex.lower()}")
        axC.fill_between(ths, lo, up, color=col[sex], alpha=0.15, lw=0)
        axC.plot(ths, [net_benefit(dd.y, np.ones(len(dd)), t) for t in ths],
                 color=col[sex], lw=1.0, ls=":", label=f"test everyone, {sex.lower()}")
    axC.axhline(0, color="0.4", lw=1.0)
    axC.set_xlabel("Threshold probability"); axC.set_ylabel("Net benefit")
    axC.set_title("Decision curve, undiagnosed diabetes", loc="left", fontsize=10)
    axC.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper right")
    axC.set_xlim(0.01, 0.25)
    for sp in ("top", "right"):
        axC.spines[sp].set_visible(False)
    axC.text(-0.085, 1.04, "C", transform=axC.transAxes, fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(OUT, "Figure5.png"), dpi=400, bbox_inches="tight")
    print("wrote Figure5.png")
    return fig


def figure6():
    """Single-predictor discrimination, diagnosed versus undiagnosed diabetes, as points with
    95% bootstrap intervals. This is the contrast on which the leakage argument rests."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from figure_style import apply_figure_style; apply_figure_style()
    except Exception:
        pass
    L = pd.read_csv(os.path.join(OUT, "leakage_per_predictor.csv")).sort_values("gap")
    sig = ((L.gap_lo > 0) | (L.gap_hi < 0)).values
    SHORT = {"Red blood cell (erythrocyte) distribution width": "Red cell distribution width",
             "Red blood cell (erythrocyte) count": "Red blood cell count",
             "White blood cell (leukocyte) count": "White blood cell count",
             "Systolic blood pressure, automated reading": "Systolic blood pressure",
             "Diastolic blood pressure, automated reading": "Diastolic blood pressure",
             "Pulse rate, automated reading": "Pulse rate",
             "Body mass index (BMI)": "Body mass index",
             "Neutrophill count": "Neutrophil count",
             "Haemoglobin concentration": "Haemoglobin",
             "Haematocrit percentage": "Haematocrit",
             "Gamma glutamyltransferase": "Gamma-glutamyltransferase"}
    lab = [SHORT.get(p, p) for p in L.predictor]
    y = np.arange(len(L))
    fig, ax = plt.subplots(figsize=(7.0, 8.6))
    ax.axvline(0, color="0.45", lw=1.0, zorder=0)
    for i, s in enumerate(sig):
        c = "#1f4e79" if s else "0.62"
        ax.plot([L.gap_lo.iloc[i], L.gap_hi.iloc[i]], [y[i], y[i]], color=c, lw=1.5,
                solid_capstyle="butt")
        ax.plot(L.gap.iloc[i], y[i], "o", color=c, ms=4.5)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8)
    ax.set_ylim(-1, len(L))
    ax.set_xlabel("Difference in single-predictor AUROC\n(diagnosed minus undiagnosed diabetes)")
    ax.set_title("Each predictor evaluated alone, in the same NHANES fasting subsample",
                 loc="left")
    ax.text(0.015, 0.975, "filled markers: interval excludes zero", transform=ax.transAxes,
            fontsize=7.5, color="0.35", va="top")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(OUT, "Figure6.png"), dpi=400, bbox_inches="tight")
    print("wrote Figure6.png")
    return fig


if __name__ == "__main__":
    import sys
    if "--figures" in sys.argv:
        main(make_figure=True); figure6()
    else:
        main()
