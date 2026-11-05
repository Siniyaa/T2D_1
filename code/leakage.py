"""Supplementary Table S13: difference in single-predictor AUROC, diagnosed vs undiagnosed.

The reported quantity is a DESCRIPTIVE CONTRAST in discrimination between two outcome
definitions evaluated in one survey. It is not a decomposition of the multivariable model and
not an estimate of a treatment effect.

BOTH outcomes are evaluated inside the NHANES fasting subsample, the frame in which undiagnosed
diabetes is defined, so the contrast is not confounded by the change of frame or by the
structural absence of LDL cholesterol outside that subsample. The two outcome samples then share
their non-diabetic members, so the two AUROCs are dependent. Each bootstrap replicate therefore resamples the
UNIQUE participants of each sex ONCE and derives both eligible outcome samples from that single
resampled frame, preserving the dependence. Sex-specific AUROCs are averaged with equal weight,
and that definition is applied identically in every replicate.

Two forms are reported. The signed AUROC keeps the direction of association as fitted. The
reflected AUROC maps values below 0.5 to 1 - AUROC so that every value measures strength of
association irrespective of direction; reflection discards direction and, for predictors near
chance, biases the reflected value upward, so the signed columns should be read alongside it.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import (load_prevalent, load_nhanes, score_nhanes, nhanes_diagnosed,
                    nhanes_undiagnosed, MAP, OUT)


def diagnosed_fasting(frame, sex):
    """Diagnosed diabetes restricted to the fasting subsample.

    nhanes_undiagnosed defines its sample by LBDGLUSI being observed, which is exactly the
    fasting subsample; applying the same restriction to the diagnosed outcome puts both outcomes
    in one frame with one set of shared controls.
    """
    d = nhanes_diagnosed(frame, sex)
    return d[d.LBDGLUSI.notna()]


def _auc(frame_fn, Nsex, nh_col, reflect):
    dd = frame_fn(Nsex)
    v = pd.to_numeric(dd[nh_col], errors="coerce")
    ok = v.notna().values
    y, x = dd.y.values[ok], v.values[ok]
    if len(y) <= 50 or len(np.unique(y)) < 2:
        return np.nan, 0, 0
    a = roc_auc_score(y, x)
    return (max(a, 1 - a) if reflect else a), len(y), int(y.sum())


def _pair(N, nh_col, reflect, sexes=("Male", "Female")):
    """Sex-averaged AUROC for both outcomes from ONE frame."""
    ad, au, nd, ed, nu, eu = [], [], 0, 0, 0, 0
    for sex in sexes:
        Ns = N[N.sexlab == sex]
        a1, n1, e1 = _auc(lambda f: diagnosed_fasting(f, sex), Ns, nh_col, reflect)
        a2, n2, e2 = _auc(lambda f: nhanes_undiagnosed(f, sex), Ns, nh_col, reflect)
        if np.isfinite(a1): ad.append(a1); nd += n1; ed += e1
        if np.isfinite(a2): au.append(a2); nu += n2; eu += e2
    if not ad or not au:
        return np.nan, np.nan, nd, ed, nu, eu
    return float(np.mean(ad)), float(np.mean(au)), nd, ed, nu, eu


def leakage_table(N, B=500, seed=11):
    rng = np.random.default_rng(seed)
    # one set of resampled frames, reused for every predictor, so the contrast across
    # predictors is computed on the same replicates
    idx = {s: np.where(N.sexlab.values == s)[0] for s in ("Male", "Female")}
    reps = []
    for _ in range(B):
        take = np.concatenate([rng.choice(v, len(v), replace=True) for v in idx.values()])
        reps.append(N.iloc[take])
    rows = []
    for ukb_col, nh_col in MAP.items():
        r = {"predictor": ukb_col.split(" [")[0]}
        for tag, reflect in (("", True), ("_signed", False)):
            d, u, nd, ed, nu, eu = _pair(N, nh_col, reflect)
            r[f"diagnosed{tag}"], r[f"undiagnosed{tag}"] = d, u
            r[f"gap{tag}"] = d - u
            if not tag:
                r.update({"n_diagnosed": nd, "cases_diagnosed": ed,
                          "n_undiagnosed": nu, "cases_undiagnosed": eu})
        if np.isfinite(r["gap"]):
            g = []
            for Nb in reps:
                d, u, *_ = _pair(Nb, nh_col, True)
                if np.isfinite(d) and np.isfinite(u): g.append(d - u)
            r["gap_lo"], r["gap_hi"] = ((float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5)))
                                        if g else (np.nan, np.nan))
        rows.append(r)
    return pd.DataFrame(rows).dropna(subset=["gap"])


def shared_controls(N):
    """Controls common to both outcome samples, per sex -- the basis of the paired bootstrap."""
    out = {}
    for sex in ("Male", "Female"):
        Ns = N[N.sexlab == sex]
        a = diagnosed_fasting(Ns, sex); b = nhanes_undiagnosed(Ns, sex)
        ca = set(a.loc[a.y == 0, "SEQN"]); cb = set(b.loc[b.y == 0, "SEQN"])
        out[sex] = (len(ca & cb), len(ca), len(cb), int((a.y == 1).sum()), int((b.y == 1).sum()),
                    len(a), len(b))
    return out


def main():
    N = score_nhanes(load_nhanes(), load_prevalent())
    for sex, (sh, ca, cb, ka, kb, na, nb) in shared_controls(N).items():
        print(f"{sex}: diagnosed-in-fasting n={na} ({ka} cases, {ca} controls); "
              f"undiagnosed n={nb} ({kb} cases, {cb} controls); shared controls={sh}")
    L = leakage_table(N).sort_values("gap", ascending=False).round(4)
    p = os.path.join(OUT, "leakage_per_predictor.csv")
    L.to_csv(p, index=False)
    print(L.head(6).to_string(index=False)); print("wrote", p)


if __name__ == "__main__":
    main()
