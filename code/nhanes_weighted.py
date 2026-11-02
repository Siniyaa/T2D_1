"""Survey-weighted external validation with design-based confidence intervals.

Weighted AUROC uses the NHANES examination weight (diagnosed diabetes) or fasting-subsample
weight (undiagnosed diabetes). Confidence intervals come from 1,000 bootstrap replicates that
resample primary sampling units within design strata, respecting the complex survey design.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, numpy as np, pandas as pd
from common import (load_prevalent, load_nhanes, score_nhanes, nhanes_diagnosed,
                    nhanes_undiagnosed, OUT)


def weighted_auc(y, p, w):
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w, float)
    pos, neg = w[y == 1], w[y == 0]
    pp, pn = p[y == 1], p[y == 0]
    order = np.argsort(pn); pn_s, neg_s = pn[order], neg[order]
    cw = np.concatenate([[0], np.cumsum(neg_s)])
    num = 0.0
    for pi, wi in zip(pp, pos):
        lt = np.searchsorted(pn_s, pi, "left")
        eq = np.searchsorted(pn_s, pi, "right") - lt
        num += wi * (cw[lt] + 0.5 * (cw[lt + eq] - cw[lt]))
    return num / (pos.sum() * neg.sum())


def design_boot(dd, pcol, wcol, B=1000, seed=7):
    rng = np.random.default_rng(seed)
    d = dd.dropna(subset=[wcol]); d = d[d[wcol] > 0]
    y, p, w = d.y.values, d[pcol].values, d[wcol].values.astype(float)
    st, psu = d.SDMVSTRA.values, d.SDMVPSU.values
    keys = pd.DataFrame({"s": st, "u": psu}).drop_duplicates()
    by_s = {s: g.u.values for s, g in keys.groupby("s")}
    idx = {(s, u): np.where((st == s) & (psu == u))[0] for s, u in zip(keys.s, keys.u)}
    out = []
    for _ in range(B):
        sel = [idx[(s, u)] for s, us in by_s.items() for u in rng.choice(us, len(us), replace=True)]
        b = np.concatenate(sel)
        if len(np.unique(y[b])) > 1:
            out.append(weighted_auc(y[b], p[b], w[b]))
    return (weighted_auc(y, p, w), float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)), len(d), int(d.y.sum()))


def main():
    prev = load_prevalent()
    N = score_nhanes(load_nhanes(), prev)
    rows = []
    for sex in ("Male", "Female"):
        for lbl, dd, wc in [("Diagnosed diabetes, all", nhanes_diagnosed(N, sex), "WTMECPRP"),
                            ("Diagnosed diabetes, non-Hispanic White",
                             nhanes_diagnosed(N, sex, nhw=True), "WTMECPRP"),
                            ("Undiagnosed diabetes (fasting subsample)",
                             nhanes_undiagnosed(N, sex), "WTSAFPRP")]:
            for mdl, pc in [("LightGBM", "p_lgbm"), ("Logistic regression", "p_lr")]:
                a, lo, hi, n, c = design_boot(dd, pc, wc)
                rows.append({"Outcome / population": lbl, "Sex": sex, "Model": mdl,
                             "n (cases)": f"{n:,} ({c})",
                             "Weighted AUROC (95% CI)": f"{a:.3f} ({lo:.3f}-{hi:.3f})"})
    p = os.path.join(OUT, "Table4_weighted.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print("wrote", p)


if __name__ == "__main__":
    main()
