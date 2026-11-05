"""Survey-weighted external validation with design-based confidence intervals.

Weighted AUROC uses the NHANES examination weight (diagnosed diabetes) or fasting-subsample
weight (undiagnosed diabetes). Confidence intervals come from 1,000 Rao-Wu rescaled bootstrap
replicates. The stratum-by-PSU frame is built from the FULL released design, before the
analysis domain is selected, so that strata keep their design degrees of freedom; within each
stratum m_h = n_h - 1 PSUs are drawn with replacement and the weights of the selected units
are multiplied by (n_h / m_h) * r_hi, where r_hi is the selection count [Rao & Wu 1988].
Domain members are then taken from the resampled design.
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


def released_design_frame(nh_dir=None):
    """Stratum -> PSU list from the FULL released design, before any age restriction.

    common.load_nhanes() returns records already restricted to ages 40-69, so the frame is
    read here directly from P_DEMO (examination design) and P_TRIGLY (fasting subsample).
    """
    import common as _c
    d = pd.read_sas(os.path.join(nh_dir or _c.NH, "P_DEMO.xpt"), format="xport")
    k = d[["SDMVSTRA", "SDMVPSU"]].dropna().drop_duplicates()
    return {s: g.SDMVPSU.values for s, g in k.groupby("SDMVSTRA")}


def design_frame(N):
    """Stratum -> PSU list present in the supplied frame."""
    k = N[["SDMVSTRA", "SDMVPSU"]].dropna().drop_duplicates()
    return {s: g.SDMVPSU.values for s, g in k.groupby("SDMVSTRA")}


def check_frame_complete(N):
    """Assert that the age restriction lost no stratum-by-PSU cell, and report it."""
    full = released_design_frame()
    used = design_frame(N)
    f = {(s, u) for s, us in full.items() for u in us}
    v = {(s, u) for s, us in used.items() for u in us}
    print(f"design frame: {len(f)} stratum-by-PSU cells in the released design, "
          f"{len(v)} after the age restriction; identical = {f == v}")
    if f - v:
        print("  cells absent after restriction:", sorted(f - v))
    assert f == v, "age restriction dropped PSUs; rebuild the frame before domain selection"
    return used


def design_boot(dd, pcol, wcol, frame, B=1000, seed=7):
    """Rao-Wu rescaled bootstrap: m_h = n_h - 1 draws, multiplier (n_h/m_h)*r_hi."""
    rng = np.random.default_rng(seed)
    d = dd.dropna(subset=[wcol]); d = d[d[wcol] > 0]
    y, p, w = d.y.values, d[pcol].values, d[wcol].values.astype(float)
    st, psu = d.SDMVSTRA.values, d.SDMVPSU.values
    # rows of the domain belonging to each (stratum, PSU) of the full design
    idx = {(s, u): np.where((st == s) & (psu == u))[0] for s, us in frame.items() for u in us}
    out = []
    for _ in range(B):
        rows, mult = [], []
        for s, us in frame.items():
            n_h = len(us)
            if n_h < 2:
                continue
            m_h = n_h - 1
            draw = rng.choice(us, m_h, replace=True)
            u_sel, r = np.unique(draw, return_counts=True)
            for u, r_hi in zip(u_sel, r):
                i = idx.get((s, u))
                if i is not None and len(i):
                    rows.append(i)
                    mult.append(np.full(len(i), (n_h / m_h) * r_hi))
        if not rows:
            continue
        b = np.concatenate(rows); m = np.concatenate(mult)
        if len(np.unique(y[b])) > 1:
            out.append(weighted_auc(y[b], p[b], w[b] * m))
    return (weighted_auc(y, p, w), float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)), len(d), int(d.y.sum()))


def main():
    prev = load_prevalent()
    N = score_nhanes(load_nhanes(), prev)
    frame = check_frame_complete(N)
    rows = []
    for sex in ("Male", "Female"):
        for lbl, dd, wc in [("Diagnosed diabetes, all", nhanes_diagnosed(N, sex), "WTMECPRP"),
                            ("Diagnosed diabetes, non-Hispanic White",
                             nhanes_diagnosed(N, sex, nhw=True), "WTMECPRP"),
                            ("Undiagnosed diabetes (fasting subsample)",
                             nhanes_undiagnosed(N, sex), "WTSAFPRP")]:
            for mdl, pc in [("LightGBM", "p_lgbm"), ("Logistic regression", "p_lr")]:
                a, lo, hi, n, c = design_boot(dd, pc, wc, frame)
                rows.append({"Outcome / population": lbl, "Sex": sex, "Model": mdl,
                             "n (cases)": f"{n:,} ({c})",
                             "Weighted AUROC (95% CI)": f"{a:.3f} ({lo:.3f}-{hi:.3f})"})
    p = os.path.join(OUT, "Table4_weighted.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print("wrote", p)


if __name__ == "__main__":
    main()
