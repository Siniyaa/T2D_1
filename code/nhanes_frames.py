"""Frame-aligned and LDL-removed sensitivity analyses for the NHANES evaluation.

The diagnosed-diabetes outcome is evaluated in the whole examination sample, where direct LDL
cholesterol is available only for the fasting subsample (2,067 of 4,757 participants aged 40-69,
43.5%); the undiagnosed outcome is evaluated in the fasting subsample, where it is present. LDL
is among the larger contributors to the male model, so part of the diagnosed-undiagnosed
difference could reflect the change of frame rather than the outcome. Two checks are run here:

  1. the diagnosed outcome re-evaluated inside the fasting subsample, so both outcomes share one
     frame and one weight (WTSAFPRP);
  2. both outcomes with LDL removed from the predictor set and the models refitted in UK Biobank.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, numpy as np, pandas as pd
import common
from common import (load_prevalent, load_nhanes, nhanes_diagnosed, nhanes_undiagnosed,
                    harmonised_X, SAFE, NCOLS, SEX, OUT)
from nhanes_weighted import design_boot, check_frame_complete

LDL = "LDL_direct_30780"


def fit_drop(df_prev, sex, drop=()):
    """Refit the harmonised LightGBM in UK Biobank with `drop` columns removed."""
    import lightgbm as lgb
    sub = df_prev[df_prev[SEX] == sex]
    X = harmonised_X(sub).drop(columns=list(drop))
    y = sub["HasT2D"].astype(int).values
    return lgb.LGBMClassifier(learning_rate=0.09, max_depth=-1,
                              random_state=common.SEED, verbose=-1).fit(X, y)


def score(N, prev, drop=()):
    """Attach a prediction column for a model fitted without `drop`."""
    X = N[NCOLS].apply(pd.to_numeric, errors="coerce"); X.columns = SAFE
    X = X.drop(columns=list(drop))
    col = "p_nold" if drop else "p_full"
    for sex in ("Male", "Female"):
        m = N.sexlab == sex
        N.loc[m, col] = fit_drop(prev, sex, drop).predict_proba(X[m])[:, 1]
    return col


def fasting(d):
    """Restrict to the fasting subsample, the frame in which the undiagnosed outcome is defined."""
    return d[d.LBDGLUSI.notna()]



def paired_frame_diff(N, sex, pcol, frame, B=1000, seed=7):
    """Diagnosed minus undiagnosed AUROC inside the SINGLE fasting-subsample frame.

    The two outcomes are defined on overlapping participants of one subsample, so each bootstrap
    replicate resamples PSUs once and both AUROCs are recomputed on that same replicate; the
    difference is therefore paired on the design.
    """
    from nhanes_weighted import weighted_auc
    rng = np.random.default_rng(seed)
    dg = fasting(nhanes_diagnosed(N, sex)); du = nhanes_undiagnosed(N, sex)
    def prep(d):
        d = d.dropna(subset=["WTSAFPRP"]); d = d[d.WTSAFPRP > 0]
        return (d.y.values, d[pcol].values, d.WTSAFPRP.values.astype(float),
                d.SDMVSTRA.values, d.SDMVPSU.values)
    G, U = prep(dg), prep(du)
    idx = [{(s, u): np.where((d[3] == s) & (d[4] == u))[0]
            for s, us in frame.items() for u in us} for d in (G, U)]
    obs = weighted_auc(G[0], G[1], G[2]) - weighted_auc(U[0], U[1], U[2])
    out = []
    for _ in range(B):
        sel = []
        for s, us in frame.items():
            n_h = len(us)
            if n_h < 2:
                continue
            m_h = n_h - 1
            u_sel, r = np.unique(rng.choice(us, m_h, replace=True), return_counts=True)
            sel.append((s, u_sel, r, n_h / m_h))
        a = []
        for d, ix in zip((G, U), idx):
            rows, mult = [], []
            for s, u_sel, r, f in sel:
                for u, r_hi in zip(u_sel, r):
                    i = ix.get((s, u))
                    if i is not None and len(i):
                        rows.append(i); mult.append(np.full(len(i), f * r_hi))
            if not rows:
                a = None; break
            b = np.concatenate(rows); m = np.concatenate(mult)
            if len(np.unique(d[0][b])) < 2:
                a = None; break
            a.append(weighted_auc(d[0][b], d[1][b], d[2][b] * m))
        if a:
            out.append(a[0] - a[1])
    return obs, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    prev = load_prevalent()
    N = load_nhanes()
    c_full = score(N, prev)
    c_nold = score(N, prev, drop=[LDL])
    frame = check_frame_complete(N)
    rows = []
    for sex in ("Male", "Female"):
        dg = nhanes_diagnosed(N, sex)
        cases = [
            ("Diagnosed diabetes", "Examination sample", dg, "WTMECPRP"),
            ("Diagnosed diabetes", "Fasting subsample", fasting(dg), "WTSAFPRP"),
            ("Undiagnosed diabetes", "Fasting subsample", nhanes_undiagnosed(N, sex), "WTSAFPRP"),
        ]
        for outcome, fr, dd, wc in cases:
            for spec, pc in [("All 36 predictors", c_full), ("LDL removed (35)", c_nold)]:
                a, lo, hi, n, k = design_boot(dd, pc, wc, frame)
                rows.append({"Outcome": outcome, "Evaluation frame": fr, "Sex": sex,
                             "Predictor set": spec, "n (cases)": f"{n:,} ({k})",
                             "Weighted AUROC (95% CI)": f"{a:.3f} ({lo:.3f}-{hi:.3f})",
                             "_auc": a})
    t = pd.DataFrame(rows)
    # LDL availability by frame, for the table note
    for sex in ("Male", "Female"):
        dg = nhanes_diagnosed(N, sex)
        print(f"{sex}: LDL present in {dg[LDL.replace('LDL_direct_30780','LBDLDLSI')].notna().mean()*100:4.1f}% "
              f"of the examination sample (n={len(dg):,}), "
              f"{fasting(dg)['LBDLDLSI'].notna().mean()*100:4.1f}% of the fasting subsample "
              f"(n={len(fasting(dg)):,})")
    dif = []
    for sex in ("Male", "Female"):
        for spec, pc in [("All 36 predictors", c_full), ("LDL removed (35)", c_nold)]:
            o, lo, hi = paired_frame_diff(N, sex, pc, frame)
            dif.append({"Sex": sex, "Predictor set": spec,
                        "Diagnosed minus undiagnosed, same frame (95% CI)":
                        f"{o:+.3f} ({lo:+.3f} to {hi:+.3f})"})
    dd_ = pd.DataFrame(dif)
    dd_.to_csv(os.path.join(OUT, "NHANES_frame_paired_difference.csv"), index=False)
    print(dd_.to_string(index=False))
    p = os.path.join(OUT, "NHANES_frame_sensitivity.csv")
    t.drop(columns="_auc").to_csv(p, index=False)
    print(t.drop(columns="_auc").to_string(index=False)); print("wrote", p)


if __name__ == "__main__":
    main()
