"""Exploratory comparison with an ADAPTED FINDRISC score on the incident analysis sample.

This is NOT the published FINDRISC. Two of the eight components (antihypertensive medication,
history of high blood glucose) are absent from this UK Biobank extract, and the physical
activity and diet components are approximated by the closest available touchscreen items, so
a six-component score with a maximum of 19 rather than 26 points is computed. The fitted
comparators (LightGBM and the five-variable logistic model) both use the same
five-times-repeated five-fold partitions and the same averaging of five held-out predictions
per record; the adapted score is a fixed formula and is not fitted. Removing and
substituting components can move discrimination in either direction in a new population, so
the comparison is exploratory and is not evidence of superiority over FINDRISC as published.

Controls are the case-finding controls with baseline HbA1c < 48 mmol/mol, with participants
who later develop recorded T2D removed. The follow-up extract carries no participant
identifier, so later-diagnosed participants are identified by an exact covariate signature
(sex, single year of age, standing height, waist circumference, BMI); residual contamination
by participants who share a signature cannot be excluded.

The adapted score, the five-variable comparator and the full LightGBM model are evaluated on
EXACTLY the same participants, and differences in AUROC are reported with paired percentile
bootstrap confidence intervals that resample participants once per replicate.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import (load_prevalent, load_incident_cases, family_history, model_matrix,
                    OUT, SEX, AGE_U, HBA, SEED, DROP_PRED)

F = {"bmi": "Body mass index (BMI) [21001]", "waist": "Waist circumference [48]",
     "pa": "Number of days/week of moderate physical activity 10+ minutes [884]",
     "veg": "Cooked vegetable intake [1289]", "salad": "Salad / raw vegetable intake [1299]",
     "fruit": "Fresh fruit intake [1309]"}
SIMPLE = [AGE_U, "Body mass index (BMI) [21001]", "Waist circumference [48]",
          "Systolic blood pressure, automated reading [4080]"]
SIGKEY = [SEX, AGE_U, "Standing height [50]", "Waist circumference [48]",
          "Body mass index (BMI) [21001]"]


def findrisc(df):
    g = lambda c: pd.to_numeric(df[c], errors="coerce")
    age, bmi, waist = g(AGE_U), g(F["bmi"]), g(F["waist"])
    s = pd.Series(0.0, index=df.index)
    s += np.select([age < 45, age < 55, age < 65], [0, 2, 3], default=4)
    s += np.select([bmi < 25, bmi < 30], [0, 1], default=3)
    male = df[SEX].eq("Male")
    s += np.where(male, np.select([waist < 94, waist <= 102], [0, 3], default=4),
                        np.select([waist < 80, waist <= 88], [0, 3], default=4))
    s += np.where(g(F["pa"]).fillna(0) >= 5, 0, 2)
    fv = g(F["fruit"]).fillna(0) + g(F["veg"]).fillna(0) + g(F["salad"]).fillna(0)
    s += np.where(fv >= 3, 0, 1)
    s += np.where(family_history(df), 5, 0)
    s[age.isna() | bmi.isna() | waist.isna()] = np.nan
    return s


def signature(df):
    return df[SIGKEY].astype(str).agg("|".join, axis=1)


def paired_diff_ci(y, pa, pb, B=1000, seed=7):
    """Percentile bootstrap for AUROC(a) - AUROC(b) on the SAME participants."""
    rng = np.random.default_rng(seed)
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    out = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            out.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_ci_single(y, p, B=1000, seed=7):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    out = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B))
           if len(np.unique(y[i])) > 1]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    import lightgbm as lgb
    prev = load_prevalent()
    cases = load_incident_cases()
    ctrl = prev[~prev.HasT2D]
    ctrl = ctrl[~(pd.to_numeric(ctrl[HBA], errors="coerce") >= 48)]
    n_before = len(ctrl)
    ctrl = ctrl[~signature(ctrl).isin(set(signature(cases)))]
    print(f"control pool {n_before:,} -> {len(ctrl):,} after removing "
          f"{n_before - len(ctrl):,} later-diagnosed participants")

    inc = pd.concat([cases.assign(y=1), ctrl.assign(y=0)], ignore_index=True)
    inc["FINDRISC"] = findrisc(inc)
    pred = [c for c in prev.columns if c in set(cases.columns) and c not in DROP_PRED]
    assert len(pred) == 175, len(pred)

    rows, diffs = [], []
    for sex in ("Male", "Female"):
        d = inc[inc[SEX] == sex].reset_index(drop=True)
        y = d.y.values
        # full 175-predictor LightGBM, 5 x 5 repeated stratified CV, out-of-fold predictions
        Xf, cat_cols = model_matrix(d, pred)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
        pf = np.zeros(len(d))
        for tr, te in cv.split(Xf, y):
            m = lgb.LGBMClassifier(learning_rate=0.09, random_state=SEED, verbose=-1)
            m.fit(Xf.iloc[tr], y[tr], categorical_feature=cat_cols)
            pf[te] += m.predict_proba(Xf.iloc[te])[:, 1]
        pf /= 5.0   # mean of the 5 held-out predictions per participant (one per repeat);
        # AUROC is then computed ONCE on this pooled score, not averaged across folds
        # five-variable comparator
        Xs = d[SIMPLE].apply(pd.to_numeric, errors="coerce")
        Xs["family_history"] = family_history(d).astype(int)
        # same five-times-repeated five-fold scheme and the same averaging as the LightGBM model
        ps = np.zeros(len(d))
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                              random_state=SEED).split(Xs, y):
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                             ("lr", LogisticRegression(max_iter=5000, random_state=SEED))])
            pipe.fit(Xs.iloc[tr], y[tr])
            ps[te] += pipe.predict_proba(Xs.iloc[te])[:, 1]
        ps /= 5.0
        ok = d.FINDRISC.notna().values          # common evaluable sample
        fr = d.FINDRISC.values
        a_fr = roc_auc_score(y[ok], fr[ok])
        a_fu = roc_auc_score(y[ok], pf[ok])
        a_si = roc_auc_score(y[ok], ps[ok])
        lo_f, hi_f = boot_ci_single(y[ok], fr[ok])
        lo_u, hi_u = boot_ci_single(y[ok], pf[ok])
        lo_s, hi_s = boot_ci_single(y[ok], ps[ok])
        rows.append({"Sex": sex, "n (cases)": f"{int(ok.sum()):,} ({int(y[ok].sum()):,})",
                     "Adapted FINDRISC, AUROC (95% CI)": f"{a_fr:.3f} ({lo_f:.3f}-{hi_f:.3f})",
                     "LightGBM, AUROC (95% CI)": f"{a_fu:.3f} ({lo_u:.3f}-{hi_u:.3f})",
                     "Five-variable comparator, AUROC (95% CI)": f"{a_si:.3f} ({lo_s:.3f}-{hi_s:.3f})",
                     "FINDRISC computable, %": f"{ok.mean()*100:.1f}"})
        lo_d, hi_d = paired_diff_ci(y[ok], pf[ok], fr[ok])
        diffs.append({"Sex": sex, "Comparison": "LightGBM - adapted FINDRISC",
                      "Difference in AUROC": round(a_fu - a_fr, 3),
                      "95% CI (paired bootstrap)": f"{lo_d:.3f} to {hi_d:.3f}"})
        lo_d, hi_d = paired_diff_ci(y[ok], ps[ok], fr[ok])
        diffs.append({"Sex": sex, "Comparison": "Five-variable comparator - adapted FINDRISC",
                      "Difference in AUROC": round(a_si - a_fr, 3),
                      "95% CI (paired bootstrap)": f"{lo_d:.3f} to {hi_d:.3f}"})

    for frame, name in [(rows, "FINDRISC_comparison.csv"), (diffs, "FINDRISC_paired_differences.csv")]:
        pd.DataFrame(frame).to_csv(os.path.join(OUT, name), index=False)
        print(pd.DataFrame(frame).to_string(index=False)); print("wrote", os.path.join(OUT, name))


if __name__ == "__main__":
    main()
