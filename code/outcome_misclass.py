"""Supplementary Table S15: outcome misclassification sensitivity analysis.

Inpatient-only ascertainment cannot identify diabetes managed in primary care, so some
controls had undiagnosed disease at baseline. This does NOT reproduce the Eastwood et al.
algorithm (PLOS One 2016;11:e0162388), which requires self-reported diagnosis, age at
diagnosis, medication and nurse-interview fields plus primary-care linkage that are absent
from this extract. It tests the component that the available data can address, by
identifying controls meeting biochemical criteria for diabetes and re-running each stratum
with those controls excluded, and reclassified as cases, using identical folds.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, re, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from common import load_prevalent, strata, DROP_PRED, OUT, HBA, GLU, SEED


def prep(df):
    X = df.drop(columns=[c for c in DROP_PRED if c in df.columns]).copy()
    for c in X.columns:
        n = pd.to_numeric(X[c], errors="coerce")
        X[c] = n if n.notna().sum() > 0.5 * X[c].notna().sum() \
            else X[c].astype("category").cat.codes.replace(-1, np.nan)
    X.columns = [re.sub(r"[^0-9A-Za-z]+", "_", c).strip("_") for c in X.columns]
    return X


def biochemical_diabetes(df):
    """ADA-consistent: HbA1c >= 48 mmol/mol or random glucose >= 11.1 mmol/L."""
    hb = pd.to_numeric(df[HBA], errors="coerce")
    gl = pd.to_numeric(df[GLU], errors="coerce")
    return ((hb >= 48) | (gl >= 11.1)).fillna(False)


def main():
    prev = load_prevalent()
    ctl = ~prev.HasT2D
    bio = biochemical_diabetes(prev)
    print(f"controls: {ctl.sum():,}; meeting diabetes-range biochemistry: "
          f"{int((ctl & bio).sum()):,} ({bio[ctl].mean()*100:.2f}%)")
    rk = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    rows = []
    for sex, lab, sub in strata(prev):
        b = biochemical_diabetes(sub)
        scenarios = {"main": sub,
                     "exclude misclassified controls": sub[~(b & ~sub.HasT2D)],
                     "reclassify as cases": sub}
        for name, s2 in scenarios.items():
            y = s2.HasT2D.astype(int).values
            if name == "reclassify as cases":
                y = np.where(biochemical_diabetes(s2).values, 1, y)
            X = prep(s2)
            aucs = []
            for tr, te in rk.split(X, y):
                m = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-1,
                                       random_state=SEED, verbose=-1).fit(X.iloc[tr], y[tr])
                aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
            rows.append({"Sex": sex, "Age group": lab, "Definition": name, "n": len(s2),
                         "cases": int(y.sum()), "AUROC": round(float(np.mean(aucs)), 3),
                         "SD": round(float(np.std(aucs)), 3)})
            print(f"  {sex:6s} {lab:6s} {name:32s} AUROC {np.mean(aucs):.3f}")
    p = os.path.join(OUT, "outcome_misclassification.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    print("wrote", p)


if __name__ == "__main__":
    main()
