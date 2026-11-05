"""Incident T2D analysis on the reproducible comparator population.

The original pipeline removed 519 controls recorded as diagnosed during follow-up using
participant identifiers that are not present in this extract, so its population (25,685)
cannot be rebuilt. This script defines the population that CAN be rebuilt from the deposited
data: incident cases, and case-finding controls after removing baseline HbA1c >= 48 mmol/mol
and controls matching an incident case on the covariate signature (Supplementary Tables S18
and S19). Every incident estimate reported in the manuscript comes from this script.

Eligibility note: the HbA1c rule excludes participants with an OBSERVED baseline value
>= 48 mmol/mol. Participants whose baseline HbA1c is missing are retained, which is why
Table 1 shows missing HbA1c among incident cases.

Nominal inputs are given pandas `category` dtype with an explicit "NA" level and declared to
LightGBM as categorical features (common.model_matrix); applying pd.to_numeric to them would map
every level to missing and silently drop the predictor.

Two summaries of repeated cross-validation are reported and are not interchangeable:
  mean_fold  - mean AUROC across the 25 held-out folds, with between-fold SD
  pooled     - one score per participant (mean of the 5 held-out predictions, one per
               repeat), with AUROC computed once and a percentile bootstrap CI
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, re, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import (load_prevalent, load_incident_cases, family_history, model_matrix, OUT,
                    SEX, AGE_U, HBA, GLU, SEED, DROP_PRED)

SIGKEY = [SEX, AGE_U, "Standing height [50]", "Waist circumference [48]",
          "Body mass index (BMI) [21001]"]
SIMPLE = [AGE_U, "Body mass index (BMI) [21001]", "Waist circumference [48]",
          "Systolic blood pressure, automated reading [4080]"]
safe = lambda cs: [re.sub(r"[^0-9A-Za-z]+", "_", c).strip("_") for c in cs]


def population():
    prev = load_prevalent()
    cases = load_incident_cases()
    ctrl = prev[~prev.HasT2D]
    ctrl = ctrl[~(pd.to_numeric(ctrl[HBA], errors="coerce") >= 48)]
    s = lambda d: d[SIGKEY].astype(str).agg("|".join, axis=1)
    ctrl = ctrl[~s(ctrl).isin(set(s(cases)))]
    pred = [c for c in prev.columns if c in set(cases.columns) and c not in DROP_PRED]
    assert len(pred) == 175 and AGE_U in pred, len(pred)
    df = pd.concat([cases.assign(y=1), ctrl.assign(y=0)], ignore_index=True)
    return df, pred


def cv_scores(X, y, model="lgbm", cat=None):
    """Return (mean_fold, sd_fold, pooled_score_per_participant)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    pooled, folds = np.zeros(len(y)), []
    for tr, te in cv.split(X, y):
        if model == "lgbm":
            m = lgb.LGBMClassifier(learning_rate=0.09, random_state=SEED, verbose=-1)
            m.fit(X.iloc[tr], y[tr], categorical_feature=(cat or "auto"))
            p = m.predict_proba(X.iloc[te])[:, 1]
        else:
            m = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                          ("lr", LogisticRegression(max_iter=5000, random_state=SEED))])
            m.fit(X.iloc[tr], y[tr]); p = m.predict_proba(X.iloc[te])[:, 1]
        pooled[te] += p
        folds.append(roc_auc_score(y[te], p))
    return float(np.mean(folds)), float(np.std(folds)), pooled / 5.0


def bci(y, p, B=1000, seed=7):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p); o = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1: o.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main():
    df, pred = population()
    days = pd.to_numeric(df.get("Days until E11 diagnosis"), errors="coerce")
    rows, glyc = [], []
    for sex in ("Male", "Female"):
        d = df[df[SEX] == sex].reset_index(drop=True)
        y = d.y.values
        dy = pd.to_numeric(d.get("Days until E11 diagnosis"), errors="coerce").values / 365.25
        X, cat_cols = model_matrix(d, pred)
        mf, sd, pool = cv_scores(X, y, cat=cat_cols)
        lo, hi = bci(y, pool)
        rows.append({"Sex": sex, "Analysis": "LightGBM, 175 predictors", "n": len(d),
                     "cases": int(y.sum()), "mean_fold_AUROC": round(mf, 3),
                     "fold_SD": round(sd, 3), "pooled_AUROC": round(roc_auc_score(y, pool), 3),
                     "pooled_95CI": f"{lo:.3f}-{hi:.3f}",
                     "AUPRC_pooled": round(average_precision_score(y, pool), 3)})
        # horizon-specific: cases in each window vs all controls, pooled scores
        for lab, lo_y, hi_y in [("<5 years", 0, 5), ("5-10 years", 5, 10), (">=10 years", 10, 99)]:
            m = (y == 0) | ((dy >= lo_y) & (dy < hi_y))
            a = roc_auc_score(y[m], pool[m]); l2, h2 = bci(y[m], pool[m])
            rows.append({"Sex": sex, "Analysis": f"LightGBM, diagnosis {lab}", "n": int(m.sum()),
                         "cases": int(y[m].sum()), "mean_fold_AUROC": "", "fold_SD": "",
                         "pooled_AUROC": round(a, 3), "pooled_95CI": f"{l2:.3f}-{h2:.3f}",
                         "AUPRC_pooled": round(average_precision_score(y[m], pool[m]), 3)})
        # five-variable comparator
        Xs = d[SIMPLE].apply(pd.to_numeric, errors="coerce")
        Xs["family_history"] = family_history(d).astype(int)
        mf2, sd2, pool2 = cv_scores(Xs, y, model="lr")
        lo2, hi2 = bci(y, pool2)
        rows.append({"Sex": sex, "Analysis": "Five-variable comparator", "n": len(d),
                     "cases": int(y.sum()), "mean_fold_AUROC": round(mf2, 3),
                     "fold_SD": round(sd2, 3), "pooled_AUROC": round(roc_auc_score(y, pool2), 3),
                     "pooled_95CI": f"{lo2:.3f}-{hi2:.3f}",
                     "AUPRC_pooled": round(average_precision_score(y, pool2), 3)})
        # optional glycaemic comparison, same participants and same fold scheme
        Xg = d[[HBA, GLU]].apply(pd.to_numeric, errors="coerce"); Xg.columns = ["hba1c", "glucose"]
        Xg.index = X.index
        mg, sg, pg = cv_scores(Xg, y, model="lr")
        Xe = pd.concat([X, Xg], axis=1)
        me, se_, pe = cv_scores(Xe, y, cat=cat_cols)
        rng = np.random.default_rng(7); dgl, dex = [], []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) > 1:
                base = roc_auc_score(y[i], pool[i])
                dgl.append(roc_auc_score(y[i], pg[i]) - base)
                dex.append(roc_auc_score(y[i], pe[i]) - base)
        for lab, p_, m_, s_, dd in [("Glycaemic only (HbA1c, glucose)", pg, mg, sg, dgl),
                                    ("Primary + HbA1c + glucose", pe, me, se_, dex)]:
            a = roc_auc_score(y, p_); l3, h3 = bci(y, p_)
            glyc.append({"Sex": sex, "Model": lab, "n": len(d), "cases": int(y.sum()),
                         "mean_fold_AUROC": round(m_, 3), "fold_SD": round(s_, 3),
                         "pooled_AUROC": round(a, 3), "pooled_95CI": f"{l3:.3f}-{h3:.3f}",
                         "Difference vs primary": round(a - roc_auc_score(y, pool), 3),
                         "95% CI (paired)": f"{np.percentile(dd,2.5):.3f} to {np.percentile(dd,97.5):.3f}"})
        glyc.append({"Sex": sex, "Model": "Primary, 175 non-glycaemic predictors", "n": len(d),
                     "cases": int(y.sum()), "mean_fold_AUROC": round(mf, 3), "fold_SD": round(sd, 3),
                     "pooled_AUROC": round(roc_auc_score(y, pool), 3),
                     "pooled_95CI": f"{lo:.3f}-{hi:.3f}", "Difference vs primary": 0.0,
                     "95% CI (paired)": "reference"})
    for frame, name in [(rows, "Incident_primary.csv"), (glyc, "Glycaemic_comparison.csv")]:
        pd.DataFrame(frame).to_csv(os.path.join(OUT, name), index=False)
        print(pd.DataFrame(frame).to_string(index=False)); print("wrote", name)


if __name__ == "__main__":
    main()
