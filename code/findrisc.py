"""FINDRISC head-to-head on the incident analysis sample.

Two of the eight FINDRISC components (antihypertensive medication, history of high blood
glucose) are absent from this UK Biobank extract, so a six-component score with a maximum of
19 rather than 26 points is computed. Activity and fruit/vegetable questions also require proxy mappings.
This is an exploratory modified score, not a validated implementation of FINDRISC.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import (load_prevalent, load_incident_cases, family_history, boot_ci,
                    OUT, SEX, AGE_U, HBA, SEED, DATA)

F = {"bmi": "Body mass index (BMI) [21001]", "waist": "Waist circumference [48]",
     "pa": "Number of days/week of moderate physical activity 10+ minutes [884]",
     "veg": "Cooked vegetable intake [1289]", "salad": "Salad / raw vegetable intake [1299]",
     "fruit": "Fresh fruit intake [1309]"}
SIMPLE = [AGE_U, "Body mass index (BMI) [21001]", "Waist circumference [48]",
          "Systolic blood pressure, automated reading [4080]"]


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


def main():
    prev = load_prevalent()
    future = pd.read_csv(os.path.join(DATA, "medical_history_fut.tsv"), sep="\t", usecols=["EID"])
    assert prev.EID.is_unique and future.EID.is_unique
    ctrl = prev[~prev.HasT2D.astype(bool) & ~prev.EID.isin(future.EID)]
    ctrl = ctrl[~(pd.to_numeric(ctrl[HBA], errors="coerce") >= 48)]
    inc = pd.concat([load_incident_cases().assign(y=1), ctrl.assign(y=0)], ignore_index=True)
    assert inc.EID.is_unique, "Cases and controls must be disjoint"
    inc["FINDRISC"] = findrisc(inc)
    rows = []
    for sex in ("Male", "Female"):
        d = inc[inc[SEX] == sex]
        ok = d.FINDRISC.notna()
        a = roc_auc_score(d.y[ok], d.FINDRISC[ok])
        lo, hi = boot_ci(d.y[ok].values, d.FINDRISC[ok].values, 500)
        X = d[SIMPLE].apply(pd.to_numeric, errors="coerce")
        X["family_history"] = family_history(d).astype(int)
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=5000, random_state=SEED))])
        pv = cross_val_predict(pipe, X, d.y, cv=5, method="predict_proba")[:, 1]
        rows.append({"Sex": sex, "n (cases)": f"{len(d):,} ({int(d.y.sum()):,})",
                     "Modified FINDRISC, AUROC (95% CI)": f"{a:.3f} ({lo:.3f}-{hi:.3f})",
                     "Five-variable comparator, AUROC": f"{roc_auc_score(d.y, pv):.3f}",
                     "FINDRISC complete, %": f"{ok.mean()*100:.1f}"})
    p = os.path.join(OUT, "FINDRISC_comparison_revised.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print("wrote", p)


if __name__ == "__main__":
    main()

