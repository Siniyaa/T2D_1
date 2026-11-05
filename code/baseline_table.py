"""Table 1: baseline characteristics of participants."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, pandas as pd, os
from common import (load_prevalent, family_history, OUT, SEX, AGE_U, HBA)

VARS = [(AGE_U, "Age, years"), ("Body mass index (BMI) [21001]", "BMI, kg/m2"),
        ("Waist circumference [48]", "Waist circumference, cm"),
        ("Systolic blood pressure, automated reading [4080]", "Systolic BP, mmHg"),
        ("Diastolic blood pressure, automated reading [4079]", "Diastolic BP, mmHg"),
        ("Triglycerides [30870]", "Triglycerides, mmol/L"),
        ("HDL cholesterol [30760]", "HDL cholesterol, mmol/L"),
        ("LDL direct [30780]", "LDL cholesterol, mmol/L"),
        ("Cholesterol [30690]", "Total cholesterol, mmol/L"),
        ("Gamma glutamyltransferase [30730]", "GGT, U/L"),
        ("Alanine aminotransferase [30620]", "ALT, U/L"),
        ("SHBG [30830]", "SHBG, nmol/L"), ("Urate [30880]", "Urate, umol/L"),
        ("C-reactive protein [30710]", "C-reactive protein, mg/L"),
        (HBA, "HbA1c, mmol/mol")]


def describe(df):
    out = {"n": f"{len(df):,}",
           "Male, n (%)": f"{(df[SEX]=='Male').sum():,} ({(df[SEX]=='Male').mean()*100:.1f})",
           "Family history of diabetes, n (%)":
               f"{family_history(df).sum():,} ({family_history(df).mean()*100:.1f})"}
    for c, lab in VARS:
        v = pd.to_numeric(df[c], errors="coerce")
        out[lab] = f"{v.mean():.1f} ({v.std():.1f})"
    return out


def main():
    prev = load_prevalent()
    groups = {"Cases recorded by 2010": prev[prev.HasT2D],
              "Controls": prev[~prev.HasT2D]}
    desc = {g: describe(d) for g, d in groups.items()}
    miss = {lab: f"{np.mean([pd.to_numeric(d[c], errors='coerce').isna().mean() for d in groups.values()])*100:.1f}"
            for c, lab in VARS}
    rows = []
    for lab in list(desc["Controls"]):
        r = {"Characteristic": lab}
        r.update({g: desc[g][lab] for g in groups})
        r["Missing, %"] = miss.get(lab, "0.0")
        rows.append(r)
    out = os.path.join(OUT, "Table1_baseline_characteristics.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
