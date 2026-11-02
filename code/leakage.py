"""Supplementary Table S13: per-predictor discrimination for diagnosed vs undiagnosed disease.

The leakage gap is AUROC for diagnosed diabetes minus AUROC for undiagnosed diabetes, with
each predictor evaluated alone. A positive gap identifies a predictor that distinguishes
treated, diagnosed patients rather than undetected disease.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import (load_prevalent, load_nhanes, score_nhanes, nhanes_diagnosed,
                    nhanes_undiagnosed, MAP, OUT)


def leakage_table(N):
    rows = []
    for ukb_col, nh_col in MAP.items():
        r = {"predictor": ukb_col.split(" [")[0]}
        for lbl, fn in [("diagnosed", nhanes_diagnosed), ("undiagnosed", nhanes_undiagnosed)]:
            aucs = []
            for sex in ("Male", "Female"):
                dd = fn(N, sex)
                v = pd.to_numeric(dd[nh_col], errors="coerce")
                ok = v.notna()
                if ok.sum() > 50 and dd.y[ok].nunique() > 1:
                    a = roc_auc_score(dd.y[ok], v[ok])
                    aucs.append(max(a, 1 - a))          # reflect inverse associations
            r[lbl] = float(np.mean(aucs)) if aucs else np.nan
        r["gap"] = r["diagnosed"] - r["undiagnosed"]
        rows.append(r)
    return pd.DataFrame(rows).dropna(subset=["gap"])


def main():
    N = score_nhanes(load_nhanes(), load_prevalent())
    L = leakage_table(N).sort_values("gap", ascending=False).round(3)
    p = os.path.join(OUT, "leakage_per_predictor.csv")
    L.to_csv(p, index=False)
    print(L.head(6).to_string(index=False)); print("wrote", p)


if __name__ == "__main__":
    main()
