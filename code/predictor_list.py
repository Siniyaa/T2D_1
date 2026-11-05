"""Supplementary Table S11: predictor list with missingness."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, pandas as pd
from common import load_prevalent, DROP_PRED, OUT


def main():
    df = load_prevalent()
    cols = [c for c in df.columns if c not in DROP_PRED]
    numeric = [pd.to_numeric(df[c], errors="coerce").notna().sum() > 0.5 * df[c].notna().sum()
               for c in cols]
    out = pd.DataFrame({"Predictor (UK Biobank field)": cols,
                        "Type": ["numeric" if n else "categorical" for n in numeric],
                        "Missing, %": [f"{df[c].isna().mean()*100:.1f}" for c in cols]})
    p = os.path.join(OUT, "S11_predictors.csv")
    out.to_csv(p, index=False)
    print(f"wrote {p} ({len(out)} predictors)")


if __name__ == "__main__":
    main()
