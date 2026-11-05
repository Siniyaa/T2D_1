"""Supplementary Table S14 (step 1): assemble pmsampsize inputs for every model."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, pandas as pd, numpy as np
from common import load_prevalent, load_incident_cases, strata, OUT, SEX, HBA

# observed C-statistics from the main analysis (results/B_summary.csv, C_incident_folds.csv)
CSTAT = {("Male", "<50"): 0.905, ("Male", "50-54"): 0.900, ("Male", "55-59"): 0.892,
         ("Male", "60-64"): 0.873, ("Male", ">=65"): 0.857,
         ("Female", "<50"): 0.894, ("Female", "50-54"): 0.891, ("Female", "55-59"): 0.901,
         ("Female", "60-64"): 0.904, ("Female", ">=65"): 0.881}
INCIDENT = {"Male": (13805, 9293, 0.818), "Female": (11880, 6520, 0.848)}
HARMONISED = {"Male": 0.897, "Female": 0.905}


def main():
    prev = load_prevalent()
    rows = []
    for sex, lab, sub in strata(prev):
        rows.append({"stratum": f"{sex} {lab}", "n": len(sub),
                     "events": int(sub.HasT2D.sum()),
                     "prev": round(sub.HasT2D.mean(), 4),
                     "cstat": CSTAT[(sex, lab)], "params": 175})
    for sex, (n, e, c) in INCIDENT.items():
        rows.append({"stratum": f"Incident, {sex.lower()}", "n": n, "events": e,
                     "prev": round(e / n, 4), "cstat": c, "params": 175})
    for sex, c in HARMONISED.items():
        sub = prev[prev[SEX] == sex]
        rows.append({"stratum": f"Harmonised 35-predictor, {sex.lower()}", "n": len(sub),
                     "events": int(sub.HasT2D.sum()),
                     "prev": round(sub.HasT2D.mean(), 4), "cstat": c, "params": 36})
    p = os.path.join(OUT, "ss_input.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("wrote", p, "- now run: Rscript sample_size.R")


if __name__ == "__main__":
    main()
