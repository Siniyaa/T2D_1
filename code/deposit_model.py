"""Supplementary Table S16: deposit the harmonised 35-predictor model.

Writes logistic-regression coefficients on the standardised and original measurement scales
with the imputation and standardisation constants, and the fitted LightGBM boosters. Verifies
that applying the original-scale coefficients reproduces the fitted pipeline.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, json, hashlib, platform, importlib.metadata
import numpy as np, pandas as pd
from common import load_prevalent, fit_harmonised, UCOLS, NCOLS, OUT, SEX

MODELS = os.path.join(OUT, "..", "models")


def main():
    prev = load_prevalent()
    os.makedirs(MODELS, exist_ok=True)
    rows, worst, artifacts = [], 0.0, {}
    for sex in ("Male", "Female"):
        gbm, pipe, X, y = fit_harmonised(prev, sex)
        med = pipe.named_steps["imp"].statistics_
        mu = pipe.named_steps["sc"].mean_
        sd = pipe.named_steps["sc"].scale_
        b_std = pipe.named_steps["lr"].coef_[0]
        b0_std = float(pipe.named_steps["lr"].intercept_[0])
        b_orig = b_std / sd
        b0_orig = b0_std - float(np.sum(b_std * mu / sd))

        Xi = X.copy()
        for j, c in enumerate(Xi.columns):
            Xi[c] = Xi[c].fillna(med[j])
        p_manual = 1 / (1 + np.exp(-(b0_orig + Xi.values @ b_orig)))
        worst = max(worst, float(np.max(np.abs(p_manual - pipe.predict_proba(X)[:, 1]))))

        payload = {"sex": sex, "UKB_columns": UCOLS, "NHANES_columns": NCOLS,
                   "development_n": len(y), "development_cases": int(y.sum()),
                   "development_prevalence": float(np.mean(y)),
                   "imputation_median": med.tolist(), "mean": mu.tolist(), "scale": sd.tolist(),
                   "coefficient_original": b_orig.tolist(), "intercept_original": b0_orig,
                   "coefficient_standardised": b_std.tolist(), "intercept_standardised": b0_std}
        model_path = os.path.join(MODELS, f"Logistic_harmonised_{sex.lower()}.json")
        with open(model_path, "w") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
        # Test the bytes actually deposited, not just the in-memory coefficients.
        with open(model_path) as f:
            restored = json.load(f)
        restored_p = 1/(1+np.exp(-(restored["intercept_original"]
                                 + Xi.values @ np.array(restored["coefficient_original"]))))
        assert np.max(np.abs(restored_p-pipe.predict_proba(X)[:, 1])) < 1e-8
        artifacts[os.path.basename(model_path)] = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
        for j, c in enumerate(UCOLS):
            rows.append({"Sex": sex, "Predictor": c, "NHANES variable": dict(zip(UCOLS, NCOLS))[c],
                         "Imputation median": round(float(med[j]), 4),
                         "Training mean": round(float(mu[j]), 4),
                         "Training SD": round(float(sd[j]), 4),
                         "Standardised coefficient": round(float(b_std[j]), 5),
                         "Coefficient, original units": round(float(b_orig[j]), 6)})
        rows.append({"Sex": sex, "Predictor": "(Intercept)", "NHANES variable": "-",
                     "Imputation median": "-", "Training mean": "-", "Training SD": "-",
                     "Standardised coefficient": round(b0_std, 5),
                     "Coefficient, original units": round(b0_orig, 6)})
        booster_path = os.path.join(MODELS, f"LightGBM_harmonised_{sex.lower()}.txt")
        gbm.booster_.save_model(booster_path)
        artifacts[os.path.basename(booster_path)] = hashlib.sha256(open(booster_path, "rb").read()).hexdigest()

    assert worst < 1e-8, f"deposited coefficients do not reproduce the pipeline ({worst:.2e})"
    p = os.path.join(OUT, "S16_logistic_coefficients.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"wrote {p}; max |manual - pipeline| = {worst:.2e}")
    manifest = {"python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in
                             ["numpy", "pandas", "scikit-learn", "lightgbm", "scipy"]},
                "sha256": artifacts,
                "note": "CSV coefficients are rounded; use JSON for full precision."
                        " Sampling offsets are sex-specific and require matching outcome definitions."}
    with open(os.path.join(MODELS, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Full-precision regression exports and file checksums written.")


if __name__ == "__main__":
    main()

