"""Deposit the externally evaluated harmonised models.

Writes, into T2D_OUT_DIR:
  S16_coefficients_full_precision.csv  machine-readable logistic model, NO display rounding
  S16_logistic_coefficients.csv        the rounded table printed as Supplementary Table S16
and, into ../models:
  LightGBM_harmonised_{male,female}.txt

The full-precision file is then RELOADED FROM DISK and its predictions compared with the fitted
scikit-learn pipeline on records that include missing values, so the check validates the file
that is actually deposited rather than in-memory values.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, numpy as np, pandas as pd

def _r(x):
    """repr of a plain Python float: numpy scalars repr as 'np.float64(...)', which
    does not parse back on reload."""
    return repr(float(x))
from common import load_prevalent, fit_harmonised, harmonised_X, OUT, SEX

MODELS = os.path.join(OUT, "..", "models")
FULL = "S16_coefficients_full_precision.csv"
ROUND = "S16_logistic_coefficients.csv"


def export(prev):
    rows, rounded, fitted = [], [], {}
    for sex in ("Male", "Female"):
        gbm, pipe, X, y = fit_harmonised(prev, sex)
        imp, sc, lr = pipe.named_steps["imp"], pipe.named_steps["sc"], pipe.named_steps["lr"]
        med, mu, sd = (imp.statistics_.astype(float), sc.mean_.astype(float),
                       sc.scale_.astype(float))          # fitted scaler scale_, not a recompute
        b, b0 = lr.coef_[0].astype(float), float(lr.intercept_[0])
        pi1 = float((prev[prev[SEX] == sex]["HasT2D"]).mean())
        fitted[sex] = (pipe, X, med, mu, sd, b, b0)
        os.makedirs(MODELS, exist_ok=True)
        gbm.booster_.save_model(os.path.join(MODELS, f"LightGBM_harmonised_{sex.lower()}.txt"))
        for j, c in enumerate(X.columns):
            rows.append({"sex": sex, "order": j, "feature": c,
                         "imputation_median": _r(med[j]), "standardisation_mean": _r(mu[j]),
                         "standardisation_sd": _r(sd[j]), "coef_standardised": _r(b[j]),
                         "coef_original_scale": _r(b[j] / sd[j]),
                         "training_case_fraction": _r(pi1)})
            rounded.append({"Sex": sex, "Feature": c, "Imputation median": round(med[j], 4),
                            "Training mean": round(mu[j], 4), "Training SD": round(sd[j], 4),
                            "Standardised coefficient": round(b[j], 5),
                            "Coefficient, original units": round(b[j] / sd[j], 6)})
        rows.append({"sex": sex, "order": -1, "feature": "(intercept)", "imputation_median": "",
                     "standardisation_mean": "", "standardisation_sd": "",
                     "coef_standardised": _r(b0),
                     "coef_original_scale": _r(b0 - float(np.sum(b / sd * mu))),
                     "training_case_fraction": _r(pi1)})
        rounded.append({"Sex": sex, "Feature": "(Intercept)", "Imputation median": "",
                        "Training mean": "", "Training SD": "",
                        "Standardised coefficient": round(b0, 5),
                        "Coefficient, original units": round(b0 - float(np.sum(b / sd * mu)), 6)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, FULL), index=False)
    pd.DataFrame(rounded).to_csv(os.path.join(OUT, ROUND), index=False)
    return fitted


def verify(fitted, n=200, seed=3):
    """Reload the deposited file and compare its predictions with the fitted pipeline."""
    D = pd.read_csv(os.path.join(OUT, FULL))
    for c in ("imputation_median", "standardisation_mean", "standardisation_sd",
              "coef_standardised", "coef_original_scale", "training_case_fraction"):
        D[c] = pd.to_numeric(D[c], errors="raise" if c == "coef_standardised" else "coerce")
    worst, detail = 0.0, []
    rng = np.random.default_rng(seed)
    for sex, (pipe, X, *_rest) in fitted.items():
        d = D[(D.sex == sex) & (D.order >= 0)].sort_values("order")
        b0 = float(D[(D.sex == sex) & (D.order == -1)].coef_standardised.iloc[0])
        med, mu = d.imputation_median.values, d.standardisation_mean.values
        sd, b = d.standardisation_sd.values, d.coef_standardised.values
        assert list(d.feature) == list(X.columns), "deposited feature order differs from the model"
        idx = rng.choice(len(X), n, replace=False)
        sub = X.iloc[idx]
        x = np.where(np.isnan(sub.values.astype(float)), med, sub.values.astype(float))
        p_file = 1 / (1 + np.exp(-(b0 + ((x - mu) / sd) @ b)))
        p_pipe = pipe.predict_proba(sub)[:, 1]
        m = float(np.abs(p_file - p_pipe).max())
        assert np.isfinite(m), f"{sex}: comparison produced non-finite values - check the reload"
        worst = m if m > worst else worst
        detail.append((sex, n, int(sub.isna().any(axis=1).sum()), m))
    for s, k, nm, m in detail:
        print(f"{s}: {k} records ({nm} with >=1 missing value) max |file - pipeline| = {m:.3e}")
    print(f"deposited file reproduces the fitted pipeline to {worst:.3e}")
    assert worst < 1e-10
    return worst


if __name__ == "__main__":
    f = export(load_prevalent())
    verify(f)
    print("wrote", os.path.join(OUT, FULL), "and", os.path.join(OUT, ROUND))
