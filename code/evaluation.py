"""Shared evaluation primitives; no participant data or fitted models at import.

Rao–Wu rescaled bootstrap: sample n_h - 1 PSUs with replacement per stratum
and multiply original weights by n_h/(n_h-1) times the selection count.
The design frame must include all PSUs, before selecting an analysis domain.
"""
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize, brentq
from sklearn.metrics import roc_auc_score


def offset_correct(p, development_prevalence, target_prevalence):
    for value in (development_prevalence, target_prevalence):
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("Prevalences must be strictly between zero and one")
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid predictions")
    return expit(logit(np.clip(p, 1e-6, 1-1e-6))
                 - logit(development_prevalence) + logit(target_prevalence))


def calibration(y, p, weights=None):
    """Calibration-in-the-large (slope fixed at 1), free slope, and Brier.

Weighted binomial maximum likelihood, with convergence checked explicitly.
Updating and assessing on these same outcomes is apparent recalibration.
"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)
    if (y.ndim != 1 or p.shape != y.shape or w.shape != y.shape
            or not np.isfinite(np.r_[y, p, w]).all()
            or np.any(w < 0) or w.sum() <= 0
            or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all()
            or len(np.unique(y[w > 0])) != 2):
        raise ValueError("Calibration needs valid probabilities, weights and two classes")
    w = w / w.sum()
    z = logit(np.clip(p, 1e-6, 1-1e-6))
    if np.ptp(z[w > 0]) < 1e-12:
        raise ValueError("A calibration slope is unidentified for constant predictions")
    intercept = brentq(lambda b: np.sum(w*(expit(z+b)-y)), -100, 100)
    X = np.column_stack([np.ones_like(z), z])
    def objective(b):
        eta = X @ b
        return np.sum(w*(np.logaddexp(0, eta)-y*eta))
    def gradient(b):
        return X.T @ (w*(expit(X @ b)-y))
    fit = minimize(objective, [0., 1.], jac=gradient, method="BFGS",
                   options={"gtol": 1e-9, "maxiter": 2000})
    if not fit.success and np.max(np.abs(gradient(fit.x))) > 1e-7:
        raise RuntimeError("Calibration fit did not converge: " + str(fit.message))
    if np.max(np.abs(fit.x)) > 100:
        raise RuntimeError("Possible separation in calibration fit")
    return float(intercept), float(fit.x[1]), float(np.sum(w*(y-p)**2))


def net_benefit(y, p, threshold, weights=None):
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be strictly between zero and one")
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Invalid weights")
    positive = p >= threshold
    return float((w[positive & (y == 1)].sum()
                  - w[positive & (y == 0)].sum()*threshold/(1-threshold))/w.sum())


def rescaled_auc_ci(domain, design, pcol, wcol, B=1000, seed=7):
    """Weighted AUC and percentile CI using the full PSU design.

Uses a with-replacement first-stage approximation without finite-population
correction. Rejects singleton design strata rather than treating them as fixed.
"""
    d = domain.dropna(subset=[wcol, pcol, "y", "SDMVSTRA", "SDMVPSU"])
    d = d[d[wcol] > 0]
    base = design.dropna(subset=[wcol, "SDMVSTRA", "SDMVPSU"])
    base = base[base[wcol] > 0]
    keys = base[["SDMVSTRA", "SDMVPSU"]].drop_duplicates()
    groups = {s: g.SDMVPSU.to_numpy() for s, g in keys.groupby("SDMVSTRA")}
    if any(len(v) < 2 for v in groups.values()):
        raise ValueError("Singleton survey stratum: supply the complete design")
    domain_keys = list(zip(d.SDMVSTRA, d.SDMVPSU))
    full_keys = set(map(tuple, keys.to_numpy()))
    if not set(domain_keys).issubset(full_keys):
        raise ValueError("Analysis PSUs absent from the design frame")
    y, p, w = d.y.to_numpy(), d[pcol].to_numpy(), d[wcol].to_numpy(float)
    point = roc_auc_score(y, p, sample_weight=w)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(B):
        multipliers = {}
        for s, psus in groups.items():
            n = len(psus)
            counts = np.bincount(rng.choice(n, n-1, replace=True), minlength=n)
            multipliers.update({(s, u): n/(n-1)*c for u, c in zip(psus, counts)})
        rw = w * np.array([multipliers[k] for k in domain_keys])
        if rw[y == 0].sum() > 0 and rw[y == 1].sum() > 0:
            estimates.append(roc_auc_score(y, p, sample_weight=rw))
    if len(estimates) < .95*B:
        raise ValueError("More than 5% of survey replicates lost an outcome class")
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(point), float(lo), float(hi), len(d), int(d.y.sum())
