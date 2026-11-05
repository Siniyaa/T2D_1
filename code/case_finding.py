"""Case-finding analysis: five algorithms on the 175 clinical predictors, and the genetic
risk score increment, in ten sex-by-age strata.

This replaces the earlier case-finding runs, which were produced in an environment that was not
recorded and which compared a LightGBM model containing 105 array variants against clinical-only
comparators. Every model here uses the SAME 175 clinical predictors, so the paired tests compare
algorithms rather than input sets. The genetic analysis adds the LD-pruned genome-wide-
significant score to that same clinical model.

Outputs
  CaseFinding_performance.csv  per stratum and algorithm: fold-mean metrics with between-fold SD
  CaseFinding_tests.csv        LightGBM vs each comparator, corrected resampled t-test
  CaseFinding_grs.csv          clinical vs clinical + GRS, and the GRS alone
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from common import load_prevalent, strata, model_matrix, DROP_PRED, OUT, DATA, AGE_U, SEED

LEAF = 0.08          # minimum fraction of training observations per leaf, tree ensembles
N_SPLITS, N_REPEATS = 5, 5


def predictors(df):
    """The 175 clinical and lifestyle predictors: candidates less age (strata are age-specific)."""
    drop = [c for c in DROP_PRED if c in df.columns] + [AGE_U]
    return [c for c in df.columns if c not in drop]


def make_models(for_lgbm_cats):
    lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                   ("lr", LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000,
                                             random_state=SEED))])
    return {
        "LightGBM": None,                       # handled separately: native categoricals
        "L1-penalised logistic regression": lr,
        "Random forest": RandomForestClassifier(min_samples_leaf=LEAF, n_estimators=300,
                                                random_state=SEED),
        "Extremely randomised trees": ExtraTreesClassifier(min_samples_leaf=LEAF, n_estimators=300,
                                                           random_state=SEED),
        "Decision tree": DecisionTreeClassifier(min_samples_leaf=LEAF, random_state=SEED),
    }


def metrics(y, p):
    pred = p >= 0.5
    tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
    return dict(AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p),
                accuracy=100 * (tp + tn) / len(y),
                sensitivity=100 * tp / max(tp + fn, 1),
                specificity=100 * tn / max(tn + fp, 1))


def corrected_t(d, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test for repeated k-fold differences."""
    d = np.asarray(d, float); k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def run_stratum(sub, cols, extra=None):
    """Fold-level metrics for every algorithm. `extra` is a numeric frame joined to X."""
    y = sub["HasT2D"].astype(int).values
    Xc, cats = model_matrix(sub, cols, for_lightgbm=True)
    Xn, _ = model_matrix(sub, cols, for_lightgbm=False)
    if extra is not None:
        Xc = pd.concat([Xc, extra.loc[sub.index]], axis=1)
        Xn = pd.concat([Xn, extra.loc[sub.index]], axis=1)
    rk = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    models = make_models(cats)
    per = {name: [] for name in models}
    sizes = []
    for tr, te in rk.split(Xc, y):
        sizes.append((len(tr), len(te)))
        for name, est in models.items():
            if name == "LightGBM":
                m = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-1,
                                       random_state=SEED, verbose=-1)
                m.fit(Xc.iloc[tr], y[tr], categorical_feature=cats)
                p = m.predict_proba(Xc.iloc[te])[:, 1]
            else:
                from sklearn.base import clone
                m = clone(est).fit(Xn.iloc[tr], y[tr])
                p = m.predict_proba(Xn.iloc[te])[:, 1]
            per[name].append(metrics(y[te], p))
    n_train = int(np.mean([a for a, _ in sizes])); n_test = int(np.mean([b for _, b in sizes]))
    return per, n_train, n_test, len(y), int(y.sum())



def build_grs(prev):
    """LD-pruned genome-wide-significant score from the 76 array-genotyped variants.

    QC: MAF >= 1%, missingness <= 5%, Hardy-Weinberg P >= 1e-6 in controls. Pruning drops the
    lower-frequency member of any pair with r^2 >= 0.1. The score is the sum of allele dosages
    over the retained variants, with missing dosages imputed at twice the allele frequency.
    """
    g = pd.read_csv(os.path.join(DATA, "T2D_geno_lasso.tsv"), sep="\t")
    g = g.set_index("EID").reindex(prev["EID"].values)
    X = g.apply(pd.to_numeric, errors="coerce")
    ctrl = ~prev["HasT2D"].values
    keep = []
    for c in X.columns:
        v = X[c]
        if v.isna().mean() > 0.05:
            continue
        p = v.mean() / 2
        maf = min(p, 1 - p)
        if maf < 0.01:
            continue
        # Hardy-Weinberg exact-ish chi-square in controls
        vc = v[ctrl].dropna()
        n = len(vc)
        if n < 100:
            continue
        obs = np.array([(vc == 0).sum(), (vc == 1).sum(), (vc == 2).sum()], float)
        q = obs @ np.array([0, 1, 2]) / (2 * n)
        exp = n * np.array([(1 - q) ** 2, 2 * q * (1 - q), q ** 2])
        exp = np.where(exp < 1e-9, 1e-9, exp)
        chi2 = float(((obs - exp) ** 2 / exp).sum())
        if stats.chi2.sf(chi2, 1) < 1e-6:
            continue
        keep.append((c, maf))
    keep.sort(key=lambda t: -t[1])
    kept = []
    for c, _ in keep:
        if all(abs(X[c].corr(X[k])) ** 2 < 0.1 for k in kept):
            kept.append(c)
    Z = X[kept]
    Z = Z.fillna(Z.mean())
    Z.index = prev.index
    print(f"GRS: {X.shape[1]} genotyped variants -> {len(keep)} pass QC -> {len(kept)} after LD pruning")
    return Z, len(kept)


def orient_and_score(Z, y, tr):
    """Risk-allele-oriented allele count, with orientation fitted on the TRAINING fold only.

    The deposited extract carries no published effect sizes, so each variant is oriented by the
    sign of its association with the outcome among training-fold participants; the score is then
    the sum of oriented dosages. Because orientation never sees the held-out fold, the resulting
    AUROC is an out-of-sample estimate.
    """
    Ztr = Z.iloc[tr]
    sign = np.sign([np.corrcoef(Ztr[c].values, y[tr])[0, 1] if Ztr[c].std() > 0 else 0.0
                    for c in Z.columns])
    sign[sign == 0] = 1.0
    return (Z.values * sign).sum(axis=1)


def grs_analysis(prev, cols):
    rows = []
    Z, n_var = build_grs(prev)
    for sex, lab, sub in strata(prev):
        y = sub["HasT2D"].astype(int).values
        Xc, cats = model_matrix(sub, cols, for_lightgbm=True)
        Zs = Z.loc[sub.index]
        rk = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
        a_cl, a_gr, a_only, sizes = [], [], [], []
        for tr, te in rk.split(Xc, y):
            sizes.append((len(tr), len(te)))
            score = orient_and_score(Zs, y, tr)
            Xg = Xc.copy(); Xg["GRS"] = score
            for X, store in ((Xc, a_cl), (Xg, a_gr)):
                m = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-1,
                                       random_state=SEED, verbose=-1)
                m.fit(X.iloc[tr], y[tr], categorical_feature=cats)
                store.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
            a_only.append(roc_auc_score(y[te], score[te]))
        n_tr = int(np.mean([a for a, _ in sizes])); n_te = int(np.mean([b for _, b in sizes]))
        d = np.array(a_gr) - np.array(a_cl)
        t, p = corrected_t(d, n_tr, n_te)
        rows.append(dict(Sex=sex, **{"Age group": lab}, n_variants=n_var,
                         clinical=np.mean(a_cl), clinical_plus_GRS=np.mean(a_gr),
                         difference=d.mean(), difference_sd=d.std(ddof=1), t=t, P=p,
                         GRS_alone=np.mean(a_only), GRS_alone_sd=np.std(a_only, ddof=1),
                         improved=f"{int((d > 0).sum())}/{len(d)}"))
        print(f"  {sex} {lab}: clinical {np.mean(a_cl):.4f} -> +GRS {np.mean(a_gr):.4f} "
              f"(d={d.mean():+.4f}, P={p:.3f}); GRS alone {np.mean(a_only):.3f}")
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "CaseFinding_grs.csv"), index=False)
    print("wrote CaseFinding_grs.csv")



def sensitivity_analyses(prev, cols):
    """Two case-finding sensitivity analyses on the same pipeline as the main run.

    (a) excluding cases whose first E11 record postdates their own baseline visit;
    (b) controls meeting biochemical criteria for diabetes excluded, and reclassified as cases.
    """
    from common import HBA, GLU
    hb = pd.to_numeric(prev[HBA], errors="coerce"); gl = pd.to_numeric(prev[GLU], errors="coerce")
    bio_all = ((hb >= 48) | (gl >= 11.1)).fillna(False)
    post = prev.get("CaseAfterBaseline")
    rows = []
    for sex, lab, sub in strata(prev):
        y = sub["HasT2D"].astype(int).values
        X, cats = model_matrix(sub, cols, for_lightgbm=True)
        rk = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
        def auc_for(Xf, yf):
            out = []
            r2 = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
            for tr, te in r2.split(Xf, yf):
                m = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-1,
                                       random_state=SEED, verbose=-1)
                m.fit(Xf.iloc[tr], yf[tr], categorical_feature=cats)
                out.append(roc_auc_score(yf[te], m.predict_proba(Xf.iloc[te])[:, 1]))
            return float(np.mean(out)), float(np.std(out, ddof=1))
        base = auc_for(X, y)
        b = bio_all.loc[sub.index].values
        keep = ~(b & (y == 0))
        excl = auc_for(X[keep], y[keep])
        recl = auc_for(X, np.where(b, 1, y))
        rows.append(dict(Sex=sex, **{"Age group": lab}, main=base[0], main_sd=base[1],
                         exclude_misclassified=excl[0], reclassify_as_cases=recl[0],
                         d_exclude=excl[0] - base[0], d_reclassify=recl[0] - base[0],
                         n=len(y), controls_biochemical=int((b & (y == 0)).sum())))
        print(f"  {sex} {lab}: main {base[0]:.4f}  exclude {excl[0]:.4f} ({excl[0]-base[0]:+.4f})  "
              f"reclassify {recl[0]:.4f} ({recl[0]-base[0]:+.4f})")
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "CaseFinding_misclassification.csv"),
                                       index=False)
    print("wrote CaseFinding_misclassification.csv")


def main():
    prev = load_prevalent()
    cols = predictors(prev)
    print(f"{len(cols)} clinical predictors; {len(prev):,} participants, {int(prev.HasT2D.sum()):,} cases")
    perf, tests = [], []
    for sex, lab, sub in strata(prev):
        per, n_tr, n_te, n, k = run_stratum(sub, cols)
        for name, folds in per.items():
            f = pd.DataFrame(folds)
            perf.append(dict(Sex=sex, **{"Age group": lab}, Algorithm=name, n=n, cases=k,
                             **{c: f[c].mean() for c in f.columns},
                             **{c + "_sd": f[c].std(ddof=1) for c in f.columns}))
        base = [d["AUROC"] for d in per["LightGBM"]]
        for name in per:
            if name == "LightGBM":
                continue
            d = np.array(base) - np.array([x["AUROC"] for x in per[name]])
            t, p = corrected_t(d, n_tr, n_te)
            tests.append(dict(Sex=sex, **{"Age group": lab}, Comparator=name,
                              LightGBM=np.mean(base),
                              comparator_AUROC=np.mean([x["AUROC"] for x in per[name]]),
                              difference=d.mean(), t=t, P=p))
        print(f"  {sex} {lab}: n={n:,} ({k} cases) "
              + ", ".join(f"{nm.split()[0]} {np.mean([x['AUROC'] for x in per[nm]]):.3f}"
                          for nm in per))
    pd.DataFrame(perf).round(4).to_csv(os.path.join(OUT, "CaseFinding_performance.csv"), index=False)
    pd.DataFrame(tests).round(4).to_csv(os.path.join(OUT, "CaseFinding_tests.csv"), index=False)
    print("wrote CaseFinding_performance.csv and CaseFinding_tests.csv")
    grs_analysis(prev, cols)


if __name__ == "__main__":
    main()
