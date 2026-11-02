"""Evaluate one harmonised fit with and without survey weights.

This replaces the legacy pickle-based path, whose blood-count and cholesterol
mapping and regression penalty differed from common.py. Old CSVs must not be
relabelled as results from this implementation; rerun on the authorised data.
"""
import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from common import (load_prevalent, load_nhanes, load_survey_design,
                    fit_harmonised, harmonised_X, nhanes_diagnosed,
                    nhanes_undiagnosed, boot_ci, OUT, SEED, UCOLS, NCOLS)
from evaluation import rescaled_auc_ci


def main():
    prev, N = load_prevalent(), load_nhanes()
    design = load_survey_design()
    rows, internal = [], []
    for sex in ('Male', 'Female'):
        gbm, pipe, X, y = fit_harmonised(prev, sex)
        mask = N.sexlab == sex
        NX = harmonised_X(N.loc[mask], source='nhanes')
        for name, model, col in [('LightGBM', gbm, 'p_lgbm'),
                                 ('Logistic regression', pipe, 'p_lr')]:
            N.loc[mask, col] = model.predict_proba(NX)[:, 1]
            cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
            aucs = [roc_auc_score(y[te], clone(model).fit(X.iloc[tr], y[tr])
                                  .predict_proba(X.iloc[te])[:, 1])
                    for tr, te in cv.split(X, y)]
            internal.append(dict(sex=sex, model=name, n=len(y), cases=int(y.sum()),
                                 AUROC=np.mean(aucs), fold_SD=np.std(aucs, ddof=1)))
        for label, d, weight in [
            ('Diagnosed diabetes, all', nhanes_diagnosed(N, sex), 'WTMECPRP'),
            ('Diagnosed diabetes, non-Hispanic White', nhanes_diagnosed(N, sex, nhw=True), 'WTMECPRP'),
            ('Undiagnosed diabetes (fasting subsample)', nhanes_undiagnosed(N, sex), 'WTSAFPRP')]:
            # Use the same eligible participants for both weighting choices.
            d = d[d[weight].notna() & (d[weight] > 0)]
            for name, col in [('LightGBM', 'p_lgbm'), ('Logistic regression', 'p_lr')]:
                lo, hi = boot_ci(d.y, d[col])
                wa, wl, wh, n, cases = rescaled_auc_ci(d, design, col, weight)
                rows.append(dict(outcome=label, sex=sex, model=name, n=n, cases=cases,
                                 AUROC=roc_auc_score(d.y, d[col]), lower=lo, upper=hi,
                                 weighted_AUROC=wa, weighted_lower=wl, weighted_upper=wh))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'NHANES_evaluation_harmonised.csv'), index=False)
    pd.DataFrame(internal).to_csv(os.path.join(OUT, 'NHANES_internal_harmonised.csv'), index=False)
    pd.DataFrame({'UK Biobank': UCOLS, 'NHANES': NCOLS}).to_csv(
        os.path.join(OUT, 'S12_harmonisation.csv'), index=False)
    print('Wrote unified NHANES evaluation, internal evaluation, and mapping.')


if __name__ == '__main__':
    main()
