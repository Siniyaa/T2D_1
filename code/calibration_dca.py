"""Exploratory external calibration after a sex-specific sampling offset.

Target prevalence is estimated from this evaluation sample. This is apparent
updating, not independent validation of recalibration or screening thresholds.
Output files deliberately have new names to preserve the historical estimates.
"""
import argparse
import os
import numpy as np
import pandas as pd
from common import (load_prevalent, load_nhanes, score_nhanes,
                    nhanes_diagnosed, nhanes_undiagnosed, OUT, SEX)
from evaluation import offset_correct, calibration, net_benefit


def main():
    prev = load_prevalent()
    N = score_nhanes(load_nhanes(), prev)
    summaries, curves, operating = [], [], []
    for sex in ('Male', 'Female'):
        development = prev.loc[prev[SEX] == sex, 'HasT2D'].astype(int).mean()
        for label, domain, wcol in [
            ('Diagnosed', nhanes_diagnosed(N, sex), 'WTMECPRP'),
            ('Undiagnosed', nhanes_undiagnosed(N, sex), 'WTSAFPRP')]:
            d = domain[domain[wcol].notna() & (domain[wcol] > 0)]
            for weighting in ('unweighted', 'survey-weighted'):
                w = np.ones(len(d)) if weighting == 'unweighted' else d[wcol].to_numpy(float)
                y = d.y.to_numpy()
                target = float(np.average(y, weights=w))
                for model, col in [('LightGBM', 'p_lgbm'), ('Logistic regression', 'p_lr')]:
                    raw = d[col].to_numpy()
                    corrected = offset_correct(raw, development, target)
                    meta = dict(outcome=label, sex=sex, model=model, weighting=weighting,
                                n=len(d), cases=int(y.sum()), development_prevalence=development,
                                target_prevalence=target,
                                updating='Target prevalence estimated on evaluation data')
                    for stage, p in [('raw', raw), ('sex-specific offset', corrected)]:
                        intercept, slope, brier = calibration(y, p, w)
                        summaries.append(dict(meta, stage=stage, intercept=intercept,
                                              slope=slope, Brier=brier,
                                              constant_Brier=target*(1-target)))
                    for threshold in (.02, .05, .075, .10, .15, .20):
                        curves.append(dict(meta, threshold=threshold,
                                           NB_model=net_benefit(y, corrected, threshold, w),
                                           NB_test_all=net_benefit(y, np.ones(len(y)), threshold, w),
                                           NB_test_none=0.))
                        positive = corrected >= threshold
                        tp = w[positive & (y == 1)].sum(); fp = w[positive & (y == 0)].sum()
                        fn = w[~positive & (y == 1)].sum(); tn = w[~positive & (y == 0)].sum()
                        ratio = lambda a, b: float(a/b) if b else np.nan
                        operating.append(dict(meta, threshold=threshold,
                            tested_pct=100*ratio(tp+fp, w.sum()), sensitivity_pct=100*ratio(tp, tp+fn),
                            specificity_pct=100*ratio(tn, tn+fp), PPV_pct=100*ratio(tp, tp+fp),
                            NPV_pct=100*ratio(tn, tn+fn), NNT=ratio(tp+fp, tp)))
    os.makedirs(OUT, exist_ok=True)
    for rows, name in [(summaries, 'Calibration_sex_specific.csv'),
                       (curves, 'DecisionCurve_sex_specific.csv'),
                       (operating, 'OperatingPoints_sex_specific.csv')]:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, name), index=False)
        print('Wrote', name)
    print('Exploratory point estimates only; uncertainty and independent updating validation remain required.')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
