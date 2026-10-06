"""Summarise fold-level results into the tables reported in the manuscript."""
import glob, numpy as np, pandas as pd
from scipy import stats
exec(open('t2d_pipeline.py').read().split("if __name__")[0])
order = {'<50': 0, '50–54': 1, '55–59': 2, '60–64': 3, '≥65': 4}
def cp(d, ntr, nte):
    d = np.asarray(d); k = len(d); v = d.var(ddof=1)
    return np.nan if v == 0 else 2 * stats.t.sf(abs(d.mean() / np.sqrt((1 / k + nte / ntr) * v)), k - 1)
def srt(df): return df.assign(o=df.age.map(order)).sort_values(['sex', 'o'], ascending=[False, True]).drop(columns='o')
B = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(OUT + 'B_cv_folds_[0-9]*.csv'))]); B.to_csv(OUT + 'B_cv_folds_all.csv', index=False)
S = B.groupby(['sex', 'age', 'model'])[['AUROC', 'AUPRC', 'accuracy', 'sensitivity', 'specificity']].agg(['mean', 'std'])
S.columns = ['_'.join(c) for c in S.columns]; srt(S.reset_index()).to_csv(OUT + 'B_summary.csv', index=False)
def paired(D, a, b, name):
    rows = []
    for (s, ag), g in D.groupby(['sex', 'age']):
        x, z = g[g.model == a].sort_values('fold'), g[g.model == b].sort_values('fold'); d = x.AUROC.values - z.AUROC.values
        rows.append(dict(sex=s, age=ag, AUROC_a=x.AUROC.mean(), AUROC_b=z.AUROC.mean(), diff=d.mean(), diff_sd=d.std(ddof=1),
                         P=cp(d, x.n_train.mean(), x.n_test.mean())))
    srt(pd.DataFrame(rows)).to_csv(OUT + name, index=False)
paired(B, 'LightGBM', 'LightGBM (no genetic data)', 'B_genetic_comparison.csv')
paired(B, 'LightGBM', 'Logistic regression', 'B_lgbm_vs_lr.csv')
for pat, a, b, name in [('GRS74_s*.csv', 'clinical_grs', 'clinical', 'GRS74_summary.csv'), ('GRSPC_s*.csv', 'clinical_pc_grs', 'clinical_pc', 'GRSPC_summary.csv')]:
    f = sorted(glob.glob(OUT + pat))
    if not f: continue
    D = pd.concat([pd.read_csv(x) for x in f]); rows = []
    for (s, ag), g in D.groupby(['sex', 'age']):
        d = (g[a] - g[b]).values
        rows.append(dict(sex=s, age=ag, without=g[b].mean(), with_grs=g[a].mean(), dAUROC=d.mean(), dSD=d.std(ddof=1),
                         P=cp(d, g.n_train.mean(), g.n_test.mean()), grs_alone=g.grs_alone.mean()))
    srt(pd.DataFrame(rows)).to_csv(OUT + name, index=False)
print('summaries written to', OUT)
