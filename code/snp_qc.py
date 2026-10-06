"""QC and LD assessment of the 105 GWAS Catalog array variants (T2D_geno.tsv)."""
import numpy as np, pandas as pd
from scipy import stats
exec(open('t2d_pipeline.py').read().split("if __name__")[0])
g = pd.read_csv(DIR + 'T2D_geno.tsv', sep='\t', index_col=0)
lab = pd.read_csv(DIR + 'medical_history_3.tsv', sep='\t', index_col=0, low_memory=False)['HasT2D'].astype(int)
ctrl = g.loc[lab[lab == 0].index.intersection(g.index)]
def hwe_p(c):
    c = c.dropna().round().astype(int); n0, n1, n2 = (c == 0).sum(), (c == 1).sum(), (c == 2).sum(); n = n0 + n1 + n2
    p = (2 * n2 + n1) / (2 * n)
    if p in (0, 1): return 1.0
    e = np.array([(1 - p) ** 2, 2 * p * (1 - p), p ** 2]) * n
    return stats.chi2.sf(((np.array([n0, n1, n2]) - e) ** 2 / e).sum(), 1)
q = pd.DataFrame({'MAF': np.minimum(g.mean() / 2, 1 - g.mean() / 2), 'missing': g.isna().mean(),
                  'HWE_P_controls': [hwe_p(ctrl[c]) for c in g.columns]})
q['pass'] = (q.MAF >= 0.01) & (q.missing <= 0.05) & (q.HWE_P_controls >= 1e-6)
q.to_csv(OUT + 'snp105_qc.csv'); q[q['pass']].index.to_series().to_csv(OUT + 'snp_qc_pass.txt', index=False, header=False)
r = (g.corr() ** 2).values.copy(); np.fill_diagonal(r, 0)
print(f"MAF<1%: {(q.MAF < 0.01).sum()}, MAF<0.1%: {(q.MAF < 0.001).sum()}, pass QC: {q['pass'].sum()}, max pairwise r2: {np.nanmax(r):.3f}")
