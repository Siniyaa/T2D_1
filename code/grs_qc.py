"""Build the 74-variant genetic risk score variant set from PLINK files T2D_4 (GWAS Catalog array set)
and T2D_12 (genome-wide significant set): sample-order check, QC, LD pruning, TCF7L2 validation."""
import numpy as np, pandas as pd
from scipy import stats, optimize
from bedread import read_bed
exec(open('t2d_pipeline.py').read().split("if __name__")[0])
G4, b4 = read_bed(DIR + 'T2D_4', DIR + 'T2D_12.fam'); G12, b12 = read_bed(DIR + 'T2D_12', DIR + 'T2D_12.fam')
ref = pd.read_csv(DIR + 'T2D_geno.tsv', sep='\t', index_col=0); ref.index = ref.index.astype(str)
a, r = G4.loc[ref.index, 'rs10010131'], ref['rs10010131']
print('sample-order agreement: %.2f%%' % (100 * max(((a == r) | (a.isna() & r.isna())).mean(), ((2 - a == r) | (a.isna() & r.isna())).mean())))
G = pd.concat([G4, G12.drop(columns=[c for c in G12.columns if c in G4.columns])], axis=1)
y = pd.read_csv(DIR + 'medical_history_3.tsv', sep='\t', index_col=0, low_memory=False)['HasT2D'].astype(int); y.index = y.index.astype(str)
ctrl = G.loc[y[y == 0].index.intersection(G.index)]
def hwe(c):
    c = c.dropna(); n0, n1, n2 = (c == 0).sum(), (c == 1).sum(), (c == 2).sum(); n = n0 + n1 + n2; p = (2 * n2 + n1) / (2 * n)
    if p in (0, 1): return 1.
    e = np.array([(1 - p) ** 2, 2 * p * (1 - p), p ** 2]) * n; return stats.chi2.sf(((np.array([n0, n1, n2]) - e) ** 2 / e).sum(), 1)
q = pd.DataFrame({'MAF': np.minimum(G.mean() / 2, 1 - G.mean() / 2), 'miss': G.isna().mean(), 'hwe': [hwe(ctrl[c]) for c in G.columns]})
qc = q[(q.MAF >= 0.01) & (q.miss <= 0.05) & (q.hwe >= 1e-6)]
order = qc.sort_values('MAF', ascending=False).index.tolist(); Xf = G[order].fillna(G[order].mean())
R = np.corrcoef(Xf.values.T) ** 2; keep = []
for i, v in enumerate(order):
    if all(R[i, order.index(k)] < 0.1 for k in keep): keep.append(v)
print(f'variants: {len(q)}, pass QC: {len(qc)}, independent after LD pruning (r2<0.1): {len(keep)}')
G[keep].to_csv(OUT + 'grs_variants_genotypes.tsv', sep='\t')   # participant-level: do NOT share
pd.Series(keep).to_csv(OUT + 'grs_keep.txt', index=False, header=False); q.to_csv(OUT + 'grs_variant_qc.csv')
g = G['rs7903146'].dropna(); yy = y.loc[g.index].values; X = np.c_[np.ones(len(g)), g.values]
b = optimize.minimize(lambda b: -(yy * (X @ b) - np.log1p(np.exp(X @ b))).sum(), np.zeros(2), method='BFGS').x
p = 1 / (1 + np.exp(-X @ b)); se = np.sqrt(np.diag(np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)))[1]
o, lo, hi = np.exp(b[1]), np.exp(b[1] - 1.96 * se), np.exp(b[1] + 1.96 * se)
if o < 1: o, lo, hi = 1 / o, 1 / hi, 1 / lo
print(f'TCF7L2 rs7903146 OR per risk allele {o:.2f} (95% CI {lo:.2f}-{hi:.2f}), P={2 * stats.norm.sf(abs(b[1] / se)):.1e}')
