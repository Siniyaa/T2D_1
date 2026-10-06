"""Top 10 genetic principal components from the LD-pruned PLINK set (PCA.bed/.bim/.fam),
using randomised SVD of 9,536 randomly selected common variants (low-memory approximation)."""
import numpy as np, pandas as pd, os
from sklearn.utils.extmath import randomized_svd
exec(open('t2d_pipeline.py').read().split("if __name__")[0])
fam = pd.read_csv(DIR + 'PCA.fam', sep=r'\s+', header=None); n = len(fam)
bim = pd.read_csv(DIR + 'PCA.bim', sep='\t', header=None); M = len(bim); bpv = (n + 3) // 4
assert os.path.getsize(DIR + 'PCA.bed') == 3 + M * bpv
bed = np.memmap(DIR + 'PCA.bed', dtype=np.uint8, mode='r', offset=3, shape=(M, bpv)); lut = np.array([2, -1, 1, 0], dtype=np.int8)
def decode(rows):
    b = np.asarray(bed[rows]); c = np.stack([(b >> (2 * k)) & 3 for k in range(4)], axis=2).reshape(len(rows), bpv * 4)[:, :n]; return lut[c]
rng = np.random.default_rng(42); cand = np.sort(rng.choice(M, 40000, replace=False)); keep = []
for i in range(0, len(cand), 4000):
    G = decode(cand[i:i + 4000]).astype(np.float32); G[G < 0] = np.nan
    maf = np.nanmean(G, 1) / 2; maf = np.minimum(maf, 1 - maf); keep += list(cand[i:i + 4000][(maf >= 0.05) & (np.isnan(G).mean(1) <= 0.02)])
keep = np.array(keep[:12000]); X = np.empty((n, len(keep)), dtype=np.float32)
for i in range(0, len(keep), 2000):
    G = decode(keep[i:i + 2000]).astype(np.float32); G[G < 0] = np.nan
    mu = np.nanmean(G, 1, keepdims=True); sd = np.sqrt(2 * (mu / 2) * (1 - mu / 2)); sd[sd == 0] = 1
    X[:, i:i + 2000] = ((np.where(np.isnan(G), mu, G) - mu) / sd).T
U, S, _ = randomized_svd(X, n_components=10, n_iter=5, random_state=42)
pcs = pd.DataFrame(U * S, index=fam[1].astype(str).values, columns=[f'PC{k}' for k in range(1, 11)]); pcs.index.name = 'eid'
pcs.to_csv(OUT + 'pcs.tsv', sep='\t')   # participant-level: do NOT share
print('variants used:', len(keep), '| variance explained (%):', np.round(100 * S ** 2 / (X ** 2).sum(), 3).tolist())
