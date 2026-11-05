import numpy as np, pandas as pd, json, re
exec(open('t2d_pipeline.py').read().split("if __name__")[0])
# (a) prevalent: gain importance from LightGBM fitted on each full stratum
X,y,cat,snps,_=load_prevalent(); imp={}
for sex,age,idx in strata(X):
    D=X.loc[idx].drop(columns=META); m=lgbm().fit(D,y[idx])
    g=pd.Series(m.booster_.feature_importance('gain'),index=D.columns); imp[f'{sex}|{age}']=g/g.sum()*100
I=pd.DataFrame(imp); I.to_csv(OUT+'prev_gain_importance.csv')
snp_share=I.loc[[c for c in I.index if c in snps]].sum(); print('SNP share of total gain (%) per stratum:',snp_share.round(2).to_dict())
# The incident section that followed (simple clinical benchmark vs LightGBM on the same folds,
# writing C_simple_benchmark.csv) was removed with the incident-prediction analysis; see
# EXECUTION.md. It is preserved in the git history at commit 87ca782d.
