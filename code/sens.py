import sys, numpy as np, pandas as pd
exec(open('t2d_pipeline.py').read().split("if __name__")[0])
X,y,cat,snps,hb=load_prevalent(); X.index=X.index.astype(str); y.index=y.index.astype(str); hb.index=hb.index.astype(str)
fut=set(pd.read_csv(DIR+'medical_history_fut.tsv',sep='\t',index_col=0,usecols=[0]).index.astype(str))
post=[i for i in y.index if y[i]==1 and i in fut]; ctl48=[i for i in y.index if y[i]==0 and hb.get(i,np.nan)>=48]
keep=y.index.difference(post+ctl48); print(f'excluded {len(post)} post-baseline cases, {len(ctl48)} controls HbA1c>=48; remaining {len(keep)} ({int(y[keep].sum())} cases)',flush=True)
X,y=X.loc[keep],y[keep]; rcv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED)
for si,(sex,age,idx) in enumerate(strata(X)):
    if si not in [int(k) for k in sys.argv[1].split(',')]: continue
    D=X.loc[idx].drop(columns=META); yy=y[idx].values; rows=[]
    for fo,(tr,te) in enumerate(rcv.split(D,yy)):
        p=lgbm().fit(D.iloc[tr],yy[tr]).predict_proba(D.iloc[te])[:,1]
        rows.append(dict(sex=sex,age=age,fold=fo,n=len(yy),cases=int(yy.sum()),AUROC=roc_auc_score(yy[te],p)))
    pd.DataFrame(rows).to_csv(OUT+f'SENS_s{si}.csv',index=False); print(sex,age,'done',flush=True)
