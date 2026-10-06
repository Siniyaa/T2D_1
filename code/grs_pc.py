import sys, numpy as np, pandas as pd
exec(open('t2d_pipeline.py').read().split("if __name__")[0])
GV=pd.read_csv(OUT+'grs_variants_genotypes.tsv',sep='\t',index_col=0); GV.index=GV.index.astype(str)
PC=pd.read_csv(OUT+'pcs.tsv',sep='\t',index_col=0); PC.index=PC.index.astype(str)
X,y,cat,snps,_=load_prevalent(); X.index=X.index.astype(str); y.index=y.index.astype(str)
rcv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED)
for si,(sex,age,idx) in enumerate(strata(X)):
    if si not in [int(k) for k in sys.argv[1].split(',')]: continue
    D=X.loc[idx].drop(columns=META).drop(columns=snps); yy=y[idx].values; Gs=GV.loc[idx]; P=PC.loc[idx]; D=pd.concat([D,P],axis=1); rows=[]
    for fo,(tr,te) in enumerate(rcv.split(D,yy)):
        mu=Gs.iloc[tr].mean(); Gi=Gs.fillna(mu)
        Z=pd.concat([Gi,P],axis=1); Zs=(Z-Z.iloc[tr].mean())/Z.iloc[tr].std()
        w=LogisticRegression(C=0.1,max_iter=3000).fit(Zs.iloc[tr],yy[tr]).coef_[0][:Gi.shape[1]]
        grs=Zs.iloc[:,:Gi.shape[1]].values@w; DG=D.assign(GRS=grs)
        a=roc_auc_score(yy[te],lgbm().fit(D.iloc[tr],yy[tr]).predict_proba(D.iloc[te])[:,1])
        b=roc_auc_score(yy[te],lgbm().fit(DG.iloc[tr],yy[tr]).predict_proba(DG.iloc[te])[:,1])
        rows.append(dict(sex=sex,age=age,fold=fo,n_train=len(tr),n_test=len(te),clinical_pc=a,clinical_pc_grs=b,grs_alone=roc_auc_score(yy[te],grs[te]),pc1_alone=roc_auc_score(yy[te],P.PC1.values[te])))
    pd.DataFrame(rows).to_csv(OUT+f'GRSPC_s{si}.csv',index=False); print(sex,age,'done',flush=True)
