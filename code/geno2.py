import sys, numpy as np, pandas as pd
exec(open('t2d_pipeline.py').read().split("if __name__")[0])
QC=[l.strip() for l in open(OUT+'snp_qc_pass.txt')]
X,y,cat,snps,_=load_prevalent(); rcv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED)
only=[int(k) for k in sys.argv[1].split(',')]; rows=[]
for si,(sex,age,idx) in enumerate(strata(X)):
    if si not in only: continue
    D=X.loc[idx].drop(columns=META); yy=y[idx].values
    clin=D.drop(columns=snps); qcd=pd.concat([clin,D[QC]],axis=1)
    for fo,(tr,te) in enumerate(rcv.split(D,yy)):
        G=D[QC]; med=G.iloc[tr].median(); Gi=G.fillna(med)
        lr=LogisticRegression(max_iter=1000).fit(Gi.iloc[tr],yy[tr]); grs=pd.Series(Gi.values@lr.coef_[0],index=D.index,name='GRS')
        variants={'none':clin,'qc6':qcd,'grs':pd.concat([clin,grs],axis=1)}
        r=dict(sex=sex,age=age,fold=fo,n_train=len(tr),n_test=len(te),grs_alone=roc_auc_score(yy[te],grs.iloc[te]))
        for k,dat in variants.items():
            p=lgbm().fit(dat.iloc[tr],yy[tr]).predict_proba(dat.iloc[te])[:,1]; r[k]=roc_auc_score(yy[te],p)
        rows.append(r)
    pd.DataFrame([r for r in rows if r['sex']==sex and r['age']==age]).to_csv(OUT+f'G2_s{si}.csv',index=False)
    print(sex,age,'done',flush=True)
