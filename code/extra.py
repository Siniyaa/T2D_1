import numpy as np, pandas as pd, json, re
exec(open('t2d_pipeline.py').read().split("if __name__")[0])
# (a) prevalent: gain importance from LightGBM fitted on each full stratum
X,y,cat,snps,_=load_prevalent(); imp={}
for sex,age,idx in strata(X):
    D=X.loc[idx].drop(columns=META); m=lgbm().fit(D,y[idx])
    g=pd.Series(m.booster_.feature_importance('gain'),index=D.columns); imp[f'{sex}|{age}']=g/g.sum()*100
I=pd.DataFrame(imp); I.to_csv(OUT+'prev_gain_importance.csv')
snp_share=I.loc[[c for c in I.index if c in snps]].sum(); print('SNP share of total gain (%) per stratum:',snp_share.round(2).to_dict())
# (b) incident: simple clinical benchmark vs full LightGBM (same folds)
f=pd.read_csv(DIR+'medical_history_fut.tsv',sep='\t',index_col=0,low_memory=False)
m=pd.read_csv(DIR+'medical_history_3.tsv',sep='\t',index_col=0,low_memory=False)
hb='Glycated haemoglobin (HbA1c) [30750]'
f=f[(f['Ethnic background [21000]']=='British')&~(f[hb]>=48)]
c=m[(~m.HasT2D.astype(bool))&~m.index.isin(pd.read_csv(DIR+'medical_history_fut.tsv',sep='\t',index_col=0,usecols=[0]).index)&~(m[hb]>=48)]
cols=['Sex [31]','Age when attended assessment centre [21003]','Body mass index (BMI) [21001]','Waist circumference [48]',
      'Father had diabetes [20107]','Mother had diabetes [20110]','Siblings had diabetes [20111]','Systolic blood pressure, automated reading [4080]']
D=pd.concat([f[cols],c[cols]]); yy=np.r_[np.ones(len(f),int),np.zeros(len(c),int)]
D['FamHx']=D[cols[4:7]].fillna(False).astype(bool).any(axis=1).astype(float); D=D.drop(columns=cols[4:7])
full=pd.read_csv(OUT+'C_incident_folds.csv')
rcv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED); rows=[]
for sex in ['Male','Female']:
    s=(D['Sex [31]']==sex).values; Xs=D[s].drop(columns=['Sex [31]']); ys=yy[s]
    for fo,(tr,te) in enumerate(rcv.split(Xs,ys)):
        mdl=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1000)).fit(Xs.iloc[tr],ys[tr])
        rows.append(dict(sex=sex,fold=fo,AUROC_simple=roc_auc_score(ys[te],mdl.predict_proba(Xs.iloc[te])[:,1])))
R=pd.DataFrame(rows).merge(full[['sex','fold','AUROC']],on=['sex','fold']); R['d']=R.AUROC-R.AUROC_simple
R.to_csv(OUT+'C_simple_benchmark.csv',index=False)
for sex,d in R.groupby('sex'):
    k=25; v=d.d.var(ddof=1); n=(D['Sex [31]']==sex).sum(); t=d.d.mean()/np.sqrt((1/k+0.25)*v)
    print(sex,'simple LR AUROC %.3f±%.3f | LightGBM %.3f | diff %.3f P=%.2g'%(d.AUROC_simple.mean(),d.AUROC_simple.std(),d.AUROC.mean(),d.d.mean(),2*stats.t.sf(abs(t),k-1)))
