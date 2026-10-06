import sys, numpy as np, pandas as pd
exec(open('t2d_pipeline.py').read().split("if __name__")[0])
f=pd.read_csv(DIR+'medical_history_fut.tsv',sep='\t',index_col=0,low_memory=False)
m=pd.read_csv(DIR+'medical_history_3.tsv',sep='\t',index_col=0,low_memory=False)
hb='Glycated haemoglobin (HbA1c) [30750]'
f=f[(f['Ethnic background [21000]']=='British')&~(f[hb]>=48)]
c=m[(~m.HasT2D.astype(bool))&~m.index.isin(pd.read_csv(DIR+'medical_history_fut.tsv',sep='\t',index_col=0,usecols=[0]).index)&~(m[hb]>=48)]
f=f.drop(columns=['Days until E11 diagnosis']); c=c.drop(columns=['HasT2D'])
common=[k for k in f.columns if k in c.columns]
D=pd.concat([f[common],c[common]]); y=pd.Series(np.r_[np.ones(len(f),int),np.zeros(len(c),int)],index=D.index)
fh=D[['Father had diabetes [20107]','Mother had diabetes [20110]','Siblings had diabetes [20111]']].fillna(False).astype(bool).any(axis=1)
key=D['Sex [31]'].astype(str)+'|'+D['Age when attended assessment centre [21003]'].astype(int).astype(str)+'|'+fh.astype(str)
rng=np.random.default_rng(SEED); keep=[]
for k,idx in key.groupby(key).groups.items():
    ca=[i for i in idx if y[i]==1]; co=[i for i in idx if y[i]==0]; n=min(len(ca),len(co))
    if n: keep+=list(rng.choice(ca,n,replace=False))+list(rng.choice(co,n,replace=False))
D,y=D.loc[keep],y[keep]; X,_=preprocess(D)
print(f'matched set: {len(y):,} ({int(y.sum()):,} cases, {int((1-y).sum()):,} controls)',flush=True)
rcv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED); rows=[]
for sex in ['Male','Female']:
    idx=X.index[X['Sex 31']==sex]; Dx=X.loc[idx].drop(columns=['Sex 31']); yy=y[idx].values
    Dna=Dx.drop(columns=['Age when attended assessment centre 21003'])
    for fo,(tr,te) in enumerate(rcv.split(Dx,yy)):
        a=roc_auc_score(yy[te],lgbm().fit(Dx.iloc[tr],yy[tr]).predict_proba(Dx.iloc[te])[:,1])
        b=roc_auc_score(yy[te],lgbm().fit(Dna.iloc[tr],yy[tr]).predict_proba(Dna.iloc[te])[:,1])
        age_only=roc_auc_score(yy[te],Dx['Age when attended assessment centre 21003'].values[te])
        rows.append(dict(sex=sex,fold=fo,n=len(yy),cases=int(yy.sum()),AUROC=a,AUROC_no_age=b,age_alone=age_only))
    print(sex,'done',flush=True)
pd.DataFrame(rows).to_csv(OUT+'INC_matched.csv',index=False)
