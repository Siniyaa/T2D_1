import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
O=os.environ.get('T2D_OUT_DIR','../results/'); U=pd.read_pickle(O+'ukb_harm.pkl'); y=pd.read_pickle(O+'ukb_y.pkl'); N=pd.read_pickle(O+'nhanes.pkl')
F=[c for c in U.columns if c!='sex']; rng=np.random.default_rng(42)
def boot(yv,p,B=1000):
    yv=np.asarray(yv); p=np.asarray(p); s=[]
    for _ in range(B):
        i=rng.integers(0,len(yv),len(yv))
        if yv[i].min()!=yv[i].max(): s.append(roc_auc_score(yv[i],p[i]))
    return roc_auc_score(yv,p),np.percentile(s,2.5),np.percentile(s,97.5)
mk={'LightGBM':lambda: lgb.LGBMClassifier(learning_rate=0.09,random_state=42,verbose=-1),
    'Logistic regression':lambda: make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=2000))}
res=[]
for sex in ['Male','Female']:
    Us=U[U.sex==sex][F]; ys=y[U.sex==sex].values; Ns=N[N.sex==sex]
    for name,m in mk.items():
        # internal 5-fold CV of the harmonised model
        cv=[roc_auc_score(ys[te],m().fit(Us.iloc[tr],ys[tr]).predict_proba(Us.iloc[te])[:,1]) for tr,te in StratifiedKFold(5,shuffle=True,random_state=42).split(Us,ys)]
        model=m().fit(Us,ys); p=pd.Series(model.predict_proba(Ns[F])[:,1],index=Ns.index)
        r=dict(sex=sex,model=name,internal_cv=np.mean(cv),internal_sd=np.std(cv,ddof=1))
        # A: diagnosed T2D vs no diabetes (controls with FPG>=7 excluded)
        a=Ns[(Ns.diagnosed==1)|((Ns.diagnosed==0)&~(Ns.fpg>=7.0))]
        r['A_n'],r['A_cases']=len(a),int(a.diagnosed.sum()); r['A_auc'],r['A_lo'],r['A_hi']=boot(a.diagnosed,p[a.index])
        w=a[a.eth==3]; r['A_white_n'],r['A_white_cases']=len(w),int(w.diagnosed.sum()); r['A_white'],r['A_wlo'],r['A_whi']=boot(w.diagnosed,p[w.index])
        # B: undiagnosed diabetes (FPG>=7.0) vs no diabetes, fasting subsample
        b=Ns[(Ns.diagnosed==0)&Ns.fpg.notna()]; yb=(b.fpg>=7.0).astype(int)
        r['B_n'],r['B_cases']=len(b),int(yb.sum()); r['B_auc'],r['B_lo'],r['B_hi']=boot(yb,p[b.index])
        res.append(r); print(sex,name,'done',flush=True)
R=pd.DataFrame(res); R.to_csv(O+'NHANES_validation.csv',index=False)
pd.set_option('display.width',250); print(R.round(3).to_string(index=False))
