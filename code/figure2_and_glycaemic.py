"""Figure 2 inputs and the glycaemic-comparison missingness, on the current population."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, re, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import (load_prevalent, load_incident_cases, model_matrix, OUT, SEX, AGE_U, HBA, GLU, SEED, DROP_PRED)
SIGKEY=[SEX,AGE_U,"Standing height [50]","Waist circumference [48]","Body mass index (BMI) [21001]"]
safe=lambda cs:[re.sub(r"[^0-9A-Za-z]+","_",c).strip("_") for c in cs]

def population():
    prev=load_prevalent(); cases=load_incident_cases()
    ctrl=prev[~prev.HasT2D]; ctrl=ctrl[~(pd.to_numeric(ctrl[HBA],errors="coerce")>=48)]
    s=lambda d: d[SIGKEY].astype(str).agg("|".join,axis=1)
    ctrl=ctrl[~s(ctrl).isin(set(s(cases)))]
    pred=[c for c in prev.columns if c in set(cases.columns) and c not in DROP_PRED]
    return pd.concat([cases.assign(y=1),ctrl.assign(y=0)],ignore_index=True), pred

def main():
    df,pred=population(); imp_rows=[]; miss_rows=[]; comp_rows=[]
    for sex in ("Male","Female"):
        d=df[df[SEX]==sex].reset_index(drop=True); y=d.y.values
        X,cat_cols=model_matrix(d,pred)
        m=lgb.LGBMClassifier(learning_rate=0.09,random_state=SEED,verbose=-1).fit(X,y,categorical_feature=cat_cols)
        g=m.booster_.feature_importance(importance_type="gain"); g=100*g/g.sum()
        for name,val in sorted(zip(pred,g),key=lambda t:-t[1])[:12]:
            imp_rows.append({"Sex":sex,"Predictor":name,"Gain, %":round(float(val),2)})
        # missingness of the two assays in this population, by outcome
        h=pd.to_numeric(d[HBA],errors="coerce"); gl=pd.to_numeric(d[GLU],errors="coerce")
        for lab,msk in [("Incident cases",y==1),("Controls",y==0)]:
            miss_rows.append({"Sex":sex,"Group":lab,"n":int(msk.sum()),
                "HbA1c missing, n (%)":f"{int(h[msk].isna().sum()):,} ({100*h[msk].isna().mean():.1f})",
                "Glucose missing, n (%)":f"{int(gl[msk].isna().sum()):,} ({100*gl[msk].isna().mean():.1f})",
                "Either missing, n (%)":f"{int((h[msk].isna()|gl[msk].isna()).sum()):,} ({100*(h[msk].isna()|gl[msk].isna()).mean():.1f})",
                "Both measured, n (%)":f"{int((h[msk].notna()&gl[msk].notna()).sum()):,} ({100*(h[msk].notna()&gl[msk].notna()).mean():.1f})"})
        # complete-assay sensitivity analysis: restrict to records with BOTH assays measured
        ok=(h.notna()&gl.notna()).values
        ds=d[ok].reset_index(drop=True); ys=y[ok]
        Xs,cat_s=model_matrix(ds,pred)
        Xg=ds[[HBA,GLU]].apply(pd.to_numeric,errors="coerce"); Xg.columns=["hba1c","glucose"]; Xg.index=Xs.index
        Xe=pd.concat([Xs,Xg],axis=1)
        cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED)
        pools={k:np.zeros(len(ys)) for k in ("primary","glyc","ext")}
        for tr,te in cv.split(Xs,ys):
            mm=lgb.LGBMClassifier(learning_rate=0.09,random_state=SEED,verbose=-1).fit(Xs.iloc[tr],ys[tr],categorical_feature=cat_s)
            pools["primary"][te]+=mm.predict_proba(Xs.iloc[te])[:,1]
            pl=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),
                         ("lr",LogisticRegression(max_iter=5000,random_state=SEED))]).fit(Xg.iloc[tr],ys[tr])
            pools["glyc"][te]+=pl.predict_proba(Xg.iloc[te])[:,1]
            me=lgb.LGBMClassifier(learning_rate=0.09,random_state=SEED,verbose=-1).fit(Xe.iloc[tr],ys[tr],categorical_feature=cat_s)
            pools["ext"][te]+=me.predict_proba(Xe.iloc[te])[:,1]
        for k in pools: pools[k]/=5.0
        rng=np.random.default_rng(7); dg,de=[],[]
        for _ in range(1000):
            i=rng.integers(0,len(ys),len(ys))
            if len(np.unique(ys[i]))>1:
                b=roc_auc_score(ys[i],pools["primary"][i])
                dg.append(roc_auc_score(ys[i],pools["glyc"][i])-b)
                de.append(roc_auc_score(ys[i],pools["ext"][i])-b)
        base=roc_auc_score(ys,pools["primary"])
        for lab,key,dd_ in [("Primary, 175 non-glycaemic inputs","primary",None),
                            ("Glycaemic only (HbA1c, glucose)","glyc",dg),
                            ("Primary + HbA1c + glucose","ext",de)]:
            a=roc_auc_score(ys,pools[key])
            comp_rows.append({"Sex":sex,"Model":lab,"n (cases)":f"{len(ys):,} ({int(ys.sum()):,})",
                "AUROC, pooled":round(a,3),
                "Difference vs primary (95% CI)":("reference" if dd_ is None else
                   f"{a-base:+.3f} ({np.percentile(dd_,2.5):.3f} to {np.percentile(dd_,97.5):.3f})")})
    for fr,name in [(imp_rows,"Figure2_importance.csv"),(miss_rows,"Glycaemic_missingness.csv"),
                    (comp_rows,"Glycaemic_complete_assay.csv")]:
        pd.DataFrame(fr).to_csv(os.path.join(OUT,name),index=False); print("wrote",name)
    print(pd.DataFrame(miss_rows).to_string(index=False))
    print(pd.DataFrame(comp_rows).to_string(index=False))

if __name__=="__main__": main()
