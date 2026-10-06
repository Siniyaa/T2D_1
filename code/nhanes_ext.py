import re
def fid(n):
    m=re.search(r'(\d+)$',n); return m.group(1) if m else ''
import numpy as np, pandas as pd, glob, re, json
exec(open('t2d_pipeline.py').read().split("if __name__")[0])
import os
E=os.environ.get('NHANES_DIR','../data/nhanes/')   # NHANES 2017-March 2020 XPT files (public, CDC)
def xpt(name):
    p=[f for f in glob.glob(E+'**/'+name+'.xpt',recursive=True)+glob.glob(E+'**/'+name+'.XPT',recursive=True)][0]
    return pd.read_sas(p,format='xport').set_index('SEQN')
demo,bmx,bio,cbc,hdl,tg,glu,diq,bp,crp=[xpt(n) for n in ['P_DEMO','P_BMX','P_BIOPRO','P_CBC','P_HDL','P_TRIGLY','P_GLU','P_DIQ','P_BPXO','P_HSCRP']]
N=pd.DataFrame(index=demo.index)
N['age']=demo.RIDAGEYR; N['sex']=demo.RIAGENDR.map({1:'Male',2:'Female'}); N['eth']=demo.RIDRETH3
mean3=lambda d,p: d[[c for c in d.columns if re.fullmatch(p+r'[123]',c)]].mean(axis=1)
# UK Biobank field ID -> NHANES variable (SI units)
MAP={'48':bmx.BMXWAIST,'49':bmx.get('BMXHIP'),'21001':bmx.BMXBMI,'21002':bmx.BMXWT,'50':bmx.BMXHT,
 '4080':mean3(bp,'BPXOSY'),'4079':mean3(bp,'BPXODI'),'102':mean3(bp,'BPXOPLS'),
 '30620':bio.LBXSATSI,'30650':bio.LBXSASSI,'30610':bio.LBXSAPSI,'30730':bio.LBXSGTSI,'30880':bio.LBDSUASI,'30700':bio.LBDSCRSI,
 '30670':bio.LBDSBUSI,'30600':bio.LBDSALSI,'30860':bio.LBDSTPSI,'30840':bio.LBDSTBSI,'30680':bio.LBDSCASI,'30810':bio.LBDSPHSI,
 '30690':bio.LBDSCHSI,'30870':bio.LBDSTRSI,'30760':hdl.LBDHDDSI,'30780':tg.LBDLDLSI,'30710':crp.LBXHSCRP,
 '30000':cbc.LBXWBCSI,'30010':cbc.LBXRBCSI,'30020':cbc.LBXHGB,'30030':cbc.LBXHCT,'30040':cbc.LBXMCVSI,'30070':cbc.LBXRDW,
 '30080':cbc.LBXPLTSI,'30120':cbc.LBDLYMNO,'30130':cbc.LBDMONO,'30140':cbc.LBDNENO}
MAP={k:v for k,v in MAP.items() if v is not None}
for k,v in MAP.items(): N['f'+k]=v.reindex(N.index)
# outcomes
dx=diq.DIQ010.reindex(N.index); onset=diq.DID040.reindex(N.index); insulin=diq.DIQ050.reindex(N.index)
fpg=glu.LBDGLUSI.reindex(N.index)
probT1=(dx==1)&(onset<30)&(insulin==1)
N['diagnosed']=np.where(probT1,np.nan,np.where(dx==1,1,np.where(dx.isin([2,3]),0,np.nan)))
N['fpg']=fpg
N=N[(N.age>=40)&(N.age<=69)]
# ---- UK Biobank training set with harmonised features
X,y,cat,snps,_=load_prevalent()
col={fid(c):c for c in X.columns}; feats=[k for k in MAP if k in col]
print('harmonised predictors:',len(feats))
U=X[[col[k] for k in feats]].copy(); U.columns=['f'+k for k in feats]; U['age']=X['Age when attended assessment centre 21003'].values
U['sex']=X['Sex 31'].values
comp=pd.DataFrame({'UKB_median':U.drop(columns='sex').median(),'NHANES_median':N[U.drop(columns='sex').columns].median()}).round(2)
print(comp.to_string())
U.to_pickle(OUT+'ukb_harm.pkl'); y.to_pickle(OUT+'ukb_y.pkl'); N.to_pickle(OUT+'nhanes.pkl')
print('NHANES 40-69: n=',len(N),'| diagnosed T2D:',int((N.diagnosed==1).sum()),'| no diabetes:',int((N.diagnosed==0).sum()),'| fasting glucose available:',int(N.fpg.notna().sum()))
