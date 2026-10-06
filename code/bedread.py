import numpy as np, pandas as pd
def read_bed(prefix, fam_path):
    fam=pd.read_csv(fam_path,sep=r'\s+',header=None); n=len(fam)
    bim=pd.read_csv(prefix+'.bim',sep='\t',header=None,names=['chr','id','cm','pos','a1','a2'])
    raw=np.fromfile(prefix+'.bed',dtype=np.uint8); assert raw[0]==0x6c and raw[1]==0x1b and raw[2]==1
    bpv=(n+3)//4; M=raw[3:].reshape(len(bim),bpv)
    codes=np.stack([(M>>(2*k))&3 for k in range(4)],axis=2).reshape(len(bim),bpv*4)[:,:n]
    lut=np.array([2.,np.nan,1.,0.])   # PLINK: 00=hom A1, 01=missing, 10=het, 11=hom A2 -> count of A1
    G=pd.DataFrame(lut[codes].T,index=fam[1].astype(str).values,columns=bim.id.values)
    return G,bim
