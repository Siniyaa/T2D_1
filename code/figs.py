import pandas as pd, numpy as np, re, matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
import os
O=os.environ.get('T2D_OUT_DIR','../results/'); F='../figures/'; import os; os.makedirs(F,exist_ok=True)
AGES=['<50','50–54','55–59','60–64','≥65']
def save(fig,name):
    for ext in ('png','pdf','tiff'):
        kw={'dpi':600} if ext!='pdf' else {}
        if ext=='tiff': kw['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(F+name+'.'+ext,bbox_inches='tight',**kw)
    plt.close(fig)
# ---------- labels / domains
PRETTY={'30780':'LDL cholesterol','30690':'Total cholesterol','30760':'HDL cholesterol','30870':'Triglycerides','30640':'Apolipoprotein B','30630':'Apolipoprotein A',
 '30730':'GGT','30620':'Alanine aminotransferase','30650':'Aspartate aminotransferase','30610':'Alkaline phosphatase','30840':'Total bilirubin','30660':'Direct bilirubin','30600':'Albumin','30860':'Total protein',
 '30700':'Creatinine','30720':'Cystatin C','30880':'Urate','30670':'Urea','30510':'Urinary creatinine','30520':'Urinary potassium','30530':'Urinary sodium',
 '30830':'SHBG','30850':'Testosterone','30770':'IGF-1','30890':'Vitamin D','30680':'Calcium','30810':'Phosphate','30710':'C-reactive protein',
 '102':'Pulse rate','4079':'Diastolic BP','4080':'Systolic BP','3062':'FVC','3063':'FEV1','3064':'Peak expiratory flow','46':'Grip strength (left)','47':'Grip strength (right)',
 '48':'Waist circumference','49':'Hip circumference','50':'Standing height','21001':'BMI','23104':'BMI (impedance)','21002':'Weight','23098':'Weight (impedance)','23099':'Body fat %',
 '23100':'Whole-body fat mass','23101':'Whole-body fat-free mass','23127':'Trunk fat %','23128':'Trunk fat mass','21003':'Age','1807':"Father's age at death"}
def fid(n): m=re.search(r'(\d+)$',n); return m.group(1) if m else ''
def label(n):
    if n.startswith('rs'): return n
    return PRETTY.get(fid(n), re.sub(r'\s+\d+$','',n).replace('  ',' / '))
def domain(n):
    i=fid(n)
    if n.startswith('rs'): return 'Genetic'
    if i in {'30690','30760','30780','30870','30630','30640'}: return 'Lipids'
    if i in {'30700','30720','30880','30670','30510','30520','30530'}: return 'Renal'
    if i in {'30600','30610','30620','30650','30660','30730','30840','30860'}: return 'Liver'
    if i in {'30680','30770','30830','30850','30890','30810'}: return 'Endocrine/mineral'
    if i.startswith('300') or i.startswith('301') or i.startswith('302') or i=='30710': return 'Haematology/inflammation'
    if i in {'102','4079','4080','3062','3063','3064','46','47'}: return 'Cardiorespiratory'
    if i in {'48','49','50','51','20015','21001','21002'} or i.startswith('231') or i.startswith('2309') or i.startswith('2310'): return 'Anthropometry'
    if i=='21003': return 'Demographic'
    return 'Lifestyle/family history'
COL={'Lipids':'#1f4e79','Renal':'#2e8b57','Liver':'#c0504d','Endocrine/mineral':'#8064a2','Haematology/inflammation':'#7f7f7f',
     'Cardiorespiratory':'#e69f00','Anthropometry':'#4bacc6','Demographic':'#333333','Lifestyle/family history':'#9c6b30','Genetic':'#d62728'}
# ---------- Figure 2: CV performance + genetics
S=pd.read_csv(O+'B_summary.csv'); G=pd.read_csv(O+'B_genetic_comparison.csv')
MOD=[('LightGBM','#1f4e79','o'),('Logistic regression','#e69f00','s'),('Random forest','#2e8b57','^'),('Extremely randomised trees','#8064a2','v'),('Decision tree','#9e9e9e','D')]
fig,ax=plt.subplots(1,3,figsize=(7.2,2.7),gridspec_kw={'width_ratios':[1,1,1.05],'wspace':0.55})
x=np.arange(5)
for k,sex in enumerate(['Male','Female']):
    a=ax[k]
    for j,(m,c,mk) in enumerate(MOD):
        d=S[(S.sex==sex)&(S.model==m)].set_index('age').loc[AGES]
        a.errorbar(x+(j-2)*0.07,d.AUROC_mean,yerr=d.AUROC_std,color=c,marker=mk,ms=3.5,lw=1,capsize=1.5,label=m)
    a.set_xticks(x,AGES,fontsize=6.8,rotation=35,ha='right'); a.set_ylim(0.74,0.94); a.set_xlabel('Age group (years)'); a.grid(axis='y',lw=0.3,alpha=0.5)
    a.set_title(f"{'AB'[k]}  {sex}s",loc='left',fontweight='bold',fontsize=8.5)
ax[0].set_ylabel('AUROC (mean ± SD, 25 folds)')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,fontsize=6.8,frameon=False,loc='lower center',ncol=5,bbox_to_anchor=(0.45,-0.16))
a=ax[2]; GR=pd.read_csv(O+'GRS74_summary.csv')
for k,(sex,c) in enumerate([('Male','#1f4e79'),('Female','#c0504d')]):
    d=GR[GR.sex==sex].set_index('age').loc[AGES]
    a.errorbar(x+(k-0.5)*0.18,d.dAUROC,yerr=d.dSD,fmt='o',color=c,ms=4,capsize=2,lw=1,label=sex+'s')
a.axhline(0,color='black',lw=0.7); a.set_xticks(x,AGES,fontsize=6.8,rotation=35,ha='right'); a.set_ylim(-0.03,0.03)
a.set_xlabel('Age group (years)'); a.set_ylabel('ΔAUROC (clinical + GRS − clinical)'); a.grid(axis='y',lw=0.3,alpha=0.5)
a.legend(fontsize=6.5,frameon=False,loc='upper left'); a.set_title('C  Genetic risk score',loc='left',fontweight='bold',fontsize=8.5)
a.text(0.98,0.03,'All P ≥ 0.06',transform=a.transAxes,ha='right',fontsize=6.5,style='italic')
save(fig,'Figure2_performance')
# ---------- Figure 3: prevalent gain heatmap
I=pd.read_csv(O+'prev_gain_importance.csv',index_col=0)
top=set()
for c in I.columns: top|=set(I[c].nlargest(8).index)
DORD=list(COL)
rows=sorted(top,key=lambda n:(DORD.index(domain(n)),-I.loc[n].sum()))
fig,axs=plt.subplots(1,2,figsize=(7.2,0.26*len(rows)+1.6),gridspec_kw={'wspace':0.04})
cmap=LinearSegmentedColormap.from_list('b',['#deebf7','#08306b']); cmap.set_bad('#f4f4f4')
vmax=np.nanmax(I.loc[rows].values)
for k,sex in enumerate(['Male','Female']):
    a=axs[k]; M=I.loc[rows,[f'{sex}|{g}' for g in AGES]].values.copy()
    mask=np.zeros_like(M,bool)
    for j in range(5):
        col=I[f'{sex}|{AGES[j]}']; t8=set(col.nlargest(8).index)
        mask[:,j]=[r not in t8 for r in rows]
    im=a.imshow(np.ma.masked_array(M,mask),cmap=cmap,vmin=0,vmax=vmax,aspect='auto')
    for i in range(len(rows)):
        for j in range(5):
            if not mask[i,j]: a.text(j,i,f'{M[i,j]:.1f}',ha='center',va='center',fontsize=6.3,color='white' if M[i,j]>vmax*0.55 else 'black')
    a.set_xticks(range(5),AGES,fontsize=7.5); a.xaxis.tick_top(); a.set_xlabel('Age group (years)',fontsize=7.5); a.xaxis.set_label_position('top')
    a.set_yticks(range(len(rows)),[label(r) if k==0 else '' for r in rows],fontsize=7); a.tick_params(axis='y',length=0,pad=24 if k==0 else 0)
    a.set_xticks(np.arange(-.5,5),minor=True); a.set_yticks(np.arange(-.5,len(rows)),minor=True); a.grid(which='minor',color='white',lw=1.2); a.tick_params(which='minor',length=0)
    for s in a.spines.values(): s.set_visible(False)
    a.set_title(f"{'AB'[k]}  {sex}s",loc='left',fontweight='bold',fontsize=9,pad=6)
for i,r in enumerate(rows): axs[0].add_patch(plt.Rectangle((-0.95,i-0.5),0.35,1,color=COL[domain(r)],clip_on=False))
axs[0].set_xlim(-0.5,4.5)
cb=fig.colorbar(im,ax=axs,orientation='horizontal',fraction=0.025,pad=0.02,aspect=45); cb.set_label('Share of total LightGBM gain (%); grey = outside top 8 in that stratum',fontsize=7); cb.ax.tick_params(labelsize=6.5)
used=[d for d in DORD if any(domain(r)==d for r in rows)]
fig.legend(handles=[Patch(color=COL[d],label=d) for d in used],loc='lower center',ncol=4,frameon=False,fontsize=7,bbox_to_anchor=(0.5,0.035))
save(fig,'Figure3_importance_prevalent')
# ---------- Figure 4: incident
H=pd.read_csv(O+'C_incident_horizon.csv'); C=pd.read_csv(O+'C_incident_folds.csv'); B=pd.read_csv(O+'C_simple_benchmark.csv')
Ii=pd.read_csv(O+'C_incident_importance.csv',index_col=0)
fig=plt.figure(figsize=(7.2,3.6)); gs=fig.add_gridspec(1,3,width_ratios=[1.25,1,1],wspace=0.95)
a=fig.add_subplot(gs[0]); hz=['<5 years','5–10 years','≥10 years']
for k,(sex,c) in enumerate([('Male','#1f4e79'),('Female','#c0504d')]):
    m=[C[C.sex==sex].AUROC.mean()]+[H[(H.sex==sex)&(H.horizon==h)].AUROC.mean() for h in hz]
    s=[C[C.sex==sex].AUROC.std()]+[H[(H.sex==sex)&(H.horizon==h)].AUROC.std() for h in hz]
    a.errorbar(np.arange(4)+(k-0.5)*0.15,m,yerr=s,fmt='o-',color=c,ms=4,capsize=2,lw=1,label=sex+'s')
    a.errorbar([4+(k-0.5)*0.15],[B[B.sex==sex].AUROC_simple.mean()],yerr=[B[B.sex==sex].AUROC_simple.std()],fmt='s',mfc='white',color=c,ms=4,capsize=2)
a.set_xticks(range(5),['All','<5 y','5–10 y','≥10 y','Simple\nmodel'],fontsize=6.5,rotation=45,ha='right'); a.set_ylim(0.74,0.9)
a.set_ylabel('AUROC (mean ± SD)'); a.grid(axis='y',lw=0.3,alpha=0.5); a.axvline(3.5,color='grey',lw=0.5,ls=':')
a.legend(fontsize=6.5,frameon=False,loc='lower left'); a.set_title('A  Incident T2D',loc='left',fontweight='bold',fontsize=8.5)
for k,sex in enumerate(['Male','Female']):
    a=fig.add_subplot(gs[k+1]); v=Ii[[c for c in Ii.columns if c.startswith(sex)]].mean(axis=1); v=(v/v.sum()*100).nlargest(12)[::-1]
    a.barh(range(len(v)),v.values,color=[COL[domain(n)] for n in v.index]); a.set_yticks(range(len(v)),[label(n) for n in v.index],fontsize=6.8)
    for i,val in enumerate(v.values): a.text(val+0.3,i,f'{val:.1f}',va='center',fontsize=6)
    a.set_xlabel('Share of total gain (%)',fontsize=7); a.set_xlim(0,v.max()*1.3); a.tick_params(axis='x',labelsize=6.5)
    a.set_title(f"{'BC'[k]}  Top predictors, {sex.lower()}s",loc='left',fontweight='bold',fontsize=8.5)
save(fig,'Figure4_incident')
print('ok')
