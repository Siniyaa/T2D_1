"""Figures 2 and 3 of the manuscript.

Figure 2  algorithm comparison and incremental value of the 69-variant genetic risk score,
          from results/CaseFinding_performance.csv and results/CaseFinding_grs.csv.
Figure 3  share of total LightGBM gain for the eight most important predictors per stratum,
          from results/prev_gain_importance.csv (written by code/extra.py).

Figure 1 is drawn by code/figure1_design.py, Figures 4 and 5 by code/calibration_dca.py --figures.
The incident-prediction panel that this script used to draw was removed with that analysis.
"""
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
# ---------- Figure 2 (manuscript Fig 2): algorithm comparison + 69-variant GRS
# Drawn from the recomputed case-finding run (code/case_finding.py), in which all five
# algorithms share the same 175 clinical predictors and the same folds. This supersedes the
# earlier panel built from B_summary.csv / GRS74_summary.csv, which used a different predictor
# set and the 74-variant score and is no longer reported.
try:
    from figure_style import apply_figure_style; apply_figure_style()
except Exception: pass
from scipy import stats as _st
AGE_CF=['<50','50-54','55-59','60-64','>=65']
P=pd.read_csv(O+'CaseFinding_performance.csv'); GR=pd.read_csv(O+'CaseFinding_grs.csv')
P['Age group']=pd.Categorical(P['Age group'],AGE_CF,ordered=True)
GR['Age group']=pd.Categorical(GR['Age group'],AGE_CF,ordered=True)
STY={'LightGBM':('#1f4e79','o','-'),'L1-penalised logistic regression':('#2a9d9d','s','-'),
     'Random forest':('#777777','^','-'),'Extremely randomised trees':('#9e9e9e','v','-'),
     'Decision tree':('#c4c4c4','D','-')}
LAB={'L1-penalised logistic regression':'Logistic regression','Extremely randomised trees':'Extra trees'}
gg=GR.sort_values(['Sex','Age group'],ascending=[False,True]).reset_index(drop=True)
se=(gg.difference/gg.t).abs(); tcrit=_st.t.ppf(0.975,24)
lo=gg.difference-tcrit*se; hi=gg.difference+tcrit*se
fig=plt.figure(figsize=(7.2,7.6))
gs=fig.add_gridspec(2,2,height_ratios=[1.0,1.15],hspace=0.45,wspace=0.22)
axA,axB=fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[1,:])
for ax,sex,ttl in ((axA,'Male','Men'),(axB,'Female','Women')):
    for alg in STY:
        d=P[(P.Sex==sex)&(P.Algorithm==alg)].sort_values('Age group')
        c,mk,ls=STY[alg]
        ax.errorbar(range(5),d.AUROC,yerr=d.AUROC_sd,color=c,marker=mk,ls=ls,ms=4.5,lw=1.5,
                    capsize=2.5,label=LAB.get(alg,alg))
    ax.set_xticks(range(5)); ax.set_xticklabels(['<50','50-54','55-59','60-64','\u226565'])
    ax.set_xlabel('Age group (years)'); ax.set_title(ttl,loc='left',fontsize=10); ax.set_ylim(0.77,0.93)
    for s_ in ('top','right'): ax.spines[s_].set_visible(False)
axA.set_ylabel('AUROC (mean \u00b1 SD)'); axB.tick_params(labelleft=False)
axA.legend(frameon=False,fontsize=7.5,loc='lower left')
y=np.arange(10); cols=['#1f4e79' if s=='Male' else '#c0504d' for s in gg.Sex]
axC.axvline(0,color='0.45',lw=1.0,zorder=0)
for i_ in range(10):
    axC.plot([lo[i_],hi[i_]],[y[i_],y[i_]],color=cols[i_],lw=1.5)
    axC.plot(gg.difference[i_],y[i_],'o',color=cols[i_],ms=5,
             mfc=cols[i_] if gg.P[i_]<0.05 else 'white')
axC.set_yticks(y)
axC.set_yticklabels([f"{s}, {a}".replace('>=65','\u226565') for s,a in zip(gg.Sex,gg['Age group'])],fontsize=8)
axC.invert_yaxis(); axC.set_ylim(9.9,-1.4)
axC.set_xlabel('Change in AUROC when the 69-variant genetic risk score is added')
axC.set_title('Incremental value of the genetic risk score',loc='left',fontsize=10)
axC.text(0.02,0.02,'filled marker: P < 0.05 (corrected resampled t-test)',transform=axC.transAxes,
         ha='left',va='bottom',fontsize=7.5,color='0.35')
for s_ in ('top','right'): axC.spines[s_].set_visible(False)
for ax,l in ((axA,'A'),(axB,'B'),(axC,'C')):
    ax.text(-0.13,1.07,l,transform=ax.transAxes,fontsize=11,fontweight='bold')
save(fig,'Figure2_algorithms_grs')
print('Figure 2 written | intervals excluding zero:',int(((lo>0)|(hi<0)).sum()),
      '| P<0.05:',int((gg.P<0.05).sum()))

# ---------- Figure 3 (manuscript Fig 3): share of LightGBM gain, top eight per stratum
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
save(fig,'Figure3_importance')
print('Figure 3 written | rows:',len(rows))
print('ok')
