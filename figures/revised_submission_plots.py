"""Recreate revised manuscript figures from reported aggregate data.

No participant results are generated or re-estimated. Figure 2 reproduces the
historical weighted intervals, which require replacement after the data rerun.
Codex (GPT-6, OpenAI) assisted with this deterministic plotting code.
"""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
WORK=Path(__file__).resolve().parent
tables=[None]*12+[[['Predictor', 'AUROC, diagnosed diabetes', 'AUROC, undiagnosed diabetes', 'Leakage gap'], ['LDL direct', '0.686', '0.534', '0.152'], ['Cholesterol', '0.655', '0.537', '0.118'], ['Red blood cell (erythrocyte) distribution width', '0.584', '0.515', '0.069'], ['Albumin', '0.587', '0.536', '0.051'], ['Mean corpuscular haemoglobin concentration', '0.558', '0.512', '0.046'], ['Urea', '0.563', '0.53', '0.034'], ['Aspartate aminotransferase', '0.547', '0.522', '0.025'], ['Mean corpuscular haemoglobin', '0.575', '0.553', '0.022'], ['Haemoglobin concentration', '0.57', '0.553', '0.017'], ['Platelet count', '0.527', '0.516', '0.01'], ['Calcium', '0.539', '0.532', '0.007'], ['Waist circumference', '0.661', '0.654', '0.007'], ['Urate', '0.576', '0.57', '0.006'], ['Haematocrit percentage', '0.556', '0.556', '-0.001'], ['White blood cell (leukocyte) count', '0.597', '0.598', '-0.001'], ['Standing height', '0.535', '0.537', '-0.002'], ['C-reactive protein', '0.608', '0.61', '-0.002'], ['Phosphate', '0.521', '0.527', '-0.006'], ['Hip circumference', '0.597', '0.604', '-0.007'], ['Mean corpuscular volume', '0.566', '0.574', '-0.008'], ['Pulse rate, automated reading', '0.599', '0.607', '-0.008'], ['Mean platelet (thrombocyte) volume', '0.538', '0.546', '-0.008'], ['Alkaline phosphatase', '0.581', '0.594', '-0.012'], ['Body mass index (BMI)', '0.635', '0.649', '-0.013'], ['Weight', '0.61', '0.627', '-0.017'], ['HDL cholesterol', '0.634', '0.659', '-0.024'], ['Creatinine', '0.529', '0.561', '-0.032'], ['Diastolic blood pressure, automated reading', '0.559', '0.592', '-0.033'], ['Total bilirubin', '0.516', '0.55', '-0.034'], ['Red blood cell (erythrocyte) count', '0.532', '0.57', '-0.038'], ['Total protein', '0.504', '0.544', '-0.04'], ['Triglycerides', '0.599', '0.651', '-0.052'], ['Gamma glutamyltransferase', '0.586', '0.65', '-0.065'], ['Systolic blood pressure, automated reading', '0.529', '0.594', '-0.065'], ['Alanine aminotransferase', '0.534', '0.606', '-0.073']]]
# Scientific figures are deterministic plots of the supplied aggregate results.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(10,7.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,title,body):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',facecolor='#f2f5f7',edgecolor='#687887',linewidth=1))
 ax.text(x+.018,y+h-.03,title,va='top',weight='bold',fontsize=12)
 ax.text(x+.018,y+h-.085,body,va='top',fontsize=10.5,linespacing=1.45)
box(.025,.57,.445,.38,'A  UK Biobank recorded diagnosis',
    'E11 recorded by the end of 2010\nCases: 10,443; controls: 10,473\nTen sex–age strata; internal evaluation\n\nStricter baseline sensitivity sample\nCases: 8,669; total: 18,936')
box(.53,.57,.445,.38,'B  UK Biobank subsequent diagnosis',
    'Post-baseline diagnoses: 26,975\nBritish ethnicity: 22,254\nExclude HbA1c ≥48 mmol/mol: 6,441\nFinal cases: 15,813\nControls: 10,473 − 519 − 82 = 9,872\nInternal prognostic evaluation only')
box(.025,.05,.95,.39,'C  NHANES external diagnostic evaluation',
    'Sex-specific models trained using sample A: 35 harmonised predictors plus age\nNHANES 2017–March 2020: 4,757 participants aged 40–69 years\n\nDiagnosed diabetes: men 2,263 (462 cases); women 2,337 (393 cases)\nUndiagnosed diabetes: men 810 (62 cases); women 896 (65 cases)\nThe undiagnosed analysis uses participants with fasting glucose available.')
ax.annotate('',xy=(.25,.46),xytext=(.25,.56),arrowprops={'arrowstyle':'->','lw':1.5,'color':'#354b60'})
fig.savefig(WORK/'figure1.png',dpi=300,bbox_inches='tight');plt.close(fig)

external=[
['Diagnosed','Men','LightGBM','2,263 (462)','0.798 (0.773–0.820)','0.848 (0.824–0.872)'],
['Diagnosed','Men','Logistic regression','2,263 (462)','0.793 (0.767–0.815)','0.835 (0.813–0.856)'],
['Diagnosed','Women','LightGBM','2,337 (393)','0.793 (0.770–0.817)','0.828 (0.799–0.854)'],
['Diagnosed','Women','Logistic regression','2,337 (393)','0.772 (0.749–0.795)','0.796 (0.768–0.826)'],
['Diagnosed, White subgroup','Men','LightGBM','689 (120)','0.838 (0.798–0.875)','0.870 (0.838–0.897)'],
['Diagnosed, White subgroup','Men','Logistic regression','689 (120)','0.846 (0.803–0.884)','0.858 (0.830–0.888)'],
['Diagnosed, White subgroup','Women','LightGBM','687 (82)','0.853 (0.813–0.890)','0.855 (0.812–0.894)'],
['Diagnosed, White subgroup','Women','Logistic regression','687 (82)','0.812 (0.760–0.859)','0.832 (0.799–0.865)'],
['Undiagnosed','Men','LightGBM','810 (62)','0.670 (0.597–0.735)','0.703 (0.604–0.786)'],
['Undiagnosed','Men','Logistic regression','810 (62)','0.639 (0.570–0.715)','0.700 (0.600–0.780)'],
['Undiagnosed','Women','LightGBM','896 (65)','0.733 (0.667–0.794)','0.768 (0.692–0.847)'],
['Undiagnosed','Women','Logistic regression','896 (65)','0.716 (0.651–0.775)','0.769 (0.689–0.852)']]
fig,axes=plt.subplots(1,2,figsize=(10,4.6),sharex=True,sharey=True)
for ax,sex in zip(axes,['Men','Women']):
 dat=[r for r in external if r[1]==sex and 'subgroup' not in r[0]]
 for j,r in enumerate(dat):
  mean,lo,hi=[float(x) for x in re.findall(r'\d+\.\d+',r[5])]
  ax.errorbar(mean,3-j,xerr=[[mean-lo],[hi-mean]],fmt='o',color='#19577f',capsize=3,ms=5)
 ax.set_title(sex,weight='bold');ax.set_xlim(.50,.95);ax.set_ylim(-.5,3.5)
 ax.set_xticks([.50,.60,.70,.80,.90])
 ax.set_yticks([3,2,1,0],['Diagnosed\nLightGBM','Diagnosed\nRegression','Undiagnosed\nLightGBM','Undiagnosed\nRegression'])
 ax.axvline(.5,color='#999999',ls=':',lw=1);ax.grid(axis='x',alpha=.15)
fig.supxlabel('Reported survey-weighted AUROC (95% CI)',fontsize=11)
fig.tight_layout(w_pad=2.5);fig.savefig(WORK/'figure2.png',dpi=300,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(8,9))
data=tables[12][1:];labels=[r[0] for r in data];vals=[float(r[3]) for r in data]
ax.barh(range(len(data)),vals,color=['#9b5550' if v>0 else '#347c8c' for v in vals]);ax.invert_yaxis();ax.set_yticks(range(len(data)),labels,fontsize=7.8)
ax.axvline(0,color='#444444',lw=.8);ax.set_xlabel('AUROC for diagnosed minus undiagnosed diabetes');ax.tick_params(axis='y',length=0)
fig.tight_layout();fig.savefig(WORK/'supp_figure5.png',dpi=300,bbox_inches='tight');plt.close(fig)

