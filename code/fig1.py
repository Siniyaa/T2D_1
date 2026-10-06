import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans"})
fig,ax=plt.subplots(figsize=(7.2,5.6)); ax.set_xlim(0,100); ax.set_ylim(13,78); ax.axis('off')
C={'g':'#dbe9f6','p':'#fbe3d6','a':'#e2f0d9','b':'#ece2f4'}
def box(x,y,w,h,t,b,c,bs=5.7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.2',fc=c,ec='#404040',lw=0.8))
    ax.text(x+w/2,y+h-1.5,t,ha='center',va='top',fontsize=6.6,fontweight='bold')
    ax.text(x+w/2,y+h-4.6,b,ha='center',va='top',fontsize=bs,linespacing=1.32)
def arr(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=0.9,color='#404040'))
ax.text(1,76.5,'Genotype data',fontsize=8,fontweight='bold',color='#1f4e79')
box(1,61,22,13,'UK Biobank array','Directly genotyped calls\n(Field 22418)\nBritish participants',C['g'])
box(27,61,22,13,'Genotype QC','Individual missingness < 0.2\nHWE P > 0.001\nHeterozygosity outlier\nremoved',C['g'])
box(53,61,22,13,'SNP selection','GWAS Catalog: 2,787 SNPs\n200 present on array\n105 with non-zero variance',C['g'])
box(79,61,20,13,'Genetic features','105 SNP genotypes\n(0/1/2)',C['g'])
for a in (23.6,49.6,75.6): arr(a,67.5,a+2.8,67.5)
ax.text(1,57,'Phenotype data (baseline visit, 2006–2010)',fontsize=8,fontweight='bold',color='#9c3d10')
box(1,42,22,13,'UK Biobank phenotypes','Anthropometry, BP, spirometry,\nbiochemistry, haematology,\nlifestyle, family history',C['p'])
box(27,42,22,13,'Predictor curation','175 predictors\nExcluded: HbA1c, glucose,\nself-rated health',C['p'])
box(53,42,22,13,'Outcome','T2D: ICD-10 E11\n(hospital records)\nDays from baseline\nto diagnosis',C['p'])
box(79,42,20,13,'Integrated dataset','Phenotypes + SNPs\nStrata: sex × age\n(<50 to ≥65 years)',C['a'])
for a in (23.6,49.6,75.6): arr(a,48.5,a+2.8,48.5)
arr(89,60.4,89,55.6)
ax.text(1,38,'Analyses',fontsize=8,fontweight='bold',color='#375623')
box(1,15,47,21,'1. Prevalent T2D (case-finding)',
 '10,443 cases (E11 recorded ≤2010)\n10,473 controls (stratified downsampling\nby age, sex, family history)\n\n10 sex × age strata\n5 algorithms (LightGBM, LR, RF, ET, DT)\nRepeated stratified 5-fold CV (5 × 5)\nLightGBM with vs without SNPs\n\nAUROC, AUPRC, accuracy, sensitivity,\nspecificity; corrected resampled t-test',C['a'],bs=6.0)
box(52,15,47,21,'2. Incident T2D (risk prediction)',
 '15,813 cases diagnosed after baseline\n(median 7.5 years)\n9,872 controls never diagnosed\nBaseline HbA1c ≥ 48 mmol/mol excluded\n\nSex-specific LightGBM; 5 × 5-fold CV\nAUROC by time to diagnosis\n(<5, 5–10, ≥10 years)\nBenchmark: age, BMI, waist, SBP,\nfamily history (logistic regression)',C['b'],bs=6.0)
arr(89,41.4,75.5,36.6); arr(89,41.4,24.5,36.6)
import os; os.makedirs('../figures',exist_ok=True)
for ext in ('png','pdf','tiff'):
    kw={'dpi':600} if ext!='pdf' else {}
    if ext=='tiff': kw['pil_kwargs']={'compression':'tiff_lzw'}
    fig.savefig(f'../figures/Figure1_workflow.{ext}',bbox_inches='tight',**kw)
