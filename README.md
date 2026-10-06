# Non-glycaemic machine learning for type 2 diabetes in the UK Biobank

Code accompanying the manuscript:

> **Routine clinical data, not genetics, identify and predict type 2 diabetes without glycaemic testing: a machine learning study in the UK Biobank**
> Nedunkulathil SP, Henschel A, Zalloua P, Aung Z, Kamarul Azman S. *(submitted)*

The study develops sex- and age-specific machine learning models that identify existing (prevalent) type 2 diabetes (T2D) and predict future (incident) T2D **without glucose or HbA1c**, tests the added value of a genetic risk score, and externally validates the approach in NHANES 2017–March 2020.

## Data availability

**UK Biobank data are not included in this repository.** They are available to approved researchers on application to the UK Biobank (https://www.ukbiobank.ac.uk). This repository contains code, SNP lists and aggregate (non-identifiable) results only.

NHANES 2017–March 2020 public-use files can be downloaded from the CDC: https://wwwn.cdc.gov/nchs/nhanes/

## Repository structure

```
code/            analysis scripts (Python)
results/         aggregate results reported in the manuscript (no participant-level data)
snp_lists/       variant lists used for the genetic analyses
prs_scripts/     optional scripts to compute the PGS000014 polygenic score on the UK Biobank RAP
run_all.sh       runs the full analysis in order
requirements.txt Python dependencies
```

## Input files (place in `data/`, not shared)

| File | Content |
|---|---|
| `medical_history_3.tsv` | Prevalent analysis: baseline phenotypes and `HasT2D` (10,443 cases, 10,473 controls) |
| `medical_history_fut.tsv` | Participants diagnosed with T2D after baseline, with `Days until E11 diagnosis` |
| `T2D_geno.tsv` | Genotypes of the 105 GWAS Catalog array variants |
| `T2D_4.bed/.bim`, `T2D_12.bed/.bim/.fam` | PLINK files for the genetic risk score variant sets |
| `PCA.bed/.bim/.fam` | LD-pruned array variants for principal component analysis |
| `nhanes/*.xpt` | NHANES 2017–March 2020 files: P_DEMO, P_BMX, P_BPXO, P_BIOPRO, P_CBC, P_HDL, P_TRIGLY, P_HSCRP, P_GLU, P_DIQ |

Paths can be changed with environment variables `T2D_DATA_DIR`, `T2D_OUT_DIR` and `NHANES_DIR`.

## Analysis steps

| Step | Script | Manuscript |
|---|---|---|
| Reproduction of the original single train–test split | `t2d_pipeline.py A` | Suppl. Table S2 |
| Repeated 5 × 5-fold cross-validation, five algorithms (one sex–age group at a time, 0–9) | `t2d_pipeline.py B <k>` | Tables 1–2, Suppl. Tables S3–S5 |
| Incident T2D prediction and time-to-diagnosis analysis | `t2d_pipeline.py C` | Table 3 |
| Feature importance and simple clinical benchmark | `extra.py` | Figures 3–4 |
| Sensitivity analysis (stricter case/control definitions) | `sens.py` | Suppl. Table S9 |
| Matched incident sensitivity analysis | `inc_matched.py` | Table 3 |
| QC and LD of the 105 GWAS Catalog variants | `snp_qc.py` | Section 3.5 |
| Six quality-controlled variants and weighted GRS | `geno2.py <k>` | Suppl. Table S6 |
| 74-variant genetic risk score: QC, LD pruning, TCF7L2 validation | `grs_qc.py` | Section 3.5 |
| GRS cross-validation | `grs_cv.py <k>` | Figure 2C, Suppl. Table S7 |
| Genetic principal components | `pca.py` | Section 2.4 |
| PC-adjusted GRS analysis | `grs_pc.py <k>` | Suppl. Table S8 |
| External validation in NHANES | `nhanes_ext.py`, `nhanes_val.py` | Table 4 |
| Summary tables | `summarise.py` | Tables |
| Figures | `fig1.py`, `figs.py` | Figures 1–4 |

## Requirements

Python ≥3.9; see `requirements.txt`. The analyses were run on a single CPU core; cross-validation takes approximately 1–2 minutes per sex–age group.

## Methods summary

- Prevalent T2D: ICD-10 E11 in hospital records up to 2010; controls reduced by stratified downsampling (age, sex, family history) to the number of cases; ten sex–age groups.
- Incident T2D: first E11 diagnosis after baseline; participants with baseline HbA1c ≥48 mmol/mol excluded.
- Models: LightGBM, L1-penalised logistic regression, random forest, extremely randomised trees, decision tree.
- Evaluation: repeated stratified five-fold cross-validation (5 × 5); corrected resampled t-test (Nadeau & Bengio 2003).
- Genetics: 74 independent genome-wide significant variants (GWAS Catalog, P < 5 × 10⁻⁸; LD r² < 0.1); fold-specific weights; adjustment for 10 genetic PCs.

## Note on AI assistance

Parts of the analysis code were developed with the assistance of a large language model (Claude, Anthropic) under the direction of the authors; all code and results were reviewed and verified by the authors.

## Licence

MIT (see `LICENSE`).
