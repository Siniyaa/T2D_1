# T2D prediction without glycaemic testing — reproduction guide

Code and aggregate results for: *Diagnosed and undiagnosed type 2 diabetes compared using
prediction models without glycaemic inputs developed in UK Biobank and evaluated in NHANES.*

UK Biobank Application Number **64823**.

`EXECUTION.md` is the authoritative execution record: it gives the locked environment, the run
order and the checks that run inside the scripts. This file maps each table and figure of the
submitted manuscript to the command that produces it.

---

## 1. Software versions

| Component | Version |
|---|---|
| Python | 3.11.17 |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| statsmodels | 0.15.0 |
| matplotlib | 3.11.2 |
| R | 4.5.3 |
| pmsampsize | 1.1.3 |

Install with `pip install -r requirements-lock.txt`; the sample-size analysis additionally needs R
with `install.packages("pmsampsize")`. All analyses used `random_state = 42` / `set.seed(42)`.

## 2. Input data (not redistributable)

| File | Contents | Source |
|---|---|---|
| `medical_history_3.tsv` | 20,916 x 185. Case-finding analysis set, one row per EID | UK Biobank application 64823 |
| `T2D_geno_lasso.tsv` | Array genotypes for the 76 genome-wide-significant variants from which the 69-variant score is built | UK Biobank field 22418 |
| `P_DEMO.xpt` … `P_GHB.xpt` | NHANES 2017-March 2020 public-use files | wwwn.cdc.gov (free) |

UK Biobank data cannot be redistributed. NHANES files download automatically on first use.

Two further inputs are required only by the superseded scripts listed in section 5 and by no
analysis reported in the manuscript: `medical_history_fut.tsv` (the follow-up extract, used by
the removed incident analysis and by `sens.py`) and `T2D_geno.tsv` (the earlier 105-variant
genotype file, superseded by `T2D_geno_lasso.tsv`).

**Do not use `medical_history.tsv`.** It is a defective intermediate: an unkeyed
`study.merge(fam_hist)` in `Cohorting-1.ipynb` produced 23,188,684 rows for 20,916
participants. It never fed the modelling data and no published result depends on it. It has
been removed from this repository; the corrected merge is the one used to build
`medical_history_3.tsv`.

## 3. Reproducing each table and figure

Set the paths as in `EXECUTION.md`, then run from the repository root.

| Output | Command | Writes |
|---|---|---|
| Table 1 (baseline characteristics) | `python code/baseline_table.py` | `results/Table1_baseline_characteristics.csv` |
| Table 2 (case-finding by stratum) | `python code/case_finding.py` | `results/CaseFinding_performance.csv` (LightGBM rows) |
| Table 3 (NHANES, unweighted and weighted) | `python code/nhanes_val.py` then `python code/nhanes_weighted.py` | `results/NHANES_validation.csv`, `results/Table4_weighted.csv` |
| Tables 4A and 4B (calibration, utility) | `python code/calibration_dca.py` | `results/Calibration.csv`, `OperatingPoints.csv`, `DecisionCurve.csv` |
| Fig 1 (study design) | `python code/figure1_design.py` | `figures/Figure1.png` |
| Fig 2 (algorithms and genetic risk score) | `python code/figs.py` | `figures/Figure2_algorithms_grs.*` |
| Fig 3 (predictor importance) | `python code/extra.py` then `python code/figs.py` | `results/prev_gain_importance.csv`, `figures/Figure3_importance.*` |
| Fig 4 (calibration and clinical utility) | `python code/calibration_dca.py --figures` | `figures/Figure5.png` |
| Fig 5 (single-predictor differences) | `python code/leakage.py` then `python code/calibration_dca.py --figures` | `results/leakage_per_predictor.csv`, `figures/Figure6.png` |

The two figure filenames in the last two rows are historical and do not match the manuscript
numbering: `Figure5.png` is manuscript Fig 4 and `Figure6.png` is manuscript Fig 5.

| Supplementary item | Command | Writes |
|---|---|---|
| S1 Table (model manifest) | compiled from the settings in `code/case_finding.py` and `code/deposit_model.py`; no separate script | — |
| S2 Table (all five algorithms) | `python code/case_finding.py` | `results/CaseFinding_performance.csv` |
| S3 Table (LightGBM vs L1 logistic) | `python code/case_finding.py` | `results/CaseFinding_tests.csv` |
| S4 Table (69-variant genetic risk score) | `python code/case_finding.py` | `results/CaseFinding_grs.csv`, `results/grs_variant_qc.csv` |
| S5 Table (TRIPOD+AI checklist) | manual; item text from Collins et al. BMJ 2024;385:e078378 | — |
| S6 Table (predictor list) | `python code/predictor_list.py` | `results/S11_predictors.csv` |
| S7 Table (NHANES harmonisation) | compiled for the manuscript from the mapping in `code/nhanes_ext.py`; no separate script | — |
| S8 Table (single-predictor differences) | `python code/leakage.py` | `results/leakage_per_predictor.csv` |
| S9 Table (sample size) | `python code/sample_size.py` then `Rscript code/sample_size.R` | `results/ss_input.csv`, `results/ss_results.csv` |
| S10 Table (outcome misclassification) | `python code/outcome_misclass.py` | `results/outcome_misclassification.csv` |
| S11 Table (model coefficients) | `python code/deposit_model.py` | `results/S16_coefficients_full_precision.csv`, `results/S16_logistic_coefficients.csv`, `models/*.txt` |
| S12 Table (predictor distributions) | compiled for the manuscript from the modelling matrix and the NHANES extract; no separate script | — |
| S13 Table (software versions) | `requirements-lock.txt` | — |
| S14 Table (case-finding sampling counts) | counts of the analysis set written by `code/case_finding.py`; tabulated for the manuscript | — |
| S15 Table (frame-aligned and LDL-removed) | `python code/nhanes_frames.py` | `results/NHANES_frame_sensitivity.csv`, `results/NHANES_frame_paired_difference.csv` |
| S1 Fig (control downsampling) | `python code/figure1_design.py` | `figures/Figure1.png` panels |

Runtime on 24 cores: `case_finding.py` approximately 40 min; `outcome_misclass.py`
approximately 4 min; everything else under 2 min each.

## 4. Using the fitted model

`results/S16_logistic_coefficients.csv` contains the sex-specific 36-input logistic
regression (35 harmonised clinical predictors and age) on both the standardised and the
original measurement scale, with the imputation median and standardisation constants. To
predict: impute missing values with the stated median, form the linear predictor as
intercept + sum(coefficient_original x predictor), then apply the logistic function.
`code/deposit_model.py` reloads the deposited file and reproduces the fitted pipeline to
6.1e-16 on 400 records, 80 of them with at least one missing value.

**Recalibration is required.** The development sample was downsampled to a case proportion of
p1 = 0.4993, so predicted probabilities are not risks. Add
`log(d1/(1-d1)) - log(0.4993/(1-0.4993))` to the linear predictor, where `d1` is the
prevalence in the target population. Without this the model over-predicts roughly tenfold.
Calibration slopes were 0.33-0.73 in NHANES even after this correction
(`results/Calibration.csv`), so full model updating is advisable before any applied use.

## 5. Scripts retained for provenance but superseded

The original pipeline (`t2d_pipeline.py`, `summarise.py`, `sens.py`, `snp_qc.py`, `geno2.py`,
`grs_qc.py`, `grs_cv.py`, `grs_pc.py`, `pca.py`, `fig1.py`, `nhanes_ext.py`, `run_all.sh`) is
kept so that the development history can be inspected. It is **not** the source of any number
in the submitted manuscript. Its case-finding outputs (`B_summary.csv`, `B_lgbm_vs_lr.csv`,
`B_genetic_comparison.csv`) were superseded by `code/case_finding.py`, which fits all five
algorithms on the same 175 predictors and the same folds; its genetic outputs
(`GRS74_summary.csv`, `GRSPC_summary.csv`, `G2_summary.csv`) rest on variant sets that are no
longer reported. `SENS_summary.csv` and `CaseFinding_misclassification.csv` are alternative
sensitivity analyses that the manuscript does not report. Read any of these files as history,
not as results.

The incident-prediction analysis was removed from the submission; its scripts and outputs were
deleted from the working tree and remain in the git history at commit 87ca782d. See
`EXECUTION.md`.

## 6. Known limitations of this code

- Outcomes are hospital inpatient ICD-10 E11 only. The Eastwood et al. algorithm
  (PLOS One 2016;11:e0162388) cannot be run on this extract: fields 2443, 2976, 6153/6177,
  20002, 20003 and 4041 and primary-care linkage were not requested. `outcome_misclass.py`
  tests the component that is testable.
- QDiabetes-2018 is not implemented; roughly ten of its sixteen predictors are absent.
- Hyperparameters are fixed (Supplementary Table S1), not tuned in nested cross-validation.

## 7. Licence and citation

Code under the licence in `LICENSE`. Please cite the paper and acknowledge UK Biobank
application 64823.
