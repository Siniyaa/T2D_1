# T2D prediction without glycaemic testing — reproduction guide

Code and aggregate results for: *Detection and prediction of type 2 diabetes from routine
clinical data without glycaemic testing: sex-specific models developed in UK Biobank and
externally validated in NHANES.*

UK Biobank Application Number **64823**.

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

Install with `pip install -r requirements.txt`; the sample-size analysis additionally needs R
with `install.packages("pmsampsize")`. All analyses used `random_state = 42` / `set.seed(42)`.

## 2. Input data (not redistributable)

| File | Contents | Source |
|---|---|---|
| `medical_history_3.tsv` | 20,916 x 185. Case-finding analysis set, one row per EID | UK Biobank application 64823 |
| `medical_history_fut.tsv` | 26,975 x 183. Participants diagnosed after baseline, with `Days until E11 diagnosis` | UK Biobank application 64823 |
| `T2D_geno.tsv` | Array genotypes for the 105 selected variants | UK Biobank field 22418 |
| `P_DEMO.xpt` … `P_GHB.xpt` | NHANES 2017-March 2020 public-use files | wwwn.cdc.gov (free) |

UK Biobank data cannot be redistributed. NHANES files download automatically (see 3.5).

**Do not use `medical_history.tsv`.** It is a defective intermediate: an unkeyed
`study.merge(fam_hist)` in `Cohorting-1.ipynb` produced 23,188,684 rows for 20,916
participants. It never fed the modelling data and no published result depends on it. It has
been removed from this repository and the merge corrected (see `code/cohorting_fix.py`).

## 3. Reproducing each table and figure

Set `T2D_DATA_DIR` to the directory holding the input files, then run from `code/`.

| Output | Command | Writes |
|---|---|---|
| Table 1 (baseline characteristics) | `python baseline_table.py` | `results/Table1_baseline_characteristics.csv` |
| Table 2 (incident prediction) | `python t2d_pipeline.py C` | `results/C_incident_folds.csv`, `C_incident_horizon.csv`, `C_simple_benchmark.csv`; `python inc_matched.py` adds `INC_matched.csv` |
| Table 2, FINDRISC rows | `python findrisc.py` | `results/FINDRISC_comparison.csv` |
| Table 3 (case-finding by stratum) | `python t2d_pipeline.py B` | `results/B_summary.csv` |
| Table 4 (NHANES, unweighted + weighted) | `python nhanes_val.py` then `python nhanes_weighted.py` | `results/NHANES_validation.csv`, `results/Table4_weighted.csv` |
| Table 5 (calibration, utility) | `python calibration_dca.py` | `results/Calibration.csv`, `OperatingPoints.csv`, `DecisionCurve.csv` |
| Figure 1 (study design) | `python fig1.py` | `figures/Figure1.png` |
| Figure 2 (incident prediction) | `python figs.py --panel incident` | `figures/Figure2.png` |
| Figure 3 (algorithms + GRS) | `python figs.py --panel algorithms` | `figures/Figure3.png` |
| Figure 4 (predictor importance) | `python figs.py --panel importance` | `figures/Figure4.png` |
| Figure 5 (calibration, utility, leakage) | `python calibration_dca.py --figure` | `figures/Figure5.png` |
| Supp S1–S9 | `bash run_all.sh` | `results/*.csv` |
| Supp S10 (TRIPOD+AI) | manual; item text from Collins et al. BMJ 2024;385:e078378 | — |
| Supp S11 (predictor list) | `python predictor_list.py` | `results/S11_predictors.csv` |
| Supp S12 (NHANES harmonisation) | `python nhanes_val.py --dump-harmonisation` | `results/S12_harmonisation.csv` |
| Supp S13 (per-predictor leakage) | `python leakage.py` | `results/leakage_per_predictor.csv` |
| Supp S14 (sample size) | `python sample_size.py` then `Rscript sample_size.R` | `results/ss_results.csv` |
| Supp S15 (outcome misclassification) | `python outcome_misclass.py` | `results/outcome_misclassification.csv` |
| Supp S16 (model coefficients) | `python deposit_model.py` | `results/S16_logistic_coefficients.csv`, `models/LightGBM_harmonised_*.txt` |
| Supp S17 (UKB vs NHANES distributions) | `python nhanes_val.py --compare-distributions` | `results/S17_distribution_comparison.csv` |

Runtime on 24 cores: `t2d_pipeline.py B` approximately 40 min; `outcome_misclass.py`
approximately 4 min; everything else under 2 min each.

## 4. Using the fitted model

`results/S16_logistic_coefficients.csv` contains the sex-specific 35-predictor logistic
regression on both the standardised and the original measurement scale, with the imputation
median and standardisation constants. To predict: impute missing values with the stated
median, form the linear predictor as intercept + sum(coefficient_original x predictor), then
apply the logistic function. This reproduces the fitted pipeline to within 2e-15.

**Recalibration is required.** The development sample was downsampled to a case proportion of
p1 = 0.4993, so predicted probabilities are not risks. Add
`log(d1/(1-d1)) - log(0.4993/(1-0.4993))` to the linear predictor, where `d1` is the
prevalence in the target population. Without this the model over-predicts roughly tenfold.
Calibration slopes were 0.36-0.77 in NHANES even after this correction, so full model updating
is advisable before any applied use.

## 5. Known limitations of this code

- Outcomes are hospital inpatient ICD-10 E11 only. The Eastwood et al. algorithm
  (PLOS One 2016;11:e0162388) cannot be run on this extract: fields 2443, 2976, 6153/6177,
  20002, 20003 and 4041 and primary-care linkage were not requested. `outcome_misclass.py`
  tests the component that is testable.
- QDiabetes-2018 is not implemented; roughly ten of its sixteen predictors are absent.
- Hyperparameters are fixed (`Supplementary Table S1`), not tuned in nested cross-validation.

## 6. Licence and citation

Code under the licence in `LICENSE`. Please cite the paper and acknowledge UK Biobank
application 64823.
