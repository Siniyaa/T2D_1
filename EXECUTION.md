# Execution record

## Working directory and environment

Run all commands **from the repository root** and set the three paths explicitly:

```bash
cd /path/to/T2D_1
export T2D_DATA_DIR=/path/to/ukb_extract     # medical_history_3.tsv, medical_history_fut.tsv
export NHANES_DIR=$PWD/nhanes                # .xpt files, downloaded on first use
export T2D_OUT_DIR=$PWD/results
python -m pip install -r requirements-lock.txt
```

On Windows PowerShell use `$env:T2D_DATA_DIR = "..."` in place of `export`.

Locked environment for the runs reported here: `requirements-lock.txt`
(Python 3.11.17, numpy 2.4.6, pandas 2.3.3,
scikit-learn 1.9.1, lightgbm 4.7.0, scipy 1.17.1,
statsmodels 0.15.0). Run date 2026-11-04.
Every result in the manuscript was regenerated in this locked environment. The earlier
case-finding runs, whose environment was not recorded, have been superseded by
`code/case_finding.py` and are no longer reported. The incident-prediction analysis was removed from the submission because the follow-up extract
carries no participant identifier, so case and control records could not be shown to be distinct
individuals. Its scripts and outputs (`incident_primary.py`, `findrisc.py`,
`figure2_and_glycaemic.py`, `figure2_plot.py`, `cohorting_fix.py`, `inc_matched.py` and the
`Incident_*`, `FINDRISC_*`, `Glycaemic_*` and `C_incident_*` result files) have been deleted from
the working tree so that this repository reproduces the submitted manuscript and nothing else.
They remain in the git history at commit 87ca782d if that work is resumed.

## Order

```bash
python code/case_finding.py            # CaseFinding_performance.csv, CaseFinding_tests.csv,
                                       # CaseFinding_grs.csv
python code/calibration_dca.py         # Calibration.csv, DecisionCurve.csv, OperatingPoints.csv
python code/leakage.py                 # leakage_per_predictor.csv  (500 replicates; prints the
                                       # shared-control counts for the two outcome samples)
python code/nhanes_weighted.py         # Table4_weighted.csv   (prints the design-frame check)
python code/nhanes_frames.py           # NHANES_frame_sensitivity.csv,
                                       # NHANES_frame_paired_difference.csv
python code/deposit_model.py           # models/*.txt, S16_coefficients_full_precision.csv
                                       # (prints the reload check)

python code/figure1_design.py            # Figure1.png  (manuscript Figure 1)
python code/figure2_plot.py              # Figure2_incident.png (manuscript Figure 6)
python code/calibration_dca.py --figures # Figure5.png (Figure 4), Figure6.png (Figure 5)
```

Seeds: 42 for model fitting and partitioning; 7 for bootstrap resampling (11 in `leakage.py`)
unless a script states otherwise.

## Preprocessing

`common.model_matrix` builds the modelling matrix for every analysis. A field is numeric when
more than half of its non-missing entries parse as numbers, so ordered questionnaire responses
keep their numeric coding and "Do not know" / "Prefer not to answer" become missing. The four
genuinely nominal fields are given an explicit "NA" level, converted to pandas `category` dtype
and declared to LightGBM with `categorical_feature`; applying `pd.to_numeric` to them would map
every level to missing and silently drop the predictor.

## Checks that run inside the scripts

* `nhanes_weighted.check_frame_complete` asserts that the age restriction leaves the
  stratum-by-PSU frame unchanged (49 cells before and after) before any weighted interval is
  computed. `nhanes_frames.py` reuses it.
* `deposit_model.verify` reloads `S16_coefficients_full_precision.csv` from disk and compares its
  predictions with the fitted pipeline on 400 records, 80 with at least one missing value;
  agreement was 6.1e-16.
* `findrisc.main` asserts the control pool is disjoint from the incident case set under the
  signature rule, and reports the removal count.

## Notes on two analyses added in this revision

* `nhanes_frames.py` re-evaluates the diagnosed outcome inside the NHANES fasting subsample, so
  that both outcomes share one frame and one weight, and repeats both outcomes with LDL
  cholesterol removed and the models refitted. The paired difference uses a design bootstrap in
  which each replicate resamples PSUs once and both AUROCs are recomputed on that replicate.
* `calibration_dca.figure5` draws loess calibration curves with bootstrap bands. `it=0` is
  required in `statsmodels.lowess`: the default of three robustifying iterations treats the cases
  of a binary outcome as outliers and collapses the fitted curve onto zero.
* `leakage.diagnosed_fasting` restricts the diagnosed outcome to the fasting subsample, the frame
  in which undiagnosed diabetes is defined, so that the per-predictor contrast is frame-aligned.
  Before this change the diagnosed outcome used the whole examination sample (denominators up to
  4,244) while the undiagnosed outcome used the fasting subsample, and only LDL cholesterol --
  which is measured solely in that subsample -- was comparable. The two samples now share 748
  male and 831 female controls, which is what the paired bootstrap preserves.

## Deposited models

`models/LightGBM_harmonised_{male,female}.txt` carry 36 features in Supplementary Table S12
order, including Lymphocyte_count_30120, Monocyte_count_30130 and Neutrophill_count_30140, and
reproduce the unweighted NHANES AUROCs in manuscript Table 3 (0.798 and 0.670 in men, 0.793 and
0.733 in women).
