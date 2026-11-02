# Non-glycaemic T2D models: reproduction and revision status

Code, aggregate results and diagnostic LightGBM models for the UK Biobank / NHANES study (UK Biobank Application 64823).

## Status: corrected code requires a study-data rerun

The results in `results/` and existing LightGBM models were deposited before the methodological corrections in this branch. They have **not** been recomputed by this change. Do not relabel them as outputs of the corrected code.

The earlier unweighted NHANES pipeline used different blood-count predictors, a different cholesterol source, and default L2 regression; the weighted pipeline used `common.py` and L1 regression. Their estimates are not a controlled comparison of weighting choices. Both evaluations now use the same 36 inputs (35 plus age), fitted model and eligible participants. Original unweighted results are historical sensitivity results only.

The earlier calibration analysis used the pooled case fraction 0.4993 for two sex-specific models. Correct fractions are computed from each actual training subset. For the reported cohort they are 6,547/11,429 for men and 3,896/9,487 for women. Original calibration, Brier and decision-curve outputs must be regenerated.

## Inputs and installation

Use Python 3.11 or later and `pip install -r requirements.txt`. The sample-size analysis also needs R and `pmsampsize`. Exact installed versions are recorded in the model manifest when exports are regenerated. The old README's claimed version list was not an environment lockfile.

Set `T2D_DATA_DIR` to the directory containing:

- `medical_history_3.tsv`: 20,916 unique participants, with EID, HasT2D, baseline clinical measurements and sex.
- `medical_history_fut.tsv`: post-baseline recorded E11 cases, EID and time to recorded diagnosis.
- `T2D_geno.tsv`: genotypes for the genetic analyses.

UK Biobank inputs are restricted and must not be committed. `NHANES_DIR` holds the public 2017–March 2020 XPT files, downloaded by `common.load_nhanes()` if absent. `T2D_OUT_DIR` selects the results directory. For compatibility with the original pipeline, directory variables should end in `/`.

Run commands from `code/`. The original `medical_history.tsv` is a defective intermediate and must not be used. The keyed merge is documented in `cohorting_fix.py`.

## Regenerate the corrected external analyses

```bash
python nhanes_val.py
python calibration_dca.py
python deposit_model.py
python findrisc.py
```

`nhanes_val.py` writes `NHANES_evaluation_harmonised.csv` (paired weighted and unweighted evaluation of the same fit), `NHANES_internal_harmonised.csv` and `S12_harmonisation.csv`. The two older NHANES entry points delegate to this implementation. Legacy pickle caches are not read.

Weighted AUROC intervals use a Rao–Wu rescaled PSU bootstrap: draw n_h−1 PSUs within each stratum, then multiply weights by n_h/(n_h−1) times the PSU selection count. All PSUs from the relevant full survey design are retained before age, sex and outcome domain selection. This is a with-replacement first-stage approximation; no finite-population correction is used. Singleton design strata and excessive invalid replicates cause errors. See [survey package documentation](https://r-survey.r-forge.r-project.org/survey/html/bootweights.html) and [NHANES variance guidance](https://wwwn.cdc.gov/nchs/nhanes/tutorials/varianceestimation.aspx).

`calibration_dca.py` writes new `*_sex_specific.csv` files with unweighted and survey-weighted calibration, Brier scores, operating points and decision curves. It evaluates both models, calculates the development case fraction separately by sex and uses the corresponding target-weighted prevalence. It does not overwrite the historical calibration files. Calibration-in-the-large fixes the slope at one; the calibration slope is fitted with an intercept using binomial likelihood.

These prevalence offsets use the evaluation outcomes and therefore represent **apparent updating**. They are not independently validated recalibration. A constant offset cannot resolve slope miscalibration, predictor-dependent sampling, or the change from diagnosed to undiagnosed disease. Clinical thresholds require uncertainty assessment and independent validation.

`deposit_model.py` exports LightGBM text files, full-precision logistic JSON files and a checksum/software manifest. Its reproduction check reads the deposited JSON back from disk. Printed CSV coefficients remain rounded and do not reproduce the pipeline to machine precision.

`findrisc.py` now excludes all subsequently diagnosed participants from controls and checks that cases and controls are disjoint. Its six-component score also uses proxies for activity and fruit/vegetable intake; it is not complete validated FINDRISC. The output is `FINDRISC_comparison_revised.csv`; the old comparison used a different and potentially overlapping sample.

## Other analyses

The original internal-analysis commands remain available: `t2d_pipeline.py A`, `t2d_pipeline.py B`, `t2d_pipeline.py C`, the genetic scripts, `inc_matched.py`, `outcome_misclass.py`, and the R sample-size calculation. `run_all.sh` runs the original sequence and corrected external evaluation. Scripts may require all restricted inputs and the documented working directory.

Original table numbers in result filenames refer to the pre-revision manuscript. In the edited manuscript, Table 3 contains reported weighted external estimates; original unweighted estimates are in Supplementary Table S19 and original calibration outputs in S18. Replace these values and their method descriptions after the corrected runs, then update the manuscript, captions and checklist together.

## Validation and limitations

Run focused numerical checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

These verify the corrected numerical functions with synthetic data. They do not verify study results. The revision environment lacked UK Biobank participant inputs, LightGBM and downloadable NHANES inputs, so no full analysis rerun was completed.

Remaining scientific reporting needs include the full cohort eligibility flow, administrative follow-up end, treatment of death and missing baseline HbA1c, actual prognostic feature count, group-specific missingness and an adequate sample-size assessment. Outcomes use inpatient E11 codes; the complete Eastwood ascertainment algorithm and QDiabetes-2018 cannot be implemented with the documented extract. Hyperparameters were fixed after preliminary tuning; nested tuning was not performed.

Code is provided under `LICENSE`. Manuscript authors must verify the revised methods and all regenerated results before submission.
