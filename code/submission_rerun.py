"""Generate the aggregate outputs needed to complete the manuscript revision.

Run in the authorised UK Biobank environment. Participant records are read
locally; only an explicit list of aggregate outputs and model files is packaged.
This does not supply upstream eligibility, administrative follow-up or author
declarations that are absent from the extracts.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

RESULT_FILES = (
    'NHANES_evaluation_harmonised.csv', 'NHANES_internal_harmonised.csv',
    'S12_harmonisation.csv', 'Calibration_sex_specific.csv',
    'DecisionCurve_sex_specific.csv', 'OperatingPoints_sex_specific.csv',
    'S16_logistic_coefficients.csv', 'Revision_cohort_audit.json',
    'Revision_baseline_missingness.csv', 'Revision_prognostic_predictors.csv',
)
MODEL_FILES = (
    'LightGBM_harmonised_male.txt', 'LightGBM_harmonised_female.txt',
    'Logistic_harmonised_male.json', 'Logistic_harmonised_female.json',
    'manifest.json',
)


def cohort_audit(out):
    import pandas as pd
    from common import load_prevalent, DATA, HBA, SEX, UCOLS
    from t2d_pipeline import preprocess
    from baseline_table import VARS

    prev = load_prevalent()
    future = pd.read_csv(Path(DATA) / 'medical_history_fut.tsv', sep='\t', low_memory=False)
    if not future.EID.is_unique:
        raise ValueError('Future-diagnosis input has duplicate EIDs; resolve before analysis.')
    labels = prev.HasT2D.map({True: 1, False: 0, 1: 1, 0: 0,
                             'True': 1, 'False': 0, '1': 1, '0': 0})
    if labels.isna().any():
        raise ValueError('HasT2D contains missing or unrecognised outcome values.')
    controls = prev.loc[labels == 0].copy()
    cases = future.loc[future['Ethnic background [21000]'] == 'British'].copy()
    eligible_controls = controls.loc[~controls.EID.isin(future.EID)].copy()
    missing_before = {
        'future_british_cases': int(pd.to_numeric(cases[HBA], errors='coerce').isna().sum()),
        'controls_without_future_diagnosis': int(pd.to_numeric(eligible_controls[HBA], errors='coerce').isna().sum()),
    }
    excluded_cases = pd.to_numeric(cases[HBA], errors='coerce') >= 48
    excluded_controls = pd.to_numeric(eligible_controls[HBA], errors='coerce') >= 48
    cases = cases.loc[~excluded_cases]
    eligible_controls = eligible_controls.loc[~excluded_controls]
    if set(cases.EID) & set(eligible_controls.EID):
        raise ValueError('Prognostic cases and controls overlap.')
    groups = {
        'Recorded diagnosis cases': prev.loc[labels == 1],
        'Original controls': controls,
        'Prognostic cases': cases,
        'Prognostic controls': eligible_controls,
    }
    counts, rows = {}, []
    for group, frame in groups.items():
        counts[group] = {
            'n': len(frame), 'male_n': int(frame[SEX].eq('Male').sum()),
            'female_n': int(frame[SEX].eq('Female').sum()),
            'missing_sex_n': int(frame[SEX].isna().sum()),
            'missing_baseline_HbA1c_n': int(pd.to_numeric(frame[HBA], errors='coerce').isna().sum()),
        }
        for field, label in VARS:
            values = pd.to_numeric(frame[field], errors='coerce')
            rows.append(dict(group=group, characteristic=label, n=len(frame),
                             observed_n=int(values.notna().sum()),
                             missing_n=int(values.isna().sum()),
                             missing_pct=100 * float(values.isna().mean()),
                             mean=float(values.mean()), SD=float(values.std())))
    # Use the original pipeline's column intersection and preprocessing, without
    # fitting or storing any individual records or individual predictions.
    f = cases.set_index('EID').drop(columns=['Days until E11 diagnosis'])
    c = eligible_controls.set_index('EID').drop(columns=['HasT2D'])
    shared = [col for col in f if col in c]
    X, _ = preprocess(pd.concat([f[shared], c[shared]]))
    predictors = X.drop(columns=['Sex 31']).columns.tolist()
    days = pd.to_numeric(cases['Days until E11 diagnosis'], errors='coerce')
    audit = {
        'counts': counts, 'all_future_records_n': len(future),
        'future_british_before_HbA1c_n': int(future['Ethnic background [21000]'].eq('British').sum()),
        'controls_excluded_future_diagnosis_n': int(controls.EID.isin(future.EID).sum()),
        'cases_excluded_HbA1c_n': int(excluded_cases.sum()),
        'controls_excluded_HbA1c_n': int(excluded_controls.sum()),
        'missing_HbA1c_before_exclusion': missing_before,
        'missing_HbA1c_retained': True,
        'prognostic_predictors_including_age': len(predictors),
        'harmonised_predictors_including_age': len(UCOLS),
        'recorded_diagnosis_development': {
            sex: {'n': int(prev[SEX].eq(sex).sum()),
                  'cases': int(labels[prev[SEX].eq(sex)].sum())}
            for sex in ('Male', 'Female')},
        'diagnosis_days_missing_n': int(days.isna().sum()),
        'diagnosis_days_nonpositive_n': int(days.le(0).sum()),
        'unresolved_author_information': [
            'Upstream source-control eligibility and exclusion counts',
            'Administrative follow-up end and treatment of deaths in source extraction',
            'Author approval of ethics, contributions, funding and AI declaration',
        ],
    }
    (out / 'Revision_cohort_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False))
    pd.DataFrame(rows).to_csv(out / 'Revision_baseline_missingness.csv', index=False)
    pd.DataFrame({'Predictor': predictors}).to_csv(out / 'Revision_prognostic_predictors.csv', index=False)


def package_outputs(root, revision):
    paths = [root / 'results' / name for name in RESULT_FILES]
    paths += [root / 'models' / name for name in MODEL_FILES]
    for path in paths:
        if not path.is_file() or not path.stat().st_size or path.is_symlink():
            raise ValueError('Missing, empty or linked expected output: ' + path.name)
    manifest = {
        'code_revision': revision,
        'status': 'Generated outputs require scientific and author review before submission',
        'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    manifest_path = root / 'revision_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    archive = root / 'T2D_revision_outputs.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
        for path in paths + [manifest_path]:
            bundle.write(path, path.relative_to(root))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--nhanes-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='A new directory, outside the input directories')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.nhanes_dir = args.nhanes_dir.resolve()
    root = args.output_dir.resolve()
    problems = []
    for name in ('medical_history_3.tsv', 'medical_history_fut.tsv'):
        if not (args.data_dir / name).is_file():
            problems.append('Missing authorised input: ' + name)
    for name in ('numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm'):
        if importlib.util.find_spec(name) is None:
            problems.append('Missing Python package: ' + name)
    if root.exists():
        problems.append('Output directory already exists; use a new directory.')
    if any(root == p or p in root.parents or root in p.parents for p in (args.data_dir, args.nhanes_dir)):
        problems.append('Output and input directories must be separate.')
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 2
    if args.check_only:
        print('Local prerequisites pass. NHANES download availability is checked during the run.')
        return 0
    out = root / 'results'
    os.environ.update(T2D_DATA_DIR=str(args.data_dir) + os.sep,
                      NHANES_DIR=str(args.nhanes_dir) + os.sep,
                      T2D_OUT_DIR=str(out) + os.sep)
    here = Path(__file__).resolve().parent
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=here, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=here, text=True).strip()
        if dirty:
            raise ValueError('Use a clean checkout so output provenance identifies the executed code.')
    except subprocess.CalledProcessError as error:
        raise ValueError('Run from the checked-out repository to record the code revision.') from error
    root.mkdir(parents=True)
    out.mkdir()
    cohort_audit(out)
    for script in ('nhanes_val.py', 'calibration_dca.py', 'deposit_model.py'):
        print('Running ' + script, flush=True)
        subprocess.run([sys.executable, str(here / script)], cwd=here, check=True)
    archive = package_outputs(root, revision)
    print('Completed: ' + str(archive))
    print('Review these aggregate outputs under the applicable data-access rules before sharing.')
    print('No manuscript submission or scientific approval is implied by a successful run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
