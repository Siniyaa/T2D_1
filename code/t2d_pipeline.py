"""
T2D UK Biobank analysis pipeline (reproduces notebook preprocessing; adds CV and incident analysis)
A. Reproduction of the original single 75/25 split (LightGBM, per sex x age stratum)
B. Repeated stratified 5-fold cross-validation (5 algorithms; LightGBM with vs without SNPs)
C. Incident T2D prediction (cases diagnosed after baseline vs controls never diagnosed)
"""
import sys, time, warnings
import numpy as np, pandas as pd, sklearn
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
warnings.filterwarnings("ignore")

import os
DIR = os.environ.get("T2D_DATA_DIR", "../data/")          # UK Biobank-derived input files (not shared)
OUT = os.environ.get("T2D_OUT_DIR", "../results/")       # analysis outputs
SEED = 42
N_SPLITS, N_REPEATS = 5, 5
STRATA_AGES = [(0, 50), (50, 55), (55, 60), (60, 65), (65, 200)]
AGE_LABEL = {(0, 50): "<50", (50, 55): "50–54", (55, 60): "55–59", (60, 65): "60–64", (65, 200): "≥65"}

# ----------------------------------------------------------------- preprocessing (as notebook)
DROP = ['Date of attending assessment centre [53]', 'Ethnic background [21000]',
        'Glycated haemoglobin (HbA1c) [30750]', 'Glucose [30740]', 'Overall health rating [2178]',
        'Age first had sexual intercourse [2139]']
NANS = {'Do not know', 'Prefer not to answer'}
C2I = {'Less than an hour a day': 0.5, 'Less than one': 0.5, 'Never had sex': 0., 'Never': 0., 'None': 0.,
       'Less than once a year': 1., 'Special occasions only': 2., 'Once in the last 4 weeks': 3.,
       'One to three times a month': 4., '2-3 times in the last 4 weeks': 5., 'Less than once a week': 6.,
       'Once a week': 7., 'Once or twice a week': 8., '2-3 times a week': 9., '2-4 times a week': 10.,
       'Three or four times a week': 11., '4-5 times a week': 12., '5-6 times a week': 13.,
       'Less than one a day': 14., 'Daily or almost daily': 15., 'Every day': 16., 'Once or more daily': 17.,
       '1-5 times a day': 18., '6-10 times a day': 19., '11-15 times a day': 20., '16-20 times a day': 21.,
       'More than 20 times a day': 22., 'Excellent': 3., 'Good': 2., 'Fair': 1., 'Poor': 0.,
       'Several days': 2., 'Not at all': 0., 'More than half the days': 1., 'Nearly every day': 3.,
       'Unable to walk': -1.}
def nullify(x): return float('nan') if x in NANS else x
def to_float_spec(x):
    try: return float(x)
    except (ValueError, TypeError): return C2I[x]

def clean_cols(df):
    df.columns = (df.columns.str.replace('[(,),/]', '', regex=True)
                  .str.replace('[[]', '', regex=True).str.replace('[]]', '', regex=True))
    return df

ORDINAL = ['Number of daysweek walked 10+ minutes 864', 'Duration of walks 874',
    'Number of daysweek of moderate physical activity 10+ minutes 884', 'Duration of moderate activity 894',
    'Number of daysweek of vigorous physical activity 10+ minutes 904', 'Time spent watching television TV 1070',
    'Time spent using computer 1080', 'Time spent driving 1090', 'Cooked vegetable intake 1289',
    'Salad  raw vegetable intake 1299', 'Fresh fruit intake 1309', 'Dried fruit intake 1319', 'Bread intake 1438',
    'Cereal intake 1458', 'Tea intake 1488', 'Water intake 1528', 'Time spend outdoors in summer 1050',
    'Time spent outdoors in winter 1060', 'Childhood sunburn occasions 1737',
    'Frequency of solariumsunlamp use 2277', 'Lifetime number of sexual partners 2149', 'Sleep duration 1160',
    'Exposure to tobacco smoke at home 1269', 'Exposure to tobacco smoke outside home 1279',
    "Father's age at death 1807", 'Number of full brothers 1873', 'Number of full sisters 1883']

def preprocess(df):
    df = clean_cols(df.drop(columns=[c for c in DROP if c in df.columns]))
    for f in ORDINAL:
        if f in df: df[f] = df[f].map(nullify).map(to_float_spec)
    fam = [c for c in df.columns if (' had ' in c) and any(k in c for k in ('20107', '20110', '20111'))]
    for f in fam: df[f] = df[f].fillna(False)
    cat = ['Wears glasses or contact lenses 2207', 'Chest pain or discomfort 2335', 'Father still alive 1797',
           'Mother still alive 1835'] + fam
    cat = [c for c in dict.fromkeys(cat) if c in df.columns]
    for f in cat: df[f] = df[f].fillna('NA').astype(str).astype('category')
    return df, cat

def load_prevalent():
    m = pd.read_csv(DIR + 'medical_history_3.tsv', sep='\t', index_col=0, low_memory=False)
    hba1c = m['Glycated haemoglobin (HbA1c) [30750]'].copy()
    g = pd.read_csv(DIR + 'T2D_geno.tsv', sep='\t', index_col=0)
    m = m.join(g)
    y = m.pop('HasT2D').astype(int)
    X, cat = preprocess(m)
    return X, y, cat, list(g.columns), hba1c

def strata(X):
    for sex in ['Male', 'Female']:
        for lo, hi in STRATA_AGES:
            a = X['Age when attended assessment centre 21003']
            idx = X.index[(X['Sex 31'] == sex) & (a >= lo) & (a < hi)]
            yield sex, AGE_LABEL[(lo, hi)], idx

META = ['Sex 31', 'Age when attended assessment centre 21003']

# ----------------------------------------------------------------- models
def lgbm(): return lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=SEED, verbose=-1)
def l1_logistic():
    if tuple(int(v) for v in sklearn.__version__.split('.')[:2]) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver='liblinear', random_state=SEED)
    return LogisticRegression(penalty='l1', solver='liblinear', random_state=SEED)
def skl(model): return make_pipeline(SimpleImputer(strategy='median'), model)
MODELS = {
    'LightGBM': lgbm,
    'Random forest': lambda: skl(RandomForestClassifier(n_estimators=200, max_depth=50, min_samples_leaf=0.1, random_state=SEED, n_jobs=1)),
    'Extremely randomised trees': lambda: skl(ExtraTreesClassifier(n_estimators=200, max_depth=20, min_samples_leaf=0.06, random_state=SEED, n_jobs=1)),
    'Decision tree': lambda: skl(DecisionTreeClassifier(max_depth=50, min_samples_leaf=0.1, random_state=SEED)),
    'Logistic regression': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), l1_logistic()),
}
def numeric(X):
    X = X.copy()
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype): X[c] = X[c].cat.codes.astype(float)
        elif X[c].dtype == bool: X[c] = X[c].astype(float)
    return X

def metrics(y, p):
    yh = (p >= 0.5).astype(int)
    return dict(AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p),
                accuracy=100 * accuracy_score(y, yh), sensitivity=100 * recall_score(y, yh),
                specificity=100 * recall_score(1 - y, 1 - yh))

def corrected_repeated_cv_p(d, n_train, n_test):
    """Corrected repeated k-fold cv t-test (Bouckaert & Frank 2004; Nadeau & Bengio 2003)."""
    d = np.asarray(d); k = len(d); v = d.var(ddof=1)
    if v == 0: return np.nan
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * v)
    return 2 * stats.t.sf(abs(t), df=k - 1)

# ----------------------------------------------------------------- A. reproduce original split
def part_a(X, y, cat):
    rows = []
    for sex, age, idx in strata(X):
        D = X.loc[idx].drop(columns=META)
        np.random.seed(0xfee1600d)
        Xtr, Xte, ytr, yte = train_test_split(D, y[idx], shuffle=True)
        m = lgbm().fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        rows.append(dict(sex=sex, age=age, n=len(idx), cases=int(y[idx].sum()),
                         test_n=len(yte), test_cases=int(yte.sum()), **metrics(yte.values, p)))
    return pd.DataFrame(rows)

# ----------------------------------------------------------------- B. repeated CV
def part_b(X, y, snps, only=None):
    rows = []
    rcv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for si, (sex, age, idx) in enumerate(strata(X)):
        if only is not None and si != only: continue
        D = X.loc[idx].drop(columns=META); yy = y[idx].values
        Dn = numeric(D); Dclin = D.drop(columns=snps)
        t0 = time.time()
        for f, (tr, te) in enumerate(rcv.split(D, yy)):
            runs = [(k, mk, Dn if k != 'LightGBM' else D) for k, mk in MODELS.items()]
            runs.append(('LightGBM (no genetic data)', lgbm, Dclin))
            for name, mk, data in runs:
                mdl = mk().fit(data.iloc[tr], yy[tr])
                p = mdl.predict_proba(data.iloc[te])[:, 1]
                rows.append(dict(sex=sex, age=age, fold=f, model=name, n=len(yy), cases=int(yy.sum()),
                                 n_train=len(tr), n_test=len(te), **metrics(yy[te], p)))
        print(f'B {sex} {age}: n={len(yy)} ({time.time()-t0:.0f}s)', flush=True)
    return pd.DataFrame(rows)

# ----------------------------------------------------------------- C. incident T2D
def part_c():
    f = pd.read_csv(DIR + 'medical_history_fut.tsv', sep='\t', index_col=0, low_memory=False)
    m = pd.read_csv(DIR + 'medical_history_3.tsv', sep='\t', index_col=0, low_memory=False)
    flow = {}
    flow['incident_all'] = len(f)
    f = f[f['Ethnic background [21000]'] == 'British']; flow['incident_british'] = len(f)
    ctrl = m[(~m['HasT2D'].astype(bool))]; flow['controls_prevalent_set'] = len(ctrl)
    ctrl = ctrl[~ctrl.index.isin(f.index) & ~ctrl.index.isin(
        pd.read_csv(DIR + 'medical_history_fut.tsv', sep='\t', index_col=0, usecols=[0]).index)]
    flow['controls_never_diagnosed'] = len(ctrl)
    hb = 'Glycated haemoglobin (HbA1c) [30750]'
    f_ex = f[f[hb] >= 48]; c_ex = ctrl[ctrl[hb] >= 48]
    flow['cases_excl_hba1c48'] = len(f_ex); flow['controls_excl_hba1c48'] = len(c_ex)
    f = f.drop(f_ex.index); ctrl = ctrl.drop(c_ex.index)
    days = f['Days until E11 diagnosis']
    f = f.drop(columns=['Days until E11 diagnosis'])
    ctrl = ctrl.drop(columns=['HasT2D'])
    common = [c for c in f.columns if c in ctrl.columns]
    D = pd.concat([f[common], ctrl[common]])
    y = pd.Series(np.r_[np.ones(len(f), int), np.zeros(len(ctrl), int)], index=D.index)
    years = pd.Series(np.nan, index=D.index); years.loc[f.index] = days / 365.25
    X, cat = preprocess(D)
    flow['cases_final'] = int(y.sum()); flow['controls_final'] = int((1 - y).sum())
    flow['median_years_to_dx'] = float(years.median()); flow['iqr_years'] = [float(years.quantile(.25)), float(years.quantile(.75))]
    rows, horizon, imps = [], [], []
    rcv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for sex in ['Male', 'Female']:
        idx = X.index[X['Sex 31'] == sex]
        Dx = X.loc[idx].drop(columns=['Sex 31']); yy = y[idx].values; yr = years[idx].values
        for fo, (tr, te) in enumerate(rcv.split(Dx, yy)):
            mdl = lgbm().fit(Dx.iloc[tr], yy[tr])
            p = mdl.predict_proba(Dx.iloc[te])[:, 1]
            rows.append(dict(sex=sex, fold=fo, n=len(yy), cases=int(yy.sum()), **metrics(yy[te], p)))
            yte, yrte = yy[te], yr[te]
            for lab, sel in [('<5 years', yrte < 5), ('5–10 years', (yrte >= 5) & (yrte < 10)), ('≥10 years', yrte >= 10)]:
                keep = (yte == 0) | sel
                horizon.append(dict(sex=sex, fold=fo, horizon=lab, AUROC=roc_auc_score(yte[keep], p[keep])))
            imps.append(pd.Series(mdl.booster_.feature_importance('gain'), index=Dx.columns, name=f'{sex}{fo}'))
        print(f'C {sex}: n={len(yy)}, cases={yy.sum()}', flush=True)
    imp = pd.concat(imps, axis=1)
    return pd.DataFrame(rows), pd.DataFrame(horizon), imp, flow

if __name__ == '__main__':
    import json, os
    os.makedirs(OUT, exist_ok=True)
    part = sys.argv[1]
    if part in ('A', 'B', 'audit'):
        X, y, cat, snps, hba1c = load_prevalent()
        print('features (excl. sex/age):', X.shape[1] - 2, ' SNPs:', len(snps), flush=True)
    if part == 'audit':
        print('controls with HbA1c>=48:', int(((y == 0) & (hba1c >= 48)).sum()), 'of', int((y == 0).sum()))
        print('cases with HbA1c>=48:', int(((y == 1) & (hba1c >= 48)).sum()), 'of', int(y.sum()))
        rows = [dict(sex=s, age=a, n=len(i), cases=int(y[i].sum()), controls=int((1 - y[i]).sum())) for s, a, i in strata(X)]
        print(pd.DataFrame(rows).to_string(index=False))
    if part == 'A':
        r = part_a(X, y, cat); r.to_csv(OUT + 'A_reproduction.csv', index=False); print(r.round(3).to_string(index=False))
    if part == 'B':
        k = int(sys.argv[2]); r = part_b(X, y, snps, only=k); r.to_csv(OUT + f'B_cv_folds_{k}.csv', index=False)
    if part == 'C':
        r, h, imp, flow = part_c()
        r.to_csv(OUT + 'C_incident_folds.csv', index=False); h.to_csv(OUT + 'C_incident_horizon.csv', index=False)
        imp.to_csv(OUT + 'C_incident_importance.csv'); json.dump(flow, open(OUT + 'C_flow.json', 'w'), indent=1)
        print(flow)
