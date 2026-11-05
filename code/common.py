"""Shared loading, harmonisation and model code for the T2D revision analyses.

Environment variables
---------------------
T2D_DATA_DIR  directory holding medical_history_3.tsv and medical_history_fut.tsv
NHANES_DIR    directory holding the NHANES 2017-March 2020 .xpt files (downloaded on demand)
T2D_OUT_DIR   directory for results (default ../results)
"""
import os, re, urllib.request
import numpy as np
import pandas as pd

DATA = os.environ.get("T2D_DATA_DIR", "../data")
NH   = os.environ.get("NHANES_DIR", "../nhanes")
OUT  = os.environ.get("T2D_OUT_DIR", "../results")
SEED = 42
os.makedirs(OUT, exist_ok=True)

HBA   = "Glycated haemoglobin (HbA1c) [30750]"
GLU   = "Glucose [30740]"
SEX   = "Sex [31]"
AGE_U = "Age when attended assessment centre [21003]"
AGE_N = "RIDAGEYR"

AGE_BINS = {"<50": (0, 50), "50-54": (50, 55), "55-59": (55, 60),
            "60-64": (60, 65), ">=65": (65, 200)}

# The 36 model inputs: 35 clinical predictors plus age, in matching units.
# This mapping is Supplementary Table S12 and MUST stay identical to it.
MAP = {
 "Waist circumference [48]": "BMXWAIST", "Hip circumference [49]": "BMXHIP",
 "Body mass index (BMI) [21001]": "BMXBMI", "Weight [21002]": "BMXWT",
 "Standing height [50]": "BMXHT",
 "Systolic blood pressure, automated reading [4080]": "SBP",
 "Diastolic blood pressure, automated reading [4079]": "DBP",
 "Pulse rate, automated reading [102]": "PULSE",
 "Alanine aminotransferase [30620]": "LBXSATSI",
 "Aspartate aminotransferase [30650]": "LBXSASSI",
 "Alkaline phosphatase [30610]": "LBXSAPSI",
 "Gamma glutamyltransferase [30730]": "LBXSGTSI",
 "Urate [30880]": "LBDSUASI", "Creatinine [30700]": "LBDSCRSI",
 "Urea [30670]": "LBDSBUSI", "Albumin [30600]": "LBDSALSI",
 "Total protein [30860]": "LBDSTPSI", "Total bilirubin [30840]": "LBDSTBSI",
 "Calcium [30680]": "LBDSCASI", "Phosphate [30810]": "LBDSPHSI",
 "Cholesterol [30690]": "LBDSCHSI", "Triglycerides [30870]": "LBDSTRSI",
 "HDL cholesterol [30760]": "LBDHDDSI", "LDL direct [30780]": "LBDLDLSI",
 "C-reactive protein [30710]": "LBXHSCRP",
 "White blood cell (leukocyte) count [30000]": "LBXWBCSI",
 "Red blood cell (erythrocyte) count [30010]": "LBXRBCSI",
 "Haemoglobin concentration [30020]": "LBXHGB",
 "Haematocrit percentage [30030]": "LBXHCT",
 "Mean corpuscular volume [30040]": "LBXMCVSI",
 "Red blood cell (erythrocyte) distribution width [30070]": "LBXRDW",
 "Platelet count [30080]": "LBXPLTSI",
 "Lymphocyte count [30120]": "LBDLYMNO",
 "Monocyte count [30130]": "LBDMONO",
 "Neutrophill count [30140]": "LBDNENO"}

UCOLS = list(MAP) + [AGE_U]
NCOLS = list(MAP.values()) + [AGE_N]
SAFE  = [re.sub(r"[^0-9A-Za-z]+", "_", c).strip("_") for c in UCOLS]

# dropped from the 175-predictor set (diagnostic tests, administrative, non-predictive)
DROP_PRED = [
    "Date of attending assessment centre [53]", "Ethnic background [21000]", HBA, GLU,
    "Overall health rating [2178]", "Age first had sexual intercourse [2139]",
    "EID", "HasT2D", SEX]


def load_prevalent():
    """Case-finding analysis set: 20,916 participants, one row per EID."""
    df = pd.read_csv(os.path.join(DATA, "medical_history_3.tsv"), sep="\t", low_memory=False)
    assert df["EID"].is_unique, "medical_history_3.tsv must have one row per participant"
    return df


def load_incident_cases(df_prev=None):
    """Incident cases: British, first E11 after baseline, baseline HbA1c < 48 mmol/mol."""
    fut = pd.read_csv(os.path.join(DATA, "medical_history_fut.tsv"), sep="\t", low_memory=False)
    fut = fut[fut["Ethnic background [21000]"] == "British"]
    return fut[~(pd.to_numeric(fut[HBA], errors="coerce") >= 48)]


def family_history(df):
    cols = [c for c in df.columns if "had diabetes" in c]
    return df[cols].apply(lambda s: s.astype(str).str.lower().eq("true")).any(axis=1)


def strata(df):
    """Yield (sex, age label, subframe) for the ten case-finding strata."""
    age = pd.to_numeric(df[AGE_U], errors="coerce")
    for sex in ("Male", "Female"):
        for lab, (lo, hi) in AGE_BINS.items():
            yield sex, lab, df[(df[SEX] == sex) & (age >= lo) & (age < hi)]


def harmonised_X(df, source="ukb"):
    cols = UCOLS if source == "ukb" else NCOLS
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X.columns = SAFE
    return X


def fit_harmonised(df_prev, sex):
    """Fit the 35-predictor sex-specific models used for external validation."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    sub = df_prev[df_prev[SEX] == sex]
    X, y = harmonised_X(sub), sub["HasT2D"].astype(int).values
    gbm = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-1,
                             random_state=SEED, verbose=-1).fit(X, y)
    # L2 (scikit-learn default) penalisation: the specification used for the
    # externally evaluated harmonised model (Supplementary Table S16).
    lr = LogisticRegression(max_iter=2000, random_state=SEED)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("lr", lr)]).fit(X, y)
    return gbm, pipe, X, y


NHANES_FILES = ["P_DEMO", "P_BMX", "P_BPXO", "P_DIQ", "P_BIOPRO", "P_CBC",
                "P_HDL", "P_TRIGLY", "P_HSCRP", "P_GLU", "P_GHB"]
NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"


def load_nhanes():
    """Merge the NHANES 2017-March 2020 public-use files, downloading if absent."""
    import functools
    os.makedirs(NH, exist_ok=True)
    nh = {}
    for f in NHANES_FILES:
        p = os.path.join(NH, f + ".xpt")
        if not os.path.exists(p):
            urllib.request.urlretrieve(NHANES_BASE + f + ".xpt", p)
        nh[f] = pd.read_sas(p, format="xport")
    bp = nh["P_BPXO"].copy()
    bp["SBP"]   = bp[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    bp["DBP"]   = bp[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    bp["PULSE"] = bp[["BPXOPLS1", "BPXOPLS2", "BPXOPLS3"]].mean(axis=1)
    parts = [
        nh["P_DEMO"][["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "WTMECPRP",
                      "SDMVPSU", "SDMVSTRA"]],
        nh["P_BMX"][["SEQN", "BMXWAIST", "BMXHIP", "BMXBMI", "BMXWT", "BMXHT"]],
        bp[["SEQN", "SBP", "DBP", "PULSE"]],
        nh["P_BIOPRO"][["SEQN", "LBXSATSI", "LBXSASSI", "LBXSAPSI", "LBXSGTSI",
                        "LBDSUASI", "LBDSCRSI", "LBDSBUSI", "LBDSALSI", "LBDSTPSI",
                        "LBDSTBSI", "LBDSCASI", "LBDSPHSI", "LBDSTRSI", "LBDSCHSI"]],
        nh["P_CBC"][["SEQN", "LBXWBCSI", "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI",
                     "LBXRDW", "LBXPLTSI", "LBDLYMNO", "LBDMONO", "LBDNENO"]],
        nh["P_HDL"][["SEQN", "LBDHDDSI"]],
        nh["P_TRIGLY"][["SEQN", "LBDLDLSI", "WTSAFPRP"]],
        nh["P_HSCRP"][["SEQN", "LBXHSCRP"]], nh["P_GLU"][["SEQN", "LBDGLUSI"]],
        nh["P_GHB"][["SEQN", "LBXGH"]], nh["P_DIQ"][["SEQN", "DIQ010", "DID040", "DIQ050"]]]
    N = functools.reduce(lambda a, b: a.merge(b, on="SEQN", how="left"), parts)
    N = N[(N.RIDAGEYR >= 40) & (N.RIDAGEYR <= 69)].copy()
    N["t1d"] = (N.DID040 < 30) & (N.DIQ050 == 1)
    N["sexlab"] = N.RIAGENDR.map({1.0: "Male", 2.0: "Female"})
    return N


def score_nhanes(N, df_prev):
    """Attach LightGBM and logistic-regression predictions to the NHANES frame."""
    X = N[NCOLS].apply(pd.to_numeric, errors="coerce"); X.columns = SAFE
    for sex in ("Male", "Female"):
        m = N.sexlab == sex
        gbm, pipe, _, _ = fit_harmonised(df_prev, sex)
        N.loc[m, "p_lgbm"] = gbm.predict_proba(X[m])[:, 1]
        N.loc[m, "p_lr"] = pipe.predict_proba(X[m])[:, 1]
    return N


def nhanes_diagnosed(N, sex, nhw=False):
    """Diagnosed diabetes vs no diabetes. 'Borderline' responses count as controls."""
    d = N[N.sexlab == sex]
    if nhw:
        d = d[d.RIDRETH3 == 3]
    case = (d.DIQ010 == 1) & (~d.t1d)
    ctrl = d.DIQ010.isin([2.0, 3.0]) & (~((d.LBDGLUSI >= 7.0).fillna(False)))
    out = d[case | ctrl].copy()
    out["y"] = case[case | ctrl].astype(int)
    return out


def nhanes_undiagnosed(N, sex, ada=False):
    """Undiagnosed diabetes within the fasting subsample."""
    d = N[N.sexlab == sex]
    base = d.LBDGLUSI.notna() & d.DIQ010.isin([2.0, 3.0])
    hi = (d.LBDGLUSI >= 7.0).fillna(False)
    if ada:
        hi = hi | (d.LBXGH >= 6.5).fillna(False)
    out = d[base].copy()
    out["y"] = hi[base].astype(int)
    return out


def boot_ci(y, p, n=1000, seed=7):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx, out = np.arange(len(y)), []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) > 1:
            out.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

# ---------------------------------------------------------------------------
# Shared modelling-matrix construction.
#
# A column is treated as NUMERIC if more than half of its non-null values parse
# as numbers; non-parsing entries ("Do not know", "Prefer not to answer") then
# become missing, which is the cleaning described in Section 2.3.  A column
# whose levels are genuinely nominal (e.g. "No"/"Yes") would be destroyed by
# that rule, so it is instead given pandas `category` dtype with an explicit
# "NA" level, which LightGBM consumes natively as a categorical feature.
# Every analysis script uses this one function so that the matrix is identical
# across them.
# ---------------------------------------------------------------------------
NOMINAL_TO_NA = ("do not know", "prefer not to answer", "none of the above")


def model_matrix(df, cols, for_lightgbm=True):
    """Return (X, categorical_columns) with safe names, from the raw frame."""
    out, cats = {}, []
    for c in cols:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        n_nonnull = s.notna().sum()
        if n_nonnull and num.notna().sum() > 0.5 * n_nonnull:
            out[c] = num                                 # numeric; text -> missing
        else:
            v = s.astype(str).str.strip()
            v = v.where(~v.str.lower().isin(NOMINAL_TO_NA) & s.notna(), "NA")
            if for_lightgbm:
                out[c] = pd.Categorical(v)
            else:                                        # models needing numbers
                codes = pd.Series(pd.Categorical(v).codes, index=df.index, dtype=float)
                out[c] = codes.where(~v.eq("NA").values, np.nan)
            cats.append(c)
    X = pd.DataFrame(out, index=df.index)
    safe = lambda n: re.sub(r"[^0-9A-Za-z]+", "_", n).strip("_")
    X.columns = [safe(c) for c in X.columns]
    return X, [safe(c) for c in cats]
