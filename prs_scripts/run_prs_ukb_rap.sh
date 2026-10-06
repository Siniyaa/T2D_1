#!/usr/bin/env bash
# =============================================================================
# T2D polygenic risk score (PGS000014; Khera et al., Nat Genet 2018) in UK Biobank
# Run on the UK Biobank Research Analysis Platform (DNAnexus) in a JupyterLab or
# Swiss Army Knife instance (>= 16 GB RAM recommended), or on any server that
# holds the UK Biobank imputed BGEN files (Field 22828).
#
# Inputs : keep.txt  -> one column of participant IDs (eid), made by make_keep.py
# Output : prs_t2d.tsv -> eid, PRS_raw, PRS_z, n_variants_used   (upload this file)
# =============================================================================
set -euo pipefail

# ---------------- CONFIG (edit if your paths differ) ----------------
BGEN_DIR="/mnt/project/Bulk/Imputation/UKB imputation from genotype"   # RAP default
BGEN_PREFIX="ukb22828_c"            # files: ukb22828_c{CHR}_b0_v3.bgen / .sample
BGEN_SUFFIX="_b0_v3"
KEEP="keep.txt"
THREADS=8
MEM_MB=14000
WORK="prs_work"
# --------------------------------------------------------------------

mkdir -p "$WORK"
command -v plink2 >/dev/null 2>&1 || {
  echo "Installing plink2..."
  wget -q https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20241206.zip -O plink2.zip
  unzip -o -q plink2.zip && chmod +x plink2 && export PATH="$PWD:$PATH"
}

# 1. Download the harmonised (GRCh37) scoring file from the PGS Catalog
SCORE_GZ="$WORK/PGS000014_hmPOS_GRCh37.txt.gz"
[ -s "$SCORE_GZ" ] || wget -q -O "$SCORE_GZ" \
  "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000014/ScoringFiles/Harmonized/PGS000014_hmPOS_GRCh37.txt.gz"

# 2. Build a plink2 score file with allele-order-independent IDs  chr:pos:A1:A2 (alleles sorted)
python3 - "$SCORE_GZ" "$WORK" << 'PY'
import sys, gzip, pandas as pd
src, work = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, sep="\t", comment="#", low_memory=False)
chr_col = "hm_chr" if "hm_chr" in df else "chr_name"
pos_col = "hm_pos" if "hm_pos" in df else "chr_position"
oth = df["other_allele"] if "other_allele" in df else pd.Series(pd.NA, index=df.index)
if "hm_inferOtherAllele" in df:
    oth = oth.fillna(df["hm_inferOtherAllele"])
df = df.assign(chr=df[chr_col].astype(str), pos=pd.to_numeric(df[pos_col], errors="coerce"),
               ea=df["effect_allele"].astype(str).str.upper(), oa=oth.astype(str).str.upper())
ok = df["pos"].notna() & df["oa"].str.fullmatch(r"[ACGT]+") & df["ea"].str.fullmatch(r"[ACGT]+") \
     & df["chr"].isin([str(c) for c in range(1, 23)])
df = df[ok].copy()
df["pos"] = df["pos"].astype(int)
a = df[["ea", "oa"]].min(axis=1); b = df[["ea", "oa"]].max(axis=1)
df["ID"] = df["chr"] + ":" + df["pos"].astype(str) + ":" + a + ":" + b
df = df.drop_duplicates("ID", keep=False)
for c, g in df.groupby("chr"):
    g[["ID", "ea", "effect_weight"]].to_csv(f"{work}/score_chr{c}.txt", sep="\t", index=False)
print(f"Scoring variants usable: {len(df):,}")
PY

# 3. Score each chromosome
for CHR in $(seq 1 22); do
  [ -s "$WORK/prs_chr${CHR}.sscore" ] && continue
  echo "Chromosome ${CHR}..."
  plink2 --bgen "${BGEN_DIR}/${BGEN_PREFIX}${CHR}${BGEN_SUFFIX}.bgen" ref-first \
         --sample "${BGEN_DIR}/${BGEN_PREFIX}${CHR}${BGEN_SUFFIX}.sample" \
         --keep "$KEEP" \
         --set-all-var-ids '@:#:$1:$2' --new-id-max-allele-len 50 truncate \
         --extract <(cut -f1 "$WORK/score_chr${CHR}.txt" | tail -n +2) \
         --score "$WORK/score_chr${CHR}.txt" 1 2 3 header cols=+scoresums,+nallele list-variants \
         --threads "$THREADS" --memory "$MEM_MB" \
         --out "$WORK/prs_chr${CHR}" > /dev/null
done

# 4. Combine chromosomes and standardise
python3 - "$WORK" << 'PY'
import sys, glob, pandas as pd
work = sys.argv[1]; tot = None; nvar = 0
for f in sorted(glob.glob(f"{work}/prs_chr*.sscore")):
    s = pd.read_csv(f, sep=r"\s+")
    s = s.rename(columns={"#IID": "IID", "#FID": "FID"})
    s = s.set_index("IID")["SCORE1_SUM"]
    tot = s if tot is None else tot.add(s, fill_value=0)
    v = f.replace(".sscore", ".sscore.vars")
    nvar += sum(1 for _ in open(v)) if glob.glob(v) else 0
out = pd.DataFrame({"eid": tot.index, "PRS_raw": tot.values})
out["PRS_z"] = (out.PRS_raw - out.PRS_raw.mean()) / out.PRS_raw.std()
out["n_variants_used"] = nvar
out.to_csv("prs_t2d.tsv", sep="\t", index=False)
print(f"Done: {len(out):,} participants, {nvar:,} variants used -> prs_t2d.tsv")
PY
