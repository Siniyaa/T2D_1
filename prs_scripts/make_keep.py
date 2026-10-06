"""Create keep.txt (participants to score) from the two analysis files.
Usage: python make_keep.py medical_history_3.tsv medical_history_fut.tsv"""
import sys, pandas as pd
ids = set()
for f in sys.argv[1:]:
    ids |= set(pd.read_csv(f, sep="\t", index_col=0, usecols=[0]).index.astype(str))
with open("keep.txt", "w") as fh:
    for i in sorted(ids):
        fh.write(f"{i}\t{i}\n")          # FID IID (UK Biobank .sample uses eid for both)
print(f"keep.txt written: {len(ids):,} participants")
