T2D POLYGENIC RISK SCORE (PGS000014) - HOW TO RUN
=================================================
Where : UK Biobank Research Analysis Platform (DNAnexus) - JupyterLab or Swiss Army Knife,
        or any server with the UK Biobank imputed BGEN files (Field 22828).
Time  : roughly 2-6 hours (22 chromosomes).

1. Copy medical_history_3.tsv and medical_history_fut.tsv to the same folder as these scripts.
2. python make_keep.py medical_history_3.tsv medical_history_fut.tsv
3. bash run_prs_ukb_rap.sh
      - downloads the PGS000014 scoring file from the PGS Catalog
      - scores every chromosome with plink2
      - writes prs_t2d.tsv
4. If your BGEN files are in another location, edit BGEN_DIR at the top of run_prs_ukb_rap.sh.
5. Optional but recommended: also export genetic principal components 1-10 (Field 22009)
   for the same participants as pcs.tsv (columns: eid, PC1..PC10).
6. Upload prs_t2d.tsv (and pcs.tsv if available) back to the chat.

The script writes progress to the screen; if it stops, re-run it - finished chromosomes are skipped.
