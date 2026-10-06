#!/usr/bin/env bash
# Runs the full analysis. Input files must be in ../data (see README).
set -e
cd code
python3 t2d_pipeline.py A
for k in $(seq 0 9); do python3 t2d_pipeline.py B $k; done
python3 t2d_pipeline.py C
python3 extra.py
python3 sens.py 0,1,2,3,4,5,6,7,8,9
python3 inc_matched.py
python3 snp_qc.py
for k in $(seq 0 9); do python3 geno2.py $k; done
python3 grs_qc.py
for k in $(seq 0 9); do python3 grs_cv.py $k; done
python3 pca.py
for k in $(seq 0 9); do python3 grs_pc.py $k; done
python3 nhanes_ext.py && python3 nhanes_val.py
python3 summarise.py
python3 fig1.py && python3 figs.py
echo "Done. Results in ../results, figures in ../figures"
