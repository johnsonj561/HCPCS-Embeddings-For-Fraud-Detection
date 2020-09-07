#!/bin/bash

#SBATCH -J xgb_none
#SBATCH -A jjohn273
#SBATCH -e errors.none.%j.txt
#SBATCH -o output.none.%j.txt
#SBATCH -p longq7
#SBATCH --mem-per-cpu=100000
#SBATCH -w node013

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=6 \
  n_jobs=4 \
  sample_size=4000000 \
  max_depth=8 \
  drop_columns=state_code,hcpcs_code \
  embedding_type=none

