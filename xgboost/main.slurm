#!/bin/bash

#SBATCH -J xgb_state
#SBATCH -A jjohn273
#SBATCH -e errors.hcpcs_state.%j.txt
#SBATCH -o output.hcpcs_state.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH -w node088
#SBATCH --mem=50gb
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=5 \
  n_jobs=8 \
  embedding_type=onehot \
  drop_columns=hcpcs_code
