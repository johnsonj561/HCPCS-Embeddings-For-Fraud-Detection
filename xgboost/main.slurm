#!/bin/bash

#SBATCH -J xgb_hcpcs
#SBATCH -A jjohn273
#SBATCH -e errors.hcpcs.%j.txt
#SBATCH -o output.hcpcs.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem-per-cpu=90000
#SBATCH -w node096

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=5 \
  n_jobs=4 \
  sample_size=5000000 \
  max_depth=8 \
  embedding_type=onehot \
  l1_reg=5
