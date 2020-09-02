#!/bin/bash

#SBATCH -J xgb_onehot
#SBATCH -A jjohn273
#SBATCH -e errors.onehot.%j.txt
#SBATCH -o output.onehot.%j.txt
#SBATCH -p longq7
#SBATCH --mem-per-cpu=100000
#SBATCH -w node008

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=3 \
  n_jobs=4 \
  sample_size=4000000 \
  max_depth=8 \
  drop_columns=state_code \
  embedding_type=onehot

