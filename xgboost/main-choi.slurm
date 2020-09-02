#!/bin/bash

#SBATCH -J xgb_choi300
#SBATCH -A jjohn273
#SBATCH -e errors.choi300.%j.txt
#SBATCH -o output.choi300.%j.txt
#SBATCH -p longq7
#SBATCH --mem-per-cpu=100000
#SBATCH -w node019

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=2 \
  n_jobs=4 \
  sample_size=4000000 \
  max_depth=8 \
  drop_columns=state_code \
  embedding_type=choi300 \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/choi-mcemc-e300.kv

