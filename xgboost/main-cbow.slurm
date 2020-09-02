#!/bin/bash

#SBATCH -J xgb_cbow300
#SBATCH -A jjohn273
#SBATCH -e errors.cbow300.%j.txt
#SBATCH -o output.cbow300.%j.txt
#SBATCH -p shortq7
#SBATCH --mem-per-cpu=100000

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=1 \
  n_jobs=4 \
  sample_size=4000000 \
  max_depth=8 \
  drop_columns=state_code \
  embedding_type=cbow300_w10 \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/cbow-e300-w10-i200.kv

