#!/bin/bash

#SBATCH -J xgb_cbow300
#SBATCH -A jjohn273
#SBATCH -e errors.cbow300.%j.txt
#SBATCH -o output.cbow300.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem-per-cpu=90000
#SBATCH -w node087

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=2 \
  n_jobs=4 \
  sample_size=5000000 \
  max_depth=4 \
  drop_columns=state_code \
  embedding_type=cbow300 \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/cbow-e300-w5-i200.kv

