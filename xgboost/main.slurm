#!/bin/bash

#SBATCH -J xgb_oneho
#SBATCH -A jjohn273
#SBATCH -e errors.onehot.%j.txt
#SBATCH -o output.onehot.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem-per-cpu=90000
#SBATCH -w node088

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=2 \
  n_jobs=4 \
  sample_size=5000000 \
  max_depth=8 \
  drop_columns=state_code \
  embedding_type=onehot
  # embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/skipgram-e300-w5-i100.kv

