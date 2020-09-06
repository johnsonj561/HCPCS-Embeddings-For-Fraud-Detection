#!/bin/bash

#SBATCH -J xgb_sg300
#SBATCH -A jjohn273
#SBATCH -e errors.sg300.%j.txt
#SBATCH -o output.sg300.%j.txt
#SBATCH -p longq7
#SBATCH --mem-per-cpu=100000
#SBATCH -w node006

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=1 \
  n_jobs=4 \
  sample_size=4000000 \
  max_depth=8 \
  drop_columns=state_code \
  embedding_type=skipgram300_w5 \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/skipgram-e300-w5-i100.kv

