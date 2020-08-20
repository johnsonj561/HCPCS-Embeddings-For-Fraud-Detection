#!/bin/bash

#SBATCH -J rf_skipgram300
#SBATCH -A jjohn273
#SBATCH -e errors.skipgram300.%j.txt
#SBATCH -o output.skipgram300.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem-per-cpu=110000
#SBATCH -w node090

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=5 \
  n_jobs=4 \
  sample_size=5000000 \
  max_depth=8 \
  drop_columns=state_code \
  embedding_type=skipgram300 \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/skipgram-e300-w5-i100.kv
