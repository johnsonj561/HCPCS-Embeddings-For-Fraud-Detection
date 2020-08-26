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
  runs=1 \
  n_jobs=4 \
  sample_size=100000 \
  max_depth=8 \
  embedding_type=skipgram \
  embedding_path=/Users/jujohnson/git/HCPCS-Embeddings-For-Fraud-Detection/data/embeddings-e150-w5-i100-t080120194624.kv
