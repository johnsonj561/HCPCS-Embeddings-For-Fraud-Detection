#!/bin/bash

#SBATCH -J rf_hcpcs_state
#SBATCH -A jjohn273
#SBATCH -e errors.hcpcs_state.%j.txt
#SBATCH -o output.hcpcs_state.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem-per-cpu=90000
#SBATCH -w node088

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=5 \
  n_jobs=4 \
  sample_size=5000000 \
  max_depth=8 \
  embedding_type=skipgram \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/embeddings-e150-w5-i100-t080120194624.kv
