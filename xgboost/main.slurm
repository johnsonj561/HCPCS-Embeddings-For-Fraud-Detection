#!/bin/bash

#SBATCH -J xgb_state_hcpcs
#SBATCH -A jjohn273
#SBATCH -e errors.hcpcs_state.%j.txt
#SBATCH -o output.hcpcs_state.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem-per-cpu=80000
source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=1 \
  n_jobs=4 \
  sample_size=4000000 \
  embedding_type=skipgram \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/embeddings-e150-w5-i100-t080120194624.kv
