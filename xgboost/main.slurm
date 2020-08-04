#!/bin/bash

#SBATCH -J xgb_state
#SBATCH -A jjohn273
#SBATCH -e errors.hcpcs_state.%j.txt
#SBATCH -o output.hcpcs_state.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH -w node090
#SBATCH --exclusive
#SBATCH --mem=110gb
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=5 \
  n_jobs=2 \
  embedding_type=skipgram \
  drop_columns=nppes_provider_state \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/embeddings-e150-w5-i100-t080120194624.kv
