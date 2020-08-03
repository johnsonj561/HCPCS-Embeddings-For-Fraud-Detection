#!/bin/bash

#SBATCH -J xgb_state_hcpcs
#SBATCH -A jjohn273
#SBATCH -e errors.state_hcpcs.%j.txt
#SBATCH -o output.state_hcpcs.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem=50gb
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=5 \
  n_jobs=8 \
  embedding_type=onehot \
  embedding_path=data/embeddings-e150-w5-i100-t080120194624.kv
