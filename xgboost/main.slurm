#!/bin/bash

#SBATCH -J xgb_sg_dstate_d8
#SBATCH -A jjohn273
#SBATCH -e errors.l25.%j.txt
#SBATCH -o output.l25.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem-per-cpu=90000
#SBATCH -w node089

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=5 \
  n_jobs=4 \
  sample_size=5000000 \
  max_depth=4 \
  l2_reg=2 \
  drop_columns=state_code \
  embedding_type=skipgram \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/embeddings-e150-w5-i100-t080120194624.kv
