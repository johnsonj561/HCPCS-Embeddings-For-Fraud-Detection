#!/bin/bash

#SBATCH -J xgb_uniform75
#SBATCH -A jjohn273
#SBATCH -e errors.uniform75.%j.txt
#SBATCH -o output.uniform75.%j.txt
#SBATCH -p longq7
#SBATCH --mem-per-cpu=100000

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=2 \
  n_jobs=4 \
  sample_size=4000000 \
  max_depth=8 \
  drop_columns=state_code \
  embedding_type=uniform75 \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/uniform-e75.pickle

