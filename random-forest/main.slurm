#!/bin/bash

#SBATCH -J rf_sg150_2
#SBATCH -A jjohn273
#SBATCH -e errors.sg150_2.%j.txt
#SBATCH -o output.sg150_22.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem-per-cpu=110000
#SBATCH -w node087

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=5 \
  n_jobs=4 \
  sample_size=5000000 \
  max_depth=8 \
  drop_columns=state_code \
  embedding_type=skipgram150_2 \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/sg-e150-w5-i100.kv
