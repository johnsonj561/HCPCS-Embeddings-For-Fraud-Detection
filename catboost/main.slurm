#!/bin/bash

#SBATCH -J cms_catboost
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=1 \
  embedding_type=categorical \
  x_train_filename=x-train-onehot-hcpcs.npz \
  x_test_filename=x-test-onehot-hcpcs.npz \
  sample_size=500000 \
  n_jobs=-1 \
  debug=true
