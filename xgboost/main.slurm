#!/bin/bash

#SBATCH -J cms_xgb
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem=50gb
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=5 \
  embedding_type=nohcpcs \
  n_jobs=8
