#!/bin/bash

#SBATCH -J catboost_state
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p mediumq7
#SBATCH --mem=50gb
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=5 \
  n_jobs=8 \
  embedding_type=catboost \
  drop_columns=hcpcs_code \
  categorical_columns=provider_type,nppes_provider_gender,nppes_provider_state
