#!/bin/bash

#SBATCH -J catboost_hcpcs
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH -w node089
#SBATCH --exclusive
#SBATCH --mem=80gb
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=5 \
  n_jobs=8 \
  embedding_type=catboost \
  drop_columns=nppes_provider_state \
  categorical_columns=provider_type,nppes_provider_gender,hcpcs_code
