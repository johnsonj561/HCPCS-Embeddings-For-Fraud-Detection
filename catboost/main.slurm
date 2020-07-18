#!/bin/bash

#SBATCH -J catboost_hcpcs_state
#SBATCH -A jjohn273
#SBATCH -e errors.hcpcs_state_.%j.txt
#SBATCH -o output.hcpcs_state.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH -w node089
#SBATCH --mem=50gb
#SBATCH --mail-user=jjohn273@fau.edu
#SBATCH --mail-type=ALL

source $CMS_ROOT/env/bin/activate

srun python3 main.py \
  runs=1 \
  n_jobs=8 \
  sample_size=500000 \
  debug=true \
  embedding_type=catboost \
  categorical_columns=provider_type,nppes_provider_state,nppes_provider_gender,hcpcs_code
