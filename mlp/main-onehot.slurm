#!/bin/bash

#SBATCH -J mlp_onehot
#SBATCH -A jjohn273
#SBATCH -e errors.mlp_onehot.%j.txt
#SBATCH -o output.mlp_onehot.%j.txt
#SBATCH -p longq7
#SBATCH --mem-per-cpu=110000
#SBATCH --exclusive

source $CMS_ROOT/env/bin/activate

#module load cuda-10.1.243-gcc-8.3.0-ti55azn
#module load cudnn-7.6.5.32-10.1-linux-x64-gcc-8.3.0-vldxhwt

srun python3 main.py \
  sample_size=4000000 \
  runs=5 \
  epochs=100 \
  drop_columns=state_code \
  embedding_type=onehot \
  hidden_layers=64+64 \
  batchnorm=true \
  epochs=100 \
  dropout_rate=0.5
