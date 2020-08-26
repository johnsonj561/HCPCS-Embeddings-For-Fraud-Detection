#!/bin/bash

#SBATCH -J mlp_sg150_64+64_do0pt5
#SBATCH -A jjohn273
#SBATCH -e errors.mlp_sg150_64+64_do0pt5.%j.txt
#SBATCH -o output.mlp_sg150_64+64_do0pt5.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem-per-cpu=90000
#SBATCH -w node093

source $CMS_ROOT/env/bin/activate

# module load cuda-10.1.243-gcc-8.3.0-ti55azn
# module load cudnn-7.6.5.32-10.1-linux-x64-gcc-8.3.0-vldxhwt

srun python3 main.py \
  sample_size=500000 \
  runs=2 \
  epochs=100 \
  drop_columns=state_code \
  embedding_type=skipgram150 \
  hidden_layers=64+64 \
  batchnorm=true \
  epochs=100 \
  dropout_rate=0.5 \
  embedding_path=/home/jjohn273/git/HCPCS-Embeddings-For-Fraud-Detection/data/skipgram-e150-w5-i100.kv
