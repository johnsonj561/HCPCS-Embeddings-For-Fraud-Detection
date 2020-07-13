from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score
from io import StringIO
import numpy as np
import pickle
import time
import math
import os
import sys
import pandas as pd
import h5py
import warnings

proj_dir = os.environ['CMS_ROOT']
raw_data_path = os.environ['CMS_PARTB_PATH']

def load_data(sample_size):
    columns = ['npi', 'provider_type', 'nppes_provider_state', 'nppes_provider_gender', 'hcpcs_code',
        'line_srvc_cnt', 'bene_unique_cnt', 'bene_day_srvc_cnt', 'average_submitted_chrg_amt',
        'average_medicare_payment_amt',
        'year', 'exclusion']
    df = pd.read_csv(raw_data_path, sep=chr(1), usecols=columns)
    df = df.sample(n=sample_size)
    df.reset_index(inplace=True)
    return df

def get_minority_size(df):
    maj, mino = df['exclusion'].value_counts()
    return mino / (maj+mino) * 100

def df_to_csr(df):
    df = df.to_sparse().to_coo().astype('float32')
    return df.tocsr()

def get_train_test(df, with_hcpcs=True):
  y = df['exclusion']
  df = df.drop(columns=['index', 'npi', 'year', 'exclusion'])
  if not with_hcpcs:
    df = df.drop(columns=['hcpcs_code'])
  df = pd.get_dummies(df, sparse=True)
  df = df_to_csr(df)
  train_ind, test_ind = train_test_split(
    np.arange(0, df.shape[0], 1), test_size=0.2, random_state=42)
  train_x = df[train_ind]
  test_x = df[test_ind]
  train_y = y[train_ind]
  test_y = y[test_ind]
  return train_x, test_x, train_y, test_y