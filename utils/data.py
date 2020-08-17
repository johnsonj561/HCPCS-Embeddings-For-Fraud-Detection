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
from gensim.models import KeyedVectors

proj_dir = os.environ['CMS_ROOT']
raw_data_path = os.environ['CMS_PARTB_PATH']

columns = {
    'npi': 'int64',
    'provider_type': 'category',
    'state_code': 'category',
    'gender': 'category',
    'hcpcs_code': 'category',
    'line_srvc_cnt': 'float16',
    'bene_unique_cnt': 'float16',
    'bene_day_srvc_cnt': 'float16',
    'average_submitted_chrg_amt': 'float16',
    'average_medicare_payment_amt': 'float16',
    'year': 'int16',
    'exclusion': 'int8'
}


def load_data(sample_size=None):
    df = pd.read_csv(raw_data_path, usecols=columns.keys(), dtype=columns)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f'Loaded data with shape: {df.shape}')
    df.dropna(inplace=True)
    print(f'Dropped nan, updated shape: {df.shape}')
    if sample_size != None:
        df = df.sample(n=sample_size)
    df.reset_index(inplace=True)
    return df


def get_minority_size(df):
    maj, mino = df['exclusion'].value_counts()
    return mino / (maj + mino) * 100


def df_to_csr(df):
    df = df.to_sparse().to_coo().astype('float32')
    return df.tocsr()


def safe_embedding(embeddings, key):
    try:
        return embeddings[key].astype('float16')
    except KeyError:
        return np.zeros(shape=(embeddings.vector_size), dtype='float16')


def get_embedded_data(df, embedding_type, embedding_path, drop_columns):
    y = df['exclusion']
    drop_columns = ['npi', 'year', 'exclusion', *drop_columns]
    df = df.drop(columns=drop_columns)
    if embedding_type == 'onehot':
        print('Using onehot embedding')
        df = pd.get_dummies(df, sparse=True)
        df = df_to_csr(df)
    if 'skipgram' in embedding_type or 'cbow' in embedding_type:
        print(f'Using {embedding_type} embedding')
        embeddings = KeyedVectors.load(embedding_path)
        embeddings = np.array([safe_embedding(embeddings, x)
                               for x in df['hcpcs_code'].values])
        df.drop(columns=['hcpcs_code'], inplace=True)
        for col in range(embeddings.shape[1]):
            df[f'hcpcs_{col}'] = embeddings[:, col]
        df = pd.get_dummies(df)

    return df, y


def get_train_test(df, with_hcpcs=True, with_categorical=False):
    y = df['exclusion']
    train_ind, test_ind = train_test_split(
        np.arange(0, df.shape[0], 1), test_size=0.2, random_state=42)
    if with_categorical:
        train_x = df.iloc[train_ind]
        test_x = df.iloc[test_ind]
        train_y = y[train_ind]
        test_y = y[test_ind]
        return train_x, test_x, train_y, test_y

    if not with_hcpcs:
        df = df.drop(columns=['hcpcs_code'])

    df = pd.get_dummies(df, sparse=True)
    df = df_to_csr(df)
    train_x = df[train_ind]
    test_x = df[test_ind]
    train_y = y[train_ind]
    test_y = y[test_ind]
    return train_x, test_x, train_y, test_y
