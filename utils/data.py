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


def load_data(sample_size=None):
    columns = ['npi', 'provider_type', 'nppes_provider_state', 'nppes_provider_gender', 'hcpcs_code',
               'line_srvc_cnt', 'bene_unique_cnt', 'bene_day_srvc_cnt', 'average_submitted_chrg_amt',
               'average_medicare_payment_amt',
               'year', 'exclusion']
    df = pd.read_csv(raw_data_path, sep=chr(1), usecols=columns)
    df['nppes_provider_gender'].fillna('M', inplace=True)
    if sample_size != None:
        df = df.sample(frac=sample_size)
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
        return embeddings[key]
    except KeyError:
        return np.zeros(shape=(embeddings.vector_size))


def get_embedded_data(df, embedding_type, embedding_path, drop_columns):
    y = df['exclusion']
    drop_columns = ['index', 'npi', 'year', 'exclusion', *drop_columns]
    df = df.drop(columns=drop_columns)
    if embedding_type == 'onehot':
        df = pd.get_dummies(df, sparse=True)
        df = df_to_csr(df)
    if embedding_type == 'skipgram' or embedding_type == 'cbow':
        embeddings = KeyedVectors.load(embedding_path)
        hcpcs = df['hcpcs_code'].values
        df.drop(columns=['hcpcs_code'], inplace=True)
        df = pd.concat(
            [
                df,
                pd.DataFrame([safe_embedding(embeddings, x) for x in hcpcs],
                             columns=[f'hcpcs_{i}' for i in range(
                                 embeddings.vector_size)],
                             index=df.index, dtype='float32'),
            ],
            axis=1
        )
        df = pd.get_dummies(df).values
        df = df_to_csr(df)

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
