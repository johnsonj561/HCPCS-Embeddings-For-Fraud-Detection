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
    'line_srvc_cnt': 'float32',
    'bene_unique_cnt': 'float32',
    'bene_day_srvc_cnt': 'float32',
    'average_submitted_chrg_amt': 'float32',
    'average_medicare_payment_amt': 'float32',
    'year': 'int16',
    'exclusion': 'int8'
}

#columns = {
#    'npi': 'int64',
#    'provider_type': 'category',
#    'nppes_provider_state': 'category',
#    'nppes_provider_gender': 'category',
#    'hcpcs_code': 'category',
#    'line_srvc_cnt': 'float32',
#    'bene_unique_cnt': 'float32',
#    'bene_day_srvc_cnt': 'float32',
#    'average_submitted_chrg_amt': 'float32',
#    'average_medicare_payment_amt': 'float32',
#    'year': 'int16',
#    'exclusion': 'int8'
#}

def load_data(sample_size=None):
    print(f'Loading data from path {raw_data_path}')
    df = pd.read_csv(raw_data_path, usecols=columns.keys(), dtype=columns)
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f'Loaded data with shape: {df.shape}')
    df.dropna(inplace=True)
    print(f'Dropped nan, updated shape: {df.shape}')
    if sample_size != None:
        df = df.sample(n=sample_size)
    df.reset_index(inplace=True)
    return df


def load_sampled_data(sample_size):
    raw_data_path = os.environ['CMS_PARTB_PATH']
    print(f'Loading data from path {raw_data_path}')
    df = pd.read_csv(raw_data_path, usecols=columns.keys(), dtype=columns)
    print(f'Loaded data with shape: {df.shape}')
    df.dropna(inplace=True)
    print(f'Dropped nan, updated shape: {df.shape}')
    pos_mask = df.exclusion == 1
    pos_samples = df.loc[pos_mask]
    pos_count = len(pos_samples)
    neg_count = sample_size - pos_count
    neg_samples = df.loc[~pos_mask].sample(neg_count)
    print(f'Positive sample count: {pos_count} {pos_count / (pos_count + neg_count) * 100}%')
    print(f'Negative sample count: {neg_count} {neg_count / (pos_count + neg_count) * 100}%')
    return pd.concat([pos_samples, neg_samples], axis=0) \
        .sample(frac=1) \
        .reset_index(drop=True)



def get_minority_size(df):
    maj, mino = df['exclusion'].value_counts()
    return mino / (maj + mino) * 100


def df_to_csr(df):
    df = df.to_sparse().to_coo().astype('float32')
    return df.tocsr()


def safe_embedding(embeddings, key, embedding_type):
    try:
        return np.array(embeddings[key]).astype('float32')
    except KeyError:
        if 'uniform' in embedding_type or 'choi' in embedding_type:
            d = len(list(embeddings.values())[0])
        else:
            d = embeddings.vector_size
        return np.random.uniform(low=0, high=1, size=d)


def get_embedded_data(df, embedding_type, embedding_path, drop_columns):
    y = df['exclusion']
    drop_columns = ['npi', 'year', 'exclusion', *drop_columns]
    df = df.drop(columns=drop_columns)
    print(f'Using columns {df.columns}')
    if 'onehot' in embedding_type or 'none' in embedding_type:
        print('Using onehot embedding')
        df = pd.get_dummies(df, sparse=True)
        df = df_to_csr(df)
    if 'skipgram' in embedding_type or 'cbow' in embedding_type or 'choi' in embedding_type or 'uniform' in embedding_type:
        print(f'Using {embedding_type} embedding')
        if 'choi' in embedding_type or 'uniform' in embedding_type:
          with open(embedding_path, 'rb') as fin:
            embeddings = pickle.load(fin)
        else:
          embeddings = KeyedVectors.load(embedding_path)

        embeddings = np.array([safe_embedding(embeddings, x, embedding_type)
                                for x in df['hcpcs_code'].values])
        df.drop(columns=['hcpcs_code'], inplace=True)
        for col in range(embeddings.shape[1]):
            df[f'hcpcs_{col}'] = embeddings[:, col]
        df = pd.get_dummies(df)

    print(f'Embedded data shape: {df.shape}')
    return df, y

