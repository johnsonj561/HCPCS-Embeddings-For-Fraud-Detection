import os
import pandas as pd
import numpy as np


proj_dir = os.environ['HCPCS_PROJ_DIR']
raw_data_path = os.environ['HCPCS_RAW_DATA']


def load_data(sample_size):
    path = os.environ['DATA_2012_2015_CMS']
    columns = ['npi', 'provider_type', 'nppes_provider_state', 'nppes_provider_gender', 'hcpcs_code',
        'line_srvc_cnt', 'bene_unique_cnt', 'bene_day_srvc_cnt', 'average_submitted_chrg_amt',
        'average_medicare_payment_amt',
        'year', 'exclusion']
    return pd.read_csv(path, sep=chr(1), usecols=columns) \
        .sample(n=sample_size) \
        .reset_index(inplace=True)


def get_minority_size(df):
    maj, mino = sample['exclusion'].value_counts()
    return mino / (maj+mino) * 100
