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
from sklearn.exceptions import UndefinedMetricWarning


proj_dir = os.environ['CMS_ROOT']
raw_data_path = os.environ['CMS_PARTB_PATH']

# caused by divide by zero during metrics calcultiong
# ignoring  because it is saturating the error logs
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def args_to_dict(args, skip_filename=True):
    '''Splits command line args in the form of key=value
    Returns dictionary of key value pairs
    '''
    if skip_filename:
        args = args[1:]
    result = {}
    for arg in args:
        key, value = arg.split('=')
        result[key] = value
    return result


def split_on_binary_attribute(df, attribute, pos_label, neg_label):
    if attribute == None:
        pos_df = df.loc[df == pos_label]
        neg_df = df.loc[df == neg_label]
    else:
        pos_df = df.loc[df[attribute] == pos_label]
        neg_df = df.loc[df[attribute] == neg_label]
    return pos_df, neg_df


def get_imbalance_description(col, posLabel=1, negLabel=0):
    neg_count, pos_count = len(col[col == negLabel]), len(col[col == posLabel])
    minority_ratio = (pos_count / (pos_count + neg_count)) * 100
    return ('Negative Samples: ' + str(neg_count) +
            '\nPositive Samples: ' + str(pos_count) +
            '\nPostive Class Ratio: ' + str(minority_ratio))


def get_minority_ratio(col, posLabel=1, negLabel=0):
    neg_count, pos_count = len(col[col == negLabel]), len(col[col == posLabel])
    positive_ratio = (pos_count / (pos_count + neg_count)) * 100
    return positive_ratio, 1 - positive_ratio


def apply_ros_rus(pos_data, neg_data, ros_rate, rus_rate):
    replaceNeg = rus_rate > 1
    pos = pos_data.sample(
        frac=ros_rate, replace=True) if ros_rate != None else pos_data
    neg = neg_data.sample(
        frac=rus_rate, replace=replaceNeg) if rus_rate != None else neg_data
    return pd.DataFrame(pos.append(neg), columns=pos.columns)


def train_valid_split(data, valid_size, target_col, shuffle=True, rand_state=None):
    train_data, valid_data = train_test_split(
        data,
        test_size=valid_size,
        shuffle=shuffle,
        stratify=data[target_col],
        random_state=rand_state)

    # separate targets from features
    train_y = train_data[target_col]
    train_x = train_data.drop(columns=[target_col])
    valid_y = valid_data[target_col]
    valid_x = valid_data.drop(columns=[target_col])
    return train_x, train_y, valid_x, valid_y


def train_valid_split_w_sampling(data, valid_size, target_col, ros_rate, rus_rate, pos_label=1, neg_label=0, shuffle=True, rand_state=None):
    train_data, valid_data = train_test_split(
        data,
        test_size=valid_size,
        shuffle=shuffle,
        stratify=data[target_col],
        random_state=rand_state)

    # apply ROS and RUS
    train_pos, train_neg = split_on_binary_attribute(
        train_data, target_col, pos_label=pos_label, neg_label=neg_label)
    sampled_train_data = apply_ros_rus(
        train_pos, train_neg, ros_rate=ros_rate, rus_rate=rus_rate)

    # separate targets from features
    train_y = sampled_train_data[target_col]
    train_x = sampled_train_data.drop(columns=[target_col])
    valid_y = valid_data[target_col]
    valid_x = valid_data.drop(columns=[target_col])
    return train_x, train_y, valid_x, valid_y


def dict_to_hdf5(dict, path):
    file = h5py.File(path, 'w')
    for key, value in dict.items():
        file.create_dataset(key, data=value, maxshape=(None,))
    file.close()


def dict_from_hdf5(path):
    file = h5py.File(path, 'r')
    result = {}
    for key, value in file.items():
        result[key] = list(value)
    file.close()
    return result


def get_next_run_description(path):
    runs = filter(lambda x: ('run' in x), os.listdir(path))
    run_ids = list(map(lambda x: int(x.replace('run-', '')), runs))
    next_id = (max(run_ids) + 1) if len(run_ids) > 0 else 1
    return 'run-' + str(next_id)


def model_summary_to_string(model):
    # keep track of the original sys.stdout
    origStdout = sys.stdout
    # replace sys.stdout temporarily with our own buffer
    outputBuf = StringIO()
    sys.stdout = outputBuf
    # print the model summary
    model.summary()
    # put back the original stdout
    sys.stdout = origStdout
    # get the model summary as a string
    return 'Model Summary:\n' + outputBuf.getvalue()


def get_class_weights(labels):
    counts = labels.value_counts()
    majority = max(counts)
    return {key: float(majority / counts[key]) for key in counts.keys()}


def rounded_str(num, precision=6):
    if type(num) == str:
        return num
    return str(round(num, precision))


perf_metrics = ['key', 'depth', 'width', 'tp', 'fp', 'tn', 'fn', 'tpr', 'tnr',
                'roc_auc', 'geometric_mean', 'arithmetic_mean', 'f1_score', 'precision']


def write_dnn_perf_metrics(y_true, y_prob, threshold, key, depth, width, path):
    # include header if file doesn't exist yet
    out = ",".join(perf_metrics) if not os.path.isfile(path) else ''

    predictions = np.where(y_prob > threshold, 1.0, 0.0)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    tpr = (tp) / (tp + fn)
    tnr = (tn) / (tn + fp)
    roc_auc = roc_auc_score(y_true, y_prob)
    geometric_mean = math.sqrt(tpr * tnr)
    arithmetic_mean = 0.5 * (tpr + tnr)
    f1 = f1_score(y_true, predictions)
    precision = precision_score(y_true, predictions)

    results = [key, depth, width, tp, fp, tn, fn, tpr, tnr,
               roc_auc, geometric_mean, arithmetic_mean, f1, precision]
    results = [rounded_str(x) for x in results]
    out += '\n' + ','.join(results)

    with open(path, 'a') as outfile:
        outfile.write(out)


def write_perf_metrics(path, y_true, y_prob, threshold, extras):
    extra_cols = list(extras.keys())
    extra_vals = list(extras.values())
    cols = ['roc_auc', 'tp', 'fp', 'tn', 'fn',
            'tpr', 'tnr', 'geometric_mean', *extra_cols]

    out = ",".join(cols) if not os.path.isfile(path) else ''

    predictions = np.where(y_prob > threshold, 1.0, 0.0)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    tpr = (tp) / (tp + fn)
    tnr = (tn) / (tn + fp)
    roc_auc = roc_auc_score(y_true, y_prob)
    geometric_mean = math.sqrt(tpr * tnr)

    results = [roc_auc, tp, fp, tn, fn, tpr, tnr, geometric_mean * extra_vals]
    results = [rounded_str(x) for x in results]
    out += '\n' + ','.join(results)

    with open(path, 'a') as outfile:
        outfile.write(out)


def get_best_threshold(train_x, train_y, model, delta=0.1):
    curr_thresh = 0.0
    best_thresh = 0.0
    best_gmean = 0.0

    y_prob = model.predict_proba(train_x)[:, 1]

    while True:
        y_pred = np.where(y_prob > curr_thresh, np.ones_like(
            y_prob), np.zeros_like(y_prob))
        tn, fp, fn, tp = confusion_matrix(train_y, y_pred).ravel()
        tpr = (tp) / (tp + fn)
        tnr = (tn) / (tn + fp)
        if tnr > tpr:
            return best_thresh
        gmean = math.sqrt(tpr * tnr)
        if gmean > best_gmean:
            best_gmean = gmean
            best_thresh = curr_thresh
        curr_thresh += delta

    return best_thresh


class Timer():
    def __init__(self):
        self.times = []
        self.reset()

    def reset(self):
        self.t0 = time.time()

    def lap(self):
        interval = time.time() - self.t0
        self.t0 = time.time()
        self.times.append(interval)
        return interval

    def write_to_file(self, out):
        with open(out, 'a') as fout:
            times = [rounded_str(t) for t in self.times]
            fout.write(','.join([*times, '\n']))
