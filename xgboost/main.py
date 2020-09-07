from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd
import sys
import os

proj_dir = os.environ['CMS_ROOT']
sys.path.append(proj_dir)
from utils.utils import args_to_dict, write_perf_metrics, Timer
from utils.data import load_sampled_data, get_embedded_data

# parse arguments
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)
debug = cli_args.get('debug') == 'true'
runs = int(cli_args.get('runs', 1))
embedding_type = cli_args.get('embedding_type')
embedding_path = cli_args.get('embedding_path')
drop_columns = cli_args.get('drop_columns', '')
drop_columns = drop_columns.split(',') if len(drop_columns) > 0 else []
max_depth = int(cli_args.get('max_depth', 8))
sample_size = cli_args.get('sample_size')
if sample_size != None:
    sample_size = int(sample_size)
n_jobs = int(cli_args.get('n_jobs', 4))
l2_reg = cli_args.get('l2_reg')
if l2_reg != None:
  l2_reg = float(l2_reg)
l1_reg = cli_args.get('l1_reg')
if l1_reg != None:
  l1_reg = float(l1_reg)
print(f'Running job with arguments\n{cli_args}')

# define configs
train_perf_filename = 'train-results.csv'
test_perf_filename = 'test-results.csv'

n_estimators = 5 if debug else 100
print(f'n_estimators: {n_estimators}')
print(f'max_depth: {max_depth}')

# init timer
timer = Timer()

# iterate over runs
for run in range(runs):
    print(f'Starting run {run}')

    # load data
    data = load_sampled_data(sample_size)
    print(f'Loaded data with shape {data.shape}')

    # drop columns, onehot encode, or lookkup embeddings
    x, y = get_embedded_data(data, embedding_type,
                             embedding_path, drop_columns)
    del data
    print(f'Encoded data shape: {x.shape}')

    # apply 5 fold stratified cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for fold, (train_index, test_index) in enumerate(skf.split(x, y)):
        print(f'Starting fold {fold}')
        if 'onehot' in embedding_type or 'none' in embedding_type:
          train_x, test_x = x[train_index], x[test_index]
        else:
          train_x, test_x = x.iloc[train_index], x.iloc[test_index]
        train_y, test_y = y[train_index], y[test_index]
        minority_size = (train_y == 1).sum() / len(train_y) * 100
        threshold = minority_size / 100
        print(f'Train shape: {train_x.shape}')
        print(f'Test shape: {test_x.shape}')
        print(f'Minority size: {minority_size}')
        print(f'Threshold: {threshold}')

        timer.reset()
        model = XGBClassifier(
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            max_depth=max_depth,
            reg_alpha=l1_reg,
            reg_lambda=l2_reg
        )
        model.fit(train_x, train_y)
        elapsed = timer.lap()
        print(f'Training completed in {elapsed}')

        train_y_prob = model.predict_proba(train_x)[:, 1]
        test_y_prob = model.predict_proba(test_x)[:, 1]

        extras = {
            'elapsed': elapsed,
            'minority_size': minority_size,
            'embedding_type': embedding_type,
            'dropped_columns': '|'.join(drop_columns),
            'max_depth': max_depth,
            'threshold': threshold,
            'l2_reg': l2_reg,
            'l1_reg': l1_reg
        }

        write_perf_metrics(train_perf_filename, train_y,
                           train_y_prob, threshold, extras)
        write_perf_metrics(test_perf_filename, test_y,
                           test_y_prob, threshold, extras)

    # free up memory
    del train_x, train_y, test_x, test_y
print('Job complete')
