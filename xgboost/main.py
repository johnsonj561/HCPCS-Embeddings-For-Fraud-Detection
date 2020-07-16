from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier
import pandas as pd
import sys
import os

proj_dir = os.environ['CMS_ROOT']
sys.path.append(proj_dir)
from utils.utils import args_to_dict, write_perf_metrics, Timer
from utils.utils import get_best_threshold, get_imbalance_description
from utils.data import load_data, get_train_test

# parse arguments
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)
debug = cli_args.get('debug') == 'true'
runs = int(cli_args.get('runs', 1))
embedding_type = cli_args.get('embedding_type')
x_train_filename = cli_args.get('x_train')
x_test_filename = cli_args.get('x_test')
sample_size = int(cli_args.get('sample_size', 500000))
n_jobs = int(cli_args.get('n_jobs', 4))
print(f'Running job with arguments\n{cli_args}')

# define configs
train_perf_filename = 'train-results.csv'
test_perf_filename = 'test-results.csv'

n_estimators = 100
param_grid = {
    'max_depth': [2, 4, 8],
}

# if debug, reduce workload
if debug:
    n_estimators = 5
    param_grid = {
        'max_depth': [4]
    }

print(f'Using param grid\n{param_grid}')

# init timer
timer = Timer()

# iterate over runs
for run in range(runs):

    print(f'Starting run {run}')

    # load data
    data = load_data(sample_size)
    print(f'Loaded data with shape {data.shape}')

    # create train-test split
    train_x, test_x, train_y, test_y = get_train_test(data, with_hcpcs=True)
    print(
        f'Created train-test split with train shape {train_x.shape} and test shape {test_x.shape}')
    print(
        f'Training class imbalance description:\n{get_imbalance_description(train_y)}')

    # enumerate and evaluate all hyperparameters
    for config in ParameterGrid(param_grid):
        max_depth = config['max_depth']
        print(f'Training model with config\n{config}')

        # train model
        timer.reset()
        model = XGBClassifier(
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            max_depth=max_depth)
        model.fit(train_x, train_y)
        elapsed = timer.lap()
        print(f'Training completed in {elapsed}')

        print('Computing the best threshold using test data')
        delta = 0.05
        optimal_threshold = round(get_best_threshold(
            train_y_prob, train_y, model, delta), 4)

        print(
            f'Writing performance metrics using threshold {optimal_threshold}')
        train_y_prob = model.predict_proba(train_x)[:, 1]
        test_y_prob = model.predict_proba(test_x)[:, 1]

        write_perf_metrics(train_perf_filename, train_y,
                           train_y_prob, elapsed, max_depth, embedding_type, optimal_threshold)
        write_perf_metrics(test_perf_filename, test_y,
                           test_y_prob, elapsed, max_depth, embedding_type, optimal_threshold)

    # free up memory
    del train_x, train_y, test_x, test_y

print('Job complete')
