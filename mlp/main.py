from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys
import os

proj_dir = os.environ['CMS_ROOT']
sys.path.append(proj_dir)
from utils.utils import args_to_dict, write_perf_metrics, Timer
from utils.utils import get_best_threshold, get_imbalance_description
from utils.data import load_data, get_embedded_data

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
print(f'Running job with arguments\n{cli_args}')

# define configs
train_perf_filename = 'train-results.csv'
test_perf_filename = 'test-results.csv'

# init timer
timer = Timer()

# iterate over runs
for run in range(runs):
    print(f'Starting run {run}')

    # load data
    data = load_data(sample_size)
    print(f'Loaded data with shape {data.shape}')

    # drop columns, onehot encode, or lookkup embeddings
    x, y = get_embedded_data(data, embedding_type,
                             embedding_path, drop_columns)
    del data
    print(f'Encoded data shape: {x.shape}')

    minority_size = (y == 1).sum() / len(y) * 100
    threshold = minority_size / 100
    print(f'Train shape: {x.shape}')
    print(f'Minority size: {minority_size}')
    print(f'Threshold: {threshold}')

    timer.reset()

    # RECORD EPOCH TIMES
    # -------------------------------------------------- #
    epochTimer = EpochTimerCallback(timings_results_file)

    # BUILD MODEL
    # -------------------------------------------------- #
    _, input_dim = train_x.shape

    model = Sequential()
    model.add(Dense(width, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    for _ in range(depth - 1):
        model.add(Dense(width))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(dropout_rate))

    # classification layer
    model.add(Dense(1, activation='sigmoid'))

    # TRAIN MODEL
    # -------------------------------------------------- #
    optimizer = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['acc'])

    print('MODEL COMPILED')
    print('BEGINNING TRAINING')

    history = model.fit(
        x=x,
        y=y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=256,
        verbose=0,
        callbacks=[epochTimer],
    )

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
    }

    write_perf_metrics(train_perf_filename, train_y,
                       train_y_prob, threshold, extras)
    write_perf_metrics(test_perf_filename, test_y,
                       test_y_prob, threshold, extras)

    # free up memory
    del train_x, train_y, test_x, test_y
print('Job complete')
