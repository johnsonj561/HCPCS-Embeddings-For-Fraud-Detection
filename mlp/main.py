import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import datetime

import tensorflow as tf
K = tf.keras.backend
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
TensorBoard = tf.keras.callbacks.TensorBoard

proj_dir = os.environ['CMS_ROOT']
sys.path.append(proj_dir)
from utils.utils import model_summary_to_string, args_to_dict, write_dnn_perf_metrics
from utils.logging import Logger
from utils.keras_callbacks import KerasRocAucCallback
from utils.data import load_data, load_sampled_data, get_embedded_data
from utils.mlp import create_model



############################################
# Parse CLI Args & Create DNN Config
############################################

config = {}
cli_args = args_to_dict(sys.argv)

debug = cli_args.get('debug') == 'true'

hidden_layers_markup = cli_args.get('hidden_layers')
config['hidden_layers'] = [int(nodes) for nodes in hidden_layers_markup.split('+')]
config['learn_rate'] = float(cli_args.get('learn_rate', 1e-3))
config['batch_size'] = int(cli_args.get('batch_size', 128))
dropout_rate = cli_args.get('dropout_rate')
config['dropout_rate'] = float(dropout_rate) if dropout_rate != None else dropout_rate
batchnorm = cli_args.get('batchnorm', 'false')
config['batchnorm'] = True if batchnorm.lower() == 'true' else False
epochs = int(cli_args.get('epochs', 10))
use_lr_reduction = cli_args.get('use_lr_reduction') == 'true'

embedding_type = cli_args.get('embedding_type')
embedding_path = cli_args.get('embedding_path')
drop_columns = cli_args.get('drop_columns', '')
drop_columns = drop_columns.split(',') if len(drop_columns) > 0 else []
sample_size = cli_args.get('sample_size')
if sample_size != None:
    sample_size = int(sample_size)

runs = int(cli_args.get('runs', 1))

    
    
############################################
# Initialize I/O
############################################

now = datetime.datetime.today()

validation_auc_outputs = 'validation-auc-results.csv'
train_auc_outputs = 'train-auc-results.csv'
results_file = 'results.csv'

config_value = f'embedding:{embedding_type}-layers:{hidden_layers_markup}-learn_rate:{config.get("learn_rate")}'
config_value += f'-batch_size:{config.get("batch_size")}-dropout_rate:{config.get("dropout_rate")}-bathcnorm:{config.get("batchnorm")}'

if not os.path.isfile(train_auc_outputs):
    results_header = 'config,' + ','.join([f'ep_{i}' for i in range(epochs)])
    output_files = [train_auc_outputs, validation_auc_outputs]
    output_headers = [results_header,results_header]
    for file, header in zip(output_files, output_headers):
        with open(file, 'w') as fout:
            fout.write(header + '\n')

def write_results(file, results):
    with open(file, 'a') as fout:
        fout.write(results + '\n')




############################################
# Iterate Over Runs
############################################
for run in range(runs):
    ts = now.strftime("%m%d%y-%H%M%S")
    log_file = f'logs/{ts}-{config_value}.txt'
    logger = Logger(log_file)
    logger.log_time(f'Starting run {run}')
    logger.log_time('Using ts: {ts}')
    logger.log_time(f'Outputs being written to {[validation_auc_outputs,train_auc_outputs]}')
    logger.write_to_file()
    
    
    ############################################
    # Load Data
    ############################################

    data = load_sampled_data(sample_size)

    # drop columns, onehot encode, or lookkup embeddings
    x, y = get_embedded_data(data, embedding_type, embedding_path, drop_columns)

    del data
    logger.log_time(f'Loaded embedded data with shape {x.shape}')

    

    ############################################
    # Train/Test Split & Normalize
    ############################################

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    del x, y
    scaler = MaxAbsScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)



    ############################################
    # Setup Training Callbacks
    ############################################

    validation_auc_callback = KerasRocAucCallback(x_test, y_test, True, logger)
    train_auc_callback = KerasRocAucCallback(x_train, y_train)
    early_stopping = EarlyStopping(monitor='val_auc', min_delta=0.001, patience=10, mode='max')
    tensorboard_dir = f'tensorboard/{ts}-{config_value}'
    tensorboard = TensorBoard(log_dir=f'{tensorboard_dir}', write_graph=False)
    callbacks = [validation_auc_callback, train_auc_callback, early_stopping, tensorboard]



    ############################################
    # Train Model
    ############################################

    K.clear_session()
    input_dim = x_train.shape[1]
    model = create_model(input_dim, config)

    logger.log_time('Starting training...').write_to_file()
    history = model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks, verbose=1)
    logger.log_time('Trainin complete!').write_to_file()



    ############################################
    # Write Results
    ############################################
    validation_aucs = np.array(history.history['val_auc'], dtype=str)
    write_results(validation_auc_outputs, f'{config_value},{",".join(validation_aucs)}')
    train_aucs = np.array(history.history['train_auc'], dtype=str)
    write_results(train_auc_outputs, f'{config_value},{",".join(train_aucs)}')

    minority_size = (y_train == 1).sum() / len(y_train) * 100
    threshold = minority_size / 100
    logger.log_time(f'Using threshold {threshold}')

    y_prob = model.predict(x_test)
    write_dnn_perf_metrics(y_test, y_prob, threshold, config_value, results_file)

    # free some memory
    del history, x_test, y_test, x_train, y_train
    del model

print('Job complete')
