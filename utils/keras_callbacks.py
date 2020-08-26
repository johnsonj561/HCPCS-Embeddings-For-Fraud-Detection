from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score
from tensorflow import keras
import os
import sys
import math
import time
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class EpochTimerCallback(keras.callbacks.Callback):
  def __init__(self, output_file):
    self.output_file = output_file
    self.timings = []

  def on_epoch_end(self, epoch, logs, generator=None):
    self.timings.append(str(time.time()))

  def write_timings(self):
    with open(self.output_file, 'a') as out:
      out.write("\n" + ",".join(self.timings))

  def get_avg_epoch_time(self):
    return np.average(self.timings)


class KerasRocAucCallback(keras.callbacks.Callback):
    def __init__(self, x, y, is_validation=False, logger=None):
        super(keras.callbacks.Callback, self).__init__()
        self.x = x
        self.y = y
        self.auc_scores = []
        self.logger = logger
        self.metric_key = 'val_auc' if is_validation else 'train_auc'

    def on_epoch_end(self, epoch, logs={}, generator=None):
        probs = self.model.predict(self.x)
        auc = roc_auc_score(self.y, probs)
        self.auc_scores.append(auc)
        logs[self.metric_key] = auc
        if self.logger != None:
            self.logger.log_time(f'Epoch: {epoch}  {self.metric_key}: {auc}').write_to_file()
