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

scores = ['epochs', 'tn','fp','fn','tp', 'precision','recall','specificity','f1','roc_auc','g_mean', 'balanced_acc', 'accuracy']

class CustomMetricsCallback(keras.callbacks.Callback):
  def __init__(self, x, y, frequency=1, validation=False, silent=False, decision_threshold=0.5):
    self.frequency = frequency
    self.silent = silent
    self.title = 'Validation' if validation else 'Training'
    self.x = x
    self.y = y
    self.decision_threshold = decision_threshold
    self.scores = {}
    for score in scores:
      self.scores[score] = []

  def on_epoch_end(self, epoch, logs, generator=None):
    if epoch % self.frequency != 0:
      return

    predict = self.model.predict
    self.scores.get('epochs').append(epoch)

    y_prob = predict(self.x)
    predictions = np.where(y_prob < self.decision_threshold, 0, 1)

    tn, fp, fn, tp = confusion_matrix(self.y, predictions).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    roc_auc = roc_auc_score(self.y, y_prob)

    self.scores.get('tn').append(tn)
    self.scores.get('fp').append(fp)
    self.scores.get('fn').append(fn)
    self.scores.get('tp').append(tp)
    self.scores.get('precision').append(precision_score(self.y, predictions))
    self.scores.get('recall').append(tpr)
    self.scores.get('specificity').append(tnr)
    self.scores.get('f1').append(f1_score(self.y, predictions))
    self.scores.get('roc_auc').append(roc_auc)
    self.scores.get('g_mean').append(math.sqrt(tpr * tnr))
    self.scores.get('balanced_acc').append(0.5 * (tpr + tnr))
    self.scores.get('accuracy').append((tp + tn)/(tp + tn + fp + fn))

    if not self.silent:
      print('Epoch ' + str(epoch) + ': ' + self.title + ' Results')
      print('TN: ', tn, '\tFP: ', fp)
      print('FN: ', fn, '\tTP: ', tp)
      print('AUC: ', roc_auc)

  def get_scores(self):
    return self.scores


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
