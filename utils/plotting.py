import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def smooth_curve(points, factor=0.9):
  '''Smooths points along a curve by replacing each point
  with an exponential moving average of the previous points
  '''
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return np.array(smoothed_points)

def plot_train_vs_validation(metrics, train_scores, valid_scores, save_path=None):
  # configure plot
  sns.set_style("darkgrid")
  sns.set_palette(sns.color_palette("magma", 2))
  plt.subplots_adjust(hspace=1)
  plot_count = len(metrics)
  fig, axs = plt.subplots(nrows=plot_count, sharex=True)
  axs[0].set_title('Training vs Validation Metrics')
  fig.set_dpi(100)
  fig.set_figheight(25)
  fig.set_figwidth(10)

  epochs = train_scores.get('epochs')

  # plot metrics
  for idx, metric in enumerate(metrics):
    train_score = smooth_curve(train_scores.get(metric), 0.8)
    valid_score = smooth_curve(valid_scores.get(metric), 0.8)
    plot = sns.lineplot(epochs, train_score, ax=axs[idx], label="Train")
    sns.lineplot(epochs, valid_score, ax=axs[idx], label="Validation")
    axs[idx].set(ylabel=metric, xlabel='Epochs')

  if save_path != None:
    fig.savefig(save_path)
