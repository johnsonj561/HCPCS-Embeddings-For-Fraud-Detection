import time
import math
import os

def deltaTime(start, end):
  delta_sec = end - start
  minutes = math.floor(delta_sec / 60.0)
  seconds = round(delta_sec % 60)
  return str(minutes) + ' min, ' + str(seconds) + ' sec'


class Logger():
  def __init__(self, output_file):
    self.start_time = time.time()
    self.output_file = output_file
    self.logs = []
    first_entry = {
      'label': 'start',
      'time': 0,
    }

  def log_message(self, mssg):
    self.logs.append(mssg + '\n')
    return self

  def log_time(self, label):
    delta_time = deltaTime(self.start_time, time.time())
    self.logs.append(f'{delta_time}: {label}\n')
    return self

  def mark_dir_complete(self, path):
    file = open(os.path.join(path, 'complete.txt'), "w+")
    file.write('true')
    file.close()
    return self

  def write_to_file(self):
    with open(self.output_file, 'a') as fout:
      fout.writelines(self.logs)
    self.logs = []
