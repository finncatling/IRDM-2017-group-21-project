import numpy as np
import pandas as pd
import pickle

pickle_file = '../../data/pre_processed_data.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  x_train = save['x_train']
  y_train = save['y_train']
  x_test = save['x_test']
  del save
  print('training size', x_train.shape, y_train.shape)
  print('test size', x_test.shape)

print('training glance', x_train.head())
print('training glance', x_train.columns)
