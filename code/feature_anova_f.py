import funcs as fc
import numpy as np
import pandas as pd
import pickle
np.set_printoptions(threshold=np.Inf)
from sklearn.feature_selection import SelectKBest

pickle_file = '../../data/pre_processed_data_ff.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  x_train = save['x_train']
  y_train = save['y_train']
  x_test = save['x_test']
  del save
  print('training size', x_train.shape, y_train.shape)
  print('test size', x_test.shape)

drop_cols = ['search_term', 'product_title', 'product_description',
             'product_info', 'attr', 'brand','id','product_uid']
x_train = x_train.drop(drop_cols, axis=1)

k_best = SelectKBest()
k_best.fit(x_train, y_train)
scores = k_best.scores_
print(scores)

