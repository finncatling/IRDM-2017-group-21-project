
# coding: utf-8

# Load libraries
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

# Load Data file - file with all engineered and initial features,
#where words are not stemmed, to be used for GloVE
pickle_file = 'pre_processed_data_no_stem.pickle'


with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  x_train = save['x_train']
  y_train = save['y_train']
  x_test = save['x_test']
  del save
  print('training size', x_train.shape, y_train.shape)
  print('test size', x_test.shape)


# Getting the features which contains words
x_train = x_train.loc[:,'product_title':'brand']
x_test = x_test.loc[:,'product_title':'brand']

# Concatenate train and test
x_all = pd.concat((x_train, x_test), axis=0, ignore_index=True)

# Creating a vocabulary out of the features which has words
vocab = []
for i, r in x_all.iterrows():
    for col in ['product_title', 'search_term', 'product_description', 'brand']:
        vocab += r[col]
vocab = set(vocab)

# Save the vocabulary for reusing
pickle.dump(vocab, open( "vocab.p", "wb" ))
