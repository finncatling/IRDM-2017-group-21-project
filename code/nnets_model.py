
# coding: utf-8
# This code runs one layer neural net with relu activaiton fucntion.
# Input vectors are: search term, product title, product description and brand name
# Get 'dl_full_size.pickle' data file at https://drive.google.com/open?id=0Bylsb5Tv26G-Q01qc1REelhsYlU
# It is pre-processed features file, which contains only relevant 4 features specific for this model
# 'glove_train_caseless_without_stops.csv' - is generated in Vocab_in_GloVe.py 

# # First Upload Data Files

import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
from joblib import Parallel, delayed
import multiprocessing

# # Apply GloVe embeddings to datasets

start = time.time()

df = pd.read_csv('glove_train_caseless_without_stops.csv', encoding="ISO-8859-1")

# Load Data file
data = 'dl_full_size.pickle'

with open(data, 'rb') as f:
  save = pickle.load(f)
  x_train = save['x_train']
  y_train = save['y_train']
  x_test = save['x_test']
  del save
  print('training size', x_train.shape, y_train.shape)
  print('test size', x_test.shape)

# Creating dictionary  
df['Values'] = df.ix[:, df.columns != 'Word_name'].values.tolist()
df['Mean'] = df.ix[:, df.columns != 'Word_name'].mean(axis = 1)

print("About to create a final dictionary!")
dict1 = df.set_index('Word_name')['Mean'].to_dict()

print("Created Dictionary!")


# Concatenate train and test
x_all = pd.concat((x_train, x_test), axis=0, ignore_index=True)

print("About to convert words to numerical values")

# Converting words inside each search_term/product_title/... to numeric values
def processInput(col, i, s):
    for w in range(0, len(s[col].iloc[i])):
        if s[col].iloc[i][w] in dict1:
            s[col].iloc[i][w] = str(dict1[s[col].iloc[i][w]])
        else:
            s[col].iloc[i][w] = '0'
    s[col][i] = sum(float(j) for j in s[col][i])

# Running in parallel cpus, to save time, but still slow...
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(col, i, x_all)
                                     for col in ['product_title', 'search_term', 'product_description', 'brand']
                                     for i in range(0, len(x_all[col])))

# def word2num(s):
#     for cols in ['product_title', 'search_term', 'product_description', 'brand']:
#         for i in range(0, len(s[cols])):
#             for w in range(0, len(s[cols].iloc[i])):
#                 if s[cols].iloc[i][w] in dict1:
#                     s[cols].iloc[i][w] = s[cols].iloc[i][w].replace(s[cols].iloc[i][w],str(dict1[s[cols].iloc[i][w]]))
#                 else:
#                     s[cols].iloc[i][w] = '0'
#             s[cols][i] = sum(float(j) for j in s[cols][i])
#     return
#
#
# word2num(x_all)

print("Converted words to numerical values")

# Separate train and test
train_size = x_train.shape[0]  # TODO: check train/test sizes are as expected
x_train = x_all[:][:train_size]
x_test = x_all[:][train_size:]
print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)


# # Modeling
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(3, input_dim=4, kernel_initializer='normal', activation='relu',
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


# build
#model = baseline_model()
#model.fit(x_train, y_train, nb_epoch=5, batch_size=5)


# # Cross Validation

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

print("About to build a model")
model = baseline_model()
print("Cross validating")
results = cross_val_score(model, x_train, y_train, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

print("Predicting relevance values of test data set")
preds = model.predict(x_test)

# Saving predicted values for test dataset
solution = pd.DataFrame({"id":x_test.Id, "relevance":preds})
solution.to_csv("DR.csv", index=False, header=True)


print('Finished in', round((time.time() - start) / 60, 2), 'minutes.')

# Getting weights used for linear combinations
for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays

print(weights)

