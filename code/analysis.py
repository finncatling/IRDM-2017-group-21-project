
import funcs as fc
import numpy as np
import pandas as pd
import pickle
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
#from sklearn import pipeline, grid_search
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn import svm
from sklearn.svm import LinearSVR

start = time.time()

pickle_file = '../../data/pre_processed_data.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  x_train = save['x_train']
  y_train = save['y_train']
  x_test = save['x_test']
  del save
  print('training size', x_train.shape, y_train.shape)
  print('test size', x_test.shape)

#print('training glance', x_train.head())
#print('training glance', x_train.columns)

y_test = pd.read_csv('../../data/solution.csv', encoding="ISO-8859-1")

x_train = x_train.drop(['search_term','product_title','product_description','product_info','attr','brand'],axis=1)
x_test = x_test.drop(['search_term','product_title','product_description','product_info','attr','brand'],axis=1)

#print('training glance', x_train.head())
#print('training glance', x_train.columns)

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
#clf = svm.SVC()
#clf = LinearSVR()

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

pd.DataFrame({"id": x_test['id'], "relevance": y_pred}).to_csv('submission.csv',index=False)

public_idx = y_test['Usage']=='Public'
private_idx = y_test['Usage']=='Private'

y_public = y_test[public_idx]['relevance']
y_private = y_test[private_idx]['relevance']

y_pred_public = y_pred[public_idx]
y_pred_private = y_pred[private_idx]

print('public score',fc.fmean_squared_error(y_public,y_pred_public))
print('private score',fc.fmean_squared_error(y_private,y_pred_private))

print(y_pred_public[:100])

duration = time.time() - start
print(duration)