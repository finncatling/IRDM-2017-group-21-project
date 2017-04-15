import funcs as fc
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor


np.set_printoptions(threshold=np.Inf)
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

y_test = pd.read_csv('../../data/solution.csv', encoding="ISO-8859-1")
test_id = x_test['id']

#output = x_train.head(100)
#output.to_csv('output.csv')

x_train = x_train.drop(['search_term','product_title','product_description','product_info','attr','brand'],axis=1)
x_test = x_test.drop(['search_term','product_title','product_description','product_info','attr','brand'],axis=1)

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)

# convert labels to classes for classifcation
'''
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1.0,1.25,1.33,1.5,1.67,1.75,2.0,2.25,2.33,2.5,2.67,2.75, 3])
y_train = le.transform(y_train)
'''

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

public_idx = y_test['Usage']=='Public'
private_idx = y_test['Usage']=='Private'

y_public = y_test[public_idx]['relevance']
y_private = y_test[private_idx]['relevance']

y_pred_public = y_pred[public_idx]
y_pred_private = y_pred[private_idx]

print('public score',fc.ms_error(y_public,y_pred_public))
print('private score',fc.ms_error(y_private,y_pred_private))

print(y_pred_public[:100])

duration = time.time() - start
print(duration)





