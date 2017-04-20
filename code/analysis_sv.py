import funcs as fc
import numpy as np
import pandas as pd
import pickle
import time
np.set_printoptions(threshold=np.Inf)
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV

pickle_file = '../../data/pre_processed_data_ff.pickle'

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
ps = fc.k_folds_generator(3, x_train, y_train, 'search_term', start_seed=seed)

drop_cols = ['search_term', 'product_title', 'product_description',
             'product_info', 'attr', 'brand']

x_train = x_train.drop(drop_cols, axis=1)
x_test = x_test.drop(drop_cols, axis=1)

# normailize features
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.fit_transform(x_test)

# CLASSIFCATION
#convert to labels for classifcation
#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#le.fit([1.0,1.25,1.33,1.5,1.67,1.75,2.0,2.25,2.33,2.5,2.67,2.75, 3])
#y_train = le.transform(y_train)
#clf = SVC()

# REGRESSION
clf = SVR()

parameters = {'C': [10**x for x in np.arange(-5,4,0.25)], 
'epsilon': [10**x for x in np.arange(-7,0,0.25)], 
'gamma': [10**x for x in np.arange(-7,0,0.25)]}

grid_obj = RandomizedSearchCV(
    clf,
    parameters,
    100,
    cv=ps,
    random_state=seed
)

# fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(x_train, y_train)

# print the estimator
print(grid_obj.best_estimator_)

# get prediction
y_pred = grid_obj.predict(x_test)

# if classification
#y_pred = le.inverse_transform(y_pred)

public_rmse, private_rmse = fc.test_rmse(y_test,y_pred)

print('public score',public_rmse)
print('private score',private_rmse)


