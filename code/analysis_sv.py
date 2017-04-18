import funcs as fc
import numpy as np
import pandas as pd
import pickle
import time
np.set_printoptions(threshold=np.Inf)
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
start = time.time()
seed = 1

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
columns = [column for column in x_train]

# normailize features
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.fit_transform(x_test)

'''
#convert to labels for classifcation
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1.0,1.25,1.33,1.5,1.67,1.75,2.0,2.25,2.33,2.5,2.67,2.75, 3])
y_train = le.transform(y_train)
'''

#clf = SVC()
clf = SVR()
#clf = LinearSVC()
#clf = LinearSVR(random_state=seed)

'''
parameters = {'C': [10**x for x in np.arange(-5,3,0.25)], 
'epsilon': [10**x for x in np.arange(-5,2,0.25)], 
'fit_intercept': [True,False],
'loss':['epsilon_insensitive','squared_epsilon_insensitive']}


parameters = {'C': [10**x for x in np.arange(-5,3,0.25)], 
'epsilon': [10**x for x in np.arange(-5,2,0.25)], 
'degree':[x for x in np.arange(1,10,1)],
'gamma': [10**x for x in np.arange(-5,2,0.25)]}
'''

print([10**x for x in np.arange(-4,3,1.0)])

parameters = {'C': [1]}#10**x for x in np.arange(-4,3,1.0)]}

# create cross-validation sets from the training data
#cv = ShuffleSplit(x_train.shape[0], n_iter = 3, test_size = 0.33, random_state = seed)

# perform grid search on the regressor using the r^2 as the scoring method
#grid_obj = GridSearchCV(clf, parameters, scoring=fc.RMSE, cv=cv)

grid_obj = RandomizedSearchCV(
    clf,
    parameters,
    1,
    cv=ps,
    random_state=seed
)


# fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(x_train, y_train)

# print the estimator
print(grid_obj.best_estimator_)

y_pred = grid_obj.best_estimator_.predict(x_test)

# if classification
#y_pred = le.inverse_transform(y_pred)

public_rmse, private_rmse = fc.test_rmse(y_test,y_pred)

print('public score',public_rmse)
print('private score',private_rmse)

print(y_pred[:100])

duration = time.time() - start
print(duration)


#abc = [10**x for x in np.arange(-5,2,0.25)]

#print(abc)

# delete at some point
'''
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
clf = MLPRegressor()
clf = MLPClassifier()


#from sklearn.feature_selection import SelectKBest
#k_best = SelectKBest(k=10)
#k_best.fit(x_train, y_train)
#scores = k_best.scores_
#print(scores)

#pd.DataFrame({"id": test_id, "relevance": y_pred}).to_csv('submission_4.csv',index=False)

clf = RandomForestRegressor()
clf = clf.fit(x_train, y_train)
print(columns)
print(clf.feature_importances_)
columns

print('len',len(y_pred))
y_pred = 2.38*np.ones(166693)
y_ = np.array(y_train)
print('mean',y_.mean())

#x_train = x_train['ratio_title']
#x_train = x_train.values.reshape(-1, 1)
#x_test = x_test['ratio_title']
#x_test = x_test.values.reshape(-1, 1)
#columns = x_train.columns

import csv

myfile = open('features.csv', 'w')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(columns)


myfile = open('features2.csv', 'w')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(clf.feature_importances_)

myfile = open('features3.csv', 'w')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(scores)
'''




