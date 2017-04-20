import funcs as fc
import numpy as np
import pandas as pd
import pickle
import time
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats


seed = 1
start = time.time()
pickle_file = '../../data/pre_processed_data.pickle'

print('Loading data...', round((time.time() - start) / 60, 2))
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

# define distribution for l2 reg parameter during CV
lower, upper = 0.01, 1000
mu, sigma = 1, 2
l2_dist = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

print('Finding best parameters...', round((time.time() - start) / 60, 2))
parameters = {
    # 'gamma': np.linspace(0.0, 1.4, num=200),
    'n_estimators': np.arange(100, 1001),
    'reg_lambda': l2_dist,
    'max_depth': np.arange(3, 14),
    'min_child_weight': stats.uniform(40, 110),  # uniform between 40 and 150
    'subsample': stats.uniform(0.55, 0.45)  # uniform between 0.55 and 1.0
}

xgb_mod = xgb.XGBRegressor(nthread=1)  # one core per parameter/fold combination
clf = RandomizedSearchCV(
    xgb_mod,
    parameters,
    1200,
    cv=ps,
    n_jobs=-1,  # one core per parameter/fold combination
    verbose=2,
    random_state=seed
)
clf.fit(x_train, y_train)
print('Best params:', clf.best_params_)
"""
Best params: {
    'max_depth': 8,
    'min_child_weight': 72.7303825614961,
    'n_estimators': 102,
    'reg_lambda': 0.78590954897833598,
    'subsample': 0.77313908783830443
}
"""

print('Finding error from best model...', round((time.time() - start) / 60, 2))
y_pred = clf.best_estimator_.predict(x_test)
scores = fc.test_rmse(y_test, y_pred)
print('public score:', scores[0])
print('private score:', scores[1])
"""
public score: 0.472809165232
private score: 0.472050535249
"""

print('Finished.', round((time.time() - start) / 60, 2))
