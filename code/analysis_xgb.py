import funcs as fc
import numpy as np
import pandas as pd
import pickle
import time
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


seed = 1
np.set_printoptions(threshold=np.Inf)
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

drop_cols = ['search_term', 'product_title', 'product_description',
             'product_info', 'attr', 'brand']
x_train = x_train.drop(drop_cols, axis=1)
x_test = x_test.drop(drop_cols, axis=1)


print('Finding best parameters...', round((time.time() - start) / 60, 2))
parameters = {
    'gamma': np.linspace(0.0, 1.4, num=200),
    'max_depth': np.arange(1, 14),
    'min_child_weight': np.linspace(0, 200, num=200),
    'subsample': np.linspace(0.5, 1, num=200)
}

xgb_mod = xgb.XGBClassifier()
clf = RandomizedSearchCV(
    xgb_mod,
    parameters,
    5000,
    scoring=fc.ms_error,
    n_jobs=-1,
    verbose=1,
    random_state=seed
)
clf.fit(x_train, y_train)
print('Best params:', clf.best_params_)


print('Finding error from best model...', round((time.time() - start) / 60, 2))
y_pred = clf.best_estimator_.predict(x_test)

public_idx = y_test['Usage']=='Public'
private_idx = y_test['Usage']=='Private'

y_public = y_test[public_idx]['relevance']
y_private = y_test[private_idx]['relevance']

y_pred_public = y_pred[public_idx]
y_pred_private = y_pred[private_idx]

print('public score', fc.ms_error(y_public, y_pred_public))
print('private score', fc.ms_error(y_private, y_pred_private))
print('Finished.', round((time.time() - start) / 60, 2))
