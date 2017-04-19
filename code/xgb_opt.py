import sys
import numpy as np
from numpy.random import RandomState
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import mean_squared_error
from scipy import stats


def test_rmse(y_test, y_pred):

    public_idx = y_test['Usage'] == 'Public'
    private_idx = y_test['Usage'] == 'Private'

    y_public = y_test[public_idx]['relevance']
    y_private = y_test[private_idx]['relevance']

    y_pred_public = y_pred[public_idx[public_idx == True].index]
    y_pred_private = y_pred[private_idx[private_idx == True].index]

    public_rmse = ms_error(y_public, y_pred_public)
    private_rmse = ms_error(y_private, y_pred_private)

    return public_rmse, private_rmse


def ms_error(y, y_hat):
    ms_error_cal = mean_squared_error(y, y_hat)**0.5
    return ms_error_cal


def split_val_set(x_, y_, val_ratio, col_name, seed=None):
    num_points = x_.shape[0]
    val_size_needed = np.floor(num_points * val_ratio).astype(int)
    # get all unique values and how many times they occur
    dif_items, item_counts = np.unique(x_[col_name].values, return_counts=True)
    # get a random permutation of indices for the unique values
    rnd = RandomState(seed)
    rand_perm = rnd.permutation(len(dif_items)) - 1
    val_size = 0
    i = 0
    val_items = []  # values to go into validation set
    # put values from the random permutation into validation set until there are enough rows
    while val_size < val_size_needed:
        val_items.append(dif_items[rand_perm[i]])
        val_size += item_counts[rand_perm[i]]
        i += 1
    # get an array that says whether a row is in the validation set
    val_rows = np.in1d(x_[col_name].values, np.asarray(val_items))
    # split validation and training sets
    x_val = x_[val_rows]
    y_val = y_[val_rows]
    x_train = x_[np.invert(val_rows)]
    y_train = y_[np.invert(val_rows)]
    return x_train, y_train, x_val, y_val


def k_folds_generator(k, x_, y_, col_name, start_seed=1):
    """Adapts get_k_folds into a generator object for use with sklearn.

    :param k: number of folds
    :param x_: pandas data frame as loaded from pickle after pre-processing
    :param y_: pandas series of true scores as loaded from pickle
    :param col_name: name of column to split on e.g. 'product_uid'
    :param start_seed: integer, allows reproducible splitting
    :return: cross validation generator object
    """
    val_sets = dict()
    df = pd.DataFrame({'test_fold': np.zeros(len(x_))})
    x_t = x_
    y_t = y_
    for i in range(k - 1):
        ratio_ = 1 / (k - i)
        if start_seed:
            x_train, y_train, x_val, y_val = split_val_set(
                x_t, y_t, ratio_, col_name, start_seed + i)
        else:
            x_train, y_train, x_val, y_val = split_val_set(
                x_t, y_t, ratio_, col_name)
        val_sets[i] = x_val.index
        x_t = x_train
        y_t = y_train
    val_sets[k - 1] = x_t.index  # the last iteration split the set in half

    for i in range(k):
        df.loc[val_sets[i]] = i
    df.test_fold.astype(int, inplace=True)

    return PredefinedSplit(df.test_fold.values)


seed = 1
pickle_file = '../../data/pre_processed_data.pickle'

print('Loading data...')
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
ps = k_folds_generator(3, x_train, y_train, 'search_term', start_seed=seed)

drop_cols = ['search_term', 'product_title', 'product_description',
             'product_info', 'attr', 'brand']
x_train = x_train.drop(drop_cols, axis=1)
x_test = x_test.drop(drop_cols, axis=1)

# define distribution for l2 reg parameter during CV
lower, upper = 0.01, 1000
mu, sigma = 1, 2
l2_dist = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

print('Finding best parameters...')
parameters = {
    # 'gamma': np.linspace(0.0, 1.4, num=200),
    'n_estimators': np.arange(100, 1001),
    'reg_lambda': l2_dist,
    'max_depth': np.arange(3, 14),
    'min_child_weight': stats.uniform(40, 110),  # uniform between 40 and 150
    'subsample': stats.uniform(0.55, 0.45)  # uniform between 0.55 and 1.0
}

sys.stdout.flush()

xgb_mod = xgb.XGBRegressor(nthread=1)
clf = RandomizedSearchCV(
    xgb_mod,
    parameters,
    1200,
    # scoring=ms_error,
    cv=ps,
    n_jobs=-1,  # quicker parallelising this way
    verbose=2,
    random_state=seed
)
clf.fit(x_train, y_train)
print('Best params:', clf.best_params_)

print('Finding error from best model...')
y_pred = clf.best_estimator_.predict(x_test)
scores = test_rmse(y_test, y_pred)
print('public score:', scores[0])
print('private score:', scores[1])
