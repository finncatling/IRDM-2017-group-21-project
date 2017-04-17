import logging
import logging.handlers
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
import itertools
from algo import PRank
import funcs as uf

DATE = '%d-%b-%Y %I:%M:%S'
logger = logging.getLogger("PartAlogger")
fh = logging.handlers.RotatingFileHandler('../logs/column_search3.log', maxBytes=2048000, backupCount=5)
formatter = logging.Formatter('%(asctime)s %(message)s')
formatter.datefmt = DATE
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.setLevel(logging.INFO)

logger.info("Starting new column search")

pickle_file = '../../data/pre_processed_data.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    x_train_full = save['x_train']
    y_train_full = save['y_train']
    # x_test = save['x_test']
    del save

unique_scores = [1., 1.33, 1.67, 2., 2.33, 2.67, 3.]

def rmse(pred_y, true_y):
    RMSE = mean_squared_error(true_y, pred_y) ** 0.5
    return RMSE


def list_2_string(list_):
    return ', '.join(str(x) for x in list_)

def run_validation(x_train_, y_train_, x_val, y_val):
    # Drop the values that are not in the reduced list of scores
    relev_ind = np.in1d(y_train_.values, unique_scores)
    x_train = x_train_[relev_ind]
    y_train = y_train_[relev_ind]

    # Take only the columns we are testing on
    numeric_train = x_train[columns]
    numeric_val = x_val[columns]
    # Convert the true scores into the classes
    true_classes = np.asarray([unique_scores.index(x) + 1 for x in y_train.values])

    model = PRank(len(columns), len(unique_scores))
    model.train(numeric_train.values, true_classes)

    num_val_points = numeric_val.shape[0]
    y_pred = np.zeros(num_val_points)
    for i in range(num_val_points):
        y_pred[i] = unique_scores[model.predict(numeric_val.values[i, :]) - 1]

    return rmse(y_pred, y_val.values)


full_columns = ['id', 'product_uid', 'len_of_query', 'len_of_title',
       'len_of_description', 'len_of_brand',
       'query_in_title', 'query_in_description', 'word_in_title',
       'word_in_description', 'word_in_brand', 'ratio_title',
       'ratio_description', 'ratio_brand', 'search_term_feature',
       'title_cos_sim', 'description_cos_sim', 'bm25_title',
       'bm25_description']

cross_val_scores = []
cross_val_columns = []

for num_col in range(1, len(full_columns)+1):
    for comb in itertools.combinations(full_columns, num_col):
        columns = np.asarray(comb)
        logger.info("Trying columns: {}".format(list_2_string(columns)))

        total_score_ = 0
        for v in range(2):
            score_ = 0
            folds = uf.get_k_folds(3, x_train_full, y_train_full, 'search_term')
            for i in range(3):
                score_ += run_validation(folds[i][0], folds[i][1], folds[i][2], folds[i][3])
            total_score_ += score_/3

        score_ = 0
        folds = uf.get_k_folds(3, x_train_full, y_train_full, 'product_uid')
        for i in range(3):
            score_ += run_validation(folds[i][0], folds[i][1], folds[i][2], folds[i][3])
        total_score_ += score_ / 3

        error = total_score_ /3
        logger.info("RMSE was {}".format(error))
        cross_val_columns.append(columns)
        cross_val_scores.append(error)


with open('../trained_models/cv_columns.pkl', 'wb') as output:
    pickle.dump(cross_val_columns, output, pickle.HIGHEST_PROTOCOL)

with open('../trained_models/cv_scores.pkl', 'wb') as output:
    pickle.dump(cross_val_scores, output, pickle.HIGHEST_PROTOCOL)