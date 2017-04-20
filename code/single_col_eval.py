import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import logging.handlers
from algo import PRank

DATE = '%d-%b-%Y %I:%M:%S'
logger = logging.getLogger("PartAlogger")
fh = logging.handlers.RotatingFileHandler('../logs/sing_col.log', maxBytes=2048000, backupCount=5)
formatter = logging.Formatter('%(asctime)s %(message)s')
formatter.datefmt = DATE
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.setLevel(logging.INFO)

logger.info("Starting new column search")

def rmse(pred_y, true_y):
    RMSE = mean_squared_error(true_y, pred_y) ** 0.5
    return RMSE

full_columns = ['id', 'product_uid', 'len_of_query', 'len_of_title',
       'len_of_description', 'len_of_brand',
       'query_in_title', 'query_in_description', 'word_in_title',
       'word_in_description', 'word_in_brand', 'ratio_title',
       'ratio_description', 'ratio_brand', 'search_term_feature',
       'title_cos_sim', 'description_cos_sim', 'bm25_title',
       'bm25_description']

pickle_file = '../../data/pre_processed_data.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    x_train_full = save['x_train']
    y_train_full = save['y_train']
    del save

unique_scores = [1., 1.33, 1.67, 2., 2.33, 2.67, 3.]
# Drop the values that are not in the reduced list of scores
relev_ind = np.in1d(y_train_full.values, unique_scores)
x_train = x_train_full[relev_ind]
y_train = y_train_full[relev_ind]
# Convert the true scores into the classes
true_classes = np.asarray([unique_scores.index(x) + 1 for x in y_train.values])

models = []
for col in range(len(full_columns)):
    model_ = PRank(1, len(unique_scores))
    model_.train(x_train[[full_columns[col]]].values, true_classes)
    models.append(model_)



pickle_file = '../../data/pre_processed_data.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  x_test = save['x_test']
  del save

y_test_full = pd.read_csv('../../data/solution.csv', encoding="ISO-8859-1")
y_test_public = y_test_full[y_test_full['Usage'] == "Public"]
y_test_private = y_test_full[y_test_full['Usage'] == "Private"]

index_list_public = y_test_full['Usage'] == "Public"
index_list_private = y_test_full['Usage'] == "Private"

x_test_public = x_test[np.in1d(x_test.values[:, 0], y_test_public.values[:, 0])]
x_test_private = x_test[np.in1d(x_test.values[:, 0], y_test_private.values[:, 0])]

# numeric_test = x_test_public[["id", "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
#                           "search_term_feature", "description_cos_sim"]] # 0.500


num_pub = x_test_public.shape[0]
num_priv = x_test_private.shape[0]
for col in range(len(full_columns)):
    model_ = models[col]
    logger.info("For column {}".format(full_columns[col]))
    p_y_pub = np.zeros(num_pub)
    for i in range(num_pub):
        p_y_pub[i] = unique_scores[model_.predict(x_test_public[[full_columns[col]]].values[i, :]) - 1]
    pub_error = rmse(p_y_pub, y_test_public.values[:, 1])
    logger.info("Public RMSE is {}".format(pub_error))

    p_y_priv = np.zeros(num_priv)
    for i in range(num_priv):
        p_y_priv[i] = unique_scores[model_.predict(x_test_private[[full_columns[col]]].values[i, :]) - 1]
    priv_error = rmse(p_y_priv, y_test_private.values[:, 1])
    logger.info("Private RMSE is {}".format(priv_error))
