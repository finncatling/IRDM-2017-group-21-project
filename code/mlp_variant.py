from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error

pickle_file = '../../data/pre_processed_data.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  x_train = save['x_train']
  y_train = save['y_train']
  x_test = save['x_test']
  del save


y_test_full = pd.read_csv('../../data/solution.csv', encoding="ISO-8859-1")
# y_test_public = y_test_full[y_test_full['Usage'] == "Public"]
# index_list = y_test_full['Usage'] == "Public"

y_test_public = y_test_full[y_test_full['Usage'] == "Private"]
index_list = y_test_full['Usage'] == "Private"

x_test_public = x_test[np.in1d(x_test.values[:, 0], y_test_public.values[:, 0])]

model_clf = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(10,))

# columns = ["id", 'query_in_title', 'query_in_description', 'word_in_title',
#        'word_in_description', 'word_in_brand', 'ratio_title',
#        'ratio_description', 'ratio_brand', 'search_term_feature',
#        'title_cos_sim', 'description_cos_sim', 'bm25_title',
#        'bm25_description']
# columns = ['id', 'product_uid', 'len_of_query', 'len_of_title',
#       'len_of_description', 'len_of_brand',
#       'query_in_title', 'query_in_description', 'word_in_title',
#       'word_in_description', 'word_in_brand', 'ratio_title',
#       'ratio_description', 'ratio_brand', 'search_term_feature',
#       'title_cos_sim', 'description_cos_sim', 'bm25_title',
#       'bm25_description']
# columns = ["id", "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
#                           "search_term_feature"]

# columns = ['id','len_of_query', 'len_of_title',
#        'len_of_description', 'len_of_brand',
#       'query_in_title', 'query_in_description', 'word_in_title',
#       'word_in_description', 'word_in_brand', 'ratio_title',
#       'ratio_description', 'ratio_brand', 'search_term_feature',
#       'title_cos_sim', 'description_cos_sim', 'bm25_title',
#       'bm25_description']

columns = ['id',
      "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
                          "search_term_feature", "description_cos_sim"]

numeric_train = x_train[columns]
numeric_test = x_test_public[columns]


reduced_scores = [1., 2., 3.]
# Drop the values that are not in the reduced list of scores
relev_ind = np.in1d(y_train.values, reduced_scores)
x_train_red = x_train[relev_ind]
y_train_red = y_train[relev_ind]
numeric_train_red = x_train_red[columns]
print("Size for reduced train ", x_train_red.shape[0])
# This was the one with very good score (tanh is better activation though)
# model_clf_red = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(10,))

model_clf_red = MLPClassifier(solver='lbfgs', activation="tanh", alpha=0.001, hidden_layer_sizes=(10,))
# model_clf_red = MLPClassifier(solver='lbfgs', activation="identity", alpha=0.001, hidden_layer_sizes=(10,))

true_classes_red = np.asarray([reduced_scores.index(x)+1 for x in y_train_red.values])
model_clf_red.fit(numeric_train_red.values[:, 1:], true_classes_red)
#model_clf_red.fit(numeric_train_red.values, true_classes_red)
print(model_clf_red.classes_)
y_prob = model_clf_red.predict_proba(numeric_test.values[:, 1:])
#y_prob = model_clf_red.predict_proba(numeric_test.values)
# y_pred = np.asarray([reduced_scores[x - 1] for x in y_pred_classes])
class_weights = np.asarray([1, 2, 3])
y_pred = np.matmul(y_prob, class_weights)

RMSE = mean_squared_error(y_test_public.values[:, 1], y_pred) ** 0.5

print("Test error classification reduced")
print(RMSE)