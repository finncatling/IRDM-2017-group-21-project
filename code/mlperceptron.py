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

model_reg = MLPRegressor(hidden_layer_sizes=(15,),  activation='tanh', solver='lbfgs',    alpha=0.001,batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.01, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

model_clf = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(10,))

# columns = ["id", 'query_in_title', 'query_in_description', 'word_in_title',
#        'word_in_description', 'word_in_brand', 'ratio_title',
#        'ratio_description', 'ratio_brand', 'search_term_feature',
#        'title_cos_sim', 'description_cos_sim', 'bm25_title',
#        'bm25_description']
#
# columns = ["id", "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
#                           "search_term_feature"]
# columns = ['id',
#       'query_in_title', 'query_in_description', 'word_in_title',
#       'word_in_description', 'word_in_brand', 'ratio_title',
#       'ratio_description', 'ratio_brand', 'search_term_feature',
#       'title_cos_sim', 'description_cos_sim', 'bm25_title',
#       'bm25_description']

columns = ['id',
      "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
                          "search_term_feature", "description_cos_sim"]
# columns = ['id', 'product_uid', 'len_of_query', 'len_of_title',
#       'len_of_description', 'len_of_brand',
#       'query_in_title', 'query_in_description', 'word_in_title',
#       'word_in_description', 'word_in_brand', 'ratio_title',
#       'ratio_description', 'ratio_brand', 'search_term_feature',
#       'title_cos_sim', 'description_cos_sim', 'bm25_title',
#       'bm25_description']

numeric_train = x_train[columns]
numeric_test = x_test_public[columns]

#model_reg.fit(numeric_train.values, y_train.values)
model_reg.fit(numeric_train.values[:, 1:], y_train.values)
y_pred = model_reg.predict(numeric_test.values[:, 1:])
#y_pred = model_reg.predict(numeric_test.values)

RMSE = mean_squared_error(y_test_public.values[:, 1], y_pred) ** 0.5

print("Test error regression")
print(RMSE)

unique_scores = [1., 1.25, 1.33, 1.5, 1.67, 1.75, 2., 2.25, 2.33, 2.5, 2.67, 2.75, 3.]
true_classes = np.asarray([unique_scores.index(x)+1 for x in y_train.values])
model_clf.fit(numeric_train.values[:, 1:], true_classes)
y_pred_classes = model_clf.predict(numeric_test.values[:, 1:])
# model_clf.fit(numeric_train.values, true_classes)
# y_pred_classes = model_clf.predict(numeric_test.values)
y_pred = np.asarray([unique_scores[x - 1] for x in y_pred_classes])

RMSE = mean_squared_error(y_test_public.values[:, 1], y_pred) ** 0.5

print("Test error classification")
print(RMSE)

reduced_scores = [1., 1.33, 1.67, 2., 2.33, 2.67, 3.]
# Drop the values that are not in the reduced list of scores
relev_ind = np.in1d(y_train.values, reduced_scores)
x_train_red = x_train[relev_ind]
y_train_red = y_train[relev_ind]
numeric_train_red = x_train_red[columns]

model_clf_red = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(10,))

true_classes_red = np.asarray([reduced_scores.index(x)+1 for x in y_train_red.values])
model_clf_red.fit(numeric_train_red.values[:, 1:], true_classes_red)
y_pred_classes = model_clf_red.predict(numeric_test.values[:, 1:])
# model_clf_red.fit(numeric_train_red.values, true_classes_red)
# y_pred_classes = model_clf_red.predict(numeric_test.values)
y_pred = np.asarray([reduced_scores[x - 1] for x in y_pred_classes])

RMSE = mean_squared_error(y_test_public.values[:, 1], y_pred) ** 0.5

print("Test error classification reduced")
print(RMSE)