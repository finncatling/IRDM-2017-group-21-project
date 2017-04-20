from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
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

model_clf = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(10, ))

# columns = ["id", 'query_in_title', 'query_in_description', 'word_in_title',
#        'word_in_description', 'word_in_brand', 'ratio_title',
#        'ratio_description', 'ratio_brand', 'search_term_feature',
#        'title_cos_sim', 'description_cos_sim', 'bm25_title',
#        'bm25_description']

# columns = ["id", "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
#                           "search_term_feature"]
# columns = ['id', 'len_of_query', 'len_of_title',
#       'len_of_description', 'len_of_brand',
#       'query_in_title', 'query_in_description', 'word_in_title',
#       'word_in_description', 'word_in_brand', 'ratio_title',
#       'ratio_description', 'ratio_brand', 'search_term_feature',
#       'title_cos_sim', 'description_cos_sim', 'bm25_title',
#       'bm25_description']
# columns = ['id',
#       'query_in_title', 'query_in_description', 'word_in_title',
#       'word_in_description', 'word_in_brand', 'ratio_title',
#       'ratio_description', 'ratio_brand', 'search_term_feature',
#       'title_cos_sim', 'description_cos_sim', 'bm25_title',
#       'bm25_description']
columns = ['id',
      "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
                          "search_term_feature", "description_cos_sim"]


#unique_scores = [1., 2., 3.]
#unique_scores = [1., 1.33, 1.67, 2., 2.33, 2.67, 3.]
unique_scores = [1., 1.25, 1.33, 1.5, 1.67, 1.75, 2., 2.25, 2.33, 2.5, 2.67, 2.75, 3.]
# Drop the values that are not in the reduced list of scores
relev_ind = np.in1d(y_train.values, unique_scores)
x_train = x_train[relev_ind]
y_train = y_train[relev_ind]

numeric_train = x_train[columns]
numeric_test = x_test_public[columns]

true_classes_ = []
for i in range(numeric_train.values.shape[0]):
    classes_ = np.zeros(len(unique_scores))
    for j in range(unique_scores.index(y_train.values[i])+1):
        classes_[j] = 1
    true_classes_.append(classes_)
true_classes = np.asarray(true_classes_)
print("Got here")
model_clf.fit(numeric_train.values[:, 1:], true_classes)
#model_clf.fit(numeric_train.values, true_classes)
#y_pred_classes = model_clf.predict(numeric_test.values[:, 1:])
# print(y_pred_classes[0])
# print(y_pred_classes[1])
# print(y_pred_classes[2])

print("Proba")
y_pred_probas = model_clf.predict_proba(numeric_test.values[:, 1:])
#y_pred_probas = model_clf.predict_proba(numeric_test.values)
print(y_pred_probas[0])
print(y_pred_probas[1])
print(y_pred_probas[2])


th = 0.508
print("Predicting ")
y_pred = np.zeros(numeric_test.values.shape[0])
for i in range(len(y_pred)):
    index_ = 0
    for j in range(len(unique_scores)):
        # index_ += y_pred_classes[i][j]
        # index_ += y_pred_classes[i][j]
        # if y_pred_classes[i][j] == 0:
        #     break
        if y_pred_probas[i][j] >= th:
            index_ += 1
        else:
            break
    if index_ > 0 :
        index_ -= 1
    y_pred[i] = unique_scores[index_]
# y_pred = np.asarray([unique_scores[x - 1] for x in y_pred_classes])


RMSE = mean_squared_error(y_test_public.values[:, 1], y_pred) ** 0.5

print("Test error classification")
print(RMSE)


# y_pred_transf = y_pred_probas * unique_scores
#
# y_pred_normal = normalize(y_pred_transf, axis=1, norm='l1')
# y_pred = np.matmul(y_pred_normal, unique_scores)
#
# RMSE = mean_squared_error(y_test_public.values[:, 1], y_pred) ** 0.5
#
# print("Test error classification +")
# print(RMSE)

# k = 10
# for i in range(len(y_pred)):
#     if y_test_public.values[i, 1] != y_pred[i]:
#         print("Mistake")
#         print(y_test_public.values[i, 1])
#         print(y_pred_probas[i])
#         k -= 1
#         if k == 0:
#             break