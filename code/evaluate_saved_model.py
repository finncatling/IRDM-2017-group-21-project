import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from algo import PRank


def rmse(pred_y, true_y):
    RMSE = mean_squared_error(true_y, pred_y) ** 0.5
    return RMSE


pickle_file = '../../data/pre_processed_data.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  x_test = save['x_test']
  del save

y_test_full = pd.read_csv('../../data/solution.csv', encoding="ISO-8859-1")
y_test_public = y_test_full[y_test_full['Usage'] == "Public"]
y_test_private = y_test_full[y_test_full['Usage'] == "Private"]

index_list = y_test_full['Usage'] == "Public"

x_test_public = x_test[np.in1d(x_test.values[:, 0], y_test_public.values[:, 0])]
x_test_private = x_test[np.in1d(x_test.values[:, 0], y_test_private.values[:, 0])]

unique_scores = [1., 1.25, 1.33, 1.5, 1.67, 1.75, 2., 2.25, 2.33, 2.5, 2.67, 2.75, 3.]
# with open('../trained_models/pred_76.pkl', 'rb') as input:
#     y_pred = pickle.load(input)

# y_pred_public = y_pred[np.in1d(y_pred[:, 0], y_test_public.values[:, 0]), :]
print(y_test_public.shape)

# numeric_test = x_test_public[["id", "query_in_title", "query_in_description", "word_in_title",
#                          "ratio_title", "word_in_brand",
#                         "ratio_brand"]]

# numeric_test = x_test_public[["id", "query_in_description", "word_in_title", "word_in_description", "word_in_brand",
#                          "ratio_title", "ratio_description", "search_term_feature", "title_cos_sim",
#                          "description_cos_sim", "bm25_description"]]
# numeric_test = x_test_public[["id", "query_in_title", "query_in_description", "word_in_brand", "ratio_title",
#                          "search_term_feature"]] # Score 0.52

# numeric_test = x_test_public[["id", "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
#                          "search_term_feature"]] # Score 0.50

# numeric_test = x_test_public[["id", "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
#                          "search_term_feature"]]
# numeric_test = x_test_public[['id',
#        'len_of_description',
#      'ratio_title',
#        'search_term_feature',
#        'description_cos_sim', 'bm25_title',
#        'bm25_description']]
# numeric_test = x_test_public[["id", "len_of_query", "len_of_title", "word_in_description", "ratio_title", "search_term_feature", "title_cos_sim", "description_cos_sim"]] # 0.509
# numeric_test = x_test_public[["id", "len_of_query", "len_of_title", "query_in_title", "query_in_description",
#                          "word_in_brand", "ratio_title", "ratio_description",
#                           "search_term_feature"]]

# numeric_test = x_test_public[["id", "query_in_description", "word_in_title", "ratio_title",
# "len_of_brand", "bm25_title"]]
# numeric_test = x_test_public[["id", "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
#                           "search_term_feature", "description_cos_sim"]] # 0.500

# numeric_test = x_test_public[["id", "len_of_title", "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
#                           "search_term_feature", "description_cos_sim"]] # 0.514
# columns = ["id", "len_of_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
# #                           "search_term_feature", "description_cos_sim"] # 0.547
# columns = ['id', 'product_uid', 'len_of_query', 'len_of_title',
#        'len_of_description', 'len_of_brand',
#        'query_in_title', 'query_in_description', 'word_in_title',
#        'word_in_description', 'word_in_brand', 'ratio_title',
#        'ratio_description', 'ratio_brand', 'search_term_feature',
#        'title_cos_sim', 'description_cos_sim', 'bm25_title',
#        'bm25_description']
# columns = ['id',
#        'query_in_title', 'query_in_description', 'word_in_title',
#        'word_in_description', 'word_in_brand', 'ratio_title',
#        'ratio_description', 'ratio_brand', 'search_term_feature',
#        'title_cos_sim', 'description_cos_sim', 'bm25_title',
#        'bm25_description']
columns = ['id',
       "query_in_title", "query_in_description", "word_in_brand", "ratio_title", "ratio_description",
                           "search_term_feature", "description_cos_sim"]
# columns = ['id', 'product_uid', 'len_of_query', 'len_of_title',
#        'len_of_description', 'len_of_brand',
#        'query_in_title', 'query_in_description', 'word_in_title',
#        'word_in_description', 'word_in_brand', 'ratio_title',
#        'ratio_description', 'ratio_brand', 'search_term_feature',
#        'title_cos_sim', 'description_cos_sim', 'bm25_title',
#        'bm25_description']

numeric_test_pub = x_test_public[columns]
numeric_test_priv = x_test_private[columns]

with open('../trained_models/prank77.pkl', 'rb') as input:
    model = pickle.load(input)

#reduced_classes = [1., 1.33, 1.67, 2., 2.33, 2.67, 3.]
reduced_classes = unique_scores
num_q_pub = numeric_test_pub.shape[0]
print('public test size', num_q_pub)
p_y_pub = np.zeros(num_q_pub)
for i in range(num_q_pub):
    p_y_pub[i] = reduced_classes[model.predict(numeric_test_pub.values[i, 1:])-1]
    # p_y_pub[i] = reduced_classes[model.predict(numeric_test_pub.values[i, :]) - 1]

error_pub = rmse(p_y_pub, y_test_public.values[:, 1])

print("Public error ", error_pub)
print(np.unique(p_y_pub, return_counts=True))
print(model.w)
print(model.b)

num_q_priv = numeric_test_priv.shape[0]
print('private test size', num_q_priv)
p_y_priv = np.zeros(num_q_priv)
for i in range(num_q_priv):
    p_y_priv[i] = reduced_classes[model.predict(numeric_test_priv.values[i, 1:])-1]
    # p_y_priv[i] = reduced_classes[model.predict(numeric_test_priv.values[i, :]) - 1]

error_priv = rmse(p_y_priv, y_test_private.values[:, 1])

print("Private error ", error_priv)



