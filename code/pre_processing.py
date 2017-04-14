import numpy as np
import pandas as pd
import code.funcs as f
import time
import pickle
from sklearn.utils import shuffle


stem = True  # consider skipping in neural models as GloVe embeddings are mapped to words
remove_stopwords = True  # consider skipping in neural models
remove_punctuation = True  # consider skipping in neural models

seed = 1
start = time.time()

# train and test sets
x_train = pd.read_csv('../../data/train.csv', encoding="ISO-8859-1")
x_test = pd.read_csv('../../data/test.csv', encoding="ISO-8859-1")

# contains lots/different attributes for products
attributes = pd.read_csv('../../data/attributes.csv')

# extract brand from attributes
brand = attributes[attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(
    columns={"value": "brand"})

# product descriptions
description = pd.read_csv('../../data/product_descriptions.csv')

# separate input and labels
y_train = x_train['relevance']
x_train = x_train.drop('relevance', axis=1)

# concatenate train and test
x_all = pd.concat((x_train, x_test), axis=0, ignore_index=True)

# merge description and brand to x_all
x_all = pd.merge(x_all, description, how='left', on='product_uid')
x_all = pd.merge(x_all, brand, how='left', on='product_uid')

# define corpora
x_all['search_term'] = x_all['search_term'].map(
    lambda x: f.process_words(x, stem, remove_stopwords, remove_punctuation))
x_all['product_title'] = x_all['product_title'].map(
    lambda x: f.process_words(x, stem, remove_stopwords, remove_punctuation))
x_all['product_description'] = x_all['product_description'].map(
    lambda x: f.process_words(x, stem, remove_stopwords, remove_punctuation))
x_all['brand'] = x_all['brand'].map(
    lambda x: f.process_words(x, stem, remove_stopwords, remove_punctuation))

# create length features
x_all['len_of_query'] = x_all['search_term'].map(lambda x: len(x)).astype(np.int64)
x_all['len_of_title'] = x_all['product_title'].map(lambda x: len(x)).astype(np.int64)
x_all['len_of_description'] = x_all['product_description'].map(lambda x: len(x)).astype(np.int64)
x_all['len_of_brand'] = x_all['brand'].map(lambda x: len(x)).astype(np.int64)

# does search term appear in: (brand added no information)
x_all['query_in_title'] = x_all.apply(
    lambda x: f.term_intersection(x['search_term'], x['product_title']))
x_all['query_in_description'] = x_all.apply(
    lambda x: f.term_intersection(x['search_term'], x['product_description']))

# how many common words
x_all['word_in_title'] = x_all.apply(
    lambda x: f.word_intersection(x['search_term'], x['product_title']))
x_all['word_in_description'] = x_all.apply(
    lambda x: f.word_intersection(x['search_term'], x['product_description']))
x_all['word_in_brand'] = x_all.apply(
    lambda x: f.word_intersection(x['search_term'], x['brand']))

# concatenate corpora search term, product title and product description
# brand is defined separately in attr
# x_all['product_info'] = (
#     x_all['search_term'] +
#     ["\t"] +
#     x_all['product_title'] +
#     ["\t"] +
#     x_all['product_description']
# )
# x_all['attr'] = x_all['search_term'] + ["\t"] + x_all['brand']

# ratio of words in target to search term
x_all['ratio_title'] = x_all['word_in_title'] / x_all['len_of_query']
x_all['ratio_description'] = x_all['word_in_description'] / x_all['len_of_query']
x_all['ratio_brand'] = x_all['word_in_brand'] / x_all['len_of_brand']

# letters in search term
x_all['search_term_feature'] = x_all['search_term'].map(lambda x: f.count_letters(x))

print('x_all shape', x_all.shape)

# save to csv file to view
x_all.to_csv('../../data/x_all.csv')
x_all = pd.read_csv('../../data/x_all.csv', encoding="ISO-8859-1", index_col=0)

# separate train and test
train_size = x_train.shape[0]
x_train = x_all[:train_size]
x_test = x_all[train_size:]
print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)

# shuffle data
x_train, y_train = shuffle(x_train, y_train, random_state=seed)
x_test = shuffle(x_test, random_state=seed)

# pickle the data
pickle_file = '../../data/pre_processed_data.pickle'

f = open(pickle_file, 'wb')
save = {
  'x_train': x_train,
  'y_train': y_train,
  'x_test': x_test
  }
pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
f.close()

duration = time.time() - start
print(duration)
