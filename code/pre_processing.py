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


print('Loading data...')
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
print('Processing strings...', round((time.time() - start) / 60, 2))
x_all['search_term'] = x_all['search_term'].map(
    lambda x: f.process_words(x, stem, remove_stopwords, remove_punctuation))
x_all['product_title'] = x_all['product_title'].map(
    lambda x: f.process_words(x, stem, remove_stopwords, remove_punctuation))
x_all['product_description'] = x_all['product_description'].map(
    lambda x: f.process_words(x, stem, remove_stopwords, remove_punctuation))
x_all['brand'] = x_all['brand'].map(
    lambda x: f.process_words(x, stem, remove_stopwords, remove_punctuation))

# create length features
print('Counting lengths...', round((time.time() - start) / 60, 2))
x_all['len_of_query'] = x_all['search_term'].map(lambda x: len(x)).astype(np.int64)
x_all['len_of_title'] = x_all['product_title'].map(lambda x: len(x)).astype(np.int64)
x_all['len_of_description'] = x_all['product_description'].map(lambda x: len(x)).astype(np.int64)
x_all['len_of_brand'] = x_all['brand'].map(lambda x: len(x)).astype(np.int64)

# pickle the data
pickle_file = '../../data/pre_processed_data_1.pickle'

f = open(pickle_file, 'wb')
save = {
    'x_all': x_all,
    'x_train': x_train,
    'y_train': y_train,
    'x_test': x_test
}
pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
f.close()

duration = time.time() - start
print(duration)
