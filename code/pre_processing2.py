import numpy as np
import pandas as pd
import code.funcs as f
import time
import pickle
from sklearn.utils import shuffle

seed = 1
start = time.time()

print('Loading data...')
pickle_file = '../../data/pre_processed_data_1.pickle'
with open(pickle_file, 'rb') as file:
  save = pickle.load(file)
  x_all = save['x_all']
  x_train = save['x_train']
  y_train = save['y_train']
  x_test = save['x_test']
  del save

# does search term appear in: (brand added no information)
print('Temporarily re-joining string lists...', round((time.time() - start) / 60, 2))
x_all['search_term_joined'] = x_all['search_term'].str.join(" ")
x_all['product_title_joined'] = x_all['product_title'].str.join(" ")
x_all['product_description_joined'] = x_all['product_description'].str.join(" ")

print('Looking for search terms in title...', round((time.time() - start) / 60, 2))
x_all['query_in_title'] = x_all.apply(
    lambda x: f.term_intersection(x['search_term_joined'], x['product_title_joined']), axis=1)
print('Looking for search terms in description...', round((time.time() - start) / 60, 2))
x_all['query_in_description'] = x_all.apply(
    lambda x: f.term_intersection(x['search_term_joined'], x['product_description_joined']), axis=1)


# how many common words
print('Looking for common words in title...', round((time.time() - start) / 60, 2))
x_all['word_in_title'] = x_all.apply(
    lambda x: f.word_intersection(x['search_term'], x['product_title']), axis=1)
print('Looking for common words in description...', round((time.time() - start) / 60, 2))
x_all['word_in_description'] = x_all.apply(
    lambda x: f.word_intersection(x['search_term'], x['product_description']), axis=1)
print('Looking for common words in brand...', round((time.time() - start) / 60, 2))
x_all['word_in_brand'] = x_all.apply(
    lambda x: f.word_intersection(x['search_term'], x['brand']), axis=1)


# ratio of words in target to search term
print('Calculating words ratios...', round((time.time() - start) / 60, 2))
x_all['ratio_title'] = x_all['word_in_title'] / x_all['len_of_query']
x_all['ratio_description'] = x_all['word_in_description'] / x_all['len_of_query']
x_all['ratio_brand'] = x_all['word_in_brand'] / x_all['len_of_brand']


# letters in search term
print('Counting letters in search term...', round((time.time() - start) / 60, 2))
x_all['search_term_feature'] = x_all['search_term'].map(lambda x: f.count_letters(x))


print('Saving data...', round((time.time() - start) / 60, 2))
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

print('Finished in', round((time.time() - start) / 60, 2), 'minutes.')
