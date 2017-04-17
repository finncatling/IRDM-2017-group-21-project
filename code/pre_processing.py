import numpy as np
import pandas as pd
import funcs as f
import time
import pickle
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize


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
print('Processing strings (in parallel)...', round((time.time() - start) / 60, 2))
x_all = f.parallelize_dataframe(x_all, f.process_strings_search)
x_all = f.parallelize_dataframe(x_all, f.process_strings_title)
x_all = f.parallelize_dataframe(x_all, f.process_strings_description)
x_all = f.parallelize_dataframe(x_all, f.process_strings_brand)

# create length features
print('Counting lengths...', round((time.time() - start) / 60, 2))
x_all['len_of_query'] = x_all['search_term'].map(lambda x: len(x)).astype(np.int64)
x_all['len_of_title'] = x_all['product_title'].map(lambda x: len(x)).astype(np.int64)
x_all['len_of_description'] = x_all['product_description'].map(lambda x: len(x)).astype(np.int64)
x_all['len_of_brand'] = x_all['brand'].map(lambda x: len(x)).astype(np.int64)

print('Temporarily re-joining string lists...', round((time.time() - start) / 60, 2))
x_all['search_term_joined'] = x_all['search_term'].str.join(" ")
x_all['product_title_joined'] = x_all['product_title'].str.join(" ")
x_all['product_description_joined'] = x_all['product_description'].str.join(" ")
x_all['brand_joined'] = x_all['brand'].str.join(" ")

# concatenate corpora search term, product title and product description / brand is defined separately in attr
x_all['product_info'] = (
    x_all['search_term_joined'] +
    "\t" +
    x_all['product_title_joined'] +
    "\t" +
    x_all['product_description_joined']
)
x_all['attr'] = x_all['search_term_joined'] + "\t" + x_all['brand_joined']

### does search term appear in: (brand added no information)
print('Looking for search terms (in parallel)...', round((time.time() - start) / 60, 2))
x_all = f.parallelize_dataframe(x_all, f.find_search_terms_title)
x_all = f.parallelize_dataframe(x_all, f.find_search_terms_description)

### how many common words
print('Looking for common words (in parallel)...', round((time.time() - start) / 60, 2))
x_all = f.parallelize_dataframe(x_all, f.find_common_words_title)
x_all = f.parallelize_dataframe(x_all, f.find_common_words_description)
x_all = f.parallelize_dataframe(x_all, f.find_common_words_brand)

# ratio of words in target to search term
print('Calculating words ratios...', round((time.time() - start) / 60, 2))
x_all['ratio_title'] = x_all['word_in_title'] / x_all['len_of_query']
x_all['ratio_description'] = x_all['word_in_description'] / x_all['len_of_query']
x_all['ratio_brand'] = x_all['word_in_brand'] / x_all['len_of_brand']

# letters in search term
print('Counting letters in search term...', round((time.time() - start) / 60, 2))
x_all['search_term_feature'] = x_all['search_term_joined'].map(lambda x: len(x))

# Get rid of temp columns
x_all.drop([
    'search_term_joined',
    'product_title_joined',
    'product_description_joined',
    'brand_joined'
], axis=1, inplace=True)


print('Extracting vocabulary...', round((time.time() - start) / 60, 2))
vocab = []
for i, r in x_all.iterrows():
    for col in ['product_title', 'search_term', 'product_description']:
        vocab += r[col]
vocab = set(vocab)


print('Deriving bag-of-words counts...', round((time.time() - start) / 60, 2))
vectorizers, text_mats = dict(), dict()
# transformers, tfidf_mats = dict(), dict()
for col in ['product_title', 'search_term', 'product_description']:
    vectorizers[col] = CountVectorizer(vocabulary=vocab)
    text_mats[col] = vectorizers[col].fit_transform(x_all[col].str.join(' ').values)
    # transformers[col] = TfidfTransformer()
    # tfidf_mats[col] = transformers[col].fit_transform(text_mats[col])


print('Calculating cosine similarity...', round((time.time() - start) / 60, 2))
text_mats_norm = dict()
for col in [
    'product_title',
    'search_term',
    'product_description'
]:
    text_mats_norm[col] = normalize(text_mats[col])

matrix_chunks = 8  # lower numbers use more ram but compute cos similarity quicker
chunk_i = [0]
for i in range(matrix_chunks - 1):
    chunk_i.append(
        round(text_mats_norm['search_term'].shape[0] / matrix_chunks) * (i + 1)
    )
chunk_i.append(text_mats_norm['search_term'].shape[0])

title_cos_sim, description_cos_sim = [], []
x_all['title_cos_sim'], x_all['description_cos_sim'] = np.nan, np.nan
for i in range(matrix_chunks):
    title_cos_sim += list(np.dot(
        text_mats_norm['search_term'][chunk_i[i]:chunk_i[i + 1], :],
        text_mats_norm['product_title'][chunk_i[i]:chunk_i[i + 1], :].T
    ).diagonal())
    description_cos_sim += list(np.dot(
        text_mats_norm['search_term'][chunk_i[i]:chunk_i[i + 1], :],
        text_mats_norm['product_description'][chunk_i[i]:chunk_i[i + 1], :].T
    ).diagonal())
x_all['title_cos_sim'] = title_cos_sim
x_all['description_cos_sim'] = description_cos_sim


print('Calculating title BM25 (in parallel)...', round((time.time() - start) / 60, 2))
title_mean = x_all['len_of_title'].mean()
description_mean = x_all['len_of_description'].mean()


def bm25_title(x_all):
    x_all['bm25_title'] = x_all.apply(lambda x: f.bm25(
        x, 'product_title', title_mean, text_mats,
        vectorizers['search_term'].vocabulary_), axis=1)
    return x_all


def bm25_description(x_all):
    x_all['bm25_description'] = x_all.apply(lambda x: f.bm25(
        x, 'product_description', description_mean, text_mats,
        vectorizers['search_term'].vocabulary_), axis=1)
    return x_all


x_all = f.parallelize_dataframe(x_all, bm25_title)
print('Calculating description BM25 (in parallel)...', round((time.time() - start) / 60, 2))
x_all = f.parallelize_dataframe(x_all, bm25_description)


print('Saving data...', round((time.time() - start) / 60, 2))
print('x_all shape', x_all.shape)

# Pickle dataframe
x_all.to_pickle('../../data/x_all.pkl')
# x_all = pd.read_csv('../../data/x_all.csv', encoding="ISO-8859-1", index_col=0)

# separate train and test
train_size = x_train.shape[0]  # TODO: check train/test sizes are as expected
x_train = x_all[:][:train_size]
x_test = x_all[:][train_size:]
print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)

# shuffle data
x_train, y_train = shuffle(x_train, y_train, random_state=seed)

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
