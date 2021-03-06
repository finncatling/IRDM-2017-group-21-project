import string
from multiprocessing import Pool, cpu_count
import numpy as np
from numpy.random import RandomState
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import PredefinedSplit
from spelling import *


stem = True  # consider skipping in neural models as GloVe embeddings are mapped to words
# remove_stopwords = True  # consider skipping in neural models
remove_stopwords = False
remove_punctuation = True  # consider skipping in neural models


num_partitions = cpu_count()
num_cores = cpu_count()
stemmer = PorterStemmer()
stops = set(stopwords.words("english"))
punct = set(string.punctuation)
print(nltk.__version__, '<- needs to be less than 3.2.2')
print('Parallel computations will use', cpu_count(), 'cores...')


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def process_strings_search(x_all):
    x_all['search_term'] = x_all['search_term'].map(
        lambda x: process_words(x, stem, remove_stopwords, remove_punctuation))
    return x_all


def process_strings_title(x_all):
    x_all['product_title'] = x_all['product_title'].map(
        lambda x: process_words(x, stem, remove_stopwords, remove_punctuation))
    return x_all


def process_strings_description(x_all):
    x_all['product_description'] = x_all['product_description'].map(
        lambda x: process_words(x, stem, remove_stopwords, remove_punctuation))
    return x_all


def process_strings_brand(x_all):
    x_all['brand'] = x_all['brand'].map(
        lambda x: process_words(x, stem, remove_stopwords, remove_punctuation))
    return x_all


def find_search_terms_title(x_all):
    x_all['query_in_title'] = x_all['product_info'].map(
        lambda x: term_intersection(x.split('\t')[0], x.split('\t')[1]))
    return x_all


def find_search_terms_description(x_all):
    x_all['query_in_description'] = x_all['product_info'].map(
        lambda x: term_intersection(x.split('\t')[0], x.split('\t')[2]))
    return x_all


def find_common_words_title(x_all):
    x_all['word_in_title'] = x_all['product_info'].map(
        lambda x: word_intersection(x.split('\t')[0], x.split('\t')[1]))
    return x_all


def find_common_words_description(x_all):
    x_all['word_in_description'] = x_all['product_info'].map(
        lambda x: word_intersection(x.split('\t')[0], x.split('\t')[2]))
    return x_all


def find_common_words_brand(x_all):
    x_all['word_in_brand'] = x_all['attr'].map(
        lambda x: word_intersection(x.split('\t')[0], x.split('\t')[1]))
    return x_all


def process_words(words, stem=True, remove_stopwords=True, remove_punctuation=True):
    if isinstance(words, str):
        words = words.lower()
        
        # spellcheck phrases
        if words in external_data_dict.keys():
            words = external_data_dict[words]

        # tokenize, so now words is a list, not a str
        words = nltk.word_tokenize(words)

        if remove_stopwords:
            words = [z for z in words if z not in stops]

        if remove_punctuation:
            words = [z for z in words if z not in punct]

        # spellcheck words
        words = [external_data_dict[z] if z in external_data_dict.keys() else z for z in words]

        if stem:
            words = [stemmer.stem(z) for z in words]
        
        return words

    else:
        return "null"


def term_intersection(term1, term2):
    count = 0
    counter = 0

    while counter < len(term2):
        counter = term2.find(term1, counter)

        if len(term1) == 0:
            return count
        elif counter == -1:
            return count
        else:
            count += 1
            counter += len(term1)

    return count


def split_val_set(x_, y_, val_ratio, col_name, seed=None):
    num_points = x_.shape[0]
    val_size_needed = np.floor(num_points * val_ratio).astype(int)
    # get all unique values and how many times they occur
    dif_items, item_counts = np.unique(x_[col_name].values, return_counts=True)
    # get a random permutation of indices for the unique values
    rnd = RandomState(seed)
    rand_perm = rnd.permutation(len(dif_items)) - 1
    val_size = 0
    i = 0
    val_items = []  # values to go into validation set
    # put values from the random permutation into validation set until there are enough rows
    while val_size < val_size_needed:
        val_items.append(dif_items[rand_perm[i]])
        val_size += item_counts[rand_perm[i]]
        i += 1
    # get an array that says whether a row is in the validation set
    val_rows = np.in1d(x_[col_name].values, np.asarray(val_items))
    # split validation and training sets
    x_val = x_[val_rows]
    y_val = y_[val_rows]
    x_train = x_[np.invert(val_rows)]
    y_train = y_[np.invert(val_rows)]
    return x_train, y_train, x_val, y_val


def get_k_folds(k, x_, y_, col_name, start_seed=None):
    """
    Return the data set split into k folds
    :param k: number of folds
    :param x_: pandas data frame as loaded from pickle after pre-processing
    :param y_: pandas series of true scores as loaded from pickle
    :param col_name: name of column to split on e.g. 'product_uid'
    :param start_seed: integer, allows reproducible splitting
    :return: a list of folds. each fold contains 4 elements - x_train, y_train, x_val, y_val
    """
    val_sets = []
    x_t = x_
    y_t = y_
    for i in range(k-1):
        ratio_ = 1/(k-i)
        if start_seed:
            x_train, y_train, x_val, y_val = split_val_set(
                x_t, y_t, ratio_, col_name, start_seed + i)
        else:
            x_train, y_train, x_val, y_val = split_val_set(x_t, y_t, ratio_, col_name)
        val_sets.append([x_val, y_val])
        x_t = x_train
        y_t = y_train
    val_sets.append([x_t, y_t])  # the last iteration split the set in half
    folds = []
    for i in range(k):
        train_list_x = [val_sets[j][0] for j in range(
            len(val_sets)) if j != i]  # all other folds will be used for training
        train_list_y = [val_sets[j][1] for j in range(len(val_sets)) if j != i]
        x_train = pd.concat(train_list_x, axis=0, ignore_index=True)
        y_train = pd.concat(train_list_y, axis=0, ignore_index=True)
        folds.append([x_train, y_train, val_sets[i][0], val_sets[i][1]])
    return folds


def k_folds_generator(k, x_, y_, col_name, start_seed=1):
    """Adapts get_k_folds into a generator object for use with sklearn.

    :param k: number of folds
    :param x_: pandas data frame as loaded from pickle after pre-processing
    :param y_: pandas series of true scores as loaded from pickle
    :param col_name: name of column to split on e.g. 'product_uid'
    :param start_seed: integer, allows reproducible splitting
    :return: cross validation generator object
    """
    val_sets = dict()
    df = pd.DataFrame({'test_fold': np.zeros(len(x_))})
    x_t = x_
    y_t = y_
    for i in range(k - 1):
        ratio_ = 1 / (k - i)
        if start_seed:
            x_train, y_train, x_val, y_val = split_val_set(
                x_t, y_t, ratio_, col_name, start_seed + i)
        else:
            x_train, y_train, x_val, y_val = split_val_set(
                x_t, y_t, ratio_, col_name)
        val_sets[i] = x_val.index
        x_t = x_train
        y_t = y_train
    val_sets[k - 1] = x_t.index  # the last iteration split the set in half

    for i in range(k):
        df.loc[val_sets[i]] = i
    df.test_fold.astype(int, inplace=True)

    return PredefinedSplit(df.test_fold.values)


def word_intersection(term1, term2):
    words = term1.split()
    count = 0

    for word in words:
        if term2.find(word) >= 0:
            count += 1

    return count


def ms_error(y, y_hat):
    ms_error_cal = mean_squared_error(y, y_hat)**0.5
    return ms_error_cal

RMSE = make_scorer(ms_error, greater_is_better=False)


def bm25(
        row,
        collection,  # e.g. 'product_title'
        avg_length,  # average document length in collection
        text_mats,
        vocab,
        k1=1.6,  # typically set between 1.2 and 2.0
        b=0.75  # typically set at 0.75
):
    """Calculates BM25 as per Wikipedia formula when applied over DataFrame."""
    score = 0

    if collection == 'product_title':
        doc_length = row['len_of_title']
    elif collection == 'product_description':
        doc_length = row['len_of_description']

    for term in row['search_term']:
        nq = text_mats[collection][:, vocab[term]].nnz  # n docs containing term
        idf = np.log((text_mats[collection].shape[0] - nq + 0.5) / (nq + 0.5))
        term_freq = text_mats[collection][row.name, vocab[term]]
        score += idf * (
            (term_freq * (k1 + 1)) /
            (term_freq + k1 * (1 - b + b * doc_length / avg_length))
        )

    return score


def test_rmse(y_test, y_pred):

    public_idx = y_test['Usage'] == 'Public'
    private_idx = y_test['Usage'] == 'Private'

    y_public = y_test[public_idx]['relevance']
    y_private = y_test[private_idx]['relevance']

    y_pred_public = y_pred[public_idx]
    y_pred_private = y_pred[private_idx]

    public_rmse = ms_error(y_public, y_pred_public)
    private_rmse = ms_error(y_private, y_pred_private)

    return public_rmse, private_rmse


