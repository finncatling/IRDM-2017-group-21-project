import string
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error, make_scorer
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


RMSE = make_scorer(ms_error, greater_is_better=False)
