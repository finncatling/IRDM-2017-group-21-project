
# coding: utf-8
# pre_processing.py and func.py were used as a base to generated this file
import string
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from spelling import *


stem = False
remove_stopwords = True
remove_punctuation = True


num_partitions = cpu_count()
num_cores = cpu_count()
stemmer = PorterStemmer()
stops = set(stopwords.words("english"))
punct = set(string.punctuation)
print(nltk.__version__, '<- needs to be less than 3.2.2')
print('Parallel computations will use', cpu_count(), 'cores...')

# Functions
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


# Main part starts here
import time
import pickle


seed = 1
start = time.time()

print('Loading data...')
# train and test sets
x_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
x_test = pd.read_csv('test.csv', encoding="ISO-8859-1")

# contains lots/different attributes for products
attributes = pd.read_csv('attributes.csv')

# product descriptions
description = pd.read_csv('product_descriptions.csv')

# separate input and labels
y_train = x_train['relevance']
x_train = x_train.drop('relevance', axis=1)

# concatenate train and test
x_all = pd.concat((x_train, x_test), axis=0, ignore_index=True)

# merge description and brand to x_all
x_all = pd.merge(x_all, description, how='left', on='product_uid')


# define corpora
print('Processing strings (in parallel)...', round((time.time() - start) / 60, 2))
x_all = parallelize_dataframe(x_all, process_strings_search)
x_all = parallelize_dataframe(x_all, process_strings_title)
x_all = parallelize_dataframe(x_all, process_strings_description)



print('Extracting vocabulary...', round((time.time() - start) / 60, 2))
vocab = []
for i, r in x_all.iterrows():
    for col in ['product_title', 'search_term', 'product_description']:
        vocab += r[col]
vocab = set(vocab)


print('Saving data...', round((time.time() - start) / 60, 2))
print('x_all shape', x_all.shape)

# Pickle vocab
pickle.dump(vocab, open( "vocab.p", "wb" ))

print('Finished in', round((time.time() - start) / 60, 2), 'minutes.')
