import os
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
import funcs as f


start = time.time()

df = pd.read_pickle(os.path.join(os.pardir, os.pardir, 'data', 'x_all.pkl'))
df.drop(['attr', 'product_info'], axis=1, inplace=True)  # TODO: remove later

print('Extracting vocabulary...', round((time.time() - start) / 60, 2))
vocab = []
for i, r in df.iterrows():
    for col in ['product_title', 'search_term', 'product_description']:
        vocab += r[col]
vocab = set(vocab)


print('Deriving bag-of-words counts...', round((time.time() - start) / 60, 2))
vectorizers, text_mats, transformers, tfidf_mats = dict(), dict(), dict(), dict()
for col in ['product_title', 'search_term', 'product_description']:
    vectorizers[col] = CountVectorizer(vocabulary=vocab)
    text_mats[col] = vectorizers[col].fit_transform(df[col].str.join(' ').values)
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
df['title_cos_sim'], df['description_cos_sim'] = np.nan, np.nan
for i in range(matrix_chunks):
    title_cos_sim += list(np.dot(
        text_mats_norm['search_term'][chunk_i[i]:chunk_i[i + 1], :],
        text_mats_norm['product_title'][chunk_i[i]:chunk_i[i + 1], :].T
    ).diagonal())
    description_cos_sim += list(np.dot(
        text_mats_norm['search_term'][chunk_i[i]:chunk_i[i + 1], :],
        text_mats_norm['product_description'][chunk_i[i]:chunk_i[i + 1], :].T
    ).diagonal())
df['title_cos_sim'] = title_cos_sim
df['description_cos_sim'] = description_cos_sim


def bm25(
        row,
        collection,  # e.g. 'product_title'
        avg_length,  # average document length in collection
        k1=1.6,  # typically set between 1.2 and 2.0
        b=0.75,  # typically set at 0.75
        text_mats=text_mats,
        vocab=vectorizers['search_term'].vocabulary_
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


title_mean = df['len_of_title'].mean()
description_mean = df['len_of_description'].mean()


def bm25_title(df):
    df['bm25_title'] = df.apply(lambda x: bm25(
        x, 'product_title', title_mean), axis=1)
    return df


def bm25_description(df):
    df['bm25_description'] = df.apply(
        lambda x: bm25(x, 'product_description', description_mean), axis=1)
    return df


print('Calculating BM25...', round((time.time() - start) / 60, 2))
df = f.parallelize_dataframe(df, bm25_title)
df = f.parallelize_dataframe(df, bm25_description)

df.to_pickle('../../data/x_all_cos_sim_bm25.pkl')
