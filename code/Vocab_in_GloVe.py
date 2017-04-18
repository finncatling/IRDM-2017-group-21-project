
# coding: utf-8

# This matches words in Vocabulary to their GloVe representation
# This file is generated based on filter_glove_embeddings.ipynb, located in irdm main folder
# As a vocabulary the file that was generated in create_vocab.py is used
# Vocabulary in GloVe representation is saved to .csv file

import os
import pandas as pd
import pickle


# Load vocabulary
vocab = pickle.load(open( "vocab.p", "rb" ))


def parse_glove(vecs):
    """Parses embeddings."""
    print("Building dataframe...")
    for emb in vecs:
        emb = emb.split(' ')
        emb[-1] = emb[-1][:-2]  # remove newline symbol
        try:
            df[emb[0]] = [float(i) for i in emb[1:]]
        except NameError:
            df = pd.DataFrame({
                emb[0]: [float(i) for i in emb[1:]]
            })
    return df


def read_glove(
    vocab,
    path=os.path.join('path', 'to','glove.42B.300d.txt')
):
    """Loads in huge GloVe file line-by-line, filters out
        embeddings for words not in vocabulary.

    Args:
        vocab (set): Vocabulary from train/validation/test data
        path (str): Path to caseless GloVe file

    Returns:
        DataFrame: Vocab-specific embeddings

    """
    print("Vocab length is", len(vocab))
    vecs = []

    with open(path, encoding="utf8") as file:
        # loop more than length of embeddings file (~1.9M lines)
        for i in range(2000000):
            if i % 100000 == 0:
                print("Read", i, "lines...")
            try:
                line = next(file)
                if line.split(' ')[0] in vocab:
                    vecs.append(line)
            except StopIteration:
                print("Finished reading file...")
                df = parse_glove(vecs)
                break

    print("Found", len(df.columns), "embeddings.")
    return df.T

read_glove(vocab).to_csv(
    os.path.join('path', 'to', 'glove_train_caseless_without_stops.csv')
)
