import re
import pandas as pd
import numpy as np
from nltk.stem.porter import *
from sklearn.metrics import mean_squared_error, make_scorer
stemmer = PorterStemmer()
import nltk
from spelling import *
print(nltk.__version__,'<- needs to be less than 3.2.2')

def word_stem(words): 

    if isinstance(words, str):

        words = words.lower()
        
        # checks phrases
        if words in external_data_dict.keys():

            words = external_data_dict[words]

        # checks words
        words = (" ").join([external_data_dict[z] if z in external_data_dict.keys() else z for z in words.split(" ")])
        
        words = (" ").join([stemmer.stem(z) for z in words.split(" ")])
        
        return words.lower()
    else:
        return "null"



def word_intersection(term1, term2):

    words =  term1.split()
    count = 0 

    for word in words:

        if term2.find(word)>=0:

            count+=1

    return count



def term_intersection(term1, term2):

    count = 0
    counter = 0
    
    while counter < len(term2):

        counter = term2.find(term1, counter)

        if counter == -1:

            return count

        else:
            count += 1

            counter += len(term1)

    return count


def split_val_set(x_, y_, val_ratio, col_name):
    num_points = x_.shape[0]
    val_size_needed = np.floor(num_points * val_ratio).astype(int)
    # get all unique values and how many times they occur
    dif_items, item_counts = np.unique(x_[col_name].values, return_counts=True)
    # get a random permutation of indices for the unique values
    rand_perm = np.random.permutation(len(dif_items)) - 1
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


def ms_error(y, y_hat):

    ms_error_cal = mean_squared_error(y, y_hat)**0.5

    return ms_error_cal

RMSE  = make_scorer(ms_error, greater_is_better=False)

