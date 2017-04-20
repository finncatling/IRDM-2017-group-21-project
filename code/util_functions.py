import numpy as np
import pandas as pd


def split_val_set(x_, y_, val_ratio, col_name):
    num_points = x_.shape[0]
    val_size_needed = np.floor(num_points * val_ratio).astype(int)
    dif_items, item_counts = np.unique(x_[col_name].values, return_counts=True)
    rand_perm = np.random.permutation(len(dif_items)) - 1  # gets a random permutation of indices for the unique values
    val_size = 0
    i = 0
    val_items = []  # values to go into validation set
    while val_size < val_size_needed:
        val_items.append(dif_items[rand_perm[i]])
        val_size += item_counts[rand_perm[i]]
        i += 1
    val_rows = np.in1d(x_[col_name].values, np.asarray(val_items))
    x_val = x_[val_rows]
    y_val = y_[val_rows]
    x_train = x_[np.invert(val_rows)]
    y_train = y_[np.invert(val_rows)]
    return x_train, y_train, x_val, y_val


def get_k_folds(k, x_, y_, col_name):
    """
    Return the data set split into k folds
    :param k: number of folds
    :param x_: pandas data frame as loaded from pickle after pre-processing
    :param y_: pandas series of true scores as loaded from pickle
    :param col_name: name of column to split on e.g. 'product_uid'
    :return: a list of folds. each fold contains 4 elements - x_train, y_train, x_val, y_val
    """
    val_sets = []
    x_t = x_
    y_t = y_
    for i in range(k-1):
        ratio_ = 1/(k-i)
        x_train, y_train, x_val, y_val = split_val_set(x_t, y_t, ratio_, col_name)
        val_sets.append([x_val, y_val])
        x_t = x_train
        y_t = y_train
    val_sets.append([x_t, y_t]) # the last iteration split the set in half
    folds = []
    for i in range(k):
        train_list_x = [val_sets[j][0] for j in range(len(val_sets)) if j != i] # all other folds will be used for training
        train_list_y = [val_sets[j][1] for j in range(len(val_sets)) if j != i]
        x_train = pd.concat(train_list_x, axis=0, ignore_index=True)
        y_train = pd.concat(train_list_y, axis=0, ignore_index=True)
        folds.append([x_train, y_train, val_sets[i][0], val_sets[i][1]])
    return folds