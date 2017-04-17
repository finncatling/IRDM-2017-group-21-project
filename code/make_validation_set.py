import numpy as np
import pickle
import funcs


pickle_file = '../../data/pre_processed_data.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    x_train = save['x_train']
    y_train = save['y_train']
    del save

# The interview with winning team said "created two runs of 3-fold
# cross-validation with disjoint search terms among
# the folds, and one 3-fold cross-validation with disjoint
# product id sets" ('product_uid')
#
x_train_, y_train_, x_val, y_val = funcs.split_val_set(
    x_train, y_train, 0.3, 'search_term')

print(x_train.shape)
print(x_train_.shape)
print(y_train_.shape)
print(x_val.shape)
print(y_val.shape)

val_values = np.unique(x_val['search_term'])
train_values = np.unique(x_train_['search_term'])
total_values = np.unique(x_train['search_term'])
common_elements = set.intersection(
    set(map(tuple, val_values)),
    set(map(tuple, train_values))
)

print("total number of unique values", len(total_values))
print("number of unique values in validation", len(val_values))
print("number of unique values in training", len(train_values))
print("number of values in common", len(common_elements))

assert(len(common_elements) == 0)
print('Test passed so saving validation set...')
val_file = '../../data/validation_set.pickle'
f = open(val_file, 'wb')
save = {
    'x_train': x_val,
    'y_val': y_val
}
pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
f.close()
