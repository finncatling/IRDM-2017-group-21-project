import numpy as np
import pandas as pd
import funcs as f
import time
import pickle

start = time.time()

# train and test sets
x_train = pd.read_csv('../../data/train.csv', encoding="ISO-8859-1")
x_test = pd.read_csv('../../data/test.csv', encoding="ISO-8859-1")


# contains lots/different attributes for products
attributes = pd.read_csv('../../data/attributes.csv')

# extract brand from attributes
brand = attributes[attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

# product descriptions
description = pd.read_csv('../../data/product_descriptions.csv')

# separate input and labels
y_train = x_train['relevance']
x_train =  x_train.drop('relevance',axis=1)

# concatenate train and test
x_all = pd.concat((x_train, x_test), axis=0, ignore_index=True)

# merge description and brand to x_all
x_all = pd.merge(x_all, description, how='left', on='product_uid')
x_all = pd.merge(x_all, brand, how='left', on='product_uid')

# create new features
x_all['search_term'] = x_all['search_term'].map(lambda x:f.str_stem(x))
x_all['product_title'] = x_all['product_title'].map(lambda x:f.str_stem(x))
x_all['product_description'] = x_all['product_description'].map(lambda x:f.str_stem(x))
x_all['brand'] = x_all['brand'].map(lambda x:f.str_stem(x))
x_all['len_of_query'] = x_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
x_all['len_of_title'] = x_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
x_all['len_of_description'] = x_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
x_all['len_of_brand'] = x_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
x_all['product_info'] = x_all['search_term']+"\t"+x_all['product_title'] +"\t"+x_all['product_description']
x_all['query_in_title'] = x_all['product_info'].map(lambda x:f.str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
x_all['query_in_description'] = x_all['product_info'].map(lambda x:f.str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
x_all['word_in_title'] = x_all['product_info'].map(lambda x:f.str_common_word(x.split('\t')[0],x.split('\t')[1]))
x_all['word_in_description'] = x_all['product_info'].map(lambda x:f.str_common_word(x.split('\t')[0],x.split('\t')[2]))
x_all['ratio_title'] = x_all['word_in_title']/x_all['len_of_query']
x_all['ratio_description'] = x_all['word_in_description']/x_all['len_of_query']
x_all['attr'] = x_all['search_term']+"\t"+x_all['brand']
x_all['word_in_brand'] = x_all['attr'].map(lambda x:f.str_common_word(x.split('\t')[0],x.split('\t')[1]))
x_all['ratio_brand'] = x_all['word_in_brand']/x_all['len_of_brand']
brand = pd.unique(x_all.brand.ravel())
d={}
i = 1
for s in brand:
    d[s]=i
    i+=1
x_all['brand_feature'] = x_all['brand'].map(lambda x:d[x])
x_all['search_term_feature'] = x_all['search_term'].map(lambda x:len(x))

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

duration = time.time() - start
print(duration)

