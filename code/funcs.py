import re
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



def ms_error(y, y_hat):

    ms_error_cal = mean_squared_error(y, y_hat)**0.5

    return ms_error_cal

RMSE  = make_scorer(ms_error, greater_is_better=False)

