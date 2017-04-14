import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error, make_scorer
from code.spelling import *


stemmer = PorterStemmer()
stops = set(stopwords.words("english"))
punct = set(string.punctuation)
print(nltk.__version__, '<- needs to be less than 3.2.2')


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

        if counter == -1:
            return count

        else:
            count += 1
            counter += len(term1)

    return count


def word_intersection(needles, haystack):
    count = 0
    needles, haystack = set(needles), set(haystack)
    for needle in needles:
        if needle in haystack:
            count += 1
    return count


def count_letters(words):
    count = 0
    for word in words:
        count += len(word)
    return count


def ms_error(y, y_hat):
    ms_error_cal = mean_squared_error(y, y_hat)**0.5
    return ms_error_cal


RMSE = make_scorer(ms_error, greater_is_better=False)
