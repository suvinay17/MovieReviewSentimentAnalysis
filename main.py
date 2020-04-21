import numpy as np
import pandas as pd
import string
import itertools

from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from matplotlib import pyplot as plt


def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
#Refer to https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#Return a linear svm classifier based on the given
#penalty function and regularization parameter c.
    if degree == 1:
        return SVC(C=c, kernel='linear', degree=degree, class_weight=class_weight)
    else:
        return SVC(C=c, kernel='poly', degree=degree, class_weight=class_weight, coef0=r)


#TODO: create dictionary of unique words and get a dataframe
def generate_feature_matrix(df, word_dict):
#Reads a dataframe and the dictionary of unique words
#to generate a matrix of {1, 0} feature vectors for each review.
#Use the word_dict to find the correct index to set to 1 for each place
#in the feature vector. The resulting feature matrix should be of
#dimension (number of reviews, number of words).
#Input:
#df: dataframe that has the ratings and labels
##Returns:
#a matrix of size (number of reviews * number of words)

    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    #refer to https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html

    index = 0
    for review in df["reviewText"]:
        review = review.lower()

        for punct in string.punctuation:
            review = review.replace(punct, " ")

        for word in review.split(" "):
            if word != '' and word in word_dict:
                feature_matrix[index][word_dict[word]] = 1

        index = index + 1

    return feature_matrix
