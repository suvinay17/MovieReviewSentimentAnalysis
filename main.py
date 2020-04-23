import numpy as np
from sklearn.svm import SVC, LinearSVC
#from sklearn.model_selection import StratifiedKFold, GridSearchCV
#from sklearn import metrics
#from sklearn.multiclass import OneVsOneClassifier
#from matplotlib import pyplot as plt
import string
#import itertools
import glob
import os
import re

#This code is for reading the multiple files in one folder into a list
# read neg files in train folder
neg_files = glob.glob(os.path.join(os.getcwd(), "data/train/neg", "*.txt"))
neg_list = []
for neg_path in neg_files:
    with open(neg_path) as neg_input:
        neg_list.append(neg_input.read())
#test purpose
#print(neg_list[1])

# read pos files in train folder
pos_files = glob.glob(os.path.join(os.getcwd(), "data/train/pos", "*.txt"))
pos_list = []
for pos_path in pos_files:
    with open(pos_path) as pos_input:
        pos_list.append(pos_input.read())
#test purpose
#print(len(pos_list))
number_of_reviews = len(pos_list)+len(neg_list)


def extract_dictionary():
    """
    Reads list of distinct words
    mapping from each distinct word to its index .
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found

    """
    print(neg_list[0])
    index = 0
    ind = 0
    word_dict = {}
    while(index <  1): # len(pos_list)):
        regex = re.compile(r'[^a-zA-Z ]+')
        pos_list[index] = re.sub(regex, '', pos_list[index]).lower()
        # split string
        splits = pos_list[index].split()
        index = index + 1
    # for loop to iterate over words array
        for split in splits:
            if split not in word_dict:
                word_dict[split] = ind
                ind = ind + 1
    index = 0

    while(index < 1): # len(neg_list)):
        regex = re.compile(r'[^a-zA-Z ]+')
        neg_list[index] = re.sub(regex, '', neg_list[index]).lower()
        # split string
        splits = neg_list[index].split()
        # for loop to iterate over words array
        for split in splits:
             if split not in word_dict:
                word_dict[split] = ind
                ind = ind + 1
        index = index +1
    # for testing
    print(len(neg_list[0]) + len(pos_list[0]))
    # print(neg_list[0])
    return word_dict


hm = extract_dictionary()



def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
# Refer to https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# Return a linear svm classifier based on the given
# penalty function and regularization parameter c.
    if degree == 1:
        return SVC(C=c, kernel='linear', degree=degree, class_weight=class_weight)
    else:
        return SVC(C=c, kernel='poly', degree=degree, class_weight=class_weight, coef0=r)


def generate_feature_matrix(hm):
# Reads the set of unique words to generate a matrix of {1, 0} feature vectors for each review.
# Use the word_dict to find the correct index to set to 1 for each placein the feature vector.
# The resulting feature matrix should be of dimension (number of reviews, number of words).
# Returns:
# a matrix of size (number of reviews * number of words)

    number_of_words = len(hm)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # refer to https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    # testCount = 0
    index = 0
    while(index < 1):
        for word in pos_list[index].split(" "):
            if word in hm:
                feature_matrix[index][hm[word]] = 1
                # testCount = testCount + 1
        index = index + 1
    old_index = index + 0
    index = 0
    # print("first test")
    # print(testCount)
    while(index < 1):
        for word in neg_list[index].split(" "):
            if word in hm:
                feature_matrix[old_index][hm[word]] = 1
                print(word)
                # testCount = testCount + 1
        index = index + 1
        old_index = index + 1
        # print("second test")
        # print(testCount)
    #gprint(feature_matrix[1][58])
    return feature_matrix


generate_feature_matrix(hm)



