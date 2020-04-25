import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from matplotlib import pyplot as plt
import string
import itertools
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
# read pos files in train folder
pos_files = glob.glob(os.path.join(os.getcwd(), "data/train/pos", "*.txt"))
pos_list = []
for pos_path in pos_files:
    with open(pos_path) as pos_input:
        pos_list.append(pos_input.read())
number_of_reviews = len(pos_list)+len(neg_list)
# read pos files in test folder
posTest_files = glob.glob(os.path.join(os.getcwd(), "data/test/pos", "*.txt"))
pos_test = []
for posTest_path in posTest_files:
    with open(posTest_path) as posTest_input:
        pos_test.append(posTest_input.read())
# read neg files in test folder
negTest_files = glob.glob(os.path.join(os.getcwd(), "data/test/neg", "*.txt"))
neg_test = []
for negTest_path in negTest_files:
    with open(negTest_path) as negTest_input:
        neg_test.append(negTest_input.read())

def extract_dictionary():
    """
    Reads list of distinct words
    mapping from each distinct word to its index .
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found

    """
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
    #print(len(neg_list[0]) + len(pos_list[0]))
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


def bag_of_words_feature_matrix(hm):
# Reads the set of unique words to generate a matrix of {1, 0} feature vectors for each review.
# The resulting feature matrix should be of dimension (number of reviews, number of words).
# Returns:
# a matrix of size (number of reviews * number of words) (for TRAIN data set)
    number_of_reviews = len(pos_list) + len(neg_list)
    number_of_words = len(hm)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # refer to https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    # testCount = 0
    index = 0
    while index < 1 :
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
                # print(word)
                # testCount = testCount + 1
        index = index + 1
        old_index = index + 1
        print("second test")
        # print(testCount)
    # print(feature_matrix[1][58])
    return feature_matrix

def feature_matrix_test(hm):
# Reads the set of unique words to generate a matrix of {1, 0} feature vectors for each review.
# The resulting feature matrix should be of dimension (number of reviews, number of words).
# Returns:
# a matrix of size (number of reviews * number of words) (for TEST data set)
    number_of_reviews = len(pos_test) + len(neg_test)
    number_of_words = len(hm)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # refer to https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    # testCount = 0
    index = 0
    while index < 1 :
        for word in pos_test[index].split(" "):
            if word in hm:
                feature_matrix[index][hm[word]] = 1
                # testCount = testCount + 1
        index = index + 1
    old_index = index + 0
    index = 0
    print("first test")
    # print(testCount)
    while(index < 1):
        for word in neg_test[index].split(" "):
            if word in hm:
                feature_matrix[old_index][hm[word]] = 1
                # print(word)
                # testCount = testCount + 1
        index = index + 1
        old_index = index + 1
        # print("second test")
        # print(testCount)
    # print(feature_matrix[1][58])
    return feature_matrix

def normalized_wf_feature_matrix(hm):
# Reads the set of unique words to generate a matrix of normalized word frequency which is the number
# Of times a word occurs divided by the length of the review
# The resulting feature matrix should be of dimension (number of reviews, number of words).
# Returns:
# a matrix of size (number of reviews * number of words)

    number_of_words = len(hm)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # refer to https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    # testCount = 0
    index = 0
    while index < 1 :
        wordCount = 0
        for word in pos_list[index].split(" "):
            wordCount = wordCount + 1
            if word in hm:
                feature_matrix[index][hm[word]] = (feature_matrix[index][hm[word]] + 1)
                # testCount = testCount + 1
        for i in range(0 , number_of_words):
            feature_matrix[index][i] = float(feature_matrix[index][i] / wordCount)

        index = index + 1
    old_index = index + 0
    index = 0
    # print("first test")
    # print(testCount)
    while(index < 1):
        wordCount = 0
        for word in neg_list[index].split(" "):
            wordCount = wordCount + 1
            if word in hm:
                feature_matrix[old_index][hm[word]] = feature_matrix[index][hm[word]] + 1
                # print(word)
                # testCount = testCount + 1
        for i in range (0 , number_of_words):
            feature_matrix[old_index][i] = float(feature_matrix[old_index][i] / wordCount)
        index = index + 1
        old_index = index + 1
        # print("second test")
        # print(testCount)
    #print(feature_matrix[1][58])
    print(wordCount)
    return feature_matrix


def split_binary_data():
    """
    Reads in the data and returns it using
    extract_dictionary and bag_of_words_feature_matrix split into training and test sets.
    Also returns the dictionary used to create the feature matrices.
    """
    Y_train = []
    Y_test = []
    for n in range(len(neg_list)):
        Y_train.append(-1)
    for p in range(len(pos_list)):
        Y_train.append(1)
    for i in range(len(neg_test)):
        Y_test.append(-1)
    for j in range(len(pos_list)):
        Y_test.append(1)
    X_train = bag_of_words_feature_matrix(hm)
    X_test = feature_matrix_test(hm)
    return (X_train, Y_train, X_test, Y_test)


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """

    # StratifiedKFold from sklearn.model_selection
    skf = StratifiedKFold(n_splits=k)
    # Put the performance of the model on each fold in the scores array
    scores = []

    for train_index, test_index in skf.split(X, y):
        clf.fit(X[train_index], y[train_index])

        if metric != "auroc":
            Y_predicted = clf.predict(X[test_index])
        else:
            Y_predicted = clf.decision_function(X[test_index])

        scores.append(performance(y[test_index], Y_predicted, metric))

    # And return the average performance across all fold splits.
    return np.array(scores).mean()



def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    #using cv_performance function here to evaluate the performance of each SVM
    max_score = -1
    best_c = -1

    for c in C_range:
        score = cv_performance(select_classifier(c=c), X, y, metric=metric)
        if score > max_score:
            max_score = score
            best_c = c

    return best_c, max_score


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
            X: (n,d) array of feature vectors, where n is the number of examples
               and d is the number of features
            y: (n,) array of binary labels {1,-1}
            k: an int specifying the number of folds (default=5)
            metric: string specifying the performance metric (default='accuracy'
                     other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                     and 'specificity')
            parameter_values: a (num_param, 2)-sized array containing the
                parameter values to search over. The first column should
                represent the values for C, and the second column should
                represent the values for r. Each row of this array thus
                represents a pair of parameters to be tried together.
        Returns:
            The parameter value(s) for a quadratic-kernel SVM that maximize
            the average 5-fold CV performance
    """
    # similar to select_param_linear, except the type of SVM model is different...
    best_c = -1
    best_r = -1
    max_score = -1

    for x in param_range:
        score = cv_performance(select_classifier(c=x[0], r=x[1], degree=2), X, y, metric=metric)
        # print("c=" + str(x[0]) + " , r=" + str(x[1]) + " : score=" + str(score))
        if score > max_score:
            max_score = score
            best_c = x[0]
            best_r = x[1]

    return best_c, best_r, max_score


def select_svc_linear(X, y, k=5, metric="accuracy", C_range = [], class_weight='balanced'):
    max_score = -1
    best_c = -1

    for c in C_range:
        score = cv_performance(LinearSVC(C=c, class_weight=class_weight), X, y, metric=metric, k=k)
        if score > max_score:
            max_score = score
            best_c = c

    return best_c, max_score


def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an numpy float
    """
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.

    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)

    if metric == "f1-score":
        return metrics.f1_score(y_true, y_pred)

    if metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)

    if metric == "precision":
        return metrics.precision_score(y_true, y_pred)

    if metric == "sensitivity":
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1, -1])
        return confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])

    if metric == "specificity":
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1, -1])
        return confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])

    return -1

bag_of_words_feature_matrix(hm)
feature_matrix_test(hm)
split_binary_data()
normalized_wf_feature_matrix(hm)



