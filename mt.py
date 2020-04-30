# File that contains all methods to
# run sentiment analysis on our data

# Import necessary libraries
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
import math

REVIEWS = 50

def extract_data(folder_path):
    """
    This method extracts data from a folder, reading each file,
    preprocessing it by getting rid of nonewords and appending
    it to a list.
    Returns the list
    """
    # Read in files 
    files = glob.glob(os.path.join(os.getcwd(), folder_path, "*.txt"))

    reviews = []
    for path in files:
      with open(path) as text:
          reviews.append(text.read())

    # Clean data
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    reviews = reviews[:REVIEWS]

    return reviews

def extract_dictionary(reviews, word_dict, ind=0):
    """
    Reads list of distinct words
    mapping from each distinct word to its index .
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found

    """
    for i in range(len(reviews)):
        splits = reviews[i].split()
        for word in splits:
            if word not in word_dict:
                word_dict[word] = ind
                ind += 1

    return (word_dict, ind)

def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
# Refer to https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# Return a linear svm classifier based on the given
# penalty function and regularization parameter c.
    if degree == 1:
        return SVC(C=c, kernel='linear', degree=degree, class_weight=class_weight)
    else:
        return SVC(C=c, kernel='poly', degree=degree, class_weight=class_weight, coef0=r)

def bag_of_words_feature_matrix(hm, pos_list, neg_list):
# Reads the set of unique words to generate a matrix of {1, 0} feature vectors for each review.
# The resulting feature matrix should be of dimension (number of reviews, number of words).
# Returns:
# a matrix of size (number of reviews * number of words) (for TRAIN data set)

    feature_matrix = np.zeros((len(pos_list) + len(neg_list), len(hm)))
    # refer to https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html

    for i in range(len(pos_list)):
        for word in pos_list[i].split(" "):
            if word in hm:
                feature_matrix[i][hm[word]] = 1

    index = len(pos_list) 
    for i in range(len(neg_list)):
        for word in neg_list[i].split(" "):
            if word in hm:
                feature_matrix[index][hm[word]] = 1

    return feature_matrix

def normalized_wf_feature_matrix(hm, pos_list, neg_list):
# Reads the set of unique words to generate a matrix of normalized word frequency which is the number
# Of times a word occurs divided by the length of the review
# The resulting feature matrix should be of dimension (number of reviews, number of words).
# Returns:
# a matrix of size (number of reviews * number of words)

    feature_matrix = np.zeros((len(pos_list) + len(neg_list), len(hm)))
    # refer to https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    # testCount = 0
    index = 0
    for i in range(len(pos_list)):
        wordCount = 0
        for word in pos_list[i].split(" "):
            wordCount += 1
            if word in hm:
                feature_matrix[i][hm[word]] += 1

        for j in range(len(pos_list[i].split())):
            feature_matrix[i][j] /= wordCount

        index = i

    for i in range(len(neg_list)):
        wordCount = 0
        for word in neg_list[i].split(" "):
            wordCount += 1
            if word in hm:
                feature_matrix[index][hm[word]] += 1

        for j in range(len(neg_list[i].split())):
            feature_matrix[index][j] /= wordCount
        index +=1 

    return feature_matrix

def get_split_binary_data(hm, pos_list, neg_list, pos_test, neg_test):
    """
    Reads in the data and returns it using
    extract_dictionary and bag_of_words_feature_matrix split into training and test sets.
    Also returns the dictionary used to create the feature matrices.
    """
    y_train = []
    y_test = []
    for i in range(len(pos_list)):
        y_train.append(1)
    for i in range( len(neg_list)):
        y_train.append(-1)
    for i in range(len(pos_test)):
        y_test.append(1)
    for i in range(len(neg_test)):
        y_test.append(-1)

    X_train = tf_idf_feature_matrix(hm, pos_list, neg_list)
    X_test = tf_idf_feature_matrix(hm, pos_test, neg_test)

    Y_train = np.array(y_train)
    Y_test = np.array(y_test)
    #print(type(Y_test))
    return (X_train, Y_train, X_test, Y_test, hm)

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


def tf_idf_feature_matrix(hm, pos_list, neg_list):
# Reads the set of unique words to generate a matrix of normalized word frequency which is the number
# Of times a word occurs divided by the length of the review
# The resulting feature matrix should be of dimension (number of reviews, number of words).
# Returns:
# a matrix of size (number of reviews * number of words)

    feature_matrix = np.zeros((len(pos_list) + len(neg_list), len(hm)))
    # refer to https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    # testCount = 0
    N = REVIEWS * 2
    tf_dict = np.zeros((len(pos_list) + len(neg_list), len(hm)))
    for i in range(len(pos_list)):
        df_dict = {}
        for word in pos_list[i].split(" "):
            if tf_dict[i][hm[word]] == 0:
                tf_dict[i][hm[word]] = 1
                if word not in df_dict:
                    df_dict[word] = 1
                else:
                    df_dict[word] += 1
            else:
                tf_dict[i][hm[word]] = tf_dict[i][hm[word]] + 1

    for i in range(len(neg_list)):
        df_dict = {}
        for word in neg_list[i].split(" "):
             if tf_dict[i+ len(pos_list)][hm[word]] == 0:
                tf_dict[i+len(pos_list)][hm[word]] = 1
                if word not in df_dict:
                    df_dict[word] = 1
                else:
                    df_dict[word] = df_dict[word] + 1
             else:
                tf_dict[i+len(pos_list)][hm[word]] = tf_dict[i+len(pos_list)][hm[word]] + 1
    visited = []
    for i in range(len(pos_list)):
        for word in pos_list[i].split(" "):
            if word not in visited:
                visited.append(word)
                feature_matrix[i][hm[word]] = tf_dict[i][hm[word]]*math.log2(N/df_dict[word])
    for i in range(len(neg_list)):
        for word in neg_list[i].split(" "):
            if word not in visited:
                visited.append(word)
                feature_matrix[i+len(pos_list)][hm[word]] = tf_dict[i+len(pos_list)][hm[word]]*math.log2(N/df_dict[word])
    return feature_matrix
