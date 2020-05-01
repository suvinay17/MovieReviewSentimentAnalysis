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
#Import helper methods
from mt import *
#This code is for reading the multiple files in one folder into a list
# read neg files in train folder

# Comment out which kind of data extraction we want
# First one does not remove names

pos_train= extract_data("data/train/pos")
neg_train = extract_data("data/train/neg")
pos_test = extract_data("data/test/pos")
neg_test = extract_data("data/test/neg")

# This one removes names of people
"""
pos_train= extract_data_no_caps("data/train/pos")
neg_train = extract_data_no_caps("data/train/neg")
pos_test = extract_data_no_caps("data/test/pos")
neg_test = extract_data_no_caps("data/test/neg")
"""

# Extracts dictionary
hm = extract_dictionary(neg_train, {})
hm = extract_dictionary(pos_train, hm[0], hm[1])[0]

X_train, Y_train, X_test, Y_test, dictionary_binary =\
get_split_binary_data(hm, pos_train, neg_train, pos_test, neg_test)

column_sums = X_train.sum(axis=1)
total = 0
for col in column_sums:
    total += col
print(total / X_train.shape[0])

#clf = SVC(C=1.0, kernel='linear')
print("cv Performance")
#print(cv_performance(clf, X_train, Y_train))
#print("end")
#C = []
#for x in range(-3, 4):
   # C.append(10 ** x)
#print("First print")
#print(select_param_linear(X_train, Y_train, C_range=C))

metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]

#for metric in metrics:
   # c, score = select_param_linear(X_train, Y_train, C_range=C, metric=metric)
   # print(metric + " : " + str(c) + " , " + str(score))

#for metric in metrics:
#    C = []
#    for x in range(-3, 4):
 #       C.append(10 ** x)
#
 #   c, score = select_param_linear(X_test, Y_test, C_range=C, metric=metric)
  #  print(metric + " : " + str(c) + " , " + str(score))

# Accuracy, because takes into account False Positives and False Negatives


svc = SVC(C=0.01, kernel='linear', degree=1, class_weight='balanced')
for metric in metrics:
    svc.fit(X_train, Y_train)
    if metric != "auroc":
        Y_predicted = svc.predict(X_test)
    else:
        Y_predicted = svc.decision_function(X_test)
    score = performance(Y_test, Y_predicted, metric)
    print(metric + " : " + str(score))
