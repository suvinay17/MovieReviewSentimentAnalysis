#Import helper methods
from mt import *

# Comment out which kind of data extraction we want
# First one does not proper nouns

pos_train = getData("data/train/pos")
neg_train = getData("data/train/neg")
pos_test = getData("data/test/pos")
neg_test = getData("data/test/neg")

# This one removes proper nouns
"""
pos_train= getDataNoCaps("data/train/pos")
neg_train = getDataNoCaps("data/train/neg")
pos_test = getDataNoCaps("data/test/pos")
neg_test = getDataNoCaps("data/test/neg")
"""

# Comment out unwanted dictionary extraction technique
# Extracts dictionary keeping stop words
hm = getDict(neg_train, {})
hm = getDict(pos_train, hm[0], hm[1])[0]

# Extracts dictionary and does not consider stop words
"""
hm = getDictNoSw(neg_train, {})
hm = getDictNoSw(pos_train, hm[0], hm[1])[0]
"""

X_train, Y_train, X_test, Y_test, dictionary_binary =\
getSplitData(hm, pos_train, neg_train, pos_test, neg_test)

"""
print("cv Performance")
print(cv_performance(clf, X_train, Y_train))
print(select_param_linear(X_train, Y_train, C_range=C))
"""

metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]

"""
for metric in metrics:
    c, score = select_param_linear(X_train, Y_train, C_range=C, metric=metric)
    print(metric + " : " + str(c) + " , " + str(score))

for metric in metrics:
    C = []
    for x in range(-3, 4):
       C.append(10 ** x)

    c, score = select_param_linear(X_test, Y_test, C_range=C, metric=metric)
    print(metric + " : " + str(c) + " , " + str(score))
"""

svc = SVC(C=0.1, kernel='linear', degree=1, class_weight='balanced')

for metric in metrics:
    svc.fit(X_train, Y_train)
    if metric != "auroc":
        Y_predicted = svc.predict(X_test)
    else:
        Y_predicted = svc.decision_function(X_test)
    score = performance(Y_test, Y_predicted, metric)
    print(metric + " : " + str(score))
