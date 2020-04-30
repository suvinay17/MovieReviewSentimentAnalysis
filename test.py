from mt import *
import numpy as np


pos_train = extract_data_no_caps("data/train/pos")

#neg_train = mt.extract_data("data/train/neg")
#pos_test = mt.extract_data("data/test/pos")
#neg_test = mt.extract_data("data/test/")

for review in pos_train:
    print(review)

print("-----------" )
print("caps")
print("-----------" )

pos_train = extract_data("data/train/pos")

for review in pos_train:
    print(review)

"""
for i in range(len(pos_train)):
    for j in range(len(pos_test)):
        if pos_train[i] == pos_test[i]:
            print(True)
            

for review in pos_train:
    print(str(i) + "-------")
    print(review)
    print("\n")
    i+=1


out = mt.extract_dictionary(pos_train, {})
hm = mt.extract_dictionary(neg_train, out[0], out[1])[0]

fm = mt.bag_of_words_feature_matrix(hm, pos_train, neg_train)
"""
