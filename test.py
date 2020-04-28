import mt
import numpy as np

pos_train = mt.extract_data("data/train/pos")
neg_train = mt.extract_data("data/train/neg")

i = 0
for review in pos_train:
    print(str(i) + "-------")
    print(review)
    print("\n")
    i+=1


out = mt.extract_dictionary(pos_train, {})
hm = mt.extract_dictionary(neg_train, out[0], out[1])[0]

fm = mt.bag_of_words_feature_matrix(hm, pos_train, neg_train)

