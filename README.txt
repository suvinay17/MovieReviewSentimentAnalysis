Movie Review Sentiment Analysis
--------------------------------------------------------------
Group Members:
Amber, Basel, Suvinay, Yanish
--------------------------------------------------------------

This is a guide on how to run our Sentiment Analysis program

Brief overview:
Our program is run on the IMDB movie review data set that 
is found in./data. The directory contains reviews for
training purposes as well as testing purposes.

------------------------------------------------------------
Running the program:
To run the program, execute python3 main.py

------------------------------------------------------------
Output:
Performance metric with different extraction features

------------------------------------------------------------
Changing Parameters:
- To change the extraction feature, in mt.py,
    choose between 
    bowFm() (bag of words feature matrix),
    tdIdfFm() (tf-idf feature_matrix) or
    normalizedWfFm() (normalized word frequency feature matrix) in
    getSplitData() on lines 260 and 261 in in mt.py when passing in
    the training and testing data

- To remove or keep  stop words, in main.py,
    Comment out a section if-statement on line 116 in mt.py
    Change the line in the method getDictNoSw() to say this: 
    if word not in word_dict : #and word not in stop_words:

- To remove proper nouns from the review,
    use getDataNoCaps() instead of getData() 
    in main.py. Simply comment out the four lines of code
    which use getData from lines 7 to 10 or lines 14 to 17
    

- To run quadratic kernel, on line 56, change the setting
    to "poly" instead of linear, change Degree to 2, and add(and an r value coef0=1)

- To change the number of reviews the experiment
is run on changes REVIEWS in mt.py on line 17
REVIEWS represent the number of positive and negative reviews to be chosen.
REVIEWS = 2500 means, running the experiment on 5000 movie reviews.
