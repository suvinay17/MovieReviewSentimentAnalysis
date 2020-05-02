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
    getSplitData() when passing in
    the training and testing data

- To remove or keep  stop words, in main.py,
    Comment out which the unwanted if-statement on line 102

- To remove proper nouns from the review,
    use getDataNoCaps() instead of getData() 
    in main.py. Simply comment out the four lines of code
    which use extract_data
    
- To run cross- validation for linear kernel

- To run quadratic kernel, on line 56, change the setting
    to quadratic instead of linear

- To change the number of reviews the experiment
is run on changes REVIEWS in mt.py
