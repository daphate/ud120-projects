#!/usr/bin/python

import pickle
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn import tree
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb"))

tmp = []
for item in word_data:
    item = item.replace("sshacklensf", "")
    item = item.replace("sshacklmsncom", "")
    tmp.append(item)
    
word_data = tmp

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier

features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

### your code goes here

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)

print(acc)

"""
print(clf.feature_importances_.argmax(axis=0), max(clf.feature_importances_))

for i, item in enumerate(clf.feature_importances_):
    if item > 0.2:
        print(i, item, vectorizer.get_feature_names()[i])
"""