#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
from sklearn.decomposition.tests.test_nmf import random_state
from sklearn.metrics.classification import precision_score
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
data_dict = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
sort_keys = '../tools/python2_lesson14_keys.pkl'
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### it's all yours from here forward!  

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42) 

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

numpos = 0
pred = clf.predict(features_test)
for i, item in enumerate(labels_test):
    print(i, item, pred[i])
    if item == 1 and item == pred[i]:
        numpos = numpos + 1

print("Number of true positives: ", numpos)
print("Precision: ", precision_score(labels_test, pred))
print("Recall: ", recall_score(labels_test, pred))

#print(len(labels_test))
#print(clf.score(features_test, labels_test))

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print("Number of true positives: ", numpos)
print("Precision: ", precision_score(true_labels, predictions))
print("Recall: ", recall_score(true_labels, predictions))

