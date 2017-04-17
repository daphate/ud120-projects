#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### You will need to use more features
features_list = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

### Load the dictionary containing the dataset
with open("final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### If we look at enron61702insiderpay.pdf, we find "TOTAL" and some "THE TRAVEL AGENCY IN THE PARK"
### which we may assume are not POIs
### also actually we are not interested in people, who didn't obtain any money 
### (and don't have messages to or from our POIs) 

print("Before removing outliers, the data_dict contains:", len(data_dict), "records")
print("")

data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

outliers = []
for i, key in enumerate(data_dict.keys()):
    if (data_dict[key]['total_payments'] == "NaN" or data_dict[key]['total_payments'] == 0) and (data_dict[key]['total_stock_value'] == "NaN" or data_dict[key]['total_stock_value'] == 0):
        outliers.append(key) #remove persons, who didn't get any money
        print (key, "didn't get any money.")
    if (data_dict[key]['to_messages'] == "NaN" or data_dict[key]['to_messages'] == 0) and (data_dict[key]['from_messages'] == "NaN" or data_dict[key]['from_messages'] == 0):
        #outliers.append(key) #remove persons, who didn't use email. Leave commented for now.
        print (key, "didn't use email.")    

for key in outliers:
    data_dict.pop(key)

print("")
print("After removing outliers, the data_dict contains:", len(data_dict), "records")
print("")

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


#clf = GaussianNB()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#### My code

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)

print(acc)



##### -- My code

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)

