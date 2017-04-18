import sys
import pickle
import math
import numpy
from time import time
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.cluster.k_means_ import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFdr, chi2
from sklearn.cross_validation import StratifiedShuffleSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### You will need to use more features

FEATURES_LIST = ['poi', 'salary',  'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'msg_to_poi_ratio', 'msg_from_poi_ratio']

### Task 2: Remove outliers
def remove_outliers():
    print("Removing outliers.")
    with open("final_project_dataset_unix.pkl", "rb") as data_file:
        data_dict = pickle.load(data_file)
    
    data_dict.pop("TOTAL")
    data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

    outliers = []
    for i, key in enumerate(data_dict.keys()):
        if (data_dict[key]['total_payments'] == "NaN" or data_dict[key]['total_payments'] == 0) and (data_dict[key]['total_stock_value'] == "NaN" or data_dict[key]['total_stock_value'] == 0):
            outliers.append(key) #remove persons, who didn't get any money
    
        #if (data_dict[key]['to_messages'] == "NaN" or data_dict[key]['to_messages'] == 0) and (data_dict[key]['from_messages'] == "NaN" or data_dict[key]['from_messages'] == 0):
        #outliers.append(key) #remove persons, who didn't use email. Leave commented for now.

    for key in outliers:
        data_dict.pop(key)
    
    return data_dict


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

def create_new_features(my_dataset):
    print("Adding ratios of from and to POIs messages to sent and received messages numbers.")
    
    for key in my_dataset.keys():
        msg_to_poi_ratio = \
            float(my_dataset[key]['from_this_person_to_poi']) / \
            float(my_dataset[key]['from_messages'])
        msg_from_poi_ratio = \
            float(my_dataset[key]['from_poi_to_this_person']) / \
            float(my_dataset[key]['to_messages'])
    
        if math.isnan(msg_to_poi_ratio):
            my_dataset[key]['msg_to_poi_ratio'] = "NaN"
        else:
            my_dataset[key]['msg_to_poi_ratio'] = msg_to_poi_ratio
        
        if math.isnan(msg_from_poi_ratio):
            my_dataset[key]['msg_from_poi_ratio'] = "NaN"
        else:
            my_dataset[key]['msg_from_poi_ratio'] = msg_from_poi_ratio

    return my_dataset


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

def prepare_pipeline():
    
    print("Preparing pipeline.")
    pipe = Pipeline([
        ('minmax', MinMaxScaler()),
        ('reduce_dim', PCA()),
        ('classify', SVC())
    ])
   
    return pipe
   
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

def tune(pipe):    
    print("Tuning pipe, using grid.")
    
    N_FEATURES_OPTIONS = [12, 10, 8, 6, 4, 2]
    C_OPTIONS = [.001, 0.01, .1 , .2, .3]
    KERNEL = ['linear', 'rbf']
    
    param_grid = [
        {
            'minmax': [MinMaxScaler()],
            'reduce_dim': [PCA(iterated_power=7)],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS,
            'classify__kernel': KERNEL
        },
        {
            'minmax': [MinMaxScaler()],
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS,
            'classify__kernel': KERNEL
        },
    ]
    
    return GridSearchCV(pipe, cv=3, param_grid=param_grid, n_jobs=-1)
    
    
#    grid.fit(features, labels)
    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

def main():
    print("Doing some f###ng black magic.")
    
    my_dataset = remove_outliers()
    my_dataset = create_new_features(my_dataset)

### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, FEATURES_LIST, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    pipe = prepare_pipeline()
    
    clf = tune(pipe)
    
    print("Testing classifier.")
    
    clf.fit(features, labels)
    print(clf.best_estimator_)
    print(clf.best_params_)
    
    testclf = clf.best_estimator_
    
    test_classifier(testclf, my_dataset, FEATURES_LIST, folds = 1000)

    #dump_classifier_and_data(clf, my_dataset, features_list)
    
if __name__ == '__main__':
    main()