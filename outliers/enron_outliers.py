#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import numpy
import matplotlib.pyplot as plt

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset_unix.pkl", "rb") )

### your code below

labels = data_dict.keys()
features = ["poi", "salary", "bonus"]
data = featureFormat(data_dict, features)
poi, features = targetFeatureSplit(data)

salaries = []
bonuses = []

for i, item in enumerate(features):
    salaries.append(item[0])
    bonuses.append(item[1])

#print(label, salaries, bonuses)

salaries = numpy.reshape( numpy.array(salaries), (len(salaries), 1))
bonuses = numpy.reshape( numpy.array(bonuses), (len(bonuses), 1))

from sklearn.cross_validation import train_test_split
salaries_train, salaries_test, bonuses_train, bonuses_test = train_test_split(salaries, bonuses, test_size=0.1, random_state=42)

reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(salaries_train, bonuses_train)
pred = reg.predict(salaries_test)

# The coefficients
print("Coefficients: ", reg.coef_, reg.intercept_)
# The mean squared error
print("Mean squared error: %.2f"
     % numpy.mean(( pred - bonuses_test) ** 2))
print('Variance score: %.2f' % reg.score(salaries_test, salaries_test))

try:
    plt.plot(salaries, reg.predict(salaries))
except NameError:
    pass
plt.scatter(salaries, bonuses)
plt.show()

cleaned_data = []

### your code goes here

for i, item in enumerate(bonuses):
    error = pred[i] - item
    t = (int(salaries[i]), float(bonuses[i]), numpy.abs(float(error)))
    cleaned_data.append(t)

cleaned_data = sorted(cleaned_data, key = lambda cleaned_data: cleaned_data[2])
cleaned_data = cleaned_data[:int(round(len(cleaned_data)*.9))]

salaries, bonuses, errors = zip(*cleaned_data)
salaries = numpy.reshape( numpy.array(salaries), (len(salaries), 1))
bonuses = numpy.reshape( numpy.array(bonuses), (len(bonuses), 1))

try:
    reg.fit(salaries, bonuses)
    plt.plot(salaries, reg.predict(salaries))
except NameError:
    print ("Somewhat error in cleaning")

plt.scatter(ages, net_worths)
plt.xlabel("salaries")
plt.ylabel("bonuses")
plt.show()

for point in features:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()