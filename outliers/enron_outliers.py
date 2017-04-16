#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import linear_model
from sklearn.model_selection import train_test_split

### read in data dictionary, convert to np array
data_dict = pickle.load( open("../final_project/final_project_dataset_unix_1.pkl", "rb") )

### your code below

for key in data_dict.keys():
    if data_dict[key]["salary"] == "NaN":
        data_dict[key]["salary"] = 0
    if data_dict[key]["bonus"] == "NaN":
        data_dict[key]["bonus"] = 0
    if int(data_dict[key]["salary"]) > 1000000 and int(data_dict[key]["bonus"]) > 5000000:
        print(key) 

features = ["poi", "salary", "bonus"]
data = featureFormat(data_dict, features)
poi, features = targetFeatureSplit(data)

features = np.reshape( np.array(features), (len(features), 2))
salaries, bonuses = np.split(features, 2, 1)
#features = np.array(features)

salaries_train, salaries_test, bonuses_train, bonuses_test = train_test_split(salaries, bonuses, test_size=0.1, random_state=42)

reg = linear_model.LinearRegression()
reg.fit(salaries, bonuses)
pred = reg.predict(salaries)

# The coefficients
print("Coefficients: ", reg.coef_, reg.intercept_)
# The mean squared error
#print("Mean squared error: %.2f" % np.mean(( pred - bonuses_test) ** 2))
#print('Variance score: %.2f' % reg.score(salaries_test, salaries_test))

try:
    plt.plot(salaries, reg.predict(salaries))
except NameError:
    pass
plt.scatter(salaries, bonuses)
plt.xlabel("salaries")
plt.ylabel("bonuses")
plt.show()

cleaned_data = []

for i, item in enumerate(bonuses):
    error = pred[i] - item
    t = (int(salaries[i]), int(bonuses[i]), np.abs(float(error)))
    cleaned_data.append(t)

print (cleaned_data, len(cleaned_data))

cleaned_data = sorted(cleaned_data, key = lambda cleaned_data: cleaned_data[2])
cleaned_data = cleaned_data[:int(94)]
print (cleaned_data, len(cleaned_data))

salaries, bonuses, errors = zip(*cleaned_data)
salaries = np.reshape( np.array(salaries), (len(salaries), 1))
bonuses = np.reshape( np.array(bonuses), (len(bonuses), 1))

reg.fit(salaries, bonuses)
pred = reg.predict(salaries_test)

# The coefficients
print("Coefficients: ", reg.coef_, reg.intercept_)
# The mean squared error
#print("Mean squared error: %.2f" % np.mean(( pred - bonuses_test) ** 2))
#print('Variance score: %.2f' % reg.score(salaries_test, salaries_test))

try:
    plt.plot(salaries, reg.predict(salaries))
except NameError:
    print ("Somewhat error in cleaning")

plt.scatter(salaries, bonuses)
plt.xlabel("salaries")
plt.ylabel("bonuses")
plt.show()
