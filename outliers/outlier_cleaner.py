#!/usr/bin/python

import numpy

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here
    
    for i, item in enumerate(net_worths):
        error = predictions[i] - item
        t = (int(ages[i]), float(net_worths[i]), numpy.abs(float(error)))
        cleaned_data.append(t)
    
    cleaned_data = sorted(cleaned_data, key = lambda cleaned_data: cleaned_data[2])
    cleaned_data = cleaned_data[:int(round(len(cleaned_data)*.9))]

    return cleaned_data

