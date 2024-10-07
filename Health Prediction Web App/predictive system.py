# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:39:23 2024

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 2024

@author: lenovo
"""

import numpy as np
import pickle

# Loading the saved models
diabetes_model = pickle.load(open('C:/Users/lenovo/OneDrive/Desktop/deploy ML-Model(API)/both/trained_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/lenovo/OneDrive/Desktop/deploy ML-Model(API)/both/heart_disease_model.sav', 'rb'))


# Function for heart disease prediction
def heart_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = heart_disease_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return "The person does not have heart disease"
    else:
        return "The person has heart disease"

# Function for diabetes prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = diabetes_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"

# Example input data for testing
heart_input_data = (52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3)
diabetes_input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)

# Making predictions
heart_result = heart_prediction(heart_input_data)
diabetes_result = diabetes_prediction(diabetes_input_data)

print(heart_result)
print(diabetes_result)
