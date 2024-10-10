# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:21:51 2024

@author: lenovo
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('C:\\Users\\lenovo\\OneDrive\\Desktop\\deploy ML-Model(API)\\heart disease\\heart_disease_model.sav', 'rb'))

# creating the function for the prediction

def heart_prediction(input_data):
    # Change the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # Reshape the numpy array as we are predicting only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make a prediction
    prediction = loaded_model.predict(input_data_reshaped)
    
    # Interpret the prediction
    if prediction[0] == 0:
        return "The person does not have heart disease"
    else:
        return "The person has heart disease"

        
        
def main():
    # Giving a title 
    st.title('Heart Prediction Web APP')
    
    # Getting the input data from the user      
    age = st.text_input("Enter the Age")
    sex = st.text_input("Enter the Sex")
    cp = st.text_input('Enter the cp')
    trestbps = st.text_input("Enter the trestbps")
    chol = st.text_input("Enter the Chol")
    fbs = st.text_input("Enter the FBS")
    restecg = st.text_input("Enter the restecg")
    thalach = st.text_input("Enter the thalach")
    exang = st.text_input("Enter the exang")
    oldpeak = st.text_input("Enter the oldpeak")
    slope = st.text_input("Enter the Slope")
    ca = st.text_input("Enter the ca")
    thal = st.text_input("Enter the thal")
    
    # Code for prediction
    target = ''
    
    # Creating a button for prediction
    if st.button('Heart Test Result'):
        input_data = [int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]
        target = heart_prediction(input_data)
    
    st.success(target)

if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
