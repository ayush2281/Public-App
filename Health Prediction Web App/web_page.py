# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:42:53 2024

@author: lenovo
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved models
diabetes_model = pickle.load(open('C:/Users/lenovo/OneDrive/Desktop/deploy ML-Model(API)/both/trained_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/lenovo/OneDrive/Desktop/deploy ML-Model(API)/both/heart_disease_model.sav', 'rb'))


# Function for diabetes prediction
def diabetic_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = diabetes_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return "Person is not diabetic"
    else:
        return "Person is diabetic"

# Function for heart disease prediction
def heart_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = heart_disease_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return "The person does not have heart disease"
    else:
        return "The person has heart disease"

def main():
    # Title of the web app
    st.title('Health Prediction Web App')

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox("Choose a prediction model", ("Diabetes Prediction", "Heart Disease Prediction"))

    if option == "Diabetes Prediction":
        st.subheader("Diabetes Prediction")
        
        # Getting the input data from the user
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Number of Glucose')
        BloodPressure = st.text_input('Number of BloodPressure')
        SkinThickness = st.text_input('Number of SkinThickness')
        Insulin = st.text_input('Number of Insulin')
        BMI = st.text_input('Number of BMI')
        DiabetesPedigreeFunction = st.text_input('Value of DiabetesPedigreeFunction')
        Age = st.text_input('Number of Age')
        
        # Code for prediction
        diagnosis = ''
        
        # Creating a button for prediction
        if st.button('Diabetes Test Result'):
            input_data = [int(Pregnancies), int(Glucose), int(BloodPressure), int(SkinThickness), int(Insulin), float(BMI), float(DiabetesPedigreeFunction), int(Age)]
            diagnosis = diabetic_prediction(input_data)
            st.success(diagnosis)

    elif option == "Heart Disease Prediction":
        st.subheader("Heart Disease Prediction")
        
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
