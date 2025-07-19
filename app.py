# app.py
from utils import plot_feature_importance, show_user_input, display_prediction
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Diabetes Prediction App")

st.sidebar.header("Input Patient Details")

def user_input():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 1)
    Glucose = st.sidebar.slider('Glucose', 40, 200, 100)
    BloodPressure = st.sidebar.slider('Blood Pressure', 30, 130, 70)
    SkinThickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    Insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    BMI = st.sidebar.slider('BMI', 10.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    Age = st.sidebar.slider('Age', 10, 100, 33)

    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Show user input
show_user_input(input_df)


# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)

display_prediction(prediction, prediction_proba)

# Feature importance
plot_feature_importance(model, input_df.columns)
