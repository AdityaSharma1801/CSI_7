# utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_feature_importance(model, feature_names):
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    st.subheader("Feature Importance (Top)")
    st.bar_chart(coef_df.set_index('Feature'))

def show_user_input(input_df):
    st.subheader("User Input Parameters")
    st.write(input_df)

def display_prediction(prediction, proba):
    st.subheader("Prediction")
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.success(f"Result: {result}")
    st.subheader("Prediction Probability")
    st.write(f"Diabetic Probability: {proba[0][1]:.2f}")
