import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------- TRAIN THE MODEL AUTOMATICALLY (if not already saved) --------
MODEL_FILE = "model.pkl"

if not os.path.exists(MODEL_FILE):
    data = {
        "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Scores": [20, 30, 35, 40, 50, 60, 65, 78, 85]
    }
    df = pd.DataFrame(data)
    X = df[["Hours_Studied"]]
    y = df["Scores"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)

# -------- LOAD MODEL --------
model = joblib.load(MODEL_FILE)

# -------- STREAMLIT UI --------
st.title("📊 Student Score Prediction App")
st.write("Predict your exam score based on study hours.")

hours = st.slider("Select Hours of Study", min_value=1.0, max_value=10.0, step=0.5)

if st.button("Predict Score"):
    predicted_score = model.predict([[hours]])[0]
    st.success(f"🎯 Predicted Score: **{predicted_score:.2f}**")

    # Visualization
    st.subheader("Score Trend Visualization")
    hours_range = np.arange(1, 10, 0.5).reshape(-1, 1)
    scores_pred = model.predict(hours_range)

    fig, ax = plt.subplots()
    sns.lineplot(x=hours_range.flatten(), y=scores_pred, ax=ax, color="blue", label="Predicted Trend")
    ax.scatter(hours, predicted_score, color="red", s=100, label="Your Prediction")
    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Predicted Score")
    ax.legend()
    st.pyplot(fig)
