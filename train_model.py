import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Scores": [20, 30, 35, 40, 50, 60, 65, 78, 85]
}
df = pd.DataFrame(data)

# Train model
X = df[["Hours_Studied"]]
y = df["Scores"]
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
