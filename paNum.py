import streamlit as st
from sklearn.linear_model import LogisticRegression
import numpy as np

st.title("AI Prediction App")

# Inputs
age = st.slider("Age", 18, 60)
salary = st.slider("Salary", 1000, 10000)

# Dummy model (for demo)
model = LogisticRegression()
X = np.array([[25, 3000], [40, 8000]])
y = np.array([0, 1])
model.fit(X, y)

if st.button("Predict"):
    prediction = model.predict([[age, salary]])
    
    if prediction[0] == 1:
        st.success("Will Buy ✅")
    else:
        st.error("Will Not Buy ❌")