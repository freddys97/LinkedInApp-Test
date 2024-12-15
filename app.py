import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained logistic regression model
import os
model_path = os.path.join("/Users/freddysiqueiros/Documents/LinkedInApp/", "logistic_regression_model.pkl")
import os

# Explicitly set the absolute path for the model
model_path = os.path.join(os.path.dirname(__file__), "logistic_regression_model.pkl")
model = joblib.load(model_path)



# Custom labels for inputs
income_options = {
    1: "Less than $10,000",
    2: "$10,000 to $20,000",
    3: "$20,000 to $30,000",
    4: "$30,000 to $40,000",
    5: "$40,000 to $50,000",
    6: "$50,000 to $75,000",
    7: "$75,000 to $100,000",
    8: "$100,000 to $150,000",
    9: "More than $150,000"
}

education_options = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year degree",
    6: "Four-year degree",
    7: "Some postgraduate or professional schooling",
    8: "Postgraduate degree"
}

# App Title and Description
st.title("🔮 LinkedIn User Predictor")
st.write("### Enter Demographic Information")
st.write("This app predicts whether an individual uses LinkedIn based on demographic inputs.")

# Organize inputs using columns
col1, col2 = st.columns(2)

with col1:
    income = st.selectbox("Income Level (1-9)", list(income_options.keys()), format_func=lambda x: income_options[x])
    education = st.selectbox("Education Level (1-8)", list(education_options.keys()), format_func=lambda x: education_options[x])
    age = st.slider("Age (18-98)", 18, 98, value=25)

with col2:
    parent = st.radio("Parent", ["No", "Yes"])
    married = st.radio("Married", ["No", "Yes"])
    female = st.radio("Female", ["No", "Yes"])

# Convert inputs to binary/numeric for the model
parent = 1 if parent == "Yes" else 0
married = 1 if married == "Yes" else 0
female = 1 if female == "Yes" else 0

# Predict Button
if st.button("Predict LinkedIn Usage"):
    # Prepare input data
    user_input = np.array([[income, education, parent, married, female, age]])

    # Make predictions
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)[0][1]

    # Display Results
    st.subheader("🎯 Prediction Results:")
    st.write("**Predicted Category:** LinkedIn User" if prediction[0] == 1 else "**Predicted Category:** Not a LinkedIn User")
    st.write(f"**Probability of being a LinkedIn User:** {probability:.2%}")

    # Probability Visualization
    st.write("### 📊 Probability Visualization")
    fig, ax = plt.subplots()
    ax.bar(["LinkedIn User"], [probability], color="green")
    ax.bar(["Not LinkedIn User"], [1 - probability], color="red")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
