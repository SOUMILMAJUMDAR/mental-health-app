import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model/mental_health_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

st.title("🧠 Mental Health Risk Predictor (Tech Industry)")
st.write("Answer the following questions to check if you may be at risk.")

# Questionnaire
age = st.number_input("Age", min_value=18, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])
family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No", "Unknown"])
work_interfere = st.selectbox("If you have a mental health issue, does it interfere with work?",
                              ["Never", "Rarely", "Sometimes", "Often", "Unknown"])

# Prepare input
input_data = pd.DataFrame([[age, gender, family_history, work_interfere]],
                          columns=["Age", "Gender", "family_history", "work_interfere"])

# Encode categorical fields
for col in input_data.columns:
    if col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

# Predict
if st.button("Check Risk"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠ You might be at risk for mental health issues. Consider seeking support.")
    else:
        st.success("✅ You are likely not at immediate risk.")
