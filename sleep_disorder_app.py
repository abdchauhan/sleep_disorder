import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
with open("rf_model.pkl", "rb") as f:
    model, label_encoders, feature_columns = pickle.load(f)

st.set_page_config(page_title="Sleep Disorder Predictor", layout="centered")
st.title("😴 SleepSense: Forecasting Sleep Disorders with Machine Learning")
st.write("Enter your health and lifestyle information to predict potential sleep disorders.")

# Input form (excluding Person ID)
input_data = {}
for col in feature_columns:
    if col == "Person ID":
        continue  # Just in case

    if col in label_encoders:
        options = label_encoders[col].classes_
        selected = st.selectbox(f"{col}:", options)
        input_data[col] = label_encoders[col].transform([selected])[0]
    else:
        val = st.number_input(f"{col}:", step=1.0)
        input_data[col] = val

# Make prediction
if st.button("Predict Sleep Disorder"):
    input_df = pd.DataFrame([input_data])
    st.write("🔍 Input Data Sent to Model:", input_df)  # Debug: See input data
    prediction = model.predict(input_df)[0]
    result = label_encoders['Sleep Disorder'].inverse_transform([prediction])[0]
    st.success(f"🛌 Predicted Sleep Disorder: **{result}**")
