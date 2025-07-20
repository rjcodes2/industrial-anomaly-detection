import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Predict Machine Fault", layout="centered")

st.title(" Predict Machine Fault")

# Load the full pipeline (preprocessor + classifier)
pipeline_path = os.path.join(os.path.dirname(__file__), "..", "models", "rf_model.pkl")
pipeline = joblib.load(pipeline_path)

st.markdown("### Fill the details to predict if the machine is faulty.")

with st.form("prediction_form"):
    temperature = st.number_input("Temperature", min_value=0.0, format="%.2f")
    pressure = st.number_input("Pressure", min_value=0.0, format="%.2f")
    vibration = st.number_input("Vibration", min_value=0.0, format="%.2f")
    humidity = st.number_input("Humidity", min_value=0.0, format="%.2f")
    
    equipment = st.selectbox("Equipment Type", ["Pump", "Compressor", "Turbine"])
    location = st.selectbox("Location", ["New York", "San Francisco", "Houston", "Chicago"])

    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            # Ensure all values are filled
            if not all([temperature, pressure, vibration, humidity]):
                st.warning(" Please fill in all fields before submitting.")
            else:
                input_df = pd.DataFrame([{
                    "temperature": temperature,
                    "pressure": pressure,
                    "vibration": vibration,
                    "humidity": humidity,
                    "equipment": equipment,
                    "location": location
                }])

                # Predict directly using the full pipeline
                prediction = pipeline.predict(input_df)[0]
                result = "Faulty" if prediction == 1 else "Not Faulty"

                st.success(f" Prediction: **{result}**")

        except Exception as e:
            st.error(f" Prediction failed: {str(e)}")



