import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="Wine Cultivar Predictor")

st.title("🍷 Wine Cultivar Origin Prediction System")

st.write("Enter the chemical properties of the wine:")

# User inputs
alcohol = st.number_input("Alcohol", 10.0, 15.0, step=0.1)
malic_acid = st.number_input("Malic Acid", 0.5, 6.0, step=0.1)
alcalinity = st.number_input("Alcalinity of Ash", 10.0, 30.0, step=0.5)
magnesium = st.number_input("Magnesium", 70, 170, step=1)
flavanoids = st.number_input("Flavanoids", 0.1, 5.0, step=0.1)
color_intensity = st.number_input("Color Intensity", 1.0, 13.0, step=0.1)

if st.button("Predict Cultivar"):
    input_data = np.array([[alcohol, malic_acid, alcalinity,
                             magnesium, flavanoids, color_intensity]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Wine Cultivar: **Cultivar {prediction + 1}**")
