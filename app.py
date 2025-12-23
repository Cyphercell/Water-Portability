import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load model and scaler
# -------------------------------
with open("water_potability_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data['model']
scaler = data['scaler']

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Water Potability Prediction", layout="centered")

st.title("💧 Water Potability Prediction")
st.write("Enter water quality parameters to check if the water is potable.")

# -------------------------------
# User Inputs
# -------------------------------
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.0, value=200.0)
solids = st.number_input("Solids", min_value=0.0, value=20000.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.0, value=330.0)
conductivity = st.number_input("Conductivity", min_value=0.0, value=420.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=14.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=66.0)
turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    input_data = np.array([[
        ph, hardness, solids, chloramines, sulfate,
        conductivity, organic_carbon, trihalomethanes, turbidity
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.success(f"✅ Water is POTABLE")
    else:
        st.error(f"❌ Water is NOT POTABLE")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Model: Tuned XGBoost | Preprocessing: StandardScaler")
