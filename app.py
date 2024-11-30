import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load the model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('breast_cancer_scaler.pkl')

# Load the breast cancer dataset for feature names
breast_cancer = load_breast_cancer()
feature_names = breast_cancer.feature_names

# Streamlit application
st.title("Breast Cancer Prediction")

# Create input fields for each feature
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(feature, min_value=float(0), max_value=float(100), value=float(0))

# Button to make prediction
if st.button("Predict"):
    # Prepare input data
    input_data = np.array(list(inputs.values())).reshape(1, -1)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    # Display the result
    if prediction[0] == 0:
        st.success("The model predicts: Malignant")
    else:
        st.success("The model predicts: Benign")