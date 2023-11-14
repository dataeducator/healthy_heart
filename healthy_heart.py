# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 21:57:26 2023
@author: Tenicka Norwood
"""

import pandas as pd
import pickle
import os
import streamlit as st
import warnings
import sklearn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# Load models
def load_model(file_path):
    with open(file_path, 'rb') as model_file:
        return pickle.load(model_file)

# Load the trained model and feature names
most_stable_model = load_model('models/most_stable_model.sav')
feature_names = load_model('models/feature_names.sav')
random_forest_model = load_model('models/random_forest_model.sav')

def load_scaler(file_path):
    with open(file_path, 'rb') as scaler_file:
        return pickle.load(scaler_file)

X_train_scaler = load_scaler('models/X_train_scaler.sav')


#print(feature_names)
# Set page configuration with wide layout and dark theme
st.set_page_config(
    page_title="Healthy Heart: A Heart Disease Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# Adding Image to web app
st.image('images/banner.png', use_column_width=True, width=800)

# Create input fields for user input
col1, col2, col3 = st.columns(3)

# Helper function for input fields
def create_input_field(label, min_value, max_value, value, step=1, key=None):
    return st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key)

def create_float_input_field(label, min_value, max_value, value, step=0.1, key=None):
    return st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key)
# Define input fields
with col1:
    age = st.slider("Age", min_value=1, max_value=110, value=30)
    sex = st.radio("Sex", ['Male', 'Female'], key="sex", index=0)

with col2:
    chest_pain_type = st.selectbox("Chest Pain Type", ['ASY', 'ATA', 'NAP', 'TA'])
    resting_bp = create_input_field("Resting Blood Pressure (mm Hg)", 80, 200, 120)

with col3:
    cholesterol = create_input_field("Cholesterol (mg/dL)", 100, 400, 200)
    max_hr = create_input_field("Max Heart Rate", 0, 300, 150)

fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"), horizontal=True, key="fasting_bs")
fasting_bs = 1 if fasting_bs == "Yes" else 0  # Convert to 1 or 0
resting_ecg = st.radio("Resting ECG", ["LVH", "Normal", "ST"], horizontal=True)
exercise_angina = st.radio("Exercise Angina", ["Yes", "No"], horizontal=True)
exercise_angina = 1 if exercise_angina == "Yes" else 0
ldpeak = create_float_input_field("Oldpeak", -4.0, 10.0, 0.0, step=0.5)
st_slope = st.radio("ST_Slope", ['Down', 'Flat', 'Up'], horizontal=True)

# Create a button to trigger prediction
if st.button("Predict"):
    # Preprocess user input
    input_data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': ldpeak,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingECG': resting_ecg,
        'ExerciseAngina': exercise_angina,
        'ST_Slope': st_slope
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])
    # print("user_data:")
    # print(input_df)
    # Ensure categorical variables have consistent encoding
    categories = {
        'Sex': ['M', 'F'],
        'ChestPainType': ['ASY', 'ATA', 'NAP', 'TA'],
        'RestingECG': ['LVH', 'Normal', 'ST'],
        'ExerciseAngina': ['N', 'Y'],
        'ST_Slope': ['Down', 'Flat', 'Up']
    }
    for col in input_df.columns:
        if col in categories:
            input_df[col] = pd.Categorical(input_df[col], categories=categories[col])

    # Reorder columns to match the order during training
    X_input_encoded = pd.get_dummies(input_df)[feature_names]
    scaled_model = X_train_scaler
    X_input_scaled = scaled_model.transform(X_input_encoded)
    prediction = most_stable_model.predict(X_input_scaled)

    # Convert the prediction to a string and display the result
    prediction_text = "not likely" if prediction[0] == 0 else "likely"
    st.write(f"The person is {prediction_text} to have heart disease.")
