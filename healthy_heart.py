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

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# Specify models folder path
model_folder_path = 'models'

# Load models
def load_model(file_path):
    with open(file_path, 'rb') as model_file:
        return pickle.load(model_file)

most_stable_model = load_model(os.path.join(model_folder_path, 'random_forest_model.sav'))
X_train_scaled_model = load_model(os.path.join(model_folder_path, 'X_train_scaled.sav'))

# Set page configuration with wide layout and dark theme
st.set_page_config(
    page_title="Healthy Heart: A Heart Disease Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Specify image folder path
image_folder_path = 'images'

# Adding Image to web app
st.image(os.path.join(image_folder_path, 'banner.png'), use_column_width=True, width=800)

# Create input fields for user input
col1, col2, col3 = st.columns(3)

# Helper function for input fields
def create_input_field(label, min_value, max_value, value, step=1, key=None):
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
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))

# Define additional input fields
age = create_input_field("Age", 1, 110, 30, key="age_slider")
sex = st.radio("Sex", ['Male', 'Female'], horizontal=True)
chest_pain_type = st.selectbox("Chest Pain Type", ['ASY', 'ATA', 'NAP', 'TA'], key="chest_pain_type")
resting_bp = create_input_field("Resting Blood Pressure (mm Hg)", 80, 200, 120, key="resting_bp")
cholesterol = create_input_field("Cholesterol (mg/dL)", 100, 400, 200, key="cholesterol")
fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"), horizontal=True, key="fasting_bs")
fasting_bs = 1 if fasting_bs == "Yes" else 0  # Convert to 1 or 0
resting_ecg = st.radio("Resting ECG", ["LVH", "Normal", "ST"], horizontal=True)
max_hr = create_input_field("Max Heart Rate", 0, 300, 150)
exercise_angina = st.radio("Exercise Angina", ["Yes", "No"], horizontal=True)
exercise_angina = 1 if exercise_angina == "Yes" else 0
ldpeak = create_input_field("Oldpeak", -4, 10, 0,  step=1)
st_slope = st.radio("ST_Slope", ['Down', 'Flat', 'Up'], horizontal=True)

# Create a button to trigger prediction
if st.button("Predict"):
    # Preprocess user input
    input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': ldpeak,
        'ST_Slope': st_slope
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Preprocess the input data
    X_input_encoded = pd.get_dummies(input_df)
    X_input_scaled = X_train_scaled_model

    print(X_input_encoded)

    # Make prediction
    prediction = most_stable_model.predict(X_input_scaled)

    # Display the result
    st.write(f"The person has {prediction[0].lower()}")
# # -*- coding: utf-8 -*-
# """
# Created on Wed Sep 20 21:57:26 2023
#
# @author: Tenicka Norwood
# """
#
# import pandas as pd
# import pickle
# import os
# import streamlit as st
# import warnings
# warnings.filterwarnings("ignore")
# warnings.simplefilter(action ='ignore', category = DeprecationWarning)
# warnings.simplefilter(action ='ignore', category = FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
#
# # Specify models folder path
# model_folder_path = 'models'
#
# # Load the  random_forest_model
# model_path = os.path.join(model_folder_path, 'random_forest_model.sav')
# with open(model_path, 'rb') as model_file_1:
#     most_stable_model = pickle.load(model_file_1)
#
# # Load the model
# with open('models\X_train_scaled.sav', 'rb') as model_file_1:
#     X_train_scaled_model = pickle.load(model_file_1)
#
# # # Set page configuration with wide layout and dark theme
# # st.set_page_config(
# #     page_title="Healthy Heart: A Heart Disease Predictor",
# #     layout="wide",
# #     initial_sidebar_state="collapsed",
# #  )
#
# # Specify image folder model_path
# image_folder_path = 'images'
# # Adding Image to web app
# st.image(os.path.join(image_folder_path,'banner.png'), use_column_width=True, width=800)
# # Load the random_forest_model
# # model_path = os.path.join(model_folder_path, 'random_forest_model.joblib')
# # most_stable_model = joblib.load(model_path)
# #
# # # Load the model
# # scaled_model_path = os.path.join(model_folder_path, 'X_train_scaled.joblib')
# # X_train_scaled_model = joblib.load(scaled_model_path)
#
# # Set page configuration with wide layout and dark theme
# st.set_page_config(
#     page_title="Healthy Heart: A Heart Disease Predictor",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )
#
# # Specify image folder path
# image_folder_path = 'images'
#
# # Adding Image to web app
# st.image(f'{image_folder_path}/banner.png', use_column_width=True, width=800)
#
#
# # Create input fields for user input
# col1, col2, col3 = st.columns(3)
# with col1:
#     age = st.slider("Age", min_value=1, max_value=110, value=30)
#     sex = st.radio("Sex", ['Male', 'Female'], key="sex", index=0)
#
# with col2:
#     chest_pain_type = st.selectbox("Chest Pain Type", ['ASY', 'ATA', 'NAP', 'TA'])
#     resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1)
#
# with col3:
#     cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1)
#     fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))
#
# # Create input fields for user input
# age = st.slider("Age", min_value=1, max_value=110, value=30, key="age_slider")
# sex = st.radio("Sex", ['Male', 'Female'], horizontal=True)
# chest_pain_type = st.selectbox("Chest Pain Type", ['ASY', 'ATA', 'NAP', 'TA'], key="chest_pain_type")
# resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1, key="resting_bp")
# cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1, key="cholesterol")
# fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"), horizontal=True, key="fasting_bs")
# fasting_bs = 1 if fasting_bs == "Yes" else 0  # Convert to 1 or 0
# resting_ecg = st.radio("Resting ECG", ["LVH", "Normal", "ST"], horizontal=True)
# max_hr = st.number_input("Max Heart Rate", min_value=0, max_value=300, step=1, value=150)
# exercise_angina = st.radio("Exercise Angina", ["Yes", "No"], horizontal=True)
# exercise_angina = 1 if exercise_angina == "Yes" else 0
# ldpeak = st.number_input("Oldpeak", min_value=-4.0, max_value=10.0, step=0.1)
# st_slope = st.radio("ST_Slope", ['Down', 'Flat', 'Up'], horizontal=True)
#
# # Create a button to trigger prediction
# if st.button("Predict"):
#     # Preprocess user input
#     input_data = {
#         'Age': age,
#         'Sex': sex,
#         'ChestPainType': chest_pain_type,
#         'RestingBP': resting_bp,
#         'Cholesterol': cholesterol,
#         'FastingBS': fasting_bs,
#         'RestingECG': resting_ecg,
#         'MaxHR': max_hr,
#         'ExerciseAngina': exercise_angina,
#         'Oldpeak': ldpeak,
#         'ST_Slope': st_slope
#     }
#
#     # Create a DataFrame from the input data
#     input_df = pd.DataFrame([input_data])
#
#     # Preprocess the input data
#     X_input_encoded = pd.get_dummies(input_df)
#     X_input_scaled = X_train_scaled_model
#
#     print(X_input_encoded)
#
#     # Make prediction
#     prediction = most_stable_model.predict(X_input_scaled)
#
#     # Display the result
#     st.write(f"The person has {prediction[0].lower()}")
