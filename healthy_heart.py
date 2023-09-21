# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 21:57:26 2023

@author: Tenicka Norwood
"""

import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#import statsmodels.api as sm
#import sklearn
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import make_scorer, recall_score
#from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.feature_selection import RFE
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from tensorflow.keras import layers, models
#from tensorflow.keras.metrics import BinaryAccuracy, Recall
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#import tensorflow as tf
import pickle
import streamlit as st
#from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
#from keras.models import Sequential
#from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
#from keras.metrics import Precision, Accuracy, Recall
#from xgboost import XGBClassifier
#from BinaryClassifier import BinaryClassifier
#import tensorflow.keras.models as models


# Used for working with the z-score
#from scipy import stats

# Used for working with long url
#from urllib.parse import urlencode


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action ='ignore', category = DeprecationWarning)
warnings.simplefilter(action ='ignore', category = FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


# Load the  random_forest_model
with open('random_forest_model.sav', 'rb') as model_file_1:
    most_stable_model = pickle.load(model_file_1)

# Load the model
with open('X_train_scaled.sav', 'rb') as model_file_1:
    X_train_scaled_model = pickle.load(model_file_1)
   
# Set page configuration with wide layout and dark theme
st.set_page_config(
    page_title="Healthy Heart: A Heart Disease Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
 )

# Adding Image to web app
st.image('banner.png', use_column_width=True, width=800)

# Create input fields for user input
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", min_value=1, max_value=110, value=30)
    sex = st.radio("Sex", ['Male', 'Female'], key="sex", index=0)

with col2:
    chest_pain_type = st.selectbox("Chest Pain Type", ['ASY', 'ATA', 'NAP', 'TA'])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1)

with col3:
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1)
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))

# Create input fields for user input
age = st.slider("Age", min_value=1, max_value=110, value=30, key="age_slider")
sex = st.radio("Sex", ['Male', 'Female'], horizontal=True)
chest_pain_type = st.selectbox("Chest Pain Type", ['ASY', 'ATA', 'NAP', 'TA'], key="chest_pain_type")
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1, key="resting_bp")
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1, key="cholesterol")
fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"), horizontal=True, key="fasting_bs")
fasting_bs = 1 if fasting_bs == "Yes" else 0  # Convert to 1 or 0
resting_ecg = st.radio("Resting ECG", ["LVH", "Normal", "ST"], horizontal=True)
max_hr = st.number_input("Max Heart Rate", min_value=0, max_value=300, step=1, value=150)
exercise_angina = st.radio("Exercise Angina", ["Yes", "No"], horizontal=True)
exercise_angina = 1 if exercise_angina == "Yes" else 0
ldpeak = st.number_input("Oldpeak", min_value=-4.0, max_value=10.0, step=0.1)
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