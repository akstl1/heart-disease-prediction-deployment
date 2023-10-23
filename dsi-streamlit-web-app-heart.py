# Improt libraries
import streamlit as st
import pandas as pd
import joblib

# load our model pipeline object
model = joblib.load("heart_disease_model.joblib")

# insert a title for the app and instructions

st.title("Heart Disease Prediction Model")
st.subheader("Enter patient information and submit for likelihood of heart disease")

# age input form

age = st.number_input(
    label = "Enter the patient's age",
    min_value = 18,
    max_value = 120,
    value = 35) #default value



# gender input form

sex = st.radio(
    label = "Enter the patient's gender",
    options = ['M','F'])

if sex=='M':
    sex=1
else:
    sex=0

# chest pain input form

chest_pain_type = st.number_input(
    label = "Enter the patients's chest pain type (0-3)",
    min_value = 0,
    max_value = 3,
    value = 2)

resting_blood_pressure = st.number_input(
    label = "Enter the patients's resting blood pressure",
    min_value = 0,
    max_value = 200,
    value = 120)

serum_cholestoral = st.number_input(
    label = "Enter the patients's serum choleroral in mg/dl",
    min_value = 0,
    max_value = 200,
    value = 80)

fasting_blood_sugar = st.number_input(
    label = "Enter the patients's fasting blood sugar",
    min_value = 0,
    max_value = 1,
    value = 0)

resting_ecg = st.number_input(
    label = "Enter the patients's resting ECG (0-2)",
    min_value = 0,
    max_value = 2,
    value = 0)

max_hr = st.number_input(
    label = "Enter the patients's Max heart rate achieved",
    min_value = 0,
    max_value = 300,
    value = 150)

exercise_induced_angina = st.number_input(
    label = "Enter whether patient experienced Exercise Induced Angina (0-1) ",
    min_value = 0,
    max_value = 1,
    value = 0)

oldpeak = st.number_input(
    label = "Enter the patient's Oldpeak results (0-2) ",
    min_value = 0,
    max_value = 2,
    value = 0)

slope = st.number_input(
    label = "Enter the patient's Slope results (0-2) ",
    min_value = 0,
    max_value = 2,
    value = 0)

ca = st.number_input(
    label = "Enter the patient's ca results (0-4) ",
    min_value = 0,
    max_value = 4,
    value = 0)

thal = st.number_input(
    label = "Enter the patient's thal results (0-3) ",
    min_value = 0,
    max_value = 3,
    value = 0)

# submit inputs to model

if st.button("Submit For Prediction"):
    # store data into df for prediction
    new_data = pd.DataFrame({"age" : [age], "sex" : [sex], "chest_pain_type" : [chest_pain_type],"resting_blood_pressure" : [resting_blood_pressure],"serum_cholestoral" : [serum_cholestoral],"fasting_blood_sugar" : [fasting_blood_sugar],"resting_ecg" : [resting_ecg],"max_hr" : [max_hr],"exercise_induced_angina" : [exercise_induced_angina],"oldpeak" : [oldpeak],"slope" : [slope],"ca" : [ca],"thal" : [thal]})
    
    # apply model pipeline to the input data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # output prediction
    st.subheader(f"Based on these customer attributes, our model predicts a heart disease probability of {pred_proba: .0%}")