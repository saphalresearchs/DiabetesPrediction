import streamlit as st
import pickle
import numpy as np

#Open the file in read binary mode
with open("model.pkl","rb") as file:
    model = pickle.load(file)

scaler = pickle.load(open("scaler.pkl", "rb"))

#Title
st.title("Diabetes Prediction System")

st.write("Enter the required details to check for diabetes prediction.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, step=0.1)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.001)
age = st.number_input("Age", min_value=0, max_value=120, step=1)

# Prediction
if st.button("Predict"):
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age])
    input_data_reshaped = input_data.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data)
    
    if prediction[0] == 0:
        st.success("Congrats! You don't have diabetes.")
    else:
        st.error("You have diabetes, please take care.")

