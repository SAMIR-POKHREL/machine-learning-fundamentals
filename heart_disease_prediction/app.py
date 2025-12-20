import streamlit as st
import pandas as pd
import joblib as jb

model=jb.load("KNN_heart.pkl")
scaler=jb.load("heart_scaler.pkl")
expected_columns=jb.load("heart_columns.pkl")

st.title("Heart stroke prediction via samir")
st.markdown("provide the following info")

age=st.slider("Age",18,100,40)

sex=st.selectbox("SEX",['M','F'])

chestpain=st.selectbox("Chest pain Type",['ATA','NAP','TA','ASY'])

resting_bp=st.number_input("Resting blood pressure(mm/Hg)",80,200,120)

cholesterol=st.number_input("Cholesterol (Mg/Dl)",100,600,200)

FastingBS=st.selectbox("Fasting Blood sugar >120 mg/dl",[0,1])

RestingECG=st.selectbox("resting ECG",['normal','ST','LVH'])

Maxhr=st.slider("max heart rate",60,220,150)

ExerciseAngina	=st.selectbox("Exercise-Angina",['Y','N'])

Oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

st_slope=st.selectbox("ST slope",['up','flat','down'])

if st.button("predict"):
    rawinput={
        'Age':age,
        'Sex':sex,
        'ChestPainType':chestpain,
        'RestingBP':resting_bp,
        'Cholesterol':cholesterol,
        'FastingBS':FastingBS,
        'RestingECG':RestingECG,
        'MaxHR':Maxhr,
        'ExerciseAngina':ExerciseAngina,
        'Oldpeak':Oldpeak,
        'ST_Slope':st_slope
    }


    input_df=pd.DataFrame([rawinput])


    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0
        
    
    input_df=input_df[expected_columns]

    
    scaled_input=scaler.transform(input_df)
    prediction=model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("💀 High risk of heat disease ")
    else:
        st.success("😊 low risk of heat disease ")
