import streamlit as st
from test import Alzheimer_Detector

def main():
    # Title
    st.title('Alzheimer\'s Detection')
    st.header('Input patient information:')
    st.text('')

    # Input feature
    gender = st.number_input('Enter sex: Male: 1, Female: 0')
    age = st.number_input('Enter age of patient')
    educ = st.number_input('Year of education')
    ses = st.number_input('Socioeconomic Status')
    mmse = st.number_input('Mini Mental State Examination')
    etiv = st.number_input('Estimated Total Intracranial Volume')
    nwbv = st.number_input('Normalize Whole Brain Volume')
    asf = st.number_input('Atlas Scaling Factor', step=1.,format="%.2f")

    patient_input = [gender, age, educ, ses, mmse, etiv, nwbv, asf]

    # Make prediction
    model = Alzheimer_Detector()
    predict = model.user_predict(patient_input)
    print(patient_input)
    if st.button("Make prediction"):
        if predict == '0':
            st.success('Nondemented')
        else:
            st.warning('Demented')

if __name__ == '__main__':
    main()
