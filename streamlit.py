import streamlit as st
from test import Alzheimer_Detector

def main():
    # Title
    st.title('Alzheimer\'s Detection')
    st.header('Input patient information:')
    st.text('')

    # Input feature
    gender = st.text_input('Enter sex: Male: 1, Female: 0')
    age = st.text_input('Enter age of patient')
    educ = st.text_input('Year of education')
    ses = st.text_input('Socioeconomic Status')
    mmse = st.text_input('Mini Mental State Examination')
    etiv = st.text_input('Estimated Total Intracranial Volume')
    nwbv = st.text_input('Normalize Whole Brain Volume')
    asf = st.text_input('Atlas Scaling Factor')

    patient_input = [gender, age, educ, ses, mmse, etiv, nwbv, asf]
    patient_input = map(float, patient_input)

    # Make prediction
    model = Alzheimer_Detector()
    predict = model.user_predict(patient_input)

    if st.button("Make prediction"):
        if predict == '0':
            st.success('Nondemented')
        else:
            st.danger('Demented')

if __name__ == '__main__':
    main()