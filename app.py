import numpy as np
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import keras
import tensorflow
import os
from keras.models import load_model
import pandas as pd

transformer_model = load_model('./transformer_model.h5')
bidirectional_model = load_model('./bidirectional_model.h5')

st.title('Running intensity classification')

selected = st.radio(
    'Choose an option:',
    ['Data analytics', 'Data Classification', 'Upload CSV']
)

if selected == 'Data analytics':
    # page title
    st.title('Analyzing the data')

if selected == 'Data Classification':
    # page title
    st.title('Classifying the data')

if selected == 'Upload CSV':
    # page title
    st.title('Upload the desired CSV')

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Attempt to read the CSV file
            df = pd.read_csv(uploaded_file)
            st.write("File Uploaded Successfully!")

            # Display uploaded file as a DataFrame
            st.write("### Uploaded Data:")
            st.write(df)

        except pd.errors.ParserError as e:
            # Handle the ParserError
            st.error(f"Error reading CSV file: {e}")

