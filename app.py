import numpy as np
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import keras
import tensorflow
import os
from keras.models import load_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

transformer_model = load_model('./transformer_model.h5')
bidirectional_model = load_model('./bidirectional_model.h5')

uploaded_file = None  # Initialize uploaded_file

st.sidebar.title('Running intensity classification')

selected = st.sidebar.radio(
    'Choose an option:',
    ['Data analytics', 'Data Classification', 'Upload CSV']
)

if selected == 'Data analytics':
    # page title
    st.title('Analyzing the data')
    if uploaded_file is not None:
        sns.countplot(x='LABEL', data=uploaded_file)
        midpoint = len(allinone) // 2

        first_half = uploaded_file.iloc[:midpoint]
        second_half = uploaded_file.iloc[midpoint:]

        label_colors = {'Fast': 'red', 'Slow': 'blue'}

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.countplot(x='LABEL', data=first_half, palette=label_colors)
        plt.title('Distribution of Labels in the First Half')

        plt.subplot(1, 2, 2)
        sns.countplot(x='LABEL', data=second_half, palette=label_colors)
        plt.title('Distribution of Labels in the Second Half')

        plt.show()

elif selected == 'Data Classification':
    # page title
    st.title('Classifying the data')

elif selected == 'Upload CSV':
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
