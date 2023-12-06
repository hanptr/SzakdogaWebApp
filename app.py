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
from streamlit import session_state

transformer_model = load_model('./transformer_model.h5')
bidirectional_model = load_model('./bidirectional_model.h5')

# Initialize session state
if 'uploaded_data' not in session_state:
    session_state.uploaded_data = None

st.sidebar.title('Running intensity classification')

selected = st.sidebar.radio(
    'Choose an option:',
    ['Data analytics', 'Data Classification', 'Upload CSV']
)

if selected == 'Data analytics':
    # page title
    st.title('Analyzing the data')
    if session_state.uploaded_data is not None:
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(3, 1, figsize=(6, 12))

        sns.countplot(x='LABEL', data=session_state.uploaded_data, ax=axes[0])
        axes[0].set_title('Distribution of Labels')

        midpoint = len(session_state.uploaded_data) // 2
        first_half = session_state.uploaded_data.iloc[:midpoint]
        second_half = session_state.uploaded_data.iloc[midpoint:]

        label_colors = {'Fast': 'red', 'Slow': 'blue'}

        sns.countplot(x='LABEL', data=first_half, palette=label_colors, ax=axes[1])
        axes[1].set_title('Distribution of Labels in the First Half')

        sns.countplot(x='LABEL', data=second_half, palette=label_colors, ax=axes[2])
        axes[2].set_title('Distribution of Labels in the First Half')

        plt.subplots_adjust(hspace=0.5)
        
        st.pyplot(fig)

elif selected == 'Data Classification':
    # page title
    st.title('Classifying the data')

    selected = st.radio(
    'Please choose the desired model for classification:',
    ['Bidirectional', 'Transformer']
)

elif selected == 'Upload CSV':
    # page title
    st.title('Upload the desired CSV')

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Attempt to read the CSV file
            df = pd.read_csv(uploaded_file)
            st.write("File Uploaded Successfully!")

            # Save the uploaded data to session state
            session_state.uploaded_data = df

            # Display uploaded file as a DataFrame
            st.write("### Uploaded Data:")
            st.write(df)

        except pd.errors.ParserError as e:
            # Handle the ParserError
            st.error(f"Error reading CSV file: {e}")
