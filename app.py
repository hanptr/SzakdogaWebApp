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

# Load models
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
    # Page title
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
        axes[2].set_title('Distribution of Labels in the Second Half')

        plt.subplots_adjust(hspace=0.5)
        
        st.pyplot(fig)

elif selected == 'Data Classification':
    # Page title
    st.title('Classifying the data')

    model_to_use = st.radio(
        'Please choose the desired model for classification:',
        ['Bidirectional', 'Transformer']
    )
    if model_to_use == 'Bidirectional':
        model = load_model('bidirectional_model.h5')
    elif model_to_use == 'Transformer':
        model = load_model('transformer_model.h5')

    # Preprocess
    session_state.uploaded_data.loc[session_state.uploaded_data['LABEL'] == 'Slow', 'LABEL'] = 0
    session_state.uploaded_data.loc[session_state.uploaded_data['LABEL'] == 'Fast', 'LABEL'] = 1

    session_state.uploaded_data = session_state.uploaded_data.astype(float)

    session_state.uploaded_data = session_state.uploaded_data.drop(columns='TIME')

    win_size = 10

    def df_to_X_y(df, window_size=60):
        df_as_np = df.to_numpy()
        X = []
        y = []
        for i in range(len(df_as_np)-window_size):
            row = [r[:-1] for r in df_as_np[i:i+window_size]]
            X.append(row)
            label = mode(df_as_np[i:i+window_size,-1])
            y.append(label)
        return np.array(X), np.array(y)
    
    X, y = df_to_X_y(session_state.uploaded_data, win_size)

    # Classification button
    if st.button('Perform Classification'):
        if session_state.uploaded_data is not None:
            # Perform classification using the selected model
            # You need to replace this part with your actual classification logic
            # This is just a placeholder
            st.write("Performing classification...")

            # Placeholder for the classification result
            # Replace this with the actual classification result
            classification_result = model.predict(session_state.uploaded_data)
            
            st.write("### Classification Result:")
            st.write(classification_result)

        else:
            st.write("Please upload data before classification!")

elif selected == 'Upload CSV':
    # Page title
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
