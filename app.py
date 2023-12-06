import numpy as np
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import keras
import tensorflow as tf
import os
from keras.models import load_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit import session_state
from statistics import mode
from tensorflow.keras.utils import to_categorical

# Load models
transformer_model = load_model('./transformer_model.h5')
bidirectional_model = load_model('./bidirectional_model.h5')
session_state.uploaded_data = pd.read_csv('SENSOR_DATA1010.csv')

# Initialize session state
if 'uploaded_data' not in session_state:
    session_state.uploaded_data = None

st.sidebar.title('Running intensity classification')

# Create a more sophisticated dropdown menu for the main options
selected = st.sidebar.selectbox(
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
    data = session_state.uploaded_data
    # Preprocess
    data.loc[data['LABEL'] == 'Slow', 'LABEL'] = 0
    data.loc[data['LABEL'] == 'Fast', 'LABEL'] = 1

    if 'TIME' in data:
        data = data.drop(columns='TIME')

    data = data.astype(float)

    win_size = 10

    def df_to_X_y(df, window_size=60):
        df_as_np = df.to_numpy()
        X = []
        y = []
        for i in range(len(df_as_np)-window_size):
            row = [r for r in df_as_np[i:i+window_size]]
            X.append(row)
            label = df_as_np[i+window_size][-1]
            y.append(label)

        return np.array(X), np.array(y)
    
    X, y = df_to_X_y(data, win_size)

    X=X[12206:]
    y=y[12206:]
    
    #y_one_hot = to_categorical(y, num_classes=2)

    # Classification button
    if st.button('Perform Classification'):
        if data is not None:
            # Perform classification using the selected model
            # You need to replace this part with your actual classification logic
            # This is just a placeholder
            st.write("Performing classification...")

            # Placeholder for the classification result
            # Replace this with the actual classification result
            classification_result = model.predict(X)
            
            st.write("### Classification Result:")
            st.write(classification_result)
            
            true_labels = y
            predicted_labels = tf.math.round(classification_result)
            predicted_labels = np.argmax(predicted_labels, axis=1)

            num_classes = 2  # Replace with the actual number of classes in your problem
            classes = ['Slow', 'Fast']

            conf_matrix = tf.math.confusion_matrix(true_labels, predicted_labels, num_classes=num_classes)

            st.write("### Confusion Matrix:")
            st.write(conf_matrix)

            # Display raw counts in the confusion matrix
            st.write("### Confusion Matrix Visualization:")
            plt.figure(figsize=(len(classes), len(classes)))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            st.pyplot()


        else:
            st.write("Please upload data before classification!")

elif selected == 'Upload CSV':
    # Page title
    st.title('Upload the desired
