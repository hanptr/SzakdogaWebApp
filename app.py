import numpy as np
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import keras
import tensorflow
import os

file_path1 = os.path.join(os.getcwd(), 'transfromer_model.sav')

if os.path.exists(file_path1):
    transformer_model = pickle.load(open(file_path1, 'rb'))
else:
    st.error(f"File not found: {file_path1}")

file_path2 = os.path.join(os.getcwd(), 'bidirectional_model.sav')

if os.path.exists(file_path2):
    transformer_model = pickle.load(open(file_path2, 'rb'))
else:
    st.error(f"File not found: {file_path2}")

with st.sidebar:
    
   selected = option_menu('Running intensity classification',
                         
                         ['Data analytics',
                          'Classification',
                          'Upload CSV'],
                         icons=['task','heart','upload'],
                         default_index=0)

   
   if (selected == 'Data analytics'):
       # page title
       st.title('Analyzing the data')
       
   if (selected == 'Data Classification'):
       # page title
       st.title('Classifying the data')
       
   if (selected == 'Upload CSV'):
       # page title
       st.title('Upload the desired CSV')
