import numpy as np
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import keras
import tensorflow
import os
from keras.models import load_model

transformer_model = load_model('transformer_model.h5')
bidirectional_model = load_model('bidirectional_model.h5')

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
